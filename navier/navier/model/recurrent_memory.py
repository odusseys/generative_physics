# This file is executed by navier/autoreg_windowed_lora.py.
# Keep shared globals in that compatibility module namespace.

class CausalResidualConv3dBlock(nn.Module):
    """Depthwise causal temporal Conv3d residual block for reduced token grids."""

    def __init__(self, channels: int, kernel_size: tuple[int, int, int] = (3, 3, 3)):
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = tuple(int(value) for value in kernel_size)
        if len(self.kernel_size) != 3:
            raise ValueError("kernel_size must be a 3-tuple")
        if any(value < 1 for value in self.kernel_size):
            raise ValueError(f"kernel_size entries must be positive: {kernel_size}")
        self.depthwise = nn.Conv3d(
            self.channels,
            self.channels,
            kernel_size=self.kernel_size,
            groups=self.channels,
        )
        self.pointwise = nn.Conv3d(self.channels, self.channels, kernel_size=1)
        self.activation = nn.SiLU()
        nn.init.zeros_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        temporal, height, width = self.kernel_size
        x_padded = F.pad(
            x.to(dtype=self.depthwise.weight.dtype),
            (
                width // 2,
                width - 1 - width // 2,
                height // 2,
                height - 1 - height // 2,
                temporal - 1,
                0,
            ),
        )
        residual = self.pointwise(self.activation(self.depthwise(x_padded)))
        return x + residual.to(dtype=input_dtype)


class StateSpaceMemory(nn.Module):
    """Frame-level SSM memory for one Wan self-attention layer."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        *,
        decay_init: Optional[float] = None,
        input_init: Optional[float] = None,
        output_init: Optional[float] = None,
        skip_init: Optional[float] = None,
        rho_init: Optional[float] = None,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.dim = self.num_heads * self.head_dim
        self.recurrent_dim = int(SSM_RECURRENT_DIM)
        if self.recurrent_dim < 1:
            raise ValueError("SSM_RECURRENT_DIM must be positive")
        self.state_mixer_expansion = int(SSM_STATE_MIXER_EXPANSION)
        if self.state_mixer_expansion < 1:
            raise ValueError("SSM_STATE_MIXER_EXPANSION must be positive")
        decay_init = SSM_DECAY_INIT if decay_init is None else decay_init
        input_init = SSM_INPUT_INIT if input_init is None else input_init
        output_init = SSM_OUTPUT_INIT if output_init is None else output_init
        skip_init = SSM_SKIP_INIT if skip_init is None else skip_init
        rho_init = SSM_RHO_INIT if rho_init is None else rho_init
        if SSM_FREEZE_RHO_ZERO:
            rho_init = 0.0
        decay_init = min(max(float(decay_init), 1e-6), 1.0 - 1e-6)
        input_init = min(max(float(input_init), 1e-6), 1.0 - 1e-6)
        self.query_projection = nn.Linear(self.dim, self.recurrent_dim)
        self.input_projection = nn.Linear(self.dim, self.recurrent_dim)
        if bool(SSM_USE_CONVOLUTIONS):
            self.pre_ssm_conv = CausalResidualConv3dBlock(self.recurrent_dim)
        else:
            self.pre_ssm_conv = None
        self.post_ssm_resnet = CausalResidualConv3dBlock(self.recurrent_dim)
        self.output_projection = nn.Linear(self.recurrent_dim, self.dim)
        self.state_mixer = nn.Sequential(
            nn.Linear(
                self.recurrent_dim,
                self.state_mixer_expansion * self.recurrent_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                self.state_mixer_expansion * self.recurrent_dim,
                self.recurrent_dim,
            ),
        )
        nn.init.zeros_(self.state_mixer[-1].weight)
        nn.init.zeros_(self.state_mixer[-1].bias)
        self.decay_logit = nn.Parameter(
            torch.full(
                (self.recurrent_dim,),
                math.log(decay_init / (1.0 - decay_init)),
            )
        )
        self.input_logit = nn.Parameter(
            torch.full(
                (self.recurrent_dim,),
                math.log(input_init / (1.0 - input_init)),
            )
        )
        self.output_gain = nn.Parameter(
            torch.full((self.recurrent_dim,), float(output_init))
        )
        self.skip_gain = nn.Parameter(
            torch.full((self.recurrent_dim,), float(skip_init))
        )
        self.rho = nn.Parameter(
            torch.tensor(float(rho_init)),
            requires_grad=not bool(SSM_FREEZE_RHO_ZERO),
        )
        self.output_norm = nn.LayerNorm(self.recurrent_dim, elementwise_affine=False)

    def _query_scale(self) -> float:
        if SSM_QUERY_SCALE is not None:
            return float(SSM_QUERY_SCALE)
        return float(self.recurrent_dim) ** -0.5

    def _mix_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.state_mixer(
            state.to(device=state.device, dtype=self.state_mixer[0].weight.dtype)
        ).float()

    def _shared_full_grid_shape(
        self,
        grid_sizes: torch.Tensor,
        seq_lens: Optional[torch.Tensor],
        *,
        batch_size: int,
        seq_len: int,
    ) -> Optional[tuple[int, int, int]]:
        grids = [
            (int(frame_count), int(grid_h), int(grid_w))
            for frame_count, grid_h, grid_w in grid_sizes.tolist()
        ]
        if len(grids) != int(batch_size) or not grids:
            return None
        first = grids[0]
        if any(grid != first for grid in grids):
            return None
        frame_count, grid_h, grid_w = first
        if frame_count * grid_h * grid_w != int(seq_len):
            return None
        if seq_lens is not None and not torch_compiler_is_compiling():
            if not bool(torch.all(seq_lens.to(device=grid_sizes.device) == int(seq_len))):
                return None
        return first

    def _apply_grid_conv(
        self,
        x: torch.Tensor,
        conv: CausalResidualConv3dBlock,
        *,
        grid_sizes: torch.Tensor,
        seq_lens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        shared_shape = self._shared_full_grid_shape(
            grid_sizes, seq_lens, batch_size=batch_size, seq_len=seq_len
        )
        if shared_shape is not None:
            frame_count, grid_h, grid_w = shared_shape
            grid = (
                x.reshape(batch_size, frame_count, grid_h, grid_w, channels)
                .permute(0, 4, 1, 2, 3)
                .contiguous()
            )
            grid = conv(grid)
            return (
                grid.permute(0, 2, 3, 4, 1)
                .reshape(batch_size, seq_len, channels)
                .contiguous()
            )

        output = x.clone()
        for batch_index, (frame_count, grid_h, grid_w) in enumerate(
            grid_sizes.tolist()
        ):
            spatial_tokens = int(grid_h) * int(grid_w)
            valid_tokens = int(frame_count) * spatial_tokens
            if seq_lens is not None and not torch_compiler_is_compiling():
                valid_tokens = min(valid_tokens, int(seq_lens[batch_index].item()))
            if spatial_tokens <= 0 or valid_tokens <= 0:
                continue
            if valid_tokens > seq_len:
                raise ValueError(
                    f"Grid has {valid_tokens} valid tokens, but seq_len is {seq_len}"
                )
            valid_frames = valid_tokens // spatial_tokens
            if valid_frames <= 0:
                continue
            valid_tokens = valid_frames * spatial_tokens
            grid = (
                x[batch_index, :valid_tokens]
                .view(valid_frames, int(grid_h), int(grid_w), channels)
                .permute(3, 0, 1, 2)
                .unsqueeze(0)
                .contiguous()
            )
            grid = conv(grid)
            output[batch_index, :valid_tokens] = (
                grid.squeeze(0)
                .permute(1, 2, 3, 0)
                .reshape(valid_tokens, channels)
            )
        return output

    def _forward_shared_grid(
        self,
        q_float: torch.Tensor,
        v_float: torch.Tensor,
        *,
        grid_shape: tuple[int, int, int],
        decay: torch.Tensor,
        input_gain: torch.Tensor,
        output_gain: torch.Tensor,
        skip_gain: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, channels = q_float.shape
        frame_count, grid_h, grid_w = grid_shape
        spatial_tokens = int(grid_h) * int(grid_w)
        q_frames = q_float.reshape(batch_size, frame_count, spatial_tokens, channels)
        v_frames = v_float.reshape(batch_size, frame_count, spatial_tokens, channels)
        decay = decay.view(1, 1, channels)
        input_gain = input_gain.view(1, 1, channels)
        output_gain = output_gain.view(1, 1, channels)
        skip_gain = skip_gain.view(1, 1, channels)
        output_frames = []
        state = q_float.new_zeros(batch_size, spatial_tokens, channels)
        for frame_index in range(int(frame_count)):
            frame_input = v_frames[:, frame_index]
            state = (
                decay * state
                + self._mix_state(state)
                + input_gain * frame_input
            )
            read = output_gain * state + skip_gain * frame_input
            output_frames.append(q_frames[:, frame_index] * read)
        return torch.stack(output_frames, dim=1).reshape_as(q_float)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        grid_sizes: torch.Tensor,
        seq_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del k
        batch_size, seq_len, num_heads, head_dim = q.shape
        if v.shape != q.shape:
            raise ValueError("q and v must have matching shapes")
        if num_heads != self.num_heads or head_dim != self.head_dim:
            raise ValueError(
                f"StateSpaceMemory expected heads/head_dim {(self.num_heads, self.head_dim)}, "
                f"got {(num_heads, head_dim)}"
            )

        q_flat = q.flatten(2).to(dtype=self.query_projection.weight.dtype)
        v_flat = v.flatten(2).to(dtype=self.input_projection.weight.dtype)
        q_float = self.query_projection(q_flat).float() * self._query_scale()
        v_float = self.input_projection(v_flat).float()
        if self.pre_ssm_conv is not None:
            v_float = self._apply_grid_conv(
                v_float,
                self.pre_ssm_conv,
                grid_sizes=grid_sizes,
                seq_lens=seq_lens,
            )
        decay = torch.sigmoid(self.decay_logit).to(device=q.device)
        input_gain = torch.sigmoid(self.input_logit).to(device=q.device)
        output_gain = self.output_gain.to(device=q.device, dtype=q_float.dtype)
        skip_gain = self.skip_gain.to(device=q.device, dtype=q_float.dtype)
        shared_shape = self._shared_full_grid_shape(
            grid_sizes, seq_lens, batch_size=batch_size, seq_len=seq_len
        )
        if shared_shape is not None:
            output = self._forward_shared_grid(
                q_float,
                v_float,
                grid_shape=shared_shape,
                decay=decay,
                input_gain=input_gain,
                output_gain=output_gain,
                skip_gain=skip_gain,
            )
        else:
            output = q_float.new_zeros(batch_size, seq_len, self.recurrent_dim)
            decay_view = decay.view(1, self.recurrent_dim)
            input_gain_view = input_gain.view(1, self.recurrent_dim)
            output_gain_view = output_gain.view(1, self.recurrent_dim)
            skip_gain_view = skip_gain.view(1, self.recurrent_dim)
            for batch_index, (frame_count, grid_h, grid_w) in enumerate(
                grid_sizes.tolist()
            ):
                spatial_tokens = int(grid_h) * int(grid_w)
                valid_tokens = int(frame_count) * spatial_tokens
                if seq_lens is not None and not torch_compiler_is_compiling():
                    valid_tokens = min(valid_tokens, int(seq_lens[batch_index].item()))
                if spatial_tokens <= 0 or valid_tokens <= 0:
                    continue
                if valid_tokens > seq_len:
                    raise ValueError(
                        f"Grid has {valid_tokens} valid tokens, but seq_len is {seq_len}"
                    )

                valid_frames = valid_tokens // spatial_tokens
                state = q_float.new_zeros(spatial_tokens, self.recurrent_dim)
                sample_output = []
                for frame_index in range(valid_frames):
                    start = frame_index * spatial_tokens
                    end = start + spatial_tokens
                    q_frame = q_float[batch_index, start:end]
                    v_frame = v_float[batch_index, start:end]
                    frame_input = v_frame
                    state = (
                        decay_view * state
                        + self._mix_state(state)
                        + input_gain_view * frame_input
                    )
                    read = output_gain_view * state + skip_gain_view * frame_input
                    sample_output.append(q_frame * read)

                if sample_output:
                    output[batch_index, : valid_frames * spatial_tokens] = torch.cat(
                        sample_output, dim=0
                    )

        output = self.output_norm(output)
        output = self._apply_grid_conv(
            output,
            self.post_ssm_resnet,
            grid_sizes=grid_sizes,
            seq_lens=seq_lens,
        )
        output = self.output_projection(
            output.to(dtype=self.output_projection.weight.dtype)
        )
        output = output.view(batch_size, seq_len, num_heads, head_dim)
        return (self.rho.to(device=q.device, dtype=output.dtype) * output).to(
            dtype=q.dtype
        )
