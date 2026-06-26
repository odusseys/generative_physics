# This file is executed by navier/autoreg_windowed_lora.py.
# Keep shared globals in that compatibility module namespace.

def load_wan_model_module(wan_repo_root: Path = WAN_REPO_ROOT):
    """Load Wan modules without importing wan/__init__.py optional extras."""
    _, modules_root, _ = _ensure_wan_package(wan_repo_root)
    attention_module = _load_module("wan.modules.attention", modules_root / "attention.py")
    attention_module.USE_FLEX_ATTENTION = bool(USE_FLEX_ATTENTION)
    attention_module.FLEX_ATTENTION_BLOCK_SIZE = FLEX_ATTENTION_BLOCK_SIZE
    attention_module.FLEX_ATTENTION_DYNAMIC = bool(FLEX_ATTENTION_DYNAMIC)
    attention_module.FLEX_ATTENTION_RECOMPILE_LIMIT = FLEX_ATTENTION_RECOMPILE_LIMIT
    module = _load_module("wan.modules.model", modules_root / "model.py")
    return module


def load_wan_vae_class(wan_repo_root: Path = WAN_REPO_ROOT):
    _, modules_root, _ = _ensure_wan_package(wan_repo_root)
    return _load_module("wan.modules.vae2_2", modules_root / "vae2_2.py").Wan2_2_VAE


def load_wan_scheduler_module(wan_repo_root: Path = WAN_REPO_ROOT):
    _, _, utils_root = _ensure_wan_package(wan_repo_root)
    return _load_module(
        "wan.utils.fm_solvers_unipc", utils_root / "fm_solvers_unipc.py"
    )


wan_model_module = load_wan_model_module()
wan_scheduler_module = load_wan_scheduler_module()
wan_attention_module = sys.modules["wan.modules.attention"]
WanModel = wan_model_module.WanModel
FlowUniPCMultistepScheduler = wan_scheduler_module.FlowUniPCMultistepScheduler
rope_params = wan_model_module.rope_params
frame_window_mask = wan_attention_module.frame_window_mask
frame_window_attention = wan_attention_module.frame_window_attention
flex_attention_available = wan_attention_module.flex_attention_available


def make_wan_flow_scheduler(num_steps: int, device: Optional[torch.device] = None):
    _validate_flow_matching_config()
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=int(WAN_NUM_TRAIN_TIMESTEPS),
        shift=1,
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(
        int(num_steps),
        device=device,
        shift=float(WAN_SAMPLE_SHIFT),
    )
    return scheduler


def load_wan_transformer(
    checkpoint_dir: Path = WAN_CHECKPOINT_DIR,
    device: torch.device = DEVICE,
    dtype: torch.dtype = PARAM_DTYPE,
) -> WanModel:
    checkpoint_dir = Path(checkpoint_dir)
    try:
        model = WanModel.from_pretrained(
            str(checkpoint_dir),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    except Exception as exc:
        warnings.warn(
            f"Diffusers from_pretrained failed ({exc}); falling back to manual shard load."
        )
        config = json.loads((checkpoint_dir / "config.json").read_text())
        config.pop("_class_name", None)
        config.pop("_diffusers_version", None)
        model = WanModel(**config)
        state = {}
        for shard in sorted(
            checkpoint_dir.glob("diffusion_pytorch_model-*.safetensors")
        ):
            state.update(load_file(str(shard)))
        model.load_state_dict(state, strict=True)
        del state

    model.eval().requires_grad_(False)
    return model.to(device=device, dtype=dtype)


class PatchEmbeddingLoRA(nn.Module):
    def __init__(
        self,
        patch_embedding: nn.Conv3d,
        *,
        rank: int,
        alpha: Optional[float] = None,
    ):
        super().__init__()
        rank = int(rank)
        if rank < 1:
            raise ValueError(f"patch embedding LoRA rank must be positive, got {rank}")
        alpha = rank if alpha is None else float(alpha)
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(rank)
        self.down = nn.Conv3d(
            patch_embedding.in_channels,
            rank,
            kernel_size=patch_embedding.kernel_size,
            stride=patch_embedding.stride,
            padding=patch_embedding.padding,
            dilation=patch_embedding.dilation,
            groups=1,
            bias=False,
            padding_mode=patch_embedding.padding_mode,
        )
        self.up = nn.Conv3d(rank, patch_embedding.out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.down.weight
        return self.up(self.down(x.to(dtype=weight.dtype))) * self.scaling


class TimestepAdaLNLoRA(nn.Module):
    def __init__(
        self,
        output_linear: nn.Linear,
        *,
        rank: int,
        alpha: Optional[float] = None,
    ):
        super().__init__()
        rank = int(rank)
        if rank < 1:
            raise ValueError(f"timestep AdaLN LoRA rank must be positive, got {rank}")
        alpha = rank if alpha is None else float(alpha)
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(rank)
        self.down = nn.Linear(output_linear.in_features, rank, bias=False)
        self.up = nn.Linear(rank, output_linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.down.weight
        return self.up(self.down(x.to(dtype=weight.dtype))) * self.scaling


def attach_patch_embedding_lora(model: WanModel) -> WanModel:
    rank = int(PATCH_EMBEDDING_LORA_R)
    if rank < 1:
        return model
    base = base_model(model)
    patch_embedding = base.patch_embedding
    existing = getattr(base, "patch_embedding_lora", None)
    if existing is not None and int(getattr(existing, "rank", rank)) == rank:
        return model
    adapter = PatchEmbeddingLoRA(
        patch_embedding,
        rank=rank,
        alpha=PATCH_EMBEDDING_LORA_ALPHA,
    ).to(device=patch_embedding.weight.device, dtype=patch_embedding.weight.dtype)
    base.patch_embedding_lora = adapter
    return model


def attach_timestep_adaln_lora(model: WanModel) -> WanModel:
    rank = int(TIMESTEP_ADALN_LORA_R)
    if rank < 1:
        return model
    base = base_model(model)
    output_linear = base.time_projection[-1]
    if not isinstance(output_linear, nn.Linear):
        raise TypeError(
            "Wan time_projection[-1] must be nn.Linear to attach timestep AdaLN LoRA"
        )
    existing = getattr(base, "timestep_adaln_lora", None)
    if existing is not None and int(getattr(existing, "rank", rank)) == rank:
        return model
    adapter = TimestepAdaLNLoRA(
        output_linear,
        rank=rank,
        alpha=TIMESTEP_ADALN_LORA_ALPHA,
    ).to(device=output_linear.weight.device, dtype=output_linear.weight.dtype)
    base.timestep_adaln_lora = adapter
    return model


def configure_wan_conditioning(model: WanModel) -> WanModel:
    """Add trainable conditioning modules used by the native Wan forward."""
    dim = model.dim
    device = next(model.parameters()).device
    model.timestep_conditioning_enabled = True

    def make_first_frame_adaln_adapter() -> nn.Sequential:
        adapter = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, int(FIRST_FRAME_CONDITIONING_DIM)),
            nn.SiLU(),
            nn.Linear(int(FIRST_FRAME_CONDITIONING_DIM), 6 * dim),
        )
        for module in adapter.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        nn.init.normal_(adapter[-1].weight, mean=0.0, std=1e-4)
        return adapter

    active_layers = sorted(
        set(range(len(model.blocks)))
        - dropped_transformer_layer_indices(len(model.blocks))
    )
    model.dropped_transformer_layers = set(
        dropped_transformer_layer_indices(len(model.blocks))
    )
    model.first_frame_adaln = nn.ModuleDict(
        {str(index): make_first_frame_adaln_adapter() for index in active_layers}
    ).to(device=device, dtype=torch.float32)

    model.physics_adaln = nn.ModuleList(
        [
            nn.Sequential(
                nn.Linear(2, 32),
                nn.SiLU(),
                nn.Linear(32, 6 * dim),
            )
            for _ in model.blocks
        ]
    ).to(device=device, dtype=torch.float32)
    for adaln in model.physics_adaln:
        nn.init.zeros_(adaln[-1].weight)
        nn.init.zeros_(adaln[-1].bias)
    return model


def add_lora_to_wan(model: WanModel):
    exclude_modules = dropped_transformer_layer_regex(transformer_layer_count(model))
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
        exclude_modules=exclude_modules,
    )
    peft_model = get_peft_model(model, config)
    base = peft_model.get_base_model()
    for parameter in physics_conditioning_parameters(base):
        parameter.requires_grad_(True)
    for parameter in first_frame_conditioning_parameters(base):
        parameter.requires_grad_(True)
    for parameter in timestep_adaln_lora_parameters(base):
        parameter.requires_grad_(True)
    return peft_model


def base_model(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def transformer_layer_count(model) -> int:
    blocks = getattr(base_model(model), "blocks", None)
    return 0 if blocks is None else len(blocks)


def _parse_layer_index_spec(spec) -> list[int]:
    if spec is None:
        return []
    if isinstance(spec, str):
        text = spec.strip()
        if not text or text.lower() in {"none", "off", "false"}:
            return []
        parts = re.split(r"[\s,]+", text)
        indices = []
        for part in parts:
            if not part:
                continue
            if "-" in part:
                start_text, end_text = part.split("-", 1)
                start = int(start_text)
                end = int(end_text)
                step = 1 if end >= start else -1
                indices.extend(range(start, end + step, step))
            else:
                indices.append(int(part))
        return indices
    return [int(index) for index in spec]


def dropped_transformer_layer_indices(total_layers: Optional[int] = None) -> set[int]:
    indices = set(_parse_layer_index_spec(TRANSFORMER_DROPPED_LAYERS))
    if total_layers is None:
        return indices
    total_layers = int(total_layers)
    invalid = sorted(index for index in indices if index < 0 or index >= total_layers)
    if invalid:
        raise ValueError(
            f"TRANSFORMER_DROPPED_LAYERS contains out-of-range indices {invalid} "
            f"for {total_layers} transformer layers"
        )
    return indices


def active_transformer_layer_indices(total_layers: int) -> list[int]:
    total_layers = int(total_layers)
    dropped = dropped_transformer_layer_indices(total_layers)
    return [index for index in range(total_layers) if index not in dropped]


def dropped_transformer_layer_regex(total_layers: Optional[int] = None) -> Optional[str]:
    indices = sorted(dropped_transformer_layer_indices(total_layers))
    if not indices:
        return None
    alternation = "|".join(str(index) for index in indices)
    return rf"(?:^|.*\.)blocks\.(?:{alternation})\..*"


def set_dropped_transformer_layer(model, layer_index: Optional[int]) -> None:
    base = base_model(model)
    if layer_index is None:
        if hasattr(base, "_dropped_transformer_layer_index"):
            delattr(base, "_dropped_transformer_layer_index")
        return
    layer_index = int(layer_index)
    total_layers = transformer_layer_count(model)
    if layer_index < 0 or layer_index >= total_layers:
        raise ValueError(
            f"dropped layer index {layer_index} out of range for {total_layers} layers"
        )
    if layer_index in dropped_transformer_layer_indices(total_layers):
        raise ValueError(
            f"layer {layer_index} is already statically skipped by "
            "TRANSFORMER_DROPPED_LAYERS"
        )
    base._dropped_transformer_layer_index = layer_index


def clear_dropped_transformer_layer(model) -> None:
    set_dropped_transformer_layer(model, None)


def sample_training_dropped_transformer_layer(model) -> Optional[int]:
    clear_dropped_transformer_layer(model)
    if not bool(TRAIN_RANDOM_LAYER_DROP):
        return None
    total_layers = transformer_layer_count(model)
    active_layers = active_transformer_layer_indices(total_layers)
    if not active_layers:
        return None
    sampled_index = random.randrange(len(active_layers) + 1)
    if sampled_index == len(active_layers):
        return NO_LAYER_DROP_BUCKET
    layer_index = active_layers[sampled_index]
    set_dropped_transformer_layer(model, layer_index)
    return layer_index


def unwrap_compiled_module(module):
    return getattr(module, "_orig_mod", module)


def torch_compiler_is_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    is_compiling = getattr(compiler, "is_compiling", None)
    if is_compiling is not None:
        return bool(is_compiling())
    dynamo = getattr(torch, "_dynamo", None)
    is_compiling = getattr(dynamo, "is_compiling", None)
    return bool(is_compiling()) if is_compiling is not None else False


def configure_regional_compile_runtime() -> None:
    if not REGIONAL_COMPILE:
        return
    if not REGIONAL_COMPILE_CAPTURE_SCALAR_OUTPUTS:
        return
    try:
        import torch._dynamo

        torch._dynamo.config.capture_scalar_outputs = True
    except Exception as exc:
        warnings.warn(f"Could not set Dynamo capture_scalar_outputs: {exc}")


def compiler_mark_step_begin() -> None:
    if not REGIONAL_COMPILE or not _REGIONAL_COMPILE_ACTIVE or DEVICE.type != "cuda":
        return
    compiler = getattr(torch, "compiler", None)
    mark_step = getattr(compiler, "cudagraph_mark_step_begin", None)
    if mark_step is not None:
        mark_step()


def regional_compile_kwargs():
    kwargs = {
        "fullgraph": bool(REGIONAL_COMPILE_FULLGRAPH),
        "dynamic": bool(REGIONAL_COMPILE_DYNAMIC),
    }
    if REGIONAL_COMPILE_BACKEND is not None:
        kwargs["backend"] = REGIONAL_COMPILE_BACKEND
    if REGIONAL_COMPILE_MODE is not None:
        kwargs["mode"] = REGIONAL_COMPILE_MODE
    return kwargs


def remove_compiled_repeated_blocks(model) -> bool:
    base = base_model(model)
    blocks = getattr(base, "blocks", None)
    if blocks is None:
        return False

    original_blocks = getattr(blocks, "_orig_mod", None)
    if original_blocks is not None:
        base.blocks = original_blocks
        return True

    changed = False
    for block_index, block in enumerate(list(blocks)):
        original_block = getattr(block, "_orig_mod", None)
        if original_block is not None:
            blocks[block_index] = original_block
            changed = True
    return changed


def regional_compile_skip_reasons(model) -> list[str]:
    reasons = []
    base = base_model(model)
    if hasattr(base, "blocks"):
        attn_modules = [
            getattr(unwrap_compiled_module(block), "self_attn", None)
            for block in base.blocks
        ]
        if any(attn is not None and hasattr(attn, "ssm_memory") for attn in attn_modules):
            reasons.append("SSM recurrent attention is attached")
    wan_attention_module = sys.modules.get("wan.modules.attention")
    wan_flex_available = getattr(wan_attention_module, "flex_attention_available", None)
    if wan_flex_available is not None and wan_flex_available():
        reasons.append("flex attention already compiles the sparse window kernel")
    return reasons


def apply_regional_compile(model, *, label: str = "model"):
    global _REGIONAL_COMPILE_ACTIVE
    _REGIONAL_COMPILE_ACTIVE = False
    if not REGIONAL_COMPILE:
        return model

    skip_reasons = regional_compile_skip_reasons(model)
    if skip_reasons:
        removed = remove_compiled_repeated_blocks(model)
        suffix = "; removed existing compiled repeated blocks" if removed else ""
        _init_log(
            f"skipping regional compile for {label}: "
            + "; ".join(skip_reasons)
            + "; compiling the outer Wan blocks can exceed Triton resource limits"
            + suffix
        )
        return model

    if not hasattr(torch, "compile"):
        raise RuntimeError("REGIONAL_COMPILE=True requires torch.compile")

    configure_regional_compile_runtime()

    try:
        from accelerate.utils import compile_regions, has_compiled_regions
    except Exception as exc:
        raise RuntimeError(
            "REGIONAL_COMPILE=True requires accelerate.utils.compile_regions"
        ) from exc

    base = base_model(model)
    if not hasattr(base, "blocks"):
        raise AttributeError(f"{label} has no repeated `blocks` module to compile")
    if has_compiled_regions(base.blocks):
        _init_log(f"{label} repeated blocks already regionally compiled")
        _REGIONAL_COMPILE_ACTIVE = True
        return model

    kwargs = regional_compile_kwargs()
    _init_log(f"regionally compiling {label} repeated blocks with {kwargs}")
    base.blocks = compile_regions(base.blocks, **kwargs)
    _REGIONAL_COMPILE_ACTIVE = True
    _init_log(f"regionally compiled {len(base.blocks)} {label} blocks")
    return model


@contextmanager
def temporarily_uncompiled_repeated_blocks(model):
    base = base_model(model)
    blocks = getattr(base, "blocks", None)
    if blocks is None:
        yield
        return

    original_blocks = getattr(blocks, "_orig_mod", None)
    if original_blocks is not None:
        compiled_blocks = blocks
        base.blocks = original_blocks
        try:
            yield
        finally:
            base.blocks = compiled_blocks
        return

    replacements = []
    try:
        for index, block in enumerate(blocks):
            original_block = getattr(block, "_orig_mod", None)
            if original_block is not None:
                replacements.append((index, block))
                blocks[index] = original_block
        yield
    finally:
        for index, compiled_block in replacements:
            blocks[index] = compiled_block


def lora_adapter_parameters(model):
    return [
        parameter for name, parameter in model.named_parameters() if "lora_" in name
    ]


def physics_conditioning_parameters(model):
    base = base_model(model)
    if not hasattr(base, "physics_adaln"):
        return []
    params = []
    active_layers = active_transformer_layer_indices(len(base.physics_adaln))
    for index in active_layers:
        params.extend(base.physics_adaln[index].parameters())
    return list(params)


def patch_embedding_lora_parameters(model):
    base = base_model(model)
    adapter = getattr(base, "patch_embedding_lora", None)
    if adapter is None:
        return []
    return list(adapter.parameters())


def timestep_adaln_lora_parameters(model):
    base = base_model(model)
    adapter = getattr(base, "timestep_adaln_lora", None)
    if adapter is None:
        return []
    return list(adapter.parameters())


def first_frame_conditioning_parameters(model):
    base = base_model(model)
    params = []
    if hasattr(base, "first_frame_conditioner"):
        params.extend(base.first_frame_conditioner.parameters())
    if hasattr(base, "first_frame_adaln"):
        params.extend(base.first_frame_adaln.parameters())
    return list(params)


def enable_lora_adapter_training(model) -> None:
    for parameter in lora_adapter_parameters(model):
        parameter.requires_grad_(True)

    for parameter in patch_embedding_lora_parameters(model):
        parameter.requires_grad_(True)

    for parameter in timestep_adaln_lora_parameters(model):
        parameter.requires_grad_(True)

    for parameter in physics_conditioning_parameters(model):
        parameter.requires_grad_(True)

    for parameter in first_frame_conditioning_parameters(model):
        parameter.requires_grad_(True)


def ssm_layer_indices(total_blocks: int) -> set[int]:
    total_blocks = int(total_blocks)
    if total_blocks < 1:
        return set()
    active_layers = active_transformer_layer_indices(total_blocks)
    setting = SSM_LAYER_COUNT
    if setting is None:
        count = len(active_layers)
    elif isinstance(setting, str):
        normalized = setting.strip().lower()
        if normalized in {"", "all", "none"}:
            count = len(active_layers) if normalized != "none" else 0
        else:
            count = int(normalized)
    else:
        count = int(setting)
    if count < 0:
        raise ValueError(f"SSM_LAYER_COUNT must be non-negative, got {setting!r}")
    return set(active_layers[: min(count, len(active_layers))])


def attached_ssm_layer_indices(model) -> list[int]:
    base = base_model(model)
    indices = []
    for index, block in enumerate(base.blocks):
        block = unwrap_compiled_module(block)
        if hasattr(block.self_attn, "ssm_memory"):
            indices.append(index)
    return indices


def attach_ssm_attention(model):
    base = base_model(model)
    device = next(base.parameters()).device
    selected_indices = ssm_layer_indices(len(base.blocks))
    default_frame_window = causal_window()
    for index, block in enumerate(base.blocks):
        block = unwrap_compiled_module(block)
        attn = block.self_attn
        attn.default_frame_window = default_frame_window
        attn.attention_dtype = TRAIN_DTYPE
        if index not in selected_indices:
            if hasattr(attn, "ssm_memory"):
                delattr(attn, "ssm_memory")
            attn.ssm_attention_enabled = False
            continue
        if not hasattr(attn, "ssm_memory"):
            attn.ssm_memory = StateSpaceMemory(attn.num_heads, attn.head_dim).to(
                device=device
            )
        attn.ssm_attention_enabled = False
    return model


def set_ssm_attention_enabled(model, enabled: bool):
    base = base_model(model)
    previous = []
    for block in base.blocks:
        block = unwrap_compiled_module(block)
        attn = block.self_attn
        previous.append((attn, bool(getattr(attn, "ssm_attention_enabled", False))))
        attn.ssm_attention_enabled = bool(enabled and hasattr(attn, "ssm_memory"))
    return previous


@contextmanager
def ssm_attention_enabled(model, enabled: bool = True):
    previous = set_ssm_attention_enabled(model, enabled)
    try:
        yield
    finally:
        for attn, old_value in previous:
            attn.ssm_attention_enabled = bool(
                old_value and hasattr(attn, "ssm_memory")
            )


def freeze_parameters(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def ssm_attention_parameters(model):
    params = []
    base = base_model(model)
    for block in base.blocks:
        block = unwrap_compiled_module(block)
        ssm_memory = getattr(block.self_attn, "ssm_memory", None)
        if ssm_memory is not None:
            for name, parameter in ssm_memory.named_parameters():
                if SSM_FREEZE_RHO_ZERO and name == "rho":
                    parameter.data.zero_()
                    parameter.requires_grad_(False)
                    continue
                params.append(parameter)
    return params


def has_ssm_attention(model) -> bool:
    base = base_model(model)
    return any(
        hasattr(unwrap_compiled_module(block).self_attn, "ssm_memory")
        for block in base.blocks
    )


def enable_ssm_attention_training(
    model, *, train_lora: bool = False, train_base: bool = False
) -> None:
    freeze_parameters(model)
    if train_base:
        base = base_model(model)
        dropped = dropped_transformer_layer_indices(len(base.blocks))
        active_layers = set(range(len(base.blocks))) - dropped
        always_active_modules = [
            "patch_embedding",
            "head",
            "patch_embedding_lora",
            "timestep_adaln_lora",
            "physics_adaln",
            "first_frame_adaln",
        ]
        for module_name in always_active_modules:
            module = getattr(base, module_name, None)
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

        for block_index, block in enumerate(base.blocks):
            block = unwrap_compiled_module(block)
            if block_index not in active_layers:
                continue
            for name, parameter in block.named_parameters():
                if name.startswith("cross_attn.") or name.startswith("norm3."):
                    continue
                parameter.requires_grad_(True)
            ssm_memory = getattr(block.self_attn, "ssm_memory", None)
            if ssm_memory is not None and SSM_FREEZE_RHO_ZERO:
                ssm_memory.rho.data.zero_()
                ssm_memory.rho.requires_grad_(False)
        return
    for parameter in ssm_attention_parameters(model):
        parameter.requires_grad_(True)
    if train_lora:
        enable_lora_adapter_training(model)


def _disabled_gradient_checkpoint_blocks(total_blocks: int) -> set[int]:
    setting = GRADIENT_CHECKPOINT_DISABLE_BLOCKS
    if setting is None:
        return set(range(total_blocks))

    if isinstance(setting, str):
        normalized = setting.strip().lower()
        if normalized == "all":
            return set(range(total_blocks))
        if normalized == "none":
            return set()
        try:
            setting = int(normalized)
        except ValueError as exc:
            raise ValueError(
                "GRADIENT_CHECKPOINT_DISABLE_BLOCKS must be 'all', 'none', an int, or a sequence of block indices"
            ) from exc

    if isinstance(setting, int):
        n_disabled = max(0, min(total_blocks, setting))
        return set(range(n_disabled))

    if isinstance(setting, (list, tuple, set)):
        disabled = {int(index) for index in setting}
        invalid = sorted(
            index for index in disabled if index < 0 or index >= total_blocks
        )
        if invalid:
            raise ValueError(
                f"Invalid gradient-checkpoint-disabled block indices for {total_blocks} blocks: {invalid}"
            )
        return disabled

    raise TypeError(
        "GRADIENT_CHECKPOINT_DISABLE_BLOCKS must be 'all', 'none', an int, or a sequence of block indices"
    )


def configure_gradient_checkpointing(model):
    base = base_model(model)
    total_blocks = len(base.blocks)
    disabled = _disabled_gradient_checkpoint_blocks(total_blocks)
    for block_index, block in enumerate(base.blocks):
        block = unwrap_compiled_module(block)
        block.gradient_checkpointing = block_index not in disabled

    enabled_count = total_blocks - len(disabled)
    if len(disabled) == total_blocks:
        disabled_label = "all"
    elif not disabled:
        disabled_label = "none"
    else:
        disabled_label = sorted(disabled)
    print(
        f"gradient checkpointing enabled on {enabled_count}/{total_blocks} blocks; "
        f"disabled blocks: {disabled_label}"
    )
    return {
        "total_blocks": total_blocks,
        "enabled_blocks": enabled_count,
        "disabled_blocks": sorted(disabled),
    }


def require_lora_adapter_state(model, enabled: bool, label: str = "inference") -> None:
    if not hasattr(model, "get_model_status"):
        raise TypeError(f"{label} expected a PEFT model with LoRA adapters")

    status = model.get_model_status()
    if status.enabled == "irregular":
        raise RuntimeError(
            f"{label} has an irregular adapter state; some LoRA layers are enabled and others are disabled"
        )

    actual = bool(status.enabled)
    if actual != bool(enabled):
        expected = "enabled" if enabled else "disabled"
        got = "enabled" if actual else "disabled"
        raise RuntimeError(
            f"{label} expected LoRA adapters {expected}, but they are {got}"
        )


def set_frame_window(model, left_frames: Optional[int], right_frames: Optional[int]):
    base = base_model(model)
    previous = []
    if (left_frames is None) != (right_frames is None):
        raise ValueError(
            "left_frames and right_frames must both be None or both be integers"
        )
    if right_frames is not None and int(right_frames) != 0:
        raise ValueError("right_frames must be 0 for causal windowed attention")
    for block in base.blocks:
        block = unwrap_compiled_module(block)
        attn = block.self_attn
        previous.append((attn, getattr(attn, "frame_window", None)))
        attn.default_frame_window = causal_window()
        attn.attention_dtype = TRAIN_DTYPE
        attn.frame_window = (
            None if left_frames is None else (int(left_frames), int(right_frames))
        )
    return previous


@contextmanager
def attention_window(model, left_frames: Optional[int], right_frames: Optional[int]):
    previous = set_frame_window(model, left_frames, right_frames)
    try:
        yield
    finally:
        for attn, old_value in previous:
            attn.frame_window = old_value
