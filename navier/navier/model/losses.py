# This file is executed by navier/autoreg_windowed_lora.py.
# Keep shared globals in that compatibility module namespace.


def frame_mse(pred: torch.Tensor, target: torch.Tensor):
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target shapes must match, got "
            f"{tuple(pred.shape)} and {tuple(target.shape)}"
        )
    if pred.ndim != 5:
        raise ValueError(
            f"Expected tensors with shape [B, C, F, H, W], got {tuple(pred.shape)}"
        )
    return (pred.float() - target.float()).pow(2).mean(dim=(0, 1, 3, 4))


def assert_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if torch.isfinite(tensor).all():
        return
    bad_count = int((~torch.isfinite(tensor)).sum().detach().cpu())
    raise FloatingPointError(f"{name} contains {bad_count} non-finite values")


def model_velocity_with_attention(
    model,
    x_t: torch.Tensor,
    seq_len: int,
    physics: torch.Tensor,
    *,
    window_left_frames: Optional[int],
    window_right_frames: Optional[int],
    first_frame: torch.Tensor,
    timestep_tokens: Optional[torch.Tensor] = None,
    ssm_enabled: bool = False,
):
    batch_size = x_t.shape[0]
    if timestep_tokens is None:
        timestep_tokens = flow_token_timesteps(
            x_t, base_model(model).patch_size
        )
    else:
        timestep_tokens = timestep_tokens.to(device=x_t.device, dtype=torch.float32)
    with ssm_attention_enabled(model, ssm_enabled):
        with attention_window(model, window_left_frames, window_right_frames):
            with torch.amp.autocast(
                "cuda", dtype=TRAIN_DTYPE, enabled=DEVICE.type == "cuda"
            ):
                pred = model(
                    [x_t[i] for i in range(batch_size)],
                    timestep_tokens,
                    context=None,
                    seq_len=seq_len,
                    physics=physics,
                    first_frame=[first_frame[i] for i in range(batch_size)],
                )
                return torch.stack(pred, dim=0)


_FLOW_TRAINING_SCHEDULE_CACHE = {}


def wan_flow_training_schedule(device: torch.device):
    _validate_flow_matching_config()
    key = (
        int(WAN_NUM_TRAIN_TIMESTEPS),
        float(WAN_SAMPLE_SHIFT),
        str(device),
    )
    cached = _FLOW_TRAINING_SCHEDULE_CACHE.get(key)
    if cached is not None:
        return cached
    scheduler = make_wan_flow_scheduler(int(WAN_NUM_TRAIN_TIMESTEPS), device=device)
    timesteps = scheduler.timesteps.to(device=device, dtype=torch.float32)
    sigmas = scheduler.sigmas[:-1].to(device=device, dtype=torch.float32)
    if int(timesteps.numel()) != int(sigmas.numel()):
        raise RuntimeError(
            f"Wan scheduler produced {int(timesteps.numel())} timesteps but "
            f"{int(sigmas.numel())} sigmas"
        )
    cached = (timesteps, sigmas)
    _FLOW_TRAINING_SCHEDULE_CACHE[key] = cached
    return cached


def sample_flow_training_timesteps(batch_size: int, device: torch.device):
    timesteps, sigmas = wan_flow_training_schedule(device)
    indices = torch.randint(
        0,
        int(timesteps.numel()),
        (int(batch_size),),
        device=device,
    )
    return timesteps[indices], sigmas[indices]


def flow_match_noisy_latents(clean: torch.Tensor):
    batch_size = int(clean.shape[0])
    timesteps, sigmas = sample_flow_training_timesteps(batch_size, clean.device)
    noise = torch.randn_like(clean)
    sigma = sigmas.to(dtype=clean.dtype).view(batch_size, 1, 1, 1, 1)
    x_t = (1.0 - sigma) * clean + sigma * noise
    x_t[:, :, 0:1] = clean[:, :, 0:1]
    target_velocity = noise - clean
    target_velocity[:, :, 0:1] = 0
    return x_t, target_velocity, timesteps


def _flow_matching_loss_components(
    student_pred: torch.Tensor,
    target_velocity: torch.Tensor,
    *,
    loss_name: str,
):
    pred_future = student_pred[:, :, 1:].float()
    target_future = target_velocity[:, :, 1:].float()
    flow_loss = F.mse_loss(pred_future, target_future)
    assert_finite_tensor(loss_name, flow_loss)
    return flow_loss, frame_mse(pred_future, target_future).detach()


def _full_sequence_flow_matching_batch(
    clean: torch.Tensor,
    physics: torch.Tensor,
    student,
    *,
    left_frames: Optional[int],
    right_frames: Optional[int],
):
    model_input, target_velocity, timesteps = flow_match_noisy_latents(clean)
    base = base_model(student)
    seq_len = latent_seq_len(clean, base.patch_size)
    timestep_tokens = flow_token_timesteps(
        model_input,
        base.patch_size,
        timesteps,
    )
    student_pred = model_velocity_with_attention(
        student,
        model_input,
        seq_len,
        physics,
        window_left_frames=left_frames,
        window_right_frames=right_frames,
        first_frame=clean[:, :, 0:1],
        timestep_tokens=timestep_tokens,
        ssm_enabled=True,
    )
    assert_finite_tensor("SSM flow prediction", student_pred)
    return student_pred, target_velocity


def ssm_training_batch(
    batch,
    student,
    *,
    step: int = 0,
    latent_frame_count: Optional[int] = None,
):
    del step
    clean = stack_training_latents(
        batch["latents"], frame_count=latent_frame_count
    ).to(device=DEVICE, dtype=TRAIN_DTYPE)
    physics = batch["physics"].to(device=DEVICE, dtype=torch.float32)
    assert_frame_window_alignment(clean, student)

    left_frames, right_frames = causal_window()
    student_pred, target_velocity = _full_sequence_flow_matching_batch(
        clean,
        physics,
        student,
        left_frames=left_frames,
        right_frames=right_frames,
    )
    flow_loss, frame_loss = _flow_matching_loss_components(
        student_pred,
        target_velocity,
        loss_name="SSM flow matching loss",
    )
    loss_metric = flow_loss.detach()
    return flow_loss, {
        "mse": loss_metric,
        "frame_loss": frame_loss,
        "total": loss_metric,
    }
