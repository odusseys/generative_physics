# This file is executed by navier/autoreg_windowed_lora.py.
# Keep shared globals in that compatibility module namespace.

def trainable_parameters(model):
    return [param for param in model.parameters() if param.requires_grad]


def resolve_max_steps(max_steps: Optional[int] = None) -> int:
    max_steps = MAX_STEPS if max_steps is None else max_steps
    max_steps = int(max_steps)
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    return max_steps


def learning_rate_for_step(step: int) -> float:
    if LR_WARMUP_STEPS > 0 and step <= LR_WARMUP_STEPS:
        return float(LEARNING_RATE) * max(0.0, float(step) / float(LR_WARMUP_STEPS))
    if MAX_STEPS <= LR_WARMUP_STEPS:
        return float(LEARNING_RATE)
    decay_progress = (float(step) - float(LR_WARMUP_STEPS)) / float(
        MAX_STEPS - LR_WARMUP_STEPS
    )
    decay_progress = min(1.0, max(0.0, decay_progress))
    return float(LEARNING_RATE) * 0.5 * (1.0 + math.cos(math.pi * decay_progress))


def set_optimizer_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def load_wan_vae(dtype=torch.float32, device=DEVICE):
    Wan2_2_VAE = load_wan_vae_class()
    return Wan2_2_VAE(
        vae_pth=str(WAN_CHECKPOINT_DIR / "Wan2.2_VAE.pth"),
        dtype=dtype,
        device=device,
    )


