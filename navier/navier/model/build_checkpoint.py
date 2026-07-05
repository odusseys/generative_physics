# This file is executed by navier/autoreg_windowed_lora.py.
# Keep shared globals in that compatibility module namespace.

def build_ssm_training_objects():
    _init_log(
        f"mode=flow_matching device={DEVICE} param_dtype={PARAM_DTYPE} "
        f"train_dtype={TRAIN_DTYPE}"
    )
    _init_log(
        "Wan scheduler flow matching: clean frame 0, noisy future frames, "
        f"shift={float(WAN_SAMPLE_SHIFT):g}, eval_steps={int(EVAL_INFERENCE_STEPS)}"
    )
    with init_stage("discover latent samples"):
        samples = discover_latent_samples()
    _init_log(f"found {len(samples):,} samples in {LATENT_ROOT}")
    with init_stage("split dataset and compute physics stats"):
        train_samples, holdout_samples = make_splits(samples)
        physics_mean, physics_std = physics_stats(train_samples)
    _init_log(f"train={len(train_samples):,} holdout={len(holdout_samples):,}")
    _init_log("text conditioning disabled; timestep AdaLN conditioning enabled")
    with init_stage("create latent dataset and dataloader"):
        train_dataset = LatentDataset(train_samples, physics_mean, physics_std)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=DEVICE.type == "cuda",
            collate_fn=collate_batch,
            drop_last=True,
        )

    with init_stage(f"load student Wan transformer from {WAN_CHECKPOINT_DIR}"):
        student = load_wan_transformer()
    with init_stage("attach student physics conditioning and SSM memory"):
        student = configure_wan_conditioning(student)
        student = attach_patch_embedding_lora(student)
        student = attach_timestep_adaln_lora(student)
        student = attach_ssm_attention(student)
    with init_stage("attach student LoRA adapters"):
        student = add_lora_to_wan(student)
    with init_stage("configure student gradient checkpointing"):
        configure_gradient_checkpointing(student)
    with init_stage("enable SSM/windowed student training"):
        enable_ssm_attention_training(student, train_lora=True, train_base=False)
        set_frame_window(student, *causal_window())
        set_ssm_attention_enabled(student, True)
        student.train()
    with init_stage("regionally compile student repeated Wan blocks"):
        student = apply_regional_compile(student, label="SSM student")

    with init_stage("create optimizer"):
        optimizer = torch.optim.AdamW(
            trainable_parameters(student), lr=LEARNING_RATE, weight_decay=1e-4
        )
    total = sum(param.numel() for param in student.parameters())
    trainable = sum(
        param.numel() for param in student.parameters() if param.requires_grad
    )
    ssm_trainable = sum(
        param.numel()
        for param in ssm_attention_parameters(student)
        if param.requires_grad
    )
    lora_trainable = sum(
        param.numel()
        for param in lora_adapter_parameters(student)
        if param.requires_grad
    )
    patch_lora_trainable = sum(
        param.numel()
        for param in patch_embedding_lora_parameters(student)
        if param.requires_grad
    )
    timestep_lora_trainable = sum(
        param.numel()
        for param in timestep_adaln_lora_parameters(student)
        if param.requires_grad
    )
    physics_trainable = sum(
        param.numel()
        for param in physics_conditioning_parameters(student)
        if param.requires_grad
    )
    first_frame_trainable = sum(
        param.numel()
        for param in first_frame_conditioning_parameters(student)
        if param.requires_grad
    )
    ssm_layers = attached_ssm_layer_indices(student)
    dropped_layers = sorted(
        dropped_transformer_layer_indices(transformer_layer_count(student))
    )
    print(
        f"SSM window trainable params: {trainable:,} / {total:,} "
        f"(init=pretrained, ssm={ssm_trainable:,}, lora={lora_trainable:,}, "
        f"patch_lora={patch_lora_trainable:,}, "
        f"timestep_lora={timestep_lora_trainable:,}, "
        f"physics={physics_trainable:,}, first_frame={first_frame_trainable:,}, "
        f"ssm_layers={ssm_layers}, dropped_layers={dropped_layers})"
    )
    _init_log("SSM window training objects ready")

    return {
        "model": student,
        "student": student,
        "optimizer": optimizer,
        "train_loader": train_loader,
        "train_samples": train_samples,
        "holdout_samples": holdout_samples,
        "physics_mean": physics_mean,
        "physics_std": physics_std,
    }


def conditioning_state_dict(model):
    with temporarily_uncompiled_repeated_blocks(model):
        base = base_model(model)
        return {
            key: value.detach().cpu()
            for key, value in base.state_dict().items()
            if key.startswith("physics_adaln.")
            or key.startswith("first_frame_conditioner.")
            or key.startswith("first_frame_adaln.")
            or key.startswith("patch_embedding_lora.")
            or key.startswith("timestep_adaln_lora.")
        }


def save_conditioning_checkpoint(model, output_dir: Path):
    state = conditioning_state_dict(model)
    if not state:
        return None
    save_file(
        state,
        str(Path(output_dir) / "conditioning.safetensors"),
        metadata={
            "first_frame_conditioning": "patchified_per_block_adaln0",
            "first_frame_adaln_hidden_dim": str(FIRST_FRAME_CONDITIONING_DIM),
            "first_frame_adaln_final_weight_std": "0.0001",
            "patch_embedding_lora_r": str(PATCH_EMBEDDING_LORA_R),
            "patch_embedding_lora_alpha": str(PATCH_EMBEDDING_LORA_ALPHA),
            "timestep_adaln_lora_r": str(TIMESTEP_ADALN_LORA_R),
            "timestep_adaln_lora_alpha": str(TIMESTEP_ADALN_LORA_ALPHA),
            "text_conditioning": "disabled",
            "timestep_conditioning": "enabled_frame0_zero_future_wan_flow",
            "patch_input": "first_frame_noisy_future",
            "wan_num_train_timesteps": str(WAN_NUM_TRAIN_TIMESTEPS),
            "wan_sample_shift": str(WAN_SAMPLE_SHIFT),
            "eval_inference_steps": str(EVAL_INFERENCE_STEPS),
            "attention_frame0_anchor": "false",
            "window_left_frames": str(WINDOW_LEFT_FRAMES),
            "window_right_frames": "0",
            "transformer_dropped_layers": json.dumps(
                sorted(dropped_transformer_layer_indices())
            ),
        },
    )
    return Path(output_dir) / "conditioning.safetensors"


def ssm_attention_state_dict(model):
    with temporarily_uncompiled_repeated_blocks(model):
        base = base_model(model)
        return {
            key: value.detach().cpu()
            for key, value in base.state_dict().items()
            if ".ssm_memory." in key
        }


def save_ssm_attention_checkpoint(model, step: int):
    output_dir = OUTPUT_DIR / f"ssm_step_{step:06d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "training_mode": "flow_matching",
        "wan_transformer_init_mode": "pretrained",
        "text_conditioning": "disabled",
        "timestep_conditioning": "enabled_frame0_zero_future_wan_flow",
        "patch_input": "first_frame_noisy_future",
        "wan_num_train_timesteps": str(WAN_NUM_TRAIN_TIMESTEPS),
        "wan_sample_shift": str(WAN_SAMPLE_SHIFT),
        "eval_inference_steps": str(EVAL_INFERENCE_STEPS),
        "window_left_frames": str(WINDOW_LEFT_FRAMES),
        "window_right_frames": "0",
        "decay_init": str(SSM_DECAY_INIT),
        "input_init": str(SSM_INPUT_INIT),
        "output_init": str(SSM_OUTPUT_INIT),
        "skip_init": str(SSM_SKIP_INIT),
        "rho_init": str(SSM_RHO_INIT),
        "rho_frozen_zero": str(bool(SSM_FREEZE_RHO_ZERO)),
        "query_scale": "" if SSM_QUERY_SCALE is None else str(SSM_QUERY_SCALE),
        "ssm_layer_count": "" if SSM_LAYER_COUNT is None else str(SSM_LAYER_COUNT),
        "patch_embedding_lora_r": str(PATCH_EMBEDDING_LORA_R),
        "patch_embedding_lora_alpha": str(PATCH_EMBEDDING_LORA_ALPHA),
        "timestep_adaln_lora_r": str(TIMESTEP_ADALN_LORA_R),
        "timestep_adaln_lora_alpha": str(TIMESTEP_ADALN_LORA_ALPHA),
        "recurrent_dim": str(SSM_RECURRENT_DIM),
        "state_layout": "token_aligned",
        "state_mixer_expansion": str(SSM_STATE_MIXER_EXPANSION),
        "layer_indices": json.dumps(attached_ssm_layer_indices(model)),
        "transformer_dropped_layers": json.dumps(
            sorted(dropped_transformer_layer_indices())
        ),
        "use_pre_ssm_convolution": str(bool(SSM_USE_CONVOLUTIONS)),
        "post_ssm_resnet_convolution": "true",
        "lora_r": str(LORA_R),
        "lora_alpha": str(LORA_ALPHA),
        "lora_dropout": str(LORA_DROPOUT),
        "lora_target_modules": json.dumps(LORA_TARGET_MODULES),
    }
    save_file(
        ssm_attention_state_dict(model),
        str(output_dir / "ssm_attention.safetensors"),
        metadata=metadata,
    )
    if lora_adapter_parameters(model) and hasattr(model, "save_pretrained"):
        with temporarily_uncompiled_repeated_blocks(model):
            model.save_pretrained(str(output_dir / "lora"))
    save_conditioning_checkpoint(model, output_dir)

    latest_dir = OUTPUT_DIR / "ssm_latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        ssm_attention_state_dict(model),
        str(latest_dir / "ssm_attention.safetensors"),
        metadata=metadata,
    )
    if lora_adapter_parameters(model) and hasattr(model, "save_pretrained"):
        with temporarily_uncompiled_repeated_blocks(model):
            model.save_pretrained(str(latest_dir / "lora"))
    save_conditioning_checkpoint(model, latest_dir)
    return output_dir


def checkpoint_saving_enabled() -> bool:
    return SAVE_EVERY is not None and int(SAVE_EVERY) > 0


_WANDB_RUN = None
_WANDB_RUN_OWNED = False
_WANDB_UNAVAILABLE_REASON = None


def _wandb_config_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_wandb_config_value(item) for item in value]
    if isinstance(value, set):
        return [_wandb_config_value(item) for item in sorted(value)]
    if isinstance(value, dict):
        return {str(key): _wandb_config_value(item) for key, item in value.items()}
    return str(value)


def _wandb_run_config(max_steps: Optional[int] = None) -> dict:
    config = {
        key: _wandb_config_value(value)
        for key, value in get_config().items()
        if not key.startswith("WANDB_")
    }
    config["resolved_max_steps"] = int(max_steps) if max_steps is not None else None
    return config


def wandb_tracking_enabled() -> bool:
    return bool(WANDB_ENABLED)


def initialize_wandb_run(max_steps: Optional[int] = None):
    global _WANDB_RUN, _WANDB_RUN_OWNED, _WANDB_UNAVAILABLE_REASON
    if not wandb_tracking_enabled():
        return None
    if _WANDB_UNAVAILABLE_REASON is not None:
        return None
    try:
        import wandb
    except Exception as exc:
        _WANDB_UNAVAILABLE_REASON = str(exc)
        warnings.warn(f"W&B tracking disabled; could not import wandb: {exc}")
        return None

    if wandb.run is not None:
        _WANDB_RUN = wandb.run
        _WANDB_RUN_OWNED = False
        return _WANDB_RUN

    init_kwargs = {
        "project": WANDB_PROJECT,
        "config": _wandb_run_config(max_steps=max_steps),
    }
    if WANDB_ENTITY:
        init_kwargs["entity"] = WANDB_ENTITY
    if WANDB_RUN_NAME:
        init_kwargs["name"] = WANDB_RUN_NAME
    if WANDB_MODE:
        init_kwargs["mode"] = WANDB_MODE
    if WANDB_TAGS:
        init_kwargs["tags"] = list(WANDB_TAGS)

    try:
        _WANDB_RUN = wandb.init(**init_kwargs)
    except Exception as exc:
        _WANDB_UNAVAILABLE_REASON = str(exc)
        warnings.warn(f"W&B tracking disabled; wandb.init failed: {exc}")
        _WANDB_RUN = None
        _WANDB_RUN_OWNED = False
        return None
    _WANDB_RUN_OWNED = True
    return _WANDB_RUN


def finish_wandb_run() -> None:
    global _WANDB_RUN, _WANDB_RUN_OWNED
    if _WANDB_RUN is None:
        return
    if _WANDB_RUN_OWNED:
        try:
            _WANDB_RUN.finish()
        except Exception as exc:
            warnings.warn(f"W&B finish failed: {exc}")
    _WANDB_RUN = None
    _WANDB_RUN_OWNED = False


def _active_wandb_run():
    if not wandb_tracking_enabled() or _WANDB_UNAVAILABLE_REASON is not None:
        return None
    try:
        import wandb
    except Exception:
        return None
    return _WANDB_RUN if _WANDB_RUN is not None else wandb.run


def _mean_step_losses(step_losses) -> dict:
    values = {}
    for name, losses in step_losses.items():
        if losses:
            values[name] = float(sum(losses) / len(losses))
    return values


def _mean_step_frame_loss(step_frame_losses) -> Optional[torch.Tensor]:
    if not step_frame_losses:
        return None
    lengths = {int(frame_loss.numel()) for frame_loss in step_frame_losses}
    if len(lengths) != 1:
        raise ValueError(
            f"Inconsistent per-frame loss lengths in step: {sorted(lengths)}"
        )
    frame_loss = torch.stack(step_frame_losses, dim=0).mean(dim=0)
    if frame_loss.numel() < 1:
        return None
    return frame_loss.detach().to(device="cpu", dtype=torch.float32)


def wandb_log_training_step(
    state,
    step: int,
    step_losses,
    step_frame_losses,
    *,
    active_latent_frame_count: Optional[int],
    learning_rate: float,
    dropped_layer_index: Optional[int],
) -> None:
    run = _active_wandb_run()
    if run is None:
        return
    try:
        import wandb
    except Exception:
        return

    logs = {
        "train/learning_rate": float(learning_rate),
    }
    if active_latent_frame_count is not None:
        logs["train/active_latent_frame_count"] = int(active_latent_frame_count)
    if dropped_layer_index is not None:
        logs["train/dropped_layer_index"] = int(dropped_layer_index)

    losses = _mean_step_losses(step_losses)
    for name, value in losses.items():
        logs[f"train/{name}_loss_step"] = float(value)
    if "total" in losses:
        logs["train/loss_step"] = float(losses["total"])
    elif "mse" in losses:
        logs["train/loss_step"] = float(losses["mse"])

    frame_loss = _mean_step_frame_loss(step_frame_losses)
    if frame_loss is not None:
        state["last_step_frame_loss"] = frame_loss
        frame_loss_indices = list(range(1, 1 + int(frame_loss.numel())))
        state["last_step_frame_loss_indices"] = frame_loss_indices
        for timestep in WANDB_FRAME_LOSS_TIMESTEPS:
            timestep = int(timestep)
            if timestep in frame_loss_indices:
                frame_loss_offset = timestep - 1
                logs[f"train/frame_loss_t{timestep:02d}_step"] = float(
                    frame_loss[frame_loss_offset].item()
                )

    try:
        wandb.log(logs, step=int(step))
    except Exception as exc:
        warnings.warn(f"W&B training log failed at step {step}: {exc}")


def _wandb_video_from_frames(frames: torch.Tensor, caption: str):
    import wandb

    frames = frames.detach().to(torch.uint8).cpu()
    video = frames.permute(0, 3, 1, 2).numpy()
    return wandb.Video(video, fps=int(EVAL_FPS), format="mp4", caption=caption)


def wandb_log_evaluation(step: int, result: Optional[dict]) -> None:
    run = _active_wandb_run()
    if run is None or not result:
        return
    try:
        import wandb
    except Exception:
        return

    logs = {}
    sample = result.get("sample")
    if sample is not None:
        logs["eval/sample"] = sample.name
        logs["eval/nu"] = float(sample.nu)
        logs["eval/rho"] = float(sample.rho)
    if WANDB_LOG_DEBUG_VIDEOS:
        frames = result.get("debug_video_frames")
        if frames is not None:
            caption = result.get("caption") or f"debug video step {int(step)}"
            logs["eval/debug_video"] = _wandb_video_from_frames(frames, caption)
    if not logs:
        return
    try:
        wandb.log(logs, step=int(step))
    except Exception as exc:
        warnings.warn(f"W&B evaluation log failed at step {step}: {exc}")


FLOW_MATCHING_LOSS_KEYS = (
    "mse",
    "total",
)


def set_training_progress_postfix(
    progress, ema_losses, active_latent_frame_count: Optional[int] = None
) -> None:
    postfix = dict(
        mse_ema=("-" if ema_losses["mse"] is None else f"{ema_losses['mse']:.6f}"),
        total_ema=(
            "-" if ema_losses["total"] is None else f"{ema_losses['total']:.6f}"
        ),
    )
    if active_latent_frame_count is not None:
        postfix["seq"] = int(active_latent_frame_count)
    progress.set_postfix(**postfix)


def initialize_frame_loss_tracking(state) -> None:
    state["frame_loss_history"] = []
    state["frame_loss_ema"] = None
    state["frame_loss_indices"] = None


def collect_training_components(components, step_losses, step_frame_losses) -> None:
    for name, value in components.items():
        if name == "frame_loss":
            step_frame_losses.append(
                value.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
            )
            continue
        if name in step_losses:
            step_losses[name].append(float(value.detach().cpu()))


def update_frame_loss_tracking(
    state,
    step: int,
    step_frame_losses,
    active_latent_frame_count: Optional[int] = None,
) -> None:
    if not step_frame_losses:
        return
    lengths = {int(frame_loss.numel()) for frame_loss in step_frame_losses}
    if len(lengths) != 1:
        raise ValueError(
            f"Inconsistent per-frame loss lengths in step: {sorted(lengths)}"
        )
    frame_loss = torch.stack(step_frame_losses, dim=0).mean(dim=0)
    if frame_loss.numel() < 1:
        return
    previous = state.get("frame_loss_ema")
    if previous is None:
        frame_loss_ema = frame_loss
    else:
        previous = torch.as_tensor(previous, dtype=torch.float32).detach().cpu()
        output_length = max(int(previous.numel()), int(frame_loss.numel()))
        frame_loss_ema = torch.empty(output_length, dtype=torch.float32)
        if previous.numel() > 0:
            frame_loss_ema[: previous.numel()] = previous
        if frame_loss.numel() > previous.numel():
            frame_loss_ema[previous.numel() :] = frame_loss[previous.numel() :]
        overlap = min(int(previous.numel()), int(frame_loss.numel()))
        if overlap > 0:
            frame_loss_ema[:overlap] = (
                LOSS_EMA_BETA * previous[:overlap]
                + (1.0 - LOSS_EMA_BETA) * frame_loss[:overlap]
            )
    start_index = 1
    frame_indices = list(range(start_index, start_index + int(frame_loss_ema.numel())))
    observed_frame_indices = list(
        range(start_index, start_index + int(frame_loss.numel()))
    )
    state["frame_loss_ema"] = frame_loss_ema
    state["frame_loss_indices"] = frame_indices
    state["frame_loss_active_latent_frame_count"] = active_latent_frame_count
    state.setdefault("frame_loss_history", []).append(
        {
            "step": int(step),
            "active_latent_frame_count": (
                None
                if active_latent_frame_count is None
                else int(active_latent_frame_count)
            ),
            "frame_indices": frame_indices,
            "observed_frame_indices": observed_frame_indices,
            "loss": [float(value) for value in frame_loss.tolist()],
            "ema": [float(value) for value in frame_loss_ema.tolist()],
        }
    )


def initialize_layer_drop_loss_tracking(state, model) -> None:
    total_layers = transformer_layer_count(model)
    bucket_count = total_layers + 1
    state["layer_drop_loss_history"] = []
    state["layer_drop_loss_ema"] = torch.full(
        (bucket_count,), float("nan"), dtype=torch.float32
    )
    state["layer_drop_loss_counts"] = torch.zeros(bucket_count, dtype=torch.long)
    state["layer_drop_loss_indices"] = [NO_LAYER_DROP_BUCKET] + list(
        range(total_layers)
    )


def _mean_step_loss(step_losses, preferred=("total", "mse")) -> Optional[float]:
    for name in preferred:
        losses = step_losses.get(name)
        if losses:
            return float(sum(losses) / len(losses))
    for losses in step_losses.values():
        if losses:
            return float(sum(losses) / len(losses))
    return None


def update_layer_drop_loss_tracking(
    state, step: int, dropped_layer_index: Optional[int], step_losses
) -> None:
    if dropped_layer_index is None:
        return
    layer_loss = _mean_step_loss(step_losses)
    if layer_loss is None or not math.isfinite(layer_loss):
        return
    ema = state.get("layer_drop_loss_ema")
    counts = state.get("layer_drop_loss_counts")
    indices = state.get("layer_drop_loss_indices")
    if ema is None or counts is None or indices is None:
        return
    layer_index = int(dropped_layer_index)
    if layer_index == NO_LAYER_DROP_BUCKET:
        bucket_index = 0
        history_layer_index = None
        history_bucket = "none"
    else:
        bucket_index = layer_index + 1
        history_layer_index = layer_index
        history_bucket = layer_index
    if bucket_index < 0 or bucket_index >= int(ema.numel()):
        raise ValueError(
            f"dropped layer index {layer_index} out of range for "
            f"{int(ema.numel()) - 1} tracked layers plus no-drop bucket"
        )
    previous_count = int(counts[bucket_index].item())
    previous = float(ema[bucket_index].item())
    new_ema = (
        layer_loss
        if previous_count < 1 or not math.isfinite(previous)
        else LOSS_EMA_BETA * previous + (1.0 - LOSS_EMA_BETA) * layer_loss
    )
    ema[bucket_index] = float(new_ema)
    counts[bucket_index] += 1
    state.setdefault("layer_drop_loss_history", []).append(
        {
            "step": int(step),
            "bucket": history_bucket,
            "layer_index": history_layer_index,
            "loss": float(layer_loss),
            "ema": float(new_ema),
            "count": int(counts[bucket_index].item()),
        }
    )
