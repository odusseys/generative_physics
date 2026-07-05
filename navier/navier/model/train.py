# This file is executed by navier/autoreg_windowed_lora.py.
# Keep shared globals in that compatibility module namespace.

def train_ssm_window(max_steps: Optional[int] = None):
    max_steps = resolve_max_steps(max_steps)
    total_start = time.perf_counter()
    _init_log(f"train_ssm_window(flow_matching, max_steps={int(max_steps)}) starting")
    with init_stage("build SSM window objects"):
        state = build_ssm_training_objects()
    student = state["student"]
    optimizer = state["optimizer"]
    train_loader = state["train_loader"]
    holdout_samples = state["holdout_samples"]
    physics_mean = state["physics_mean"]
    physics_std = state["physics_std"]
    with init_stage("load Wan VAE"):
        vae = load_wan_vae(dtype=torch.float32, device=DEVICE)

    with init_stage("create training iterator and bookkeeping"):
        loader_iter = iter(train_loader)
        optimizer.zero_grad(set_to_none=True)
    tracked_losses = FLOW_MATCHING_LOSS_KEYS
    loss_history = {name: [] for name in tracked_losses}
    ema_losses = {name: None for name in tracked_losses}
    state["loss_history"] = loss_history
    state["loss_ema"] = ema_losses
    initialize_frame_loss_tracking(state)
    initialize_layer_drop_loss_tracking(state, student)
    initialize_wandb_run(max_steps=max_steps)
    _init_log(
        f"initialization complete in {time.perf_counter() - total_start:.1f}s; entering training loop"
    )

    try:
        progress = trange(1, int(max_steps) + 1, desc="ssm-window", dynamic_ncols=True)
        previous_active_frame_count = None
        for step in progress:
            current_frame_count = active_latent_frame_count(step)
            state["active_latent_frame_count"] = current_frame_count
            if current_frame_count != previous_active_frame_count:
                _init_log(
                    f"active latent frame count at step {step}: "
                    f"{current_frame_count if current_frame_count is not None else 'full'}"
                )
                previous_active_frame_count = current_frame_count
            lr = learning_rate_for_step(step)
            set_optimizer_lr(optimizer, lr)
            step_losses = {name: [] for name in tracked_losses}
            step_frame_losses = []
            dropped_layer_index = sample_training_dropped_transformer_layer(student)

            try:
                for _ in range(GRAD_ACCUM_STEPS):
                    try:
                        batch = next(loader_iter)
                    except StopIteration:
                        loader_iter = iter(train_loader)
                        batch = next(loader_iter)

                    compiler_mark_step_begin()
                    loss, components = ssm_training_batch(
                        batch,
                        student,
                        step=step,
                        latent_frame_count=current_frame_count,
                    )
                    collect_training_components(
                        components, step_losses, step_frame_losses
                    )
                    (loss / GRAD_ACCUM_STEPS).backward()
            finally:
                clear_dropped_transformer_layer(student)

            torch.nn.utils.clip_grad_norm_(
                trainable_parameters(student),
                CLIP_GRAD_NORM,
                error_if_nonfinite=True,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            for mode, losses in step_losses.items():
                if not losses:
                    continue
                mode_loss = sum(losses) / len(losses)
                previous = ema_losses[mode]
                ema_losses[mode] = (
                    mode_loss
                    if previous is None
                    else LOSS_EMA_BETA * previous + (1.0 - LOSS_EMA_BETA) * mode_loss
                )
                loss_history[mode].append(
                    {"step": step, "loss": mode_loss, "ema": ema_losses[mode]}
                )
            update_frame_loss_tracking(
                state,
                step,
                step_frame_losses,
                active_latent_frame_count=current_frame_count,
            )
            update_layer_drop_loss_tracking(
                state, step, dropped_layer_index, step_losses
            )
            wandb_log_training_step(
                state,
                step,
                step_losses,
                step_frame_losses,
                active_latent_frame_count=current_frame_count,
                learning_rate=lr,
                dropped_layer_index=dropped_layer_index,
            )

            set_training_progress_postfix(
                progress, ema_losses, active_latent_frame_count=current_frame_count
            )

            if step % EVAL_EVERY == 0:
                clear_dropped_transformer_layer(student)
                eval_result = evaluate_random_holdout(
                    student,
                    vae,
                    holdout_samples,
                    step,
                    physics_mean,
                    physics_std,
                    frame_loss_ema=state.get("frame_loss_ema"),
                    frame_loss_indices=state.get("frame_loss_indices"),
                    layer_drop_loss_ema=state.get("layer_drop_loss_ema"),
                    layer_drop_loss_counts=state.get("layer_drop_loss_counts"),
                    layer_drop_loss_indices=state.get("layer_drop_loss_indices"),
                    latent_frame_count=current_frame_count,
                )
                wandb_log_evaluation(step, eval_result)
                student.train()
                set_frame_window(student, *causal_window())
                set_ssm_attention_enabled(student, True)

            if checkpoint_saving_enabled() and step % int(SAVE_EVERY) == 0:
                save_ssm_attention_checkpoint(student, step)

        if checkpoint_saving_enabled():
            save_ssm_attention_checkpoint(student, int(max_steps))
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return student, state
    finally:
        finish_wandb_run()


def train(max_steps: Optional[int] = None):
    max_steps = resolve_max_steps(max_steps)
    return train_ssm_window(max_steps=max_steps)
