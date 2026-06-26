# This file is executed by navier/autoreg_windowed_lora.py.
# Keep shared globals in that compatibility module namespace.

@torch.no_grad()
def regress_latents_from_first_frame(
    model,
    condition_latents: torch.Tensor,
    *,
    physics: torch.Tensor,
    window_left_frames: Optional[int],
    window_right_frames: Optional[int],
    original_first_frame: Optional[torch.Tensor] = None,
):
    was_training = model.training
    model.eval()

    clean = condition_latents.to(device=DEVICE, dtype=TRAIN_DTYPE)
    base = base_model(model)
    clean_batch = clean.unsqueeze(0)
    assert_frame_window_alignment(clean_batch, model)
    seq_len = latent_seq_len(clean_batch, base.patch_size)
    physics = physics.to(device=DEVICE, dtype=torch.float32).view(1, 2)
    if original_first_frame is None:
        first_frame_condition = clean[:, 0:1]
    else:
        first_frame_condition = original_first_frame.to(
            device=DEVICE, dtype=TRAIN_DTYPE
        )
        if first_frame_condition.dim() == 5:
            first_frame_condition = first_frame_condition[0]
        if first_frame_condition.dim() != 4:
            raise ValueError(
                "original_first_frame must have shape [C,1,H,W] or [1,C,1,H,W]"
            )
        if int(first_frame_condition.shape[1]) != 1:
            raise ValueError("original_first_frame must contain exactly one frame")

    scheduler = make_wan_flow_scheduler(int(EVAL_INFERENCE_STEPS), device=DEVICE)
    latents = torch.randn_like(clean)
    latents[:, 0:1] = clean[:, 0:1]
    first_frame_batch = first_frame_condition.unsqueeze(0)
    for timestep in scheduler.timesteps:
        latent_batch = latents.unsqueeze(0)
        timestep_tokens = flow_token_timesteps(
            latent_batch,
            base.patch_size,
            timestep.to(device=DEVICE, dtype=torch.float32).view(1),
        )
        velocity = model_velocity_with_attention(
            model,
            latent_batch,
            seq_len,
            physics,
            window_left_frames=window_left_frames,
            window_right_frames=window_right_frames,
            first_frame=first_frame_batch,
            timestep_tokens=timestep_tokens,
            ssm_enabled=True,
        )
        velocity[:, :, 0:1] = 0.0
        assert_finite_tensor("Wan flow inference velocity", velocity)
        latents = scheduler.step(
            velocity.float(),
            timestep,
            latent_batch.float(),
            return_dict=False,
        )[0][0].to(device=DEVICE, dtype=TRAIN_DTYPE)
        latents[:, 0:1] = clean[:, 0:1]
        assert_finite_tensor("Wan flow inference latents", latents)

    if was_training:
        model.train()
    return latents.float().cpu()


@torch.no_grad()
def diagnose_inference_nans(
    model,
    state: dict,
    *,
    sample: Optional[LatentSample] = None,
    seed: int = 0,
    use_forward_hooks: bool = True,
    print_report: bool = True,
) -> dict:
    """Trace one flow-matching denoising forward and report the first non-finite source."""
    if state is None:
        raise ValueError("Pass the training state returned by awl.train()")
    physics_mean = state["physics_mean"]
    physics_std = state["physics_std"]
    holdout_samples = state["holdout_samples"]
    if sample is None:
        sample = holdout_samples[0]

    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(int(seed))
    was_training = model.training
    model.eval()

    full_clean_latents = load_sample_latents(sample)
    clean_latents = truncate_latent_frames(full_clean_latents)
    clean = clean_latents.to(device=DEVICE, dtype=TRAIN_DTYPE)
    physics = sample_physics(sample, physics_mean, physics_std).to(
        device=DEVICE, dtype=torch.float32
    )

    model_input = torch.randn(
        clean.shape,
        device=DEVICE,
        dtype=TRAIN_DTYPE,
        generator=generator,
    )
    model_input[:, 0:1] = clean[:, 0:1]

    base = base_model(model)
    clean_batch = clean.unsqueeze(0)
    assert_frame_window_alignment(clean_batch, model)
    seq_len = latent_seq_len(clean_batch, base.patch_size)
    physics = physics.view(1, 2)
    left_frames, right_frames = causal_window()
    scheduler = make_wan_flow_scheduler(int(EVAL_INFERENCE_STEPS), device=DEVICE)
    timestep = scheduler.timesteps[0]
    timestep_tokens = flow_token_timesteps(
        model_input.unsqueeze(0),
        base.patch_size,
        timestep.to(device=DEVICE, dtype=torch.float32).view(1),
    )
    ssm_attached = has_ssm_attention(model)
    previous_ssm = set_ssm_attention_enabled(model, True) if ssm_attached else None

    records = []
    result = {
        "sample": sample.name,
        "ssm_enabled": bool(ssm_attached),
        "parameter_diagnostics": recurrent_parameter_diagnostics(model),
        "records": records,
        "first_failure": None,
    }

    trace_context = (
        first_nonfinite_forward_trace(model)
        if use_forward_hooks
        else nullcontext({"first": None})
    )
    try:
        with trace_context as module_trace:
            with attention_window(model, left_frames, right_frames):
                record = {
                    "model_input": tensor_finite_stats("model_input", model_input),
                    "target": tensor_finite_stats("target", clean),
                }
                records.append(record)
                pred = model_velocity_with_attention(
                    model,
                    model_input.unsqueeze(0),
                    seq_len,
                    physics,
                    window_left_frames=left_frames,
                    window_right_frames=right_frames,
                    first_frame=clean[:, 0:1].unsqueeze(0),
                    timestep_tokens=timestep_tokens,
                    ssm_enabled=True,
                )[0]
                pred[:, 0:1] = 0.0
                record["pred"] = tensor_finite_stats("pred", pred)
                record["abs_error"] = tensor_finite_stats(
                    "abs_error", pred.float().abs()
                )
                if not record["pred"]["finite"]:
                    result["first_failure"] = {
                        "kind": "model_prediction",
                        "module": module_trace["first"],
                        "record": record,
                    }
    finally:
        if previous_ssm is not None:
            for attn, old_value in previous_ssm:
                attn.ssm_attention_enabled = old_value
        if was_training:
            model.train()

    if result["first_failure"] is None:
        result["final_latents"] = records[-1]["pred"]

    if print_report:
        print(
            f"flow NaN diagnostic: sample={result['sample']} "
            f"ssm_enabled={result['ssm_enabled']}"
        )
        for diag in result["parameter_diagnostics"][:3]:
            print(f"block {diag['block']} {diag['kind']} parameters:")
            for key, value in diag.items():
                if isinstance(value, dict):
                    print("  " + format_finite_stats(value))
        if len(result["parameter_diagnostics"]) > 3:
            print(
                f"  ... {len(result['parameter_diagnostics']) - 3} more recurrent blocks"
            )
        for record in records:
            print(format_finite_stats(record["model_input"]))
            if "pred" in record:
                print("  " + format_finite_stats(record["pred"]))
            if "abs_error" in record:
                print("  " + format_finite_stats(record["abs_error"]))
        if result["first_failure"] is None:
            print("no non-finite values found")
            print(format_finite_stats(result["final_latents"]))
        else:
            failure = result["first_failure"]
            print(f"first failure: {failure['kind']}")
            if failure["module"] is not None:
                print("first non-finite module:", failure["module"])
    return result


@torch.no_grad()
def decode_latents_to_video(vae, latents: torch.Tensor):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*"
        )
        video = vae.decode([latents.to(device=DEVICE, dtype=torch.float32)])[0]
    return video.detach().float().cpu().clamp(-1, 1)


def tensor_range_summary(name: str, tensor: torch.Tensor) -> str:
    values = tensor.detach().float()
    return (
        f"{name}: min={values.amin().item():.3f}, "
        f"max={values.amax().item():.3f}, "
        f"mean={values.mean().item():.3f}, "
        f"std={values.std(unbiased=False).item():.3f}"
    )


def tensor_finite_stats(name: str, tensor: torch.Tensor) -> dict:
    values = tensor.detach().float()
    finite = torch.isfinite(values)
    stats = {
        "name": name,
        "shape": tuple(values.shape),
        "finite": bool(finite.all().detach().cpu()),
        "bad_values": int((~finite).sum().detach().cpu()),
    }
    if bool(finite.any().detach().cpu()):
        finite_values = values[finite]
        stats.update(
            {
                "min": float(finite_values.amin().detach().cpu()),
                "max": float(finite_values.amax().detach().cpu()),
                "absmax": float(finite_values.abs().amax().detach().cpu()),
                "mean": float(finite_values.mean().detach().cpu()),
                "std": float(finite_values.std(unbiased=False).detach().cpu()),
            }
        )
    return stats


def format_finite_stats(stats: dict) -> str:
    if "min" not in stats:
        return (
            f"{stats['name']}: shape={stats['shape']} finite={stats['finite']} "
            f"bad={stats['bad_values']}"
        )
    return (
        f"{stats['name']}: shape={stats['shape']} finite={stats['finite']} "
        f"bad={stats['bad_values']} min={stats['min']:.4g} max={stats['max']:.4g} "
        f"absmax={stats['absmax']:.4g} mean={stats['mean']:.4g} std={stats['std']:.4g}"
    )


def iter_tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_tensors(item)


@contextmanager
def first_nonfinite_forward_trace(model):
    trace = {"first": None}
    handles = []

    def make_hook(module_name: str, module: nn.Module):
        def hook(_module, _inputs, output):
            if trace["first"] is not None:
                return
            for tensor_index, tensor in enumerate(iter_tensors(output)):
                if torch.isfinite(tensor).all():
                    continue
                trace["first"] = {
                    "module": module_name,
                    "module_type": type(module).__name__,
                    "tensor_index": tensor_index,
                    "output": tensor_finite_stats(
                        f"{module_name}.output[{tensor_index}]", tensor
                    ),
                }
                return

        return hook

    try:
        for module_name, module in model.named_modules():
            if not module_name:
                continue
            handles.append(module.register_forward_hook(make_hook(module_name, module)))
        yield trace
    finally:
        for handle in handles:
            handle.remove()


def recurrent_parameter_diagnostics(model) -> list[dict]:
    diagnostics = []
    base = base_model(model)
    for block_index, block in enumerate(base.blocks):
        block = unwrap_compiled_module(block)
        attn = block.self_attn
        ssm_memory = getattr(attn, "ssm_memory", None)
        if ssm_memory is not None:
            diagnostics.append(
                {
                    "block": block_index,
                    "kind": "ssm",
                    "rho": tensor_finite_stats(
                        f"blocks.{block_index}.self_attn.ssm_memory.rho",
                        ssm_memory.rho,
                    ),
                    "decay": tensor_finite_stats(
                        f"blocks.{block_index}.self_attn.ssm_memory.decay",
                        torch.sigmoid(ssm_memory.decay_logit),
                    ),
                    "input_gain": tensor_finite_stats(
                        f"blocks.{block_index}.self_attn.ssm_memory.input_gain",
                        torch.sigmoid(ssm_memory.input_logit),
                    ),
                    "output_gain": tensor_finite_stats(
                        f"blocks.{block_index}.self_attn.ssm_memory.output_gain",
                        ssm_memory.output_gain,
                    ),
                    "skip_gain": tensor_finite_stats(
                        f"blocks.{block_index}.self_attn.ssm_memory.skip_gain",
                        ssm_memory.skip_gain,
                    ),
                }
            )
    return diagnostics


def video_cthw_to_uint8_frames(
    video: torch.Tensor, output_size: int = VIDEO_OUTPUT_SIZE
) -> torch.Tensor:
    frames = ((video.float().clamp(-1, 1) + 1.0) * 0.5).permute(1, 0, 2, 3).contiguous()
    if frames.shape[-2:] != (int(output_size), int(output_size)):
        frames = F.interpolate(
            frames,
            size=(int(output_size), int(output_size)),
            mode="bilinear",
            align_corners=False,
        )
    return (
        (255.0 * frames.permute(0, 2, 3, 1)).round().clamp(0, 255).to(torch.uint8).cpu()
    )


def encode_rgb_frames_mp4(frames: torch.Tensor, fps: int = EVAL_FPS) -> bytes:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must have shape (time, height, width, 3)")
    frames = frames.detach().to(torch.uint8).cpu()
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=int(fps))
        stream.width = int(frames.shape[2])
        stream.height = int(frames.shape[1])
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "veryfast"}
        for frame in frames.numpy():
            video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


def _pad_video_frames(frames: torch.Tensor, frame_count: int) -> torch.Tensor:
    frame_count = int(frame_count)
    if frames.shape[0] >= frame_count:
        return frames[:frame_count]
    if frames.shape[0] < 1:
        raise ValueError("Cannot pad a video with zero frames")
    padding = frames[-1:].expand(frame_count - frames.shape[0], -1, -1, -1)
    return torch.cat([frames, padding], dim=0)


def error_to_uint8_frames(
    error: torch.Tensor, vmax: float = EVAL_ERROR_VMAX
) -> torch.Tensor:
    x = (error.float() / max(float(vmax), 1e-12)).clamp(0.0, 1.0)
    red = x
    green = (1.35 * x - 0.35).clamp(0.0, 1.0)
    blue = (2.0 * x - 1.5).clamp(0.0, 1.0)
    return (
        (255.0 * torch.stack((red, green, blue), dim=-1))
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .cpu()
    )


def prediction_triplet_frames(
    pred_video,
    gt_video,
    output_size: int = VIDEO_OUTPUT_SIZE,
    frame_alignment: str = "shortest",
) -> torch.Tensor:
    pred = video_cthw_to_uint8_frames(pred_video, output_size=output_size)
    gt = video_cthw_to_uint8_frames(gt_video, output_size=output_size)
    error_frame_count = min(gt.shape[0], pred.shape[0])
    if error_frame_count < 1:
        raise ValueError("prediction and ground-truth videos need at least one frame")
    error = (
        (
            gt[:error_frame_count].float() / 255.0
            - pred[:error_frame_count].float() / 255.0
        )
        .abs()
        .mean(dim=-1)
    )
    err = error_to_uint8_frames(error)

    if frame_alignment == "shortest":
        frame_count = error_frame_count
        pred = pred[:frame_count]
        gt = gt[:frame_count]
        err = err[:frame_count]
    elif frame_alignment == "longest":
        frame_count = max(gt.shape[0], pred.shape[0])
        pred = _pad_video_frames(pred, frame_count)
        gt = _pad_video_frames(gt, frame_count)
        err = _pad_video_frames(err, frame_count)
    else:
        raise ValueError(
            "frame_alignment must be either 'shortest' or 'longest', "
            f"got {frame_alignment!r}"
        )
    return torch.cat((pred, gt, err), dim=2).contiguous()


def display_rgb_frames(frames: torch.Tensor, fps: int = EVAL_FPS):
    video_b64 = base64.b64encode(encode_rgb_frames_mp4(frames, fps=fps)).decode("ascii")
    height = int(frames.shape[1])
    width = int(frames.shape[2])
    html = HTML(
        f"<video autoplay loop muted playsinline controls "
        f'width="{width}" height="{height}" '
        f'style="display:block;width:{width}px;height:{height}px;margin:0;padding:0;border:0;line-height:0">'
        f'<source src="data:video/mp4;base64,{video_b64}" type="video/mp4">'
        f"</video>"
    )
    display(html)
    return html


def display_side_by_side_videos(
    *videos,
    output_size: int = VIDEO_OUTPUT_SIZE,
    fps: int = EVAL_FPS,
    frame_alignment: str = "shortest",
):
    if len(videos) < 1:
        raise ValueError("display_side_by_side_videos expects at least one video")
    video_frames = [
        video_cthw_to_uint8_frames(video, output_size=output_size) for video in videos
    ]
    if frame_alignment == "shortest":
        frame_count = min(frames.shape[0] for frames in video_frames)
        video_frames = [frames[:frame_count] for frames in video_frames]
    elif frame_alignment == "longest":
        frame_count = max(frames.shape[0] for frames in video_frames)
        video_frames = [
            _pad_video_frames(frames, frame_count) for frames in video_frames
        ]
    else:
        raise ValueError(
            "frame_alignment must be either 'shortest' or 'longest', "
            f"got {frame_alignment!r}"
        )
    frames = torch.cat(video_frames, dim=2).contiguous()
    return display_rgb_frames(frames, fps=fps)


def display_prediction_triplet(
    pred_video,
    gt_video,
    output_size: int = VIDEO_OUTPUT_SIZE,
    fps: int = EVAL_FPS,
    frame_alignment: str = "shortest",
):
    frames = prediction_triplet_frames(
        pred_video,
        gt_video,
        output_size=output_size,
        frame_alignment=frame_alignment,
    )
    return display_rgb_frames(frames, fps=fps)


def display_frame_loss_histogram(
    frame_loss_ema,
    frame_loss_indices=None,
    *,
    step: Optional[int] = None,
    max_frame_index: Optional[int] = None,
):
    if frame_loss_ema is None:
        return None
    values = torch.as_tensor(frame_loss_ema, dtype=torch.float32).detach().cpu()
    if values.numel() < 1:
        return None
    if frame_loss_indices is None:
        indices = list(range(1, int(values.numel()) + 1))
    else:
        indices = [int(index) for index in frame_loss_indices]
    if len(indices) != int(values.numel()):
        raise ValueError(
            f"frame_loss_indices length {len(indices)} does not match "
            f"frame_loss_ema length {int(values.numel())}"
        )
    if max_frame_index is not None:
        max_frame_index = int(max_frame_index)
        keep = [
            offset for offset, index in enumerate(indices) if index <= max_frame_index
        ]
        if not keep:
            return None
        values = values[keep]
        indices = [indices[offset] for offset in keep]
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        warnings.warn(f"Could not plot frame-loss histogram: {exc}")
        return None

    width = max(6.0, min(14.0, 0.45 * len(indices) + 3.0))
    fig, ax = plt.subplots(figsize=(width, 3.5))
    ax.bar(indices, values.numpy(), width=0.8)
    title = "Training loss EMA by latent frame"
    if step is not None:
        title = f"{title} at step {int(step)}"
    if max_frame_index is not None:
        title = f"{title} (<= frame {max_frame_index})"
    ax.set_title(title)
    ax.set_xlabel("latent frame index")
    ax.set_ylabel("velocity MSE EMA")
    ax.set_xticks(indices)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    display(fig)
    plt.close(fig)
    return fig


def display_layer_drop_loss_histogram(
    layer_loss_ema,
    layer_loss_counts=None,
    layer_loss_indices=None,
    *,
    step: Optional[int] = None,
):
    if layer_loss_ema is None:
        return None
    values = torch.as_tensor(layer_loss_ema, dtype=torch.float32).detach().cpu()
    if values.numel() < 1:
        return None
    if layer_loss_indices is None:
        indices = list(range(int(values.numel())))
    else:
        indices = [int(index) for index in layer_loss_indices]
    if len(indices) != int(values.numel()):
        raise ValueError(
            f"layer_loss_indices length {len(indices)} does not match "
            f"layer_loss_ema length {int(values.numel())}"
        )
    if layer_loss_counts is None:
        counts = torch.isfinite(values).to(dtype=torch.float32)
    else:
        counts = torch.as_tensor(layer_loss_counts, dtype=torch.float32).detach().cpu()
        if counts.numel() != values.numel():
            raise ValueError(
                f"layer_loss_counts length {int(counts.numel())} does not match "
                f"layer_loss_ema length {int(values.numel())}"
            )
    observed = counts > 0
    if not bool(observed.any()):
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        warnings.warn(f"Could not plot layer-drop loss histogram: {exc}")
        return None

    plot_values = torch.where(observed, values, torch.zeros_like(values))
    colors = ["#4c78a8" if bool(seen) else "#d3d3d3" for seen in observed.tolist()]
    width = max(8.0, min(18.0, 0.35 * len(indices) + 3.0))
    fig, ax = plt.subplots(figsize=(width, 3.8))
    positions = list(range(len(indices)))
    labels = [
        "none" if int(index) == NO_LAYER_DROP_BUCKET else str(int(index))
        for index in indices
    ]
    ax.bar(positions, plot_values.numpy(), width=0.8, color=colors)
    title = "Training loss EMA by dropped transformer layer"
    if step is not None:
        title = f"{title} at step {int(step)}"
    ax.set_title(title)
    ax.set_xlabel("dropped transformer layer index")
    ax.set_ylabel("total loss EMA")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    if len(indices) > 20:
        for label in ax.get_xticklabels():
            label.set_rotation(90)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    display(fig)
    plt.close(fig)
    return fig


@torch.no_grad()
def evaluate_random_holdout(
    model,
    vae,
    holdout_samples,
    step: int,
    physics_mean,
    physics_std,
    *,
    frame_loss_ema=None,
    frame_loss_indices=None,
    layer_drop_loss_ema=None,
    layer_drop_loss_counts=None,
    layer_drop_loss_indices=None,
    latent_frame_count: Optional[int] = None,
):
    sample = random.choice(holdout_samples)
    full_clean_latents = load_sample_latents(sample)
    clean_latents = truncate_latent_frames(
        full_clean_latents, frame_count=latent_frame_count
    )
    physics = sample_physics(sample, physics_mean, physics_std)

    left_frames, right_frames = causal_window()
    if has_ssm_attention(model):
        if lora_adapter_parameters(model):
            require_lora_adapter_state(
                model, enabled=True, label="SSM LoRA windowed inference"
            )
        set_ssm_attention_enabled(model, True)
        eval_label = (
            f"ssm Wan flow {int(EVAL_INFERENCE_STEPS)} steps causal window L={left_frames}"
        )
    else:
        require_lora_adapter_state(model, enabled=True, label="LoRA windowed inference")
        eval_label = (
            f"lora Wan flow {int(EVAL_INFERENCE_STEPS)} steps causal window L={left_frames}"
        )
    lora_latents = regress_latents_from_first_frame(
        model,
        clean_latents,
        physics=physics,
        window_left_frames=left_frames,
        window_right_frames=right_frames,
    )

    lora_video = decode_latents_to_video(vae, lora_latents)
    ground_truth_video = decode_latents_to_video(vae, clean_latents)
    html = display_prediction_triplet(lora_video, ground_truth_video)
    print(
        f"step {step}: {sample.name} | "
        f"nu={sample.nu:.2e}, rho={sample.rho:.3g} | "
        f"left={eval_label} | "
        "middle=ground truth | "
        "right=absolute error"
    )
    print(
        tensor_range_summary("pred_latents", lora_latents),
        "|",
        tensor_range_summary("gt_latents", clean_latents),
        "|",
        tensor_range_summary("pred_video", lora_video),
        "|",
        tensor_range_summary("gt_video", ground_truth_video),
    )
    frame_loss_histogram = display_frame_loss_histogram(
        frame_loss_ema,
        frame_loss_indices,
        step=step,
        max_frame_index=(
            None if latent_frame_count is None else max(0, int(latent_frame_count) - 1)
        ),
    )
    layer_drop_loss_histogram = display_layer_drop_loss_histogram(
        layer_drop_loss_ema,
        layer_drop_loss_counts,
        layer_drop_loss_indices,
        step=step,
    )

    result = {
        "sample": sample,
        "html": html,
        "frame_loss_histogram": frame_loss_histogram,
        "layer_drop_loss_histogram": layer_drop_loss_histogram,
    }
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return result
