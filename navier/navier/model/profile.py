# This file is executed by navier/autoreg_windowed_lora.py.
# Keep shared globals in that compatibility module namespace.

def _synchronize_if_cuda(device: torch.device = DEVICE) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _benchmark_latent_shape(
    latent_template=None, sample: Optional[LatentSample] = None
):
    if latent_template is None:
        if sample is None:
            raise ValueError("Pass latent_template or sample to infer C/H/W")
        latent_template = load_sample_latents(sample)
    if not isinstance(latent_template, torch.Tensor):
        latent_template = torch.as_tensor(latent_template)
    if latent_template.dim() == 5:
        latent_template = latent_template[0]
    if latent_template.dim() != 4:
        raise ValueError(
            "latent_template must have shape [C, F, H, W] or [B, C, F, H, W]"
        )
    channels, _, height, width = latent_template.shape
    return int(channels), int(height), int(width)


def benchmark_forward_frame_counts(
    model,
    *,
    frame_counts=range(5, 41),
    latent_template=None,
    sample: Optional[LatentSample] = None,
    batch_size: int = 1,
    warmup: int = 2,
    repeats: int = 5,
    physics: Optional[torch.Tensor] = None,
    ssm_enabled: Optional[bool] = None,
    window_left_frames: Optional[int] = None,
    window_right_frames: Optional[int] = None,
    requires_grad: bool = False,
    train_mode: bool = False,
    seed: int = RANDOM_SEED,
):
    """Benchmark one model forward over different latent frame counts.

    The warmup passes are reported separately because torch.compile/regional
    compilation can make the first run for each sequence length much slower.
    """
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    if int(warmup) < 0 or int(repeats) < 1:
        raise ValueError("warmup must be >= 0 and repeats must be >= 1")

    base = base_model(model)
    channels, height, width = _benchmark_latent_shape(
        latent_template=latent_template, sample=sample
    )
    frame_counts = [int(frame_count) for frame_count in frame_counts]
    if any(frame_count < 1 for frame_count in frame_counts):
        raise ValueError("frame_counts must all be positive")

    old_training = model.training
    model.train(bool(train_mode))
    left_frames = (
        WINDOW_LEFT_FRAMES if window_left_frames is None else window_left_frames
    )
    right_frames = 0 if window_right_frames is None else window_right_frames
    use_ssm = has_ssm_attention(model) if ssm_enabled is None else bool(ssm_enabled)

    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(int(seed))
    grad_context = torch.enable_grad if requires_grad else torch.no_grad
    results = []

    try:
        for frame_count in frame_counts:
            clean = torch.randn(
                int(batch_size),
                channels,
                frame_count,
                height,
                width,
                device=DEVICE,
                dtype=TRAIN_DTYPE,
                generator=generator,
            )
            x_t = torch.randn_like(clean)
            x_t[:, :, 0:1] = clean[:, :, 0:1]
            if requires_grad:
                x_t.requires_grad_(True)
            seq_len = latent_seq_len(clean, base.patch_size)
            timestep_tokens = flow_token_timesteps(
                x_t,
                base.patch_size,
                torch.full(
                    (int(batch_size),),
                    0.5 * float(WAN_NUM_TRAIN_TIMESTEPS),
                    device=DEVICE,
                    dtype=torch.float32,
                ),
            )
            if physics is None:
                physics_batch = torch.zeros(
                    int(batch_size), 2, device=DEVICE, dtype=torch.float32
                )
            else:
                physics_batch = physics.to(device=DEVICE, dtype=torch.float32)
                if physics_batch.dim() == 1:
                    physics_batch = physics_batch.unsqueeze(0)
                if physics_batch.shape[0] == 1 and int(batch_size) > 1:
                    physics_batch = physics_batch.expand(int(batch_size), -1)
                if tuple(physics_batch.shape) != (int(batch_size), 2):
                    raise ValueError(
                        f"physics must broadcast to ({int(batch_size)}, 2), "
                        f"got {tuple(physics_batch.shape)}"
                    )

            def run_forward():
                with grad_context():
                    pred = model_velocity_with_attention(
                        model,
                        x_t,
                        seq_len,
                        physics_batch,
                        window_left_frames=left_frames,
                        window_right_frames=right_frames,
                        first_frame=clean[:, :, 0:1],
                        timestep_tokens=timestep_tokens,
                        ssm_enabled=use_ssm,
                    )
                    if requires_grad:
                        pred.float().mean()
                    return pred

            warmup_times = []
            for _ in range(int(warmup)):
                _synchronize_if_cuda(DEVICE)
                start = time.perf_counter()
                output = run_forward()
                _synchronize_if_cuda(DEVICE)
                warmup_times.append((time.perf_counter() - start) * 1000.0)
                del output

            if DEVICE.type == "cuda":
                torch.cuda.reset_peak_memory_stats(DEVICE)
            gpu_times = []
            wall_times = []
            for _ in range(int(repeats)):
                if DEVICE.type == "cuda":
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    _synchronize_if_cuda(DEVICE)
                    wall_start = time.perf_counter()
                    start_event.record()
                    output = run_forward()
                    end_event.record()
                    _synchronize_if_cuda(DEVICE)
                    wall_times.append((time.perf_counter() - wall_start) * 1000.0)
                    gpu_times.append(float(start_event.elapsed_time(end_event)))
                else:
                    wall_start = time.perf_counter()
                    output = run_forward()
                    wall_times.append((time.perf_counter() - wall_start) * 1000.0)
                del output

            peak_allocated_mib = None
            peak_reserved_mib = None
            if DEVICE.type == "cuda":
                peak_allocated_mib = torch.cuda.max_memory_allocated(DEVICE) / (1024**2)
                peak_reserved_mib = torch.cuda.max_memory_reserved(DEVICE) / (1024**2)

            timing_source = "cuda_event" if gpu_times else "wall"
            times = gpu_times if gpu_times else wall_times
            results.append(
                {
                    "frames": frame_count,
                    "seq_len": int(seq_len),
                    "batch_size": int(batch_size),
                    "ssm_enabled": bool(use_ssm),
                    "train_mode": bool(train_mode),
                    "requires_grad": bool(requires_grad),
                    "warmup_ms_total": sum(warmup_times),
                    "warmup_ms_last": warmup_times[-1] if warmup_times else None,
                    "forward_ms_mean": sum(times) / len(times),
                    "forward_ms_min": min(times),
                    "forward_ms_max": max(times),
                    "wall_ms_mean": sum(wall_times) / len(wall_times),
                    "timing_source": timing_source,
                    "peak_allocated_mib": peak_allocated_mib,
                    "peak_reserved_mib": peak_reserved_mib,
                }
            )

            del clean, x_t, physics_batch
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        model.train(old_training)

    return results


def print_forward_benchmark_results(results) -> None:
    if not results:
        print("No benchmark results.")
        return
    columns = [
        "frames",
        "seq_len",
        "forward_ms_mean",
        "forward_ms_min",
        "forward_ms_max",
        "warmup_ms_total",
        "peak_allocated_mib",
    ]
    header = " | ".join(columns)
    print(header)
    print(" | ".join("---" for _ in columns))
    for row in results:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            elif value is None:
                values.append("-")
            else:
                values.append(str(value))
        print(" | ".join(values))


PROFILE_DEFAULT_LEAF_TYPE_NAMES = {
    "Conv1d",
    "Conv2d",
    "Dropout",
    "Embedding",
    "GELU",
    "Identity",
    "LayerNorm",
    "Linear",
    "ModuleDict",
    "RMSNorm",
    "SiLU",
}
PROFILE_DEFAULT_INTERMEDIATE_NAMES = {
    "ffn",
    "first_frame_adaln",
    "first_frame_conditioner",
    "norm1",
    "norm2",
    "physics_adaln",
    "self_attn",
}


def _module_direct_parameter_count(module, *, trainable_only: bool = False) -> int:
    total = 0
    for parameter in module.parameters(recurse=False):
        if trainable_only and not parameter.requires_grad:
            continue
        total += int(parameter.numel())
    return total


def _is_leaf_module(module) -> bool:
    return next(module.children(), None) is None


def _profile_module_type_name(module) -> str:
    module = unwrap_compiled_module(module)
    return type(module).__name__


def _profile_layer_index_from_name(name: str) -> Optional[int]:
    match = re.search(r"(?:^|\.)blocks\.(\d+)(?:\.|$)", str(name))
    if match is None:
        return None
    return int(match.group(1))


def _profile_is_block_name(name: str) -> bool:
    return re.search(r"(?:^|\.)blocks\.\d+$", str(name)) is not None


def _iter_named_modules_once(model):
    try:
        return model.named_modules(remove_duplicate=True)
    except TypeError:
        return model.named_modules()


class ModuleRuntimeProfiler:
    """Forward-hook profiler grouped by transformer layer and module type.

    The timings are Python wall-clock timings. With ``synchronize_cuda=True``,
    CUDA is synchronized at every profiled module boundary so the numbers are
    much more interpretable, at the cost of considerable profiler overhead.
    """

    def __init__(
        self,
        model,
        *,
        include_leaf_modules: bool = True,
        include_block_modules: bool = True,
        include_intermediate_modules: bool = True,
        leaf_type_names=None,
        synchronize_cuda: bool = True,
        device: torch.device = DEVICE,
    ):
        self.model = model
        self.include_leaf_modules = bool(include_leaf_modules)
        self.include_block_modules = bool(include_block_modules)
        self.include_intermediate_modules = bool(include_intermediate_modules)
        if leaf_type_names is None:
            self.leaf_type_names = set(PROFILE_DEFAULT_LEAF_TYPE_NAMES)
        elif isinstance(leaf_type_names, str) and leaf_type_names.lower() == "all":
            self.leaf_type_names = None
        else:
            self.leaf_type_names = {str(value) for value in leaf_type_names}
        self.synchronize_cuda = bool(synchronize_cuda)
        self.device = device
        self.handles = []
        self.stack = []
        self.records = {}

    def _sync(self) -> None:
        if self.synchronize_cuda:
            _synchronize_if_cuda(self.device)

    def _should_profile(self, name: str, module) -> bool:
        if not name:
            return False
        module_type = _profile_module_type_name(module)
        if self.include_block_modules and _profile_is_block_name(name):
            return True
        if isinstance(unwrap_compiled_module(module), StateSpaceMemory):
            return True
        basename = name.rsplit(".", 1)[-1]
        if (
            self.include_intermediate_modules
            and basename in PROFILE_DEFAULT_INTERMEDIATE_NAMES
        ):
            return True
        if not self.include_leaf_modules or not _is_leaf_module(module):
            return False
        return self.leaf_type_names is None or module_type in self.leaf_type_names

    def _ensure_record(self, key, name: str, module) -> dict:
        record = self.records.get(key)
        if record is not None:
            return record
        is_block = _profile_is_block_name(name)
        layer_index = _profile_layer_index_from_name(name)
        record = {
            "name": name,
            "type": _profile_module_type_name(module),
            "layer": layer_index,
            "is_block": bool(is_block),
            "is_leaf": bool(_is_leaf_module(module)),
            "direct_params": _module_direct_parameter_count(module),
            "direct_trainable_params": _module_direct_parameter_count(
                module, trainable_only=True
            ),
            "calls": 0,
            "inclusive_ms": 0.0,
            "exclusive_ms": 0.0,
        }
        self.records[key] = record
        return record

    def _make_pre_hook(self, key):
        def hook(module, inputs):
            self._sync()
            self.stack.append(
                {
                    "key": key,
                    "start": time.perf_counter(),
                    "child_ms": 0.0,
                }
            )

        return hook

    def _make_post_hook(self, key, name: str, module):
        def hook(module, inputs, output):
            self._sync()
            end = time.perf_counter()
            if not self.stack:
                return
            frame = self.stack.pop()
            elapsed_ms = (end - frame["start"]) * 1000.0
            if frame["key"] != key:
                warnings.warn(
                    f"Profiler hook stack mismatch: expected {key}, got {frame['key']}"
                )
            exclusive_ms = max(0.0, elapsed_ms - float(frame["child_ms"]))
            record = self._ensure_record(key, name, module)
            record["calls"] += 1
            record["inclusive_ms"] += elapsed_ms
            record["exclusive_ms"] += exclusive_ms
            if self.stack:
                self.stack[-1]["child_ms"] += elapsed_ms

        return hook

    def __enter__(self):
        for name, module in _iter_named_modules_once(self.model):
            if not self._should_profile(name, module):
                continue
            key = id(module)
            self._ensure_record(key, name, module)
            self.handles.append(
                module.register_forward_pre_hook(self._make_pre_hook(key))
            )
            self.handles.append(
                module.register_forward_hook(self._make_post_hook(key, name, module))
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.stack.clear()
        return False

    def report(self):
        rows = [dict(record) for record in self.records.values()]
        for row in rows:
            calls = max(1, int(row["calls"]))
            row["inclusive_ms_per_call"] = float(row["inclusive_ms"]) / calls
            row["exclusive_ms_per_call"] = float(row["exclusive_ms"]) / calls

        by_layer = [
            dict(row)
            for row in rows
            if row.get("is_block") and row.get("layer") is not None
        ]
        by_layer.sort(key=lambda row: int(row["layer"]))

        by_type_accumulator = defaultdict(
            lambda: {
                "type": None,
                "modules": 0,
                "calls": 0,
                "inclusive_ms": 0.0,
                "exclusive_ms": 0.0,
                "direct_params": 0,
                "direct_trainable_params": 0,
            }
        )
        for row in rows:
            if row.get("is_block"):
                continue
            item = by_type_accumulator[row["type"]]
            item["type"] = row["type"]
            item["modules"] += 1
            item["calls"] += int(row["calls"])
            item["inclusive_ms"] += float(row["inclusive_ms"])
            item["exclusive_ms"] += float(row["exclusive_ms"])
            item["direct_params"] += int(row["direct_params"])
            item["direct_trainable_params"] += int(row["direct_trainable_params"])
        by_type = list(by_type_accumulator.values())
        for row in by_type:
            calls = max(1, int(row["calls"]))
            row["inclusive_ms_per_call"] = float(row["inclusive_ms"]) / calls
            row["exclusive_ms_per_call"] = float(row["exclusive_ms"]) / calls
        by_type.sort(key=lambda row: float(row["inclusive_ms"]), reverse=True)

        top_modules = [row for row in rows if int(row["calls"]) > 0]
        top_modules.sort(key=lambda row: float(row["inclusive_ms"]), reverse=True)
        return {
            "records": rows,
            "by_layer": by_layer,
            "by_module_type": by_type,
            "top_modules": top_modules,
        }


def _profile_format_value(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _print_profile_table(title: str, rows, columns, *, max_rows: Optional[int]) -> None:
    print(title)
    if not rows:
        print("  no rows")
        return
    shown = rows if max_rows is None else rows[: int(max_rows)]
    print(" | ".join(columns))
    print(" | ".join("---" for _ in columns))
    for row in shown:
        print(" | ".join(_profile_format_value(row.get(column)) for column in columns))
    if max_rows is not None and len(rows) > int(max_rows):
        print(f"... {len(rows) - int(max_rows):,} more rows")


def print_module_profile_report(report: dict, *, max_rows: int = 30) -> None:
    if not report:
        print("No module profile report.")
        return
    print("Profile totals")
    print(f"  mode: {report.get('mode')}")
    print(f"  repeats: {report.get('repeats')}")
    print(f"  include_backward: {report.get('include_backward')}")
    print(f"  synchronize_cuda: {report.get('synchronize_cuda')}")
    for key in ("forward_ms_mean", "backward_ms_mean", "total_ms_mean"):
        value = report.get(key)
        if value is not None:
            print(f"  {key}: {float(value):.3f}")
    print()
    _print_profile_table(
        "By transformer block, inclusive forward time",
        report.get("by_layer", []),
        [
            "layer",
            "calls",
            "inclusive_ms",
            "inclusive_ms_per_call",
            "exclusive_ms",
        ],
        max_rows=None,
    )
    print()
    _print_profile_table(
        "By module type, inclusive/exclusive forward time",
        report.get("by_module_type", []),
        [
            "type",
            "modules",
            "calls",
            "inclusive_ms",
            "exclusive_ms",
            "direct_trainable_params",
        ],
        max_rows=max_rows,
    )
    print()
    _print_profile_table(
        "Top individual modules by inclusive forward time",
        report.get("top_modules", []),
        [
            "name",
            "type",
            "layer",
            "calls",
            "inclusive_ms",
            "exclusive_ms",
        ],
        max_rows=max_rows,
    )


def _profile_model_from_state(state: dict):
    if "model" in state:
        return state["model"]
    if "student" in state:
        return state["student"]
    raise KeyError("state must contain 'model' or 'student'")


def _profile_batch_loss(batch, model, step: int):
    return ssm_training_batch(batch, model, step=step)


def profile_training_batch_by_module(
    model=None,
    state: Optional[dict] = None,
    *,
    batch=None,
    warmup: int = 1,
    repeats: int = 1,
    include_backward: bool = True,
    include_leaf_modules: bool = True,
    include_block_modules: bool = True,
    include_intermediate_modules: bool = True,
    leaf_type_names=None,
    synchronize_cuda: bool = True,
    print_report: bool = True,
    max_rows: int = 30,
):
    """Profile one real training batch by transformer layer and module type.

    Pass the live notebook objects to avoid rebuilding the model:

    ``profile = awl.profile_training_batch_by_module(model, state)``

    The layer and module-type breakdown is forward-hook based. If
    ``include_backward`` is true, total backward wall time is also reported, but
    backward work is not attributed to individual modules.
    """
    if state is None:
        state = build_ssm_training_objects()
    if model is None:
        model = _profile_model_from_state(state)
    train_loader = state.get("train_loader")
    if batch is None:
        if train_loader is None:
            raise KeyError("state must contain train_loader when batch is not provided")
        batch = next(iter(train_loader))

    old_training = model.training
    model.train(True)

    def run_once(*, grad_enabled: bool):
        grad_context = torch.enable_grad if grad_enabled else torch.no_grad
        with grad_context():
            return _profile_batch_loss(batch, model, step=0)

    try:
        for _ in range(int(warmup)):
            model.zero_grad(set_to_none=True)
            loss, _components = run_once(grad_enabled=bool(include_backward))
            if include_backward:
                loss.backward()
            model.zero_grad(set_to_none=True)
            _synchronize_if_cuda(DEVICE)

        profiler = ModuleRuntimeProfiler(
            model,
            include_leaf_modules=include_leaf_modules,
            include_block_modules=include_block_modules,
            include_intermediate_modules=include_intermediate_modules,
            leaf_type_names=leaf_type_names,
            synchronize_cuda=synchronize_cuda,
            device=DEVICE,
        )
        forward_times = []
        backward_times = []
        total_times = []
        losses = []
        with profiler:
            for _ in range(int(repeats)):
                model.zero_grad(set_to_none=True)
                _synchronize_if_cuda(DEVICE)
                total_start = time.perf_counter()
                forward_start = total_start
                loss, _components = run_once(grad_enabled=bool(include_backward))
                _synchronize_if_cuda(DEVICE)
                forward_end = time.perf_counter()
                forward_times.append((forward_end - forward_start) * 1000.0)
                losses.append(float(loss.detach().cpu()))
                if include_backward:
                    backward_start = time.perf_counter()
                    loss.backward()
                    _synchronize_if_cuda(DEVICE)
                    backward_times.append(
                        (time.perf_counter() - backward_start) * 1000.0
                    )
                total_times.append((time.perf_counter() - total_start) * 1000.0)
                model.zero_grad(set_to_none=True)
    finally:
        clear_dropped_transformer_layer(model)
        model.train(old_training)

    hook_report = profiler.report()
    report = {
        **hook_report,
        "mode": "ssm",
        "warmup": int(warmup),
        "repeats": int(repeats),
        "include_backward": bool(include_backward),
        "synchronize_cuda": bool(synchronize_cuda),
        "losses": losses,
        "forward_ms": forward_times,
        "backward_ms": backward_times,
        "total_ms": total_times,
        "forward_ms_mean": sum(forward_times) / max(1, len(forward_times)),
        "backward_ms_mean": (
            sum(backward_times) / len(backward_times) if backward_times else None
        ),
        "total_ms_mean": sum(total_times) / max(1, len(total_times)),
    }
    if print_report:
        print_module_profile_report(report, max_rows=max_rows)
    return report


profile_one_batch_by_module = profile_training_batch_by_module
