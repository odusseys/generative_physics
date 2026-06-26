# This file is executed by navier/autoreg_windowed_lora.py.
# Keep shared globals in that compatibility module namespace.

@dataclass(frozen=True)
class LatentSample:
    latents_path: Path
    metadata_path: Path
    nu: float
    rho: float
    latent_frame_count: int

    @property
    def name(self) -> str:
        if self.latents_path.name == "latents.safetensors":
            return self.latents_path.parent.name
        return self.latents_path.stem


def _metadata_number_from_prefix(
    metadata_path: Path,
    key: str,
    *,
    chunk_size: int = 8192,
    max_bytes: int = 262144,
) -> Optional[float]:
    pattern = re.compile(
        rb'"' + re.escape(key.encode("utf-8")) + rb'"\s*:\s*([-+0-9.eE]+)'
    )
    buffer = b""
    with Path(metadata_path).open("rb") as handle:
        while len(buffer) < max_bytes:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            buffer += chunk
            match = pattern.search(buffer)
            if match:
                return float(match.group(1))
    return None


def read_metadata_physics_fast(
    metadata_path: Path, require_rho: Optional[bool] = None
) -> Optional[tuple[float, float]]:
    require_rho = REQUIRE_RHO if require_rho is None else bool(require_rho)
    nu = _metadata_number_from_prefix(metadata_path, "nu")
    if nu is None:
        return None
    rho = _metadata_number_from_prefix(metadata_path, "rho")
    if rho is None:
        if require_rho:
            return None
        rho = 1.0
    return float(nu), float(rho)


def read_latent_frame_count(latents_path: Path) -> int:
    with safe_open(str(latents_path), framework="pt", device="cpu") as handle:
        if "latents" in handle.keys():
            key = "latents"
        else:
            keys = list(handle.keys())
            if len(keys) != 1:
                raise ValueError(
                    f"{latents_path} must contain a single tensor or a `latents` tensor; got keys={keys}"
                )
            key = keys[0]
        shape = tuple(handle.get_slice(key).get_shape())
    if len(shape) < 2:
        raise ValueError(
            f"{latents_path} tensor {key!r} must have a frame dimension; got shape={shape}"
        )
    return int(shape[1])


def _discovery_log(message: str) -> None:
    if TRAIN_INIT_LOGS:
        print(f"[sample discovery] {message}", flush=True)


def discover_latent_samples(
    latent_root: Optional[Path] = None, require_rho: Optional[bool] = None
):
    latent_root = LATENT_ROOT if latent_root is None else Path(latent_root)
    require_rho = REQUIRE_RHO if require_rho is None else bool(require_rho)
    samples = []
    latents_paths = sorted(Path(latent_root).glob("*/latents.safetensors"))
    _discovery_log(
        f"checking {len(latents_paths):,} latent files in {latent_root} "
        f"(min latent frames={MIN_LATENT_FRAME_COUNT})"
    )
    start_time = time.perf_counter()
    skipped_metadata = 0
    skipped_short = 0
    skipped_bad_latents = 0
    for index, latents_path in enumerate(latents_paths, start=1):
        checked_count = index - 1
        if TRAIN_INIT_LOGS and checked_count > 0 and checked_count % 500 == 0:
            elapsed = time.perf_counter() - start_time
            _discovery_log(
                f"checked {checked_count:,}/{len(latents_paths):,}; kept {len(samples):,}; elapsed {elapsed:.1f}s"
            )
        try:
            latent_frame_count = read_latent_frame_count(latents_path)
        except Exception:
            skipped_bad_latents += 1
            continue
        if (
            MIN_LATENT_FRAME_COUNT is not None
            and latent_frame_count < int(MIN_LATENT_FRAME_COUNT)
        ):
            skipped_short += 1
            continue
        metadata_path = latents_path.parent / "metadata.json"
        if not metadata_path.exists():
            skipped_metadata += 1
            continue
        physics = read_metadata_physics_fast(metadata_path, require_rho=require_rho)
        if physics is None:
            skipped_metadata += 1
            continue
        nu, rho = physics
        samples.append(
            LatentSample(
                latents_path=latents_path,
                metadata_path=metadata_path,
                nu=nu,
                rho=rho,
                latent_frame_count=latent_frame_count,
            )
        )
    _discovery_log(
        f"done in {time.perf_counter() - start_time:.1f}s; kept {len(samples):,}; "
        f"skipped_short={skipped_short:,}; skipped_bad_latents={skipped_bad_latents:,}; "
        f"skipped_metadata={skipped_metadata:,}"
    )
    if len(samples) <= HOLDOUT_COUNT:
        raise ValueError(
            f"Need more than {HOLDOUT_COUNT} latent files with rho/nu metadata, found {len(samples)} in {latent_root}"
        )
    return samples


def make_splits(
    samples, holdout_count: Optional[int] = None, seed: Optional[int] = None
):
    samples = list(samples)
    holdout_count = HOLDOUT_COUNT if holdout_count is None else int(holdout_count)
    seed = RANDOM_SEED if seed is None else int(seed)
    rng = random.Random(seed)
    rng.shuffle(samples)
    return samples[holdout_count:], samples[:holdout_count]


def physics_stats(samples):
    vals = torch.tensor(
        [[math.log(sample.nu), math.log(sample.rho)] for sample in samples],
        dtype=torch.float32,
    )
    mean = vals.mean(dim=0)
    std = vals.std(dim=0).clamp_min(1e-6)
    return mean, std


def sample_physics(
    sample: LatentSample, physics_mean: torch.Tensor, physics_std: torch.Tensor
) -> torch.Tensor:
    physics = torch.tensor(
        [math.log(sample.nu), math.log(sample.rho)], dtype=torch.float32
    )
    return (physics - physics_mean.float()) / physics_std.float()


def truncate_latent_frames(
    latents: torch.Tensor, frame_count: Optional[int] = None
) -> torch.Tensor:
    frame_count = LATENT_FRAME_COUNT if frame_count is None else frame_count
    if frame_count is None:
        return latents
    frame_count = int(frame_count)
    if frame_count < 1:
        raise ValueError("LATENT_FRAME_COUNT must be positive or None")
    return latents[:, :frame_count].contiguous()


def load_sample_latents(sample: LatentSample) -> torch.Tensor:
    return load_file(str(sample.latents_path))["latents"].float()


def stack_training_latents(latents, frame_count: Optional[int] = None) -> torch.Tensor:
    if isinstance(latents, torch.Tensor):
        return truncate_latent_frames(latents, frame_count=frame_count)
    return torch.stack(
        [truncate_latent_frames(latent, frame_count=frame_count) for latent in latents],
        dim=0,
    )


class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, samples, physics_mean, physics_std):
        self.samples = list(samples)
        self.physics_mean = physics_mean.float()
        self.physics_std = physics_std.float()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        latents = load_sample_latents(sample)
        physics = sample_physics(sample, self.physics_mean, self.physics_std)
        return {"latents": latents, "physics": physics, "sample_index": index}


def collate_batch(items):
    return {
        "latents": [item["latents"] for item in items],
        "physics": torch.stack([item["physics"] for item in items], dim=0),
        "sample_index": torch.tensor(
            [item["sample_index"] for item in items], dtype=torch.long
        ),
    }


def latent_patch_grid(
    latents: torch.Tensor,
    patch_size=(1, 2, 2),
    *,
    require_one_frame_tokens: bool = False,
):
    _, _, frames, height, width = latents.shape
    patch_f, patch_h, patch_w = patch_size
    patch_f, patch_h, patch_w = int(patch_f), int(patch_h), int(patch_w)
    if require_one_frame_tokens and patch_f != 1:
        raise ValueError(
            f"Frame-window attention expects temporal patch_size[0] == 1 so each token belongs to one latent frame; got {patch_size}"
        )
    if frames % patch_f != 0 or height % patch_h != 0 or width % patch_w != 0:
        raise ValueError(
            f"Latent shape {tuple(latents.shape)} is not divisible by Wan patch_size {patch_size}"
        )
    return frames // patch_f, height // patch_h, width // patch_w


def latent_seq_len(latents: torch.Tensor, patch_size=(1, 2, 2)) -> int:
    grid_f, grid_h, grid_w = latent_patch_grid(latents, patch_size)
    return grid_f * grid_h * grid_w


def spatial_tokens_per_latent_frame(latents: torch.Tensor, patch_size=(1, 2, 2)) -> int:
    _, grid_h, grid_w = latent_patch_grid(
        latents, patch_size, require_one_frame_tokens=True
    )
    return grid_h * grid_w


def assert_frame_window_alignment(latents: torch.Tensor, model) -> int:
    _validate_causal_window()
    spatial_tokens = spatial_tokens_per_latent_frame(
        latents, base_model(model).patch_size
    )
    left_frames, right_frames = causal_window()
    token_window = (left_frames + 1 + right_frames) * spatial_tokens
    if token_window % spatial_tokens != 0:
        raise AssertionError("Token window must be divisible by spatial token count")
    return spatial_tokens


def flow_token_timesteps(
    latents: torch.Tensor,
    patch_size,
    future_timesteps=0.0,
) -> torch.Tensor:
    if latents.dim() == 4:
        latents_for_grid = latents.unsqueeze(0)
        batch_size = 1
    elif latents.dim() == 5:
        latents_for_grid = latents
        batch_size = int(latents.shape[0])
    else:
        raise ValueError(
            f"Expected latents with shape [B,C,F,H,W] or [C,F,H,W], got {tuple(latents.shape)}"
        )
    grid_f, grid_h, grid_w = latent_patch_grid(
        latents_for_grid, patch_size, require_one_frame_tokens=True
    )
    future_timesteps = torch.as_tensor(
        future_timesteps,
        device=latents.device,
        dtype=torch.float32,
    )
    if future_timesteps.dim() == 0:
        future_timesteps = future_timesteps.expand(batch_size)
    else:
        future_timesteps = future_timesteps.reshape(-1)
        if future_timesteps.numel() == 1:
            future_timesteps = future_timesteps.expand(batch_size)
        elif future_timesteps.numel() != batch_size:
            raise ValueError(
                f"future_timesteps must be scalar or one value per batch item; "
                f"got {int(future_timesteps.numel())} values for batch {batch_size}"
            )

    spatial_tokens = grid_h * grid_w
    frame_timesteps = future_timesteps.view(batch_size, 1).expand(batch_size, grid_f)
    frame_timesteps = frame_timesteps.clone()
    frame_timesteps[:, 0] = 0.0
    return frame_timesteps.repeat_interleave(spatial_tokens, dim=1)

