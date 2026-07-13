import math

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


NAVIER_STOKES_GRID_SIZE = 128
NAVIER_STOKES_DOMAIN_SIZE = 1.0
NAVIER_STOKES_FINAL_TIME = 1.0
NAVIER_STOKES_MULTIPLE_TIMES = (0.25, 0.5, 0.75, 1.0)
NAVIER_STOKES_DENSITY_MIN = 0.8
NAVIER_STOKES_DENSITY_MAX = 1.2
NAVIER_STOKES_VISCOSITY_MIN = 5e-4
NAVIER_STOKES_VISCOSITY_MAX = 3e-3
NAVIER_STOKES_MIN_COMPONENTS = 3
NAVIER_STOKES_MAX_COMPONENTS = 8
NAVIER_STOKES_SIGMA_MIN = 0.055
NAVIER_STOKES_SIGMA_MAX = 0.16
NAVIER_STOKES_INITIAL_SPEED_MIN = 0.35
NAVIER_STOKES_INITIAL_SPEED_MAX = 0.75
NAVIER_STOKES_CFL = 0.4
NAVIER_STOKES_CONDITIONING_NAMES = ("density", "kinematic_viscosity")
NAVIER_STOKES_CONDITIONING_TRANSFORMS = ("linear", "log")
NAVIER_STOKES_COVERAGE_SAMPLE_SIZES = (
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
)
NAVIER_STOKES_COVERAGE_DISTANCE_TOLERANCES = (0.5, 0.6, 0.7, 0.75)
NAVIER_STOKES_COVERAGE_QUANTILES = (0.9, 0.95, 0.99)
NAVIER_STOKES_CONDITIONING_MEAN = (
    0.5 * (NAVIER_STOKES_DENSITY_MIN + NAVIER_STOKES_DENSITY_MAX),
    0.5 * (math.log(NAVIER_STOKES_VISCOSITY_MIN) + math.log(NAVIER_STOKES_VISCOSITY_MAX)),
)
NAVIER_STOKES_CONDITIONING_STD = (
    (NAVIER_STOKES_DENSITY_MAX - NAVIER_STOKES_DENSITY_MIN) / math.sqrt(12.0),
    (math.log(NAVIER_STOKES_VISCOSITY_MAX) - math.log(NAVIER_STOKES_VISCOSITY_MIN))
    / math.sqrt(12.0),
)


def _validate_range(name, minimum, maximum, *, positive=False):
    minimum = float(minimum)
    maximum = float(maximum)
    if positive and minimum <= 0.0:
        raise ValueError(f"{name} minimum must be positive.")
    if maximum < minimum:
        raise ValueError(f"{name} maximum must be at least its minimum.")
    return minimum, maximum


def _spectral_operators(grid_size, domain_size, device, dtype):
    spacing = float(domain_size) / int(grid_size)
    frequencies = 2.0 * torch.pi * torch.fft.fftfreq(
        int(grid_size),
        d=spacing,
        device=device,
        dtype=dtype,
    )
    kx = frequencies[None, :]
    ky = frequencies[:, None]
    wavenumber_squared = kx.square() + ky.square()

    integer_modes = torch.fft.fftfreq(
        int(grid_size),
        d=1.0 / int(grid_size),
        device=device,
        dtype=dtype,
    )
    mode_x = integer_modes[None, :]
    mode_y = integer_modes[:, None]
    cutoff = int(grid_size) / 3.0
    dealias = (mode_x.abs() <= cutoff) & (mode_y.abs() <= cutoff)
    return kx, ky, wavenumber_squared, dealias


def _velocity_from_vorticity_hat(vorticity_hat, kx, ky, wavenumber_squared, mean_velocity=None):
    inverse_k2 = torch.where(
        wavenumber_squared > 0.0,
        wavenumber_squared.reciprocal(),
        torch.zeros_like(wavenumber_squared),
    )
    streamfunction_hat = vorticity_hat * inverse_k2
    velocity_x = torch.fft.ifft2(1j * ky * streamfunction_hat, dim=(-2, -1)).real
    velocity_y = torch.fft.ifft2(-1j * kx * streamfunction_hat, dim=(-2, -1)).real
    velocity = torch.stack((velocity_x, velocity_y), dim=1)
    if mean_velocity is not None:
        velocity = velocity + mean_velocity
    return velocity


@torch.no_grad()
def sample_divergence_free_velocity_batch(
    seeds,
    grid_size=NAVIER_STOKES_GRID_SIZE,
    domain_size=NAVIER_STOKES_DOMAIN_SIZE,
    min_components=NAVIER_STOKES_MIN_COMPONENTS,
    max_components=NAVIER_STOKES_MAX_COMPONENTS,
    sigma_min=NAVIER_STOKES_SIGMA_MIN,
    sigma_max=NAVIER_STOKES_SIGMA_MAX,
    speed_min=NAVIER_STOKES_INITIAL_SPEED_MIN,
    speed_max=NAVIER_STOKES_INITIAL_SPEED_MAX,
    density_min=NAVIER_STOKES_DENSITY_MIN,
    density_max=NAVIER_STOKES_DENSITY_MAX,
    viscosity_min=NAVIER_STOKES_VISCOSITY_MIN,
    viscosity_max=NAVIER_STOKES_VISCOSITY_MAX,
    device=None,
    dtype=torch.float32,
):
    """Sample smooth periodic velocity fields from streamfunction density mixtures."""
    seeds = [int(seed) for seed in seeds]
    if not seeds:
        raise ValueError("seeds must contain at least one seed.")
    grid_size = int(grid_size)
    if grid_size < 16:
        raise ValueError("grid_size must be at least 16.")
    if min_components < 1 or max_components < min_components:
        raise ValueError("component count must satisfy 1 <= min_components <= max_components.")
    sigma_min, sigma_max = _validate_range("sigma", sigma_min, sigma_max, positive=True)
    speed_min, speed_max = _validate_range("speed", speed_min, speed_max)
    density_min, density_max = _validate_range("density", density_min, density_max, positive=True)
    viscosity_min, viscosity_max = _validate_range(
        "kinematic viscosity",
        viscosity_min,
        viscosity_max,
        positive=True,
    )
    if speed_min < 0.0 or speed_max > 1.0:
        raise ValueError("sampled speed bounds must lie in [0, 1].")

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    coordinates = torch.arange(grid_size, device=device, dtype=dtype) / grid_size
    x = coordinates[None, :]
    y = coordinates[:, None]
    streamfunctions = torch.zeros((len(seeds), grid_size, grid_size), device=device, dtype=dtype)
    sample_params = []

    for sample_index, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        num_components = int(rng.integers(int(min_components), int(max_components) + 1))
        density = float(rng.uniform(density_min, density_max))
        log_viscosity = rng.uniform(math.log(viscosity_min), math.log(viscosity_max))
        kinematic_viscosity = float(math.exp(log_viscosity))
        target_max_speed = float(rng.uniform(speed_min, speed_max))

        streamfunction = torch.zeros((grid_size, grid_size), device=device, dtype=dtype)
        for _ in range(num_components):
            center_x, center_y = rng.uniform(0.0, 1.0, size=2)
            sigma_x, sigma_y = rng.uniform(sigma_min, sigma_max, size=2)
            weight = float(rng.normal())
            concentration_x = 1.0 / (4.0 * math.pi**2 * sigma_x**2)
            concentration_y = 1.0 / (4.0 * math.pi**2 * sigma_y**2)
            component = torch.exp(
                concentration_x * (torch.cos(2.0 * torch.pi * (x - center_x)) - 1.0)
                + concentration_y * (torch.cos(2.0 * torch.pi * (y - center_y)) - 1.0)
            )
            streamfunction.add_(component, alpha=weight)
        streamfunctions[sample_index] = streamfunction - streamfunction.mean()
        sample_params.append(
            {
                "seed": seed,
                "density": density,
                "kinematic_viscosity": kinematic_viscosity,
                "num_components": num_components,
                "initial_max_speed": target_max_speed,
            }
        )

    kx, ky, wavenumber_squared, dealias = _spectral_operators(
        grid_size,
        domain_size,
        device,
        dtype,
    )
    streamfunction_hat = torch.fft.fft2(streamfunctions, dim=(-2, -1)) * dealias
    velocity_x = torch.fft.ifft2(1j * ky * streamfunction_hat, dim=(-2, -1)).real
    velocity_y = torch.fft.ifft2(-1j * kx * streamfunction_hat, dim=(-2, -1)).real
    velocity = torch.stack((velocity_x, velocity_y), dim=1)
    current_max_speed = velocity.square().sum(dim=1).sqrt().amax(dim=(-2, -1)).clamp_min(1e-12)
    target_max_speed = torch.tensor(
        [params["initial_max_speed"] for params in sample_params],
        device=device,
        dtype=dtype,
    )
    velocity = velocity * (target_max_speed / current_max_speed)[:, None, None, None]
    return velocity, sample_params


def _advection_hat(vorticity_hat, kx, ky, wavenumber_squared, dealias, mean_velocity):
    velocity = _velocity_from_vorticity_hat(
        vorticity_hat,
        kx,
        ky,
        wavenumber_squared,
        mean_velocity=mean_velocity,
    )
    gradient_x = torch.fft.ifft2(1j * kx * vorticity_hat, dim=(-2, -1)).real
    gradient_y = torch.fft.ifft2(1j * ky * vorticity_hat, dim=(-2, -1)).real
    advection = velocity[:, 0] * gradient_x + velocity[:, 1] * gradient_y
    return -torch.fft.fft2(advection, dim=(-2, -1)) * dealias


def _integrating_factor_rk4_step(
    vorticity_hat,
    dt,
    viscosity,
    kx,
    ky,
    wavenumber_squared,
    dealias,
    mean_velocity,
):
    half_diffusion = torch.exp(-0.5 * dt * viscosity * wavenumber_squared)
    full_diffusion = half_diffusion.square()
    k1 = _advection_hat(vorticity_hat, kx, ky, wavenumber_squared, dealias, mean_velocity)
    stage_a = half_diffusion * (vorticity_hat + 0.5 * dt * k1)
    k2 = _advection_hat(stage_a, kx, ky, wavenumber_squared, dealias, mean_velocity)
    stage_b = half_diffusion * vorticity_hat + 0.5 * dt * k2
    k3 = _advection_hat(stage_b, kx, ky, wavenumber_squared, dealias, mean_velocity)
    stage_c = full_diffusion * vorticity_hat + dt * half_diffusion * k3
    k4 = _advection_hat(stage_c, kx, ky, wavenumber_squared, dealias, mean_velocity)
    result = full_diffusion * vorticity_hat + (dt / 6.0) * (
        full_diffusion * k1
        + 2.0 * half_diffusion * (k2 + k3)
        + k4
    )
    result = result * dealias
    result[..., 0, 0] = 0.0
    return result


@torch.no_grad()
def solve_navier_stokes_2d(
    initial_velocity,
    kinematic_viscosity,
    save_times,
    domain_size=NAVIER_STOKES_DOMAIN_SIZE,
    cfl=NAVIER_STOKES_CFL,
    progress=False,
):
    """Solve periodic unforced 2D incompressible Navier-Stokes in vorticity form."""
    initial_velocity = torch.as_tensor(initial_velocity)
    if initial_velocity.ndim == 3:
        initial_velocity = initial_velocity.unsqueeze(0)
    if initial_velocity.ndim != 4 or initial_velocity.shape[1] != 2:
        raise ValueError("initial_velocity must have shape [batch, 2, ny, nx].")
    if initial_velocity.shape[-2] != initial_velocity.shape[-1]:
        raise ValueError("initial_velocity must be sampled on a square grid.")
    if not initial_velocity.is_floating_point():
        initial_velocity = initial_velocity.float()

    save_times = np.asarray(save_times, dtype=np.float64)
    if save_times.ndim != 1 or save_times.size == 0:
        raise ValueError("save_times must be a non-empty one-dimensional sequence.")
    if abs(float(save_times[0])) > 1e-12 or np.any(np.diff(save_times) <= 0.0):
        raise ValueError("save_times must start at zero and be strictly increasing.")
    if cfl <= 0.0:
        raise ValueError("cfl must be positive.")

    batch_size = initial_velocity.shape[0]
    grid_size = initial_velocity.shape[-1]
    device = initial_velocity.device
    dtype = initial_velocity.dtype
    viscosity = torch.as_tensor(kinematic_viscosity, device=device, dtype=dtype).reshape(-1)
    if viscosity.numel() == 1:
        viscosity = viscosity.repeat(batch_size)
    if viscosity.numel() != batch_size or torch.any(viscosity <= 0.0):
        raise ValueError("kinematic_viscosity must contain one positive value per sample.")
    viscosity = viscosity[:, None, None]

    kx, ky, wavenumber_squared, dealias = _spectral_operators(
        grid_size,
        domain_size,
        device,
        dtype,
    )
    mean_velocity = initial_velocity.mean(dim=(-2, -1), keepdim=True)
    velocity_x_hat = torch.fft.fft2(initial_velocity[:, 0], dim=(-2, -1))
    velocity_y_hat = torch.fft.fft2(initial_velocity[:, 1], dim=(-2, -1))
    vorticity_hat = (1j * kx * velocity_y_hat - 1j * ky * velocity_x_hat) * dealias
    vorticity_hat[..., 0, 0] = 0.0

    max_dt = float(cfl) * (float(domain_size) / grid_size)
    interval_steps = [max(1, int(math.ceil(float(delta) / max_dt))) for delta in np.diff(save_times)]
    progress_bar = tqdm(
        total=sum(interval_steps),
        desc="2D Navier-Stokes steps",
        leave=False,
        disable=not progress,
    )
    trajectory = [
        _velocity_from_vorticity_hat(
            vorticity_hat,
            kx,
            ky,
            wavenumber_squared,
            mean_velocity=mean_velocity,
        )
    ]
    try:
        for target_time, num_steps in zip(save_times[1:], interval_steps):
            start_time = float(save_times[len(trajectory) - 1])
            dt = (float(target_time) - start_time) / num_steps
            for _ in range(num_steps):
                vorticity_hat = _integrating_factor_rk4_step(
                    vorticity_hat,
                    dt,
                    viscosity,
                    kx,
                    ky,
                    wavenumber_squared,
                    dealias,
                    mean_velocity,
                )
                progress_bar.update(1)
            trajectory.append(
                _velocity_from_vorticity_hat(
                    vorticity_hat,
                    kx,
                    ky,
                    wavenumber_squared,
                    mean_velocity=mean_velocity,
                )
            )
    finally:
        progress_bar.close()
    return torch.stack(trajectory, dim=1)


@torch.no_grad()
def simulate_navier_stokes_2d_batch(
    seeds,
    grid_size=NAVIER_STOKES_GRID_SIZE,
    final_time=NAVIER_STOKES_FINAL_TIME,
    save_times=None,
    domain_size=NAVIER_STOKES_DOMAIN_SIZE,
    device=None,
    progress=False,
):
    if save_times is None:
        save_times = np.array([0.0, float(final_time)], dtype=np.float64)
    else:
        save_times = np.asarray(save_times, dtype=np.float64)
    initial_velocity, sample_params = sample_divergence_free_velocity_batch(
        seeds,
        grid_size=grid_size,
        domain_size=domain_size,
        device=device,
    )
    viscosities = [params["kinematic_viscosity"] for params in sample_params]
    trajectory = solve_navier_stokes_2d(
        initial_velocity,
        viscosities,
        save_times,
        domain_size=domain_size,
        progress=progress,
    )
    return save_times, trajectory, sample_params


def velocity_to_rgb(velocity, output_size=None):
    """Encode R=(unit_x+1)/2, G=(unit_y+1)/2, and B=speed in [0, 1]."""
    velocity = torch.as_tensor(velocity, dtype=torch.float32)
    if velocity.shape[-3] != 2:
        raise ValueError("velocity must have a two-component axis at position -3.")
    if output_size is not None and velocity.shape[-2:] != (int(output_size), int(output_size)):
        leading_shape = velocity.shape[:-3]
        velocity = F.interpolate(
            velocity.reshape(-1, 2, *velocity.shape[-2:]),
            size=(int(output_size), int(output_size)),
            mode="bilinear",
            align_corners=False,
        ).reshape(*leading_shape, 2, int(output_size), int(output_size))

    speed = velocity.square().sum(dim=-3).sqrt()
    direction = velocity / speed.clamp_min(1e-12).unsqueeze(-3)
    direction = torch.where(speed.unsqueeze(-3) > 1e-12, direction, torch.zeros_like(direction))
    red = 0.5 * (direction[..., 0, :, :] + 1.0)
    green = 0.5 * (direction[..., 1, :, :] + 1.0)
    blue = speed.clamp(0.0, 1.0)
    return torch.stack((red, green, blue), dim=-1).clamp(0.0, 1.0)


@torch.no_grad()
def _initial_condition_rgb_features(
    seeds,
    simulation_grid_size,
    feature_grid_size,
    generation_batch_size,
    device,
):
    features = []
    seeds = [int(seed) for seed in seeds]
    for start in range(0, len(seeds), int(generation_batch_size)):
        velocity, _ = sample_divergence_free_velocity_batch(
            seeds[start : start + int(generation_batch_size)],
            grid_size=simulation_grid_size,
            device=device,
        )
        rgb = velocity_to_rgb(velocity).permute(0, 3, 1, 2)
        if rgb.shape[-2:] != (int(feature_grid_size), int(feature_grid_size)):
            rgb = F.interpolate(
                rgb,
                size=(int(feature_grid_size), int(feature_grid_size)),
                mode="area",
            )
        features.append(rgb.flatten(1).cpu())
    return torch.cat(features)


def _estimate_coverage_sample_counts(sample_sizes, distance_quantiles, tolerances):
    sample_sizes = np.asarray(sample_sizes, dtype=np.float64)
    distance_quantiles = np.asarray(distance_quantiles, dtype=np.float64)
    tolerances = np.asarray(tolerances, dtype=np.float64)
    estimates = np.empty((distance_quantiles.shape[0], tolerances.size), dtype=np.float64)
    extrapolated = np.zeros_like(estimates, dtype=bool)
    effective_dimensions = np.empty(distance_quantiles.shape[0], dtype=np.float64)

    fit_count = min(5, sample_sizes.size)
    fit_x = np.log(sample_sizes[-fit_count:])
    for quantile_index, distances in enumerate(distance_quantiles):
        slope, intercept = np.polyfit(fit_x, np.log(distances[-fit_count:]), 1)
        effective_dimensions[quantile_index] = -1.0 / slope if slope < 0.0 else np.inf
        for tolerance_index, tolerance in enumerate(tolerances):
            crossing = np.flatnonzero(distances <= tolerance)
            if crossing.size and crossing[0] > 0:
                upper = int(crossing[0])
                lower = upper - 1
                fraction = (
                    (math.log(tolerance) - math.log(distances[lower]))
                    / (math.log(distances[upper]) - math.log(distances[lower]))
                )
                estimates[quantile_index, tolerance_index] = math.exp(
                    math.log(sample_sizes[lower])
                    + fraction * (math.log(sample_sizes[upper]) - math.log(sample_sizes[lower]))
                )
            elif crossing.size:
                estimates[quantile_index, tolerance_index] = sample_sizes[0]
            elif slope < 0.0:
                estimates[quantile_index, tolerance_index] = math.exp(
                    (math.log(tolerance) - intercept) / slope
                )
                extrapolated[quantile_index, tolerance_index] = True
            else:
                estimates[quantile_index, tolerance_index] = np.inf
                extrapolated[quantile_index, tolerance_index] = True
    return estimates, extrapolated, effective_dimensions


@torch.no_grad()
def estimate_navier_stokes_initial_condition_coverage(
    candidate_seeds,
    evaluation_seeds,
    simulation_grid_size=32,
    feature_grid_size=16,
    sample_sizes=NAVIER_STOKES_COVERAGE_SAMPLE_SIZES,
    distance_tolerances=NAVIER_STOKES_COVERAGE_DISTANCE_TOLERANCES,
    mse_tolerances=None,
    coverage_quantiles=NAVIER_STOKES_COVERAGE_QUANTILES,
    generation_batch_size=1024,
    distance_batch_size=64,
    device="cpu",
):
    """Estimate practical RGB condition-image coverage using unseen nearest neighbors.

    Distances are RGB pixel RMSE divided by the median RMSE between unrelated
    samples. Exact coverage is impossible because the sampler has continuous
    support; the returned sample counts are tolerance-dependent estimates.
    """
    candidate_seeds = [int(seed) for seed in candidate_seeds]
    evaluation_seeds = [int(seed) for seed in evaluation_seeds]
    if not candidate_seeds or not evaluation_seeds:
        raise ValueError("candidate_seeds and evaluation_seeds must both be non-empty.")
    if set(candidate_seeds).intersection(evaluation_seeds):
        raise ValueError("candidate and evaluation seeds must be disjoint.")
    if simulation_grid_size < feature_grid_size:
        raise ValueError("simulation_grid_size must be at least feature_grid_size.")
    if generation_batch_size < 1 or distance_batch_size < 1:
        raise ValueError("coverage batch sizes must be positive.")

    sample_sizes = np.asarray(sorted({int(size) for size in sample_sizes}), dtype=np.int64)
    if sample_sizes.size < 2 or sample_sizes[0] < 1:
        raise ValueError("sample_sizes must contain at least two positive integers.")
    if sample_sizes[-1] > len(candidate_seeds):
        raise ValueError("candidate_seeds must cover the largest requested sample size.")
    if mse_tolerances is not None and distance_tolerances is not None:
        raise ValueError("Specify either distance_tolerances or mse_tolerances, not both.")
    if mse_tolerances is not None:
        mse_tolerances = np.asarray(mse_tolerances, dtype=np.float64)
        if np.any(mse_tolerances <= 0.0):
            raise ValueError("mse_tolerances must be positive.")
    else:
        distance_tolerances = np.asarray(distance_tolerances, dtype=np.float64)
        if np.any(distance_tolerances <= 0.0):
            raise ValueError("distance_tolerances must be positive.")
    coverage_quantiles = np.asarray(coverage_quantiles, dtype=np.float64)
    if np.any((coverage_quantiles <= 0.0) | (coverage_quantiles >= 1.0)):
        raise ValueError("coverage_quantiles must lie strictly between zero and one.")

    all_features = _initial_condition_rgb_features(
        candidate_seeds + evaluation_seeds,
        simulation_grid_size=int(simulation_grid_size),
        feature_grid_size=int(feature_grid_size),
        generation_batch_size=int(generation_batch_size),
        device=device,
    )
    candidates = all_features[: len(candidate_seeds)]
    evaluation = all_features[len(candidate_seeds) :]
    pair_count = min(len(candidates), len(evaluation))
    random_pair_rmse = (
        (candidates[:pair_count] - evaluation[:pair_count])
        .square()
        .mean(dim=1)
        .sqrt()
        .median()
    )
    if random_pair_rmse <= 0.0:
        raise RuntimeError("random-pair distance was zero; use distinct sampler seeds.")
    if mse_tolerances is not None:
        distance_tolerances = np.sqrt(mse_tolerances) / float(random_pair_rmse)
    else:
        mse_tolerances = np.square(distance_tolerances * float(random_pair_rmse))

    nearest = torch.empty((len(evaluation), len(sample_sizes)), dtype=torch.float32)
    feature_scale = math.sqrt(candidates.shape[1]) * float(random_pair_rmse)
    for start in range(0, len(evaluation), int(distance_batch_size)):
        query = evaluation[start : start + int(distance_batch_size)]
        distances = torch.cdist(query, candidates) / feature_scale
        for size_index, sample_size in enumerate(sample_sizes):
            nearest[start : start + len(query), size_index] = distances[
                :, : int(sample_size)
            ].amin(dim=1)

    tolerance_tensor = torch.as_tensor(distance_tolerances, dtype=nearest.dtype)
    empirical_coverage = (
        nearest[:, :, None] <= tolerance_tensor[None, None, :]
    ).float().mean(dim=0)
    distance_quantiles = torch.quantile(
        nearest,
        torch.as_tensor(coverage_quantiles, dtype=nearest.dtype),
        dim=0,
    )
    estimates, extrapolated, effective_dimensions = _estimate_coverage_sample_counts(
        sample_sizes,
        distance_quantiles.numpy(),
        distance_tolerances,
    )
    return {
        "sample_sizes": sample_sizes,
        "distance_tolerances": distance_tolerances,
        "mse_tolerances": mse_tolerances,
        "coverage_quantiles": coverage_quantiles,
        "empirical_coverage": empirical_coverage.numpy(),
        "distance_quantiles": distance_quantiles.numpy(),
        "estimated_sample_counts": estimates,
        "estimate_extrapolated": extrapolated,
        "effective_dimensions": effective_dimensions,
        "random_pair_rgb_rmse": float(random_pair_rmse),
        "simulation_grid_size": int(simulation_grid_size),
        "feature_grid_size": int(feature_grid_size),
    }


def generate_navier_stokes_image_pairs(
    seeds,
    sim_nx=NAVIER_STOKES_GRID_SIZE,
    output_size=256,
    final_time=NAVIER_STOKES_FINAL_TIME,
    sim_device=None,
    progress=False,
):
    seeds = [int(seed) for seed in seeds]
    save_times, trajectory, sample_params = simulate_navier_stokes_2d_batch(
        seeds,
        grid_size=sim_nx,
        final_time=final_time,
        device=sim_device,
        progress=progress,
    )
    rendered = velocity_to_rgb(trajectory, output_size=output_size).cpu().numpy()
    speed = trajectory.square().sum(dim=2).sqrt()
    trajectory_max_speed = speed.amax(dim=(1, 2, 3)).cpu().numpy()
    speed_clip_fraction = (speed > 1.0).float().mean(dim=(1, 2, 3)).cpu().numpy()

    pairs = []
    for seed, images, params, max_speed, clip_fraction in zip(
        seeds,
        rendered,
        sample_params,
        trajectory_max_speed,
        speed_clip_fraction,
    ):
        density = float(params["density"])
        viscosity = float(params["kinematic_viscosity"])
        metadata = {
            **params,
            "seed": seed,
            "pde": "navier_stokes",
            "solution_target": "velocity_at_final_time",
            "boundary_condition": "periodic",
            "forcing": "none",
            "sim_nx": int(sim_nx),
            "output_size": int(output_size),
            "domain_size": float(NAVIER_STOKES_DOMAIN_SIZE),
            "final_time": float(save_times[-1]),
            "trajectory_max_speed": float(max_speed),
            "speed_clip_fraction": float(clip_fraction),
            "conditioning_names": NAVIER_STOKES_CONDITIONING_NAMES,
            "conditioning_values": (density, viscosity),
        }
        pairs.append((images[-1], images[0], metadata))
    return pairs


def generate_navier_stokes_multiple_records(
    seeds,
    sim_nx=NAVIER_STOKES_GRID_SIZE,
    output_size=256,
    sim_device=None,
    progress=False,
):
    seeds = [int(seed) for seed in seeds]
    target_times = np.asarray(NAVIER_STOKES_MULTIPLE_TIMES, dtype=np.float64)
    save_times = np.concatenate(([0.0], target_times))
    _, trajectory, sample_params = simulate_navier_stokes_2d_batch(
        seeds,
        grid_size=sim_nx,
        save_times=save_times,
        device=sim_device,
        progress=progress,
    )
    rendered = velocity_to_rgb(trajectory, output_size=output_size).cpu().numpy()
    speed = trajectory.square().sum(dim=2).sqrt()
    trajectory_max_speed = speed.amax(dim=(1, 2, 3)).cpu().numpy()
    speed_clip_fraction = (speed > 1.0).float().mean(dim=(1, 2, 3)).cpu().numpy()

    records = []
    for seed, images, params, max_speed, clip_fraction in zip(
        seeds,
        rendered,
        sample_params,
        trajectory_max_speed,
        speed_clip_fraction,
    ):
        density = float(params["density"])
        viscosity = float(params["kinematic_viscosity"])
        metadata = {
            **params,
            "seed": seed,
            "pde": "navier_stokes_multiple",
            "solution_target": "joint_velocity_sequence",
            "solution_times": NAVIER_STOKES_MULTIPLE_TIMES,
            "target_time_ids": (1, 2, 3, 4),
            "boundary_condition": "periodic",
            "forcing": "none",
            "sim_nx": int(sim_nx),
            "output_size": int(output_size),
            "domain_size": float(NAVIER_STOKES_DOMAIN_SIZE),
            "final_time": float(target_times[-1]),
            "trajectory_max_speed": float(max_speed),
            "speed_clip_fraction": float(clip_fraction),
            "conditioning_names": NAVIER_STOKES_CONDITIONING_NAMES,
            "conditioning_values": (density, viscosity),
        }
        records.append(
            {
                "initial": images[0],
                "solutions": tuple(images[1:]),
                "solution": images[-1],
                "params": metadata,
            }
        )
    return records
