from dataclasses import dataclass

import numpy as np


def _linear_to_srgb(x):
    x = np.asarray(x)
    return np.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * np.maximum(x, 0.0) ** (1.0 / 2.4) - 0.055,
    )


def _oklab_to_linear_srgb(ok):
    L, a, b = np.moveaxis(ok, -1, 0)

    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b

    l = l_**3
    m = m_**3
    s = s_**3

    r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    return np.stack([r, g, b], axis=-1)


def _oklab_to_srgb(ok):
    return _linear_to_srgb(_oklab_to_linear_srgb(ok))


def _oklch_to_oklab(L, C, h):
    return np.stack([L, C * np.cos(h), C * np.sin(h)], axis=-1)


def _in_srgb_gamut_from_oklab(ok, eps=1e-7):
    rgb = _oklab_to_linear_srgb(ok)
    return np.all((rgb >= -eps) & (rgb <= 1.0 + eps), axis=-1)


def _max_chroma_for_lh(L, h, n_iter=24):
    L = np.asarray(L)
    h = np.asarray(h)

    lo = np.zeros_like(L, dtype=float)
    hi = np.full_like(L, 0.5, dtype=float)

    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        ok = _oklch_to_oklab(L, mid, h)
        good = _in_srgb_gamut_from_oklab(ok)
        lo = np.where(good, mid, lo)
        hi = np.where(good, hi, mid)

    return lo


@dataclass(frozen=True)
class PerceptualColorRamp:
    rgb_lut: np.ndarray
    params: dict

    def __call__(self, t):
        t = np.asarray(t, dtype=float)
        t = np.clip(t, 0.0, 1.0)

        n = len(self.rgb_lut)
        x = t * (n - 1)
        i = np.floor(x).astype(int)
        j = np.minimum(i + 1, n - 1)
        w = (x - i)[..., None]

        rgb = (1.0 - w) * self.rgb_lut[i] + w * self.rgb_lut[j]
        return np.clip(rgb, 0.0, 1.0)

    def uint8(self, t):
        return (255.0 * self(t)).round().astype(np.uint8)

    def uint16(self, t):
        return (65535.0 * self(t)).round().astype(np.uint16)


def make_perceptual_color_ramps(
    N,
    n_grid=256,
    n_hues=None,
    sweeps_deg=(0, -60, 60, -120, 120),
    L_range=(0.18, 0.92),
    chroma_scale=0.88,
    objective="independent",
    sample_pairs=4096,
    seed=0,
):
    rng = np.random.default_rng(seed)

    if n_hues is None:
        n_hues = max(32, 8 * N)

    t = np.linspace(0.0, 1.0, n_grid)
    L = L_range[0] + (L_range[1] - L_range[0]) * t

    candidates_ok = []
    candidates_rgb = []
    candidates_params = []

    for h0 in np.linspace(0.0, 2.0 * np.pi, n_hues, endpoint=False):
        for sweep_deg in sweeps_deg:
            sweep = np.deg2rad(sweep_deg)
            h = h0 + sweep * (t - 0.5)

            Cmax = _max_chroma_for_lh(L, h)
            C = chroma_scale * Cmax

            ok = _oklch_to_oklab(L, C, h)
            rgb = np.clip(_oklab_to_srgb(ok), 0.0, 1.0)

            candidates_ok.append(ok)
            candidates_rgb.append(rgb)
            candidates_params.append(
                {
                    "h0_deg": float(np.rad2deg(h0) % 360),
                    "sweep_deg": float(sweep_deg),
                    "n_grid": int(n_grid),
                    "L_range": tuple(float(x) for x in L_range),
                    "chroma_scale": float(chroma_scale),
                }
            )

    candidates_ok = np.asarray(candidates_ok)
    candidates_rgb = np.asarray(candidates_rgb)
    M = len(candidates_ok)

    if N > M:
        raise ValueError("Increase n_hues or add more sweeps_deg.")

    if objective == "independent":
        ti = rng.integers(0, n_grid, size=sample_pairs)
        sj = rng.integers(0, n_grid, size=sample_pairs)

    def dist_to(idx):
        if objective == "same_t":
            d = candidates_ok - candidates_ok[idx][None, :, :]
            return np.sqrt(np.sum(d * d, axis=-1)).mean(axis=1)

        if objective == "independent":
            out = np.empty(M)
            b = candidates_ok[idx, sj, :]
            chunk = 256

            for start in range(0, M, chunk):
                end = min(start + chunk, M)
                d = candidates_ok[start:end, ti, :] - b[None, :, :]
                out[start:end] = np.sqrt(np.sum(d * d, axis=-1)).mean(axis=1)

            return out

        raise ValueError("objective must be 'independent' or 'same_t'.")

    selected = [0]
    min_dist = dist_to(0)
    min_dist[0] = -np.inf

    while len(selected) < N:
        k = int(np.argmax(min_dist))
        selected.append(k)

        d = dist_to(k)
        min_dist = np.minimum(min_dist, d)
        min_dist[selected] = -np.inf

    ramps = []
    for k in selected:
        ramps.append(PerceptualColorRamp(rgb_lut=candidates_rgb[k], params=candidates_params[k]))

    return ramps


def scalar_to_rgb_uint8(array, vmin, vmax, ramp):
    array = np.asarray(array, dtype=np.float32)
    z = np.clip((array - vmin) / max(float(vmax) - float(vmin), 1e-6), 0.0, 1.0)
    return ramp.uint8(z)


def scalar_to_rgb_uint16(array, vmin, vmax, ramp):
    array = np.asarray(array, dtype=np.float32)
    z = np.clip((array - vmin) / max(float(vmax) - float(vmin), 1e-6), 0.0, 1.0)
    return ramp.uint16(z)
