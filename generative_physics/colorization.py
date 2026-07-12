import numpy as np


def normalize_base_colors(base_colors):
    colors = np.asarray(base_colors, dtype=np.float32)
    if colors.ndim != 2 or colors.shape[1] != 3 or colors.shape[0] < 2:
        raise ValueError("base_colors must have shape (num_colors, 3) with at least two colors.")
    if colors.size and float(np.nanmax(colors)) > 1.0:
        colors = colors / 255.0
    return np.clip(colors, 0.0, 1.0)


def apply_midpoint_contrast(values, contrast=1.0, midpoint=0.5):
    contrast = float(contrast)
    midpoint = float(midpoint)
    if contrast <= 0.0:
        raise ValueError("contrast must be positive")
    if midpoint <= 0.0 or midpoint >= 1.0:
        raise ValueError("midpoint must be between 0 and 1")

    values = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    if abs(contrast - 1.0) < 1e-6:
        return values

    out = np.empty_like(values)
    left = values <= midpoint
    out[left] = midpoint * np.power(values[left] / midpoint, contrast)
    out[~left] = 1.0 - (1.0 - midpoint) * np.power((1.0 - values[~left]) / (1.0 - midpoint), contrast)
    return out


def cyclic_value_colorize(
    phase,
    value,
    base_colors,
    value_vmin=0.0,
    value_vmax=1.0,
    gamma=1.0,
    value_softness=0.0,
    mask=None,
    mask_color=(0.0, 0.0, 0.0),
    output_dtype=np.uint8,
):
    colors = normalize_base_colors(base_colors)
    phase = np.mod(np.asarray(phase, dtype=np.float32), 1.0)
    value = np.asarray(value, dtype=np.float32)

    span = max(float(value_vmax) - float(value_vmin), 1e-12)
    z = (value - float(value_vmin)) / span
    if value_softness and float(value_softness) > 0.0:
        intensity = 1.0 - np.exp(-np.maximum(z, 0.0) / float(value_softness))
    else:
        intensity = np.clip(z, 0.0, 1.0)
    intensity = np.clip(intensity, 0.0, 1.0) ** float(gamma)

    x = phase * colors.shape[0]
    i = np.floor(x).astype(np.int64) % colors.shape[0]
    j = (i + 1) % colors.shape[0]
    w = (x - np.floor(x))[..., None]

    rgb = ((1.0 - w) * colors[i] + w * colors[j]) * intensity[..., None]
    if mask is not None:
        rgb[np.asarray(mask, dtype=bool)] = normalize_base_colors([mask_color, mask_color])[0]

    rgb = np.clip(rgb, 0.0, 1.0)
    if output_dtype is None:
        return rgb.astype(np.float32, copy=False)
    if output_dtype == np.uint16:
        return (65535.0 * rgb).round().astype(np.uint16)
    if output_dtype == np.uint8:
        return (255.0 * rgb).round().astype(np.uint8)
    return rgb.astype(output_dtype)
