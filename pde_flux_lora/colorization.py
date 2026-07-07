import numpy as np


def normalize_base_colors(base_colors):
    colors = np.asarray(base_colors, dtype=np.float32)
    if colors.ndim != 2 or colors.shape[1] != 3 or colors.shape[0] < 2:
        raise ValueError("base_colors must have shape (num_colors, 3) with at least two colors.")
    if colors.size and float(np.nanmax(colors)) > 1.0:
        colors = colors / 255.0
    return np.clip(colors, 0.0, 1.0)


def cyclic_value_colorize(
    phase,
    value,
    base_colors,
    value_vmin=0.0,
    value_vmax=1.0,
    gamma=1.0,
    mask=None,
    mask_color=(0.0, 0.0, 0.0),
    output_dtype=np.uint8,
):
    colors = normalize_base_colors(base_colors)
    phase = np.mod(np.asarray(phase, dtype=np.float32), 1.0)
    value = np.asarray(value, dtype=np.float32)

    span = max(float(value_vmax) - float(value_vmin), 1e-12)
    intensity = np.clip((value - float(value_vmin)) / span, 0.0, 1.0) ** float(gamma)

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
