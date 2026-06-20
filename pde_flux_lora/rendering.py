import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image


def torch_hsv_to_rgb(h, s, v):
    h = torch.remainder(h, 1.0)
    if not torch.is_tensor(s):
        s = torch.full_like(h, float(s))
    if not torch.is_tensor(v):
        v = torch.full_like(h, float(v))

    h6 = h * 6.0
    r = torch.clamp(torch.abs(h6 - 3.0) - 1.0, 0.0, 1.0)
    g = torch.clamp(2.0 - torch.abs(h6 - 2.0), 0.0, 1.0)
    b = torch.clamp(2.0 - torch.abs(h6 - 4.0), 0.0, 1.0)
    rgb = torch.stack([r, g, b], dim=-1)
    return ((rgb - 1.0) * s[..., None] + 1.0) * v[..., None]


def float_rgb_to_uint8(rgb):
    return (rgb.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)


def complex_to_rgb_uint8_torch(U, amp_vmax=None, saturation=0.95):
    amp = U.abs().float()
    phase = torch.angle(U).float()
    if amp_vmax is None:
        amp_vmax = torch.quantile(amp.flatten(1), 0.995, dim=1).clamp_min(1e-6)
    amp_vmax = torch.as_tensor(amp_vmax, device=U.device, dtype=amp.dtype)
    view_shape = (amp.shape[0],) + (1,) * (amp.ndim - 1)

    h = (phase + torch.pi) / (2 * torch.pi)
    v = (amp / amp_vmax.view(view_shape)).clamp(0.0, 1.0)
    return float_rgb_to_uint8(torch_hsv_to_rgb(h, saturation, v))


def linear_colormap_uint8_torch(z, cmap_name="viridis"):
    if cmap_name != "viridis":
        raise ValueError("Only the torch viridis renderer is implemented for batched Burgers records.")
    colors = torch.tensor(
        [
            [68, 1, 84],
            [59, 82, 139],
            [33, 145, 140],
            [94, 201, 97],
            [253, 231, 37],
        ],
        device=z.device,
        dtype=torch.float32,
    ) / 255.0
    scaled = z.clamp(0.0, 1.0) * (len(colors) - 1)
    idx = torch.floor(scaled).to(torch.long).clamp(max=len(colors) - 2)
    frac = (scaled - idx.to(scaled.dtype))[..., None]
    rgb = colors[idx] * (1.0 - frac) + colors[idx + 1] * frac
    return float_rgb_to_uint8(rgb)


def scalar_to_rgb_uint8_torch(A, vmin, vmax, cmap_name="viridis"):
    vmin = torch.as_tensor(vmin, device=A.device, dtype=A.dtype)
    vmax = torch.as_tensor(vmax, device=A.device, dtype=A.dtype)
    view_shape = (A.shape[0],) + (1,) * (A.ndim - 1)
    z = (A - vmin.view(view_shape)) / (vmax - vmin).clamp_min(1e-6).view(view_shape)
    return linear_colormap_uint8_torch(z, cmap_name=cmap_name)


def repeat_rgb_row_view(row_rgb, output_size):
    return np.broadcast_to(row_rgb[None, :, :], (output_size, row_rgb.shape[0], row_rgb.shape[1]))


def array_to_pil(A, vmin=None, vmax=None, cmap_name="viridis"):
    A = np.asarray(A, dtype=float)
    if vmin is None or vmax is None:
        bound = np.percentile(np.abs(A), 99.5)
        bound = max(float(bound), 1e-6)
        vmin, vmax = -bound, bound
    Z = np.clip((A - vmin) / (vmax - vmin), 0.0, 1.0)
    rgba = colormaps[cmap_name](Z)
    rgb = (255 * rgba[..., :3]).astype(np.uint8)
    return Image.fromarray(rgb)


def hsv_to_rgb_np(h, s, v):
    h = np.mod(h, 1.0)
    i = np.floor(h * 6).astype(np.int32)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i = i % 6

    rgb = np.empty(h.shape + (3,), dtype=np.float32)
    masks = [i == k for k in range(6)]
    rgb[masks[0]] = np.stack([v, t, p], axis=-1)[masks[0]]
    rgb[masks[1]] = np.stack([q, v, p], axis=-1)[masks[1]]
    rgb[masks[2]] = np.stack([p, v, t], axis=-1)[masks[2]]
    rgb[masks[3]] = np.stack([p, q, v], axis=-1)[masks[3]]
    rgb[masks[4]] = np.stack([t, p, v], axis=-1)[masks[4]]
    rgb[masks[5]] = np.stack([v, p, q], axis=-1)[masks[5]]
    return rgb


def complex_to_pil(U, amp_vmax=None, saturation=0.95):
    U = np.asarray(U)
    amp = np.abs(U).astype(np.float32)
    phase = np.angle(U).astype(np.float32)
    if amp_vmax is None:
        amp_vmax = max(float(np.percentile(amp, 99.5)), 1e-6)
    h = (phase + np.pi) / (2 * np.pi)
    s = np.full_like(amp, saturation, dtype=np.float32)
    v = np.clip(amp / amp_vmax, 0.0, 1.0)
    rgb = hsv_to_rgb_np(h, s, v)
    return Image.fromarray((255 * rgb).astype(np.uint8))


def as_numpy_rgb(image):
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    else:
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        elif array.shape[-1] == 4:
            array = array[..., :3]

    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and (array.size == 0 or float(np.nanmax(array)) <= 1.0):
            array = np.clip(array * 255.0, 0.0, 255.0)
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def as_pil_image(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(np.ascontiguousarray(as_numpy_rgb(image)))


def image_size_xy(image):
    if isinstance(image, Image.Image):
        return image.size
    array = np.asarray(image)
    return (array.shape[1], array.shape[0])
