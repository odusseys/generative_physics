import torch
import torch.nn.functional as F


def periodic_linear_upsample_1d(values, target_nx):
    source_nx = values.shape[-1]
    if source_nx == target_nx:
        return values
    if torch.is_complex(values):
        real = periodic_linear_upsample_1d(values.real, target_nx)
        imag = periodic_linear_upsample_1d(values.imag, target_nx)
        return torch.complex(real, imag)

    positions = torch.arange(target_nx, device=values.device, dtype=values.dtype) * (source_nx / target_nx)
    left = torch.floor(positions).to(torch.long) % source_nx
    right = (left + 1) % source_nx
    weight = (positions - torch.floor(positions)).view(*([1] * (values.ndim - 1)), target_nx)
    return values.index_select(-1, left) * (1.0 - weight) + values.index_select(-1, right) * weight


def downsample_solution_torch(U, output_size=256):
    if U.shape[-2:] == (output_size, output_size):
        return U
    return F.interpolate(U[:, None], size=(output_size, output_size), mode="area").squeeze(1)


def downsample_complex_solution_torch(U, output_size=512):
    if U.shape[-2:] == (output_size, output_size):
        return U
    channels = torch.stack([U.real, U.imag], dim=1)
    channels = F.interpolate(channels, size=(output_size, output_size), mode="area")
    return torch.complex(channels[:, 0], channels[:, 1])
