import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .burgers import generate_burgers_image_pairs
from .cgl import generate_cgl_image_pairs
from .config import TrainingConfig
from .fourier import generate_fourier_image_pairs
from .heat import generate_heat_image_pairs
from .poisson import generate_poisson_image_pairs
from .rendering import as_numpy_rgb


def make_pde_records(
    num_pairs,
    seed_offset=0,
    image_size=None,
    initial_grid_size=None,
    save_steps=None,
    sim_nx=None,
    sim_batch_size=None,
    sim_device=None,
    config=None,
):
    config = config or TrainingConfig()
    image_size = config.output_image_size if image_size is None else image_size
    initial_grid_size = config.initial_grid_size if initial_grid_size is None else initial_grid_size
    save_steps = config.sim_save_steps if save_steps is None else save_steps
    if sim_nx is None:
        if config.pde_kind == "poisson":
            sim_nx = config.poisson_grid_size
        elif config.pde_kind == "fourier":
            sim_nx = config.fourier_grid_size
        else:
            sim_nx = config.sim_nx
    sim_batch_size = config.sim_batch_size if sim_batch_size is None else sim_batch_size
    sim_device = sim_device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = []
    seeds = list(range(seed_offset, seed_offset + num_pairs))
    chunks = range(0, num_pairs, sim_batch_size)
    for start in tqdm(chunks, desc=f"{config.pde_name} sims {seed_offset}", position=0, leave=True):
        seed_chunk = seeds[start : start + sim_batch_size]
        progress_desc = f"{config.pde_kind.upper()} {seed_chunk[0]}..{seed_chunk[-1]}"
        if config.pde_kind == "heat":
            pairs = generate_heat_image_pairs(
                seed_chunk,
                sim_nx=sim_nx,
                output_size=image_size,
                initial_grid_size=initial_grid_size,
                save_steps=save_steps,
                T=config.heat_t,
                num_modes=config.heat_num_modes,
                scale=config.heat_scale,
                forcing_num_modes=config.heat_forcing_num_modes,
                forcing_scale=config.heat_forcing_scale,
                diffusivity_min=config.heat_diffusivity_min,
                diffusivity_max=config.heat_diffusivity_max,
                sim_device=sim_device,
                progress=True,
                progress_desc=progress_desc,
                progress_position=1,
                progress_update_every=config.sim_progress_update_every,
            )
        elif config.pde_kind == "cgl":
            pairs = generate_cgl_image_pairs(
                seed_chunk,
                sim_nx=sim_nx,
                output_size=image_size,
                initial_grid_size=initial_grid_size,
                save_steps=save_steps,
                T=config.cgl_t,
                domain_length=config.cgl_domain_length,
                c1=config.cgl_c1,
                c3=config.cgl_c3,
                num_modes=config.cgl_num_modes,
                amp_scale=config.cgl_amp_scale,
                phase_scale=config.cgl_phase_scale,
                substeps_per_frame=config.cgl_substeps_per_frame,
                sim_device=sim_device,
                progress=True,
                progress_desc=progress_desc,
                progress_position=1,
                progress_update_every=config.sim_progress_update_every,
            )
        elif config.pde_kind == "burgers":
            pairs = generate_burgers_image_pairs(
                seed_chunk,
                sim_nx=sim_nx,
                output_size=image_size,
                initial_grid_size=initial_grid_size,
                save_steps=save_steps,
                sim_device=sim_device,
                adaptive_dt=config.sim_adaptive_dt,
                compile_step=config.sim_compile_step,
                progress=True,
                progress_desc=progress_desc,
                progress_position=1,
                progress_update_every=config.sim_progress_update_every,
            )
        elif config.pde_kind == "poisson":
            pairs = generate_poisson_image_pairs(
                seed_chunk,
                sim_nx=sim_nx,
                output_size=image_size,
                num_gaussian_modes=config.poisson_num_gaussian_modes,
                source_scale=config.poisson_source_scale,
                solution_vmax=config.poisson_solution_vmax,
                sim_device=sim_device,
            )
        elif config.pde_kind == "fourier":
            pairs = generate_fourier_image_pairs(
                seed_chunk,
                sim_nx=sim_nx,
                output_size=image_size,
                num_modes=config.fourier_num_modes,
                scale=config.fourier_scale,
                sigma_min=config.fourier_gaussian_sigma_min,
                sigma_max=config.fourier_gaussian_sigma_max,
                fft_shift=config.fourier_shift,
                sim_device=sim_device,
            )
        else:
            raise ValueError(
                f"Unknown pde_kind={config.pde_kind!r}; expected 'heat', 'cgl', 'burgers', 'poisson', or 'fourier'."
            )

        for pair in pairs:
            if len(pair) == 4:
                solution_img, initial_img, forcing_img, params = pair
                records.append(
                    {
                        "initial": initial_img,
                        "forcing": forcing_img,
                        "solution": solution_img,
                        "params": params,
                    }
                )
            else:
                solution_img, initial_img, params = pair
                records.append({"initial": initial_img, "solution": solution_img, "params": params})

    sim_device = torch.device(sim_device)
    if torch.cuda.is_available() and sim_device.type == "cuda":
        torch.cuda.empty_cache()
    return records


def image_to_model_tensor(image, image_size=256):
    array = as_numpy_rgb(image)
    if array.shape[:2] != (image_size, image_size):
        array = np.asarray(
            Image.fromarray(np.ascontiguousarray(array)).resize((image_size, image_size), Image.Resampling.BICUBIC)
        )
    array = np.array(array, dtype=np.float32, copy=True) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class PdePairDataset(Dataset):
    def __init__(self, records, image_size=256):
        self.records = records
        self.image_size = image_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        item = {
            "initial_pixels": image_to_model_tensor(record["initial"], self.image_size),
            "target_pixels": image_to_model_tensor(record["solution"], self.image_size),
            "index": idx,
        }
        if "forcing" in record:
            item["forcing_pixels"] = image_to_model_tensor(record["forcing"], self.image_size)
        if "thermal_diffusivity" in record["params"]:
            item["thermal_diffusivity"] = torch.tensor(record["params"]["thermal_diffusivity"], dtype=torch.float32)
        return item


def record_param_signature(record):
    params = record["params"]
    return tuple((key, params[key]) for key in sorted(params) if key not in {"amp_vmax", "vmin", "vmax"})


def assert_disjoint_record_splits(train_records, eval_records):
    train_seeds = {record["params"]["seed"] for record in train_records}
    eval_seeds = {record["params"]["seed"] for record in eval_records}
    overlapping_seeds = train_seeds & eval_seeds
    assert not overlapping_seeds, f"validation seeds leaked into train: {sorted(overlapping_seeds)}"

    train_signatures = {record_param_signature(record) for record in train_records}
    eval_signatures = {record_param_signature(record) for record in eval_records}
    overlapping_signatures = train_signatures & eval_signatures
    assert not overlapping_signatures, f"validation params leaked into train: {sorted(overlapping_signatures)[:3]}"
    return train_seeds, eval_seeds
