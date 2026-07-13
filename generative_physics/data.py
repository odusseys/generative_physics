import os
from concurrent.futures import ProcessPoolExecutor

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from tqdm.auto import tqdm

from .airfoil import generate_airfoil_image_pairs
from .burgers import generate_burgers_image_pairs
from .cgl import generate_cgl_image_pairs
from .config import TrainingConfig
from .elliptic import generate_elliptic_image_pairs
from .eikonal import generate_eikonal_image_pairs
from .elasticity import generate_elasticity_image_pairs
from .fracture import generate_fracture_image_pairs
from .heat import generate_heat_image_pairs
from .ks import generate_ks_image_pairs
from .ot import generate_ot_image_pairs
from .poisson import generate_poisson_image_pairs
from .rendering import rgb_image_to_model_tensor


CPU_ONLY_PDE_KINDS = {"airfoil", "elliptic", "elasticity", "eikonal", "ot", "fracture", "ks"}


def simulations_run_on_cpu(config=None, sim_device=None):
    config = config or TrainingConfig()
    if config.pde_kind in CPU_ONLY_PDE_KINDS:
        return True
    sim_device = torch.device(sim_device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return sim_device.type == "cpu"


def cpu_sim_worker_count(config=None, num_pairs=None):
    config = config or TrainingConfig()
    requested = max(0, int(getattr(config, "sim_num_workers", 0)))
    if requested == 0:
        return 0
    available = os.cpu_count() or requested
    count = min(requested, available)
    if num_pairs is not None:
        count = min(count, max(0, int(num_pairs)))
    return count


def simulation_grid_size(config):
    if config.pde_kind == "poisson":
        return config.poisson_grid_size
    if config.pde_kind == "ks":
        return max(config.ks_grid_size, config.output_image_size)
    if config.pde_kind == "airfoil":
        return config.airfoil_grid_size
    if config.pde_kind == "elliptic":
        return config.elliptic_grid_size
    if config.pde_kind == "elasticity":
        return config.elasticity_grid_size
    if config.pde_kind == "eikonal":
        return config.eikonal_grid_size
    if config.pde_kind == "ot":
        return config.ot_solve_grid_size
    if config.pde_kind == "fracture":
        return config.fracture_grid_size
    return config.sim_nx


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
    progress=True,
    use_cache=True,
):
    config = config or TrainingConfig()
    image_size = config.output_image_size if image_size is None else image_size
    initial_grid_size = config.initial_grid_size if initial_grid_size is None else initial_grid_size
    save_steps = config.sim_save_steps if save_steps is None else save_steps
    if sim_nx is None:
        sim_nx = simulation_grid_size(config)
    sim_batch_size = config.sim_batch_size if sim_batch_size is None else sim_batch_size
    sim_device = sim_device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = []
    seeds = list(range(seed_offset, seed_offset + num_pairs))
    if config.pde_kind == "airfoil":
        pairs = generate_airfoil_image_pairs(
            seeds,
            sim_nx=sim_nx,
            output_size=image_size,
        )
        for solution_img, initial_img, params in pairs:
            records.append({"initial": initial_img, "solution": solution_img, "params": params})
        sim_device = torch.device(sim_device)
        if torch.cuda.is_available() and sim_device.type == "cuda":
            torch.cuda.empty_cache()
        return records

    if config.pde_kind == "elliptic":
        records = generate_elliptic_image_pairs(
            seeds,
            sim_nx=sim_nx,
            output_size=image_size,
            cache_dir=config.elliptic_cache_dir if use_cache else None,
        )
        sim_device = torch.device(sim_device)
        if torch.cuda.is_available() and sim_device.type == "cuda":
            torch.cuda.empty_cache()
        return records

    if config.pde_kind == "elasticity":
        pairs = generate_elasticity_image_pairs(
            seeds,
            sim_nx=sim_nx,
            output_size=image_size,
        )
        for solution_img, mask_img, params in pairs:
            records.append({"initial": mask_img, "solution": solution_img, "params": params})
        sim_device = torch.device(sim_device)
        if torch.cuda.is_available() and sim_device.type == "cuda":
            torch.cuda.empty_cache()
        return records

    if config.pde_kind == "eikonal":
        pairs = generate_eikonal_image_pairs(
            seeds,
            sim_nx=sim_nx,
            output_size=image_size,
        )
        for solution_img, refractive_img, params in pairs:
            records.append({"initial": refractive_img, "solution": solution_img, "params": params})
        sim_device = torch.device(sim_device)
        if torch.cuda.is_available() and sim_device.type == "cuda":
            torch.cuda.empty_cache()
        return records

    if config.pde_kind == "ot":
        records = generate_ot_image_pairs(
            seeds,
            sim_nx=sim_nx,
            output_size=image_size,
        )
        sim_device = torch.device(sim_device)
        if torch.cuda.is_available() and sim_device.type == "cuda":
            torch.cuda.empty_cache()
        return records

    if config.pde_kind == "fracture":
        pairs = generate_fracture_image_pairs(
            seeds,
            sim_nx=sim_nx,
            output_size=image_size,
            steps=config.fracture_steps,
            inner_iters=config.fracture_inner_iters,
        )
        for solution_img, initial_img, params in pairs:
            records.append({"initial": initial_img, "solution": solution_img, "params": params})
        sim_device = torch.device(sim_device)
        if torch.cuda.is_available() and sim_device.type == "cuda":
            torch.cuda.empty_cache()
        return records

    chunks = range(0, num_pairs, sim_batch_size)
    if progress:
        chunks = tqdm(chunks, desc=f"{config.pde_name} sims {seed_offset}", position=0, leave=True)
    for start in chunks:
        seed_chunk = seeds[start : start + sim_batch_size]
        progress_desc = f"{config.pde_kind.upper()} {seed_chunk[0]}..{seed_chunk[-1]}"
        if config.pde_kind == "heat":
            pairs = generate_heat_image_pairs(
                seed_chunk,
                sim_nx=sim_nx,
                output_size=image_size,
                initial_grid_size=initial_grid_size,
                save_steps=save_steps,
                sim_device=sim_device,
                progress=progress,
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
                sim_device=sim_device,
                progress=progress,
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
                progress=progress,
                progress_desc=progress_desc,
                progress_position=1,
                progress_update_every=config.sim_progress_update_every,
            )
        elif config.pde_kind == "poisson":
            pairs = generate_poisson_image_pairs(
                seed_chunk,
                sim_nx=sim_nx,
                output_size=image_size,
                sim_device=sim_device,
            )
        elif config.pde_kind == "ks":
            pairs = generate_ks_image_pairs(
                seed_chunk,
                sim_nx=sim_nx,
                output_size=image_size,
                condition_encoding=config.ks_condition_encoding,
                sim_device=sim_device,
            )
        else:
            raise ValueError(
                f"Unknown pde_kind={config.pde_kind!r}; expected 'heat', 'cgl', 'burgers', 'poisson', 'ks', "
                "'airfoil', 'elliptic', 'elasticity', 'eikonal', 'ot', or 'fracture'."
            )

        for pair in pairs:
            if isinstance(pair, dict):
                records.append(pair)
                continue
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


def _make_single_pde_record_worker(args):
    seed, kwargs = args
    torch.set_num_threads(1)
    return make_pde_records(1, seed_offset=seed, progress=False, **kwargs)[0]


def make_pde_records_with_workers(
    num_pairs,
    seed_offset=0,
    image_size=None,
    initial_grid_size=None,
    save_steps=None,
    sim_nx=None,
    sim_batch_size=None,
    sim_device=None,
    config=None,
    progress=True,
    use_cache=True,
    num_workers=None,
):
    config = config or TrainingConfig()
    sim_device = sim_device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_generated = simulations_run_on_cpu(config, sim_device)
    if num_workers is None:
        num_workers = cpu_sim_worker_count(config, num_pairs=num_pairs)
    else:
        available = os.cpu_count() or int(num_workers)
        num_workers = min(max(0, int(num_workers)), available, max(0, int(num_pairs)))

    if not cpu_generated or num_workers <= 1 or num_pairs <= 1:
        return make_pde_records(
            num_pairs,
            seed_offset=seed_offset,
            image_size=image_size,
            initial_grid_size=initial_grid_size,
            save_steps=save_steps,
            sim_nx=sim_nx,
            sim_batch_size=sim_batch_size,
            sim_device=sim_device,
            config=config,
            progress=progress,
            use_cache=use_cache,
        )

    worker_kwargs = {
        "image_size": image_size,
        "initial_grid_size": initial_grid_size,
        "save_steps": save_steps,
        "sim_nx": sim_nx,
        "sim_batch_size": sim_batch_size,
        "sim_device": "cpu",
        "config": config,
        "use_cache": use_cache,
    }
    seeds = range(int(seed_offset), int(seed_offset) + int(num_pairs))
    tasks = ((seed, worker_kwargs) for seed in seeds)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        records = executor.map(_make_single_pde_record_worker, tasks)
        if progress:
            records = tqdm(
                records,
                total=num_pairs,
                desc=f"{config.pde_name} CPU sims {seed_offset}",
                position=0,
                leave=True,
            )
        return list(records)


def image_to_model_tensor(image, image_size=256):
    return rgb_image_to_model_tensor(image, image_size=image_size)


def record_to_item(record, image_size=256, index=0):
    condition_images = record.get("condition_images")
    if condition_images is None:
        condition_images = [record["initial"]]
        if "forcing" in record:
            condition_images.append(record["forcing"])

    item = {
        "condition_pixels": [image_to_model_tensor(image, image_size) for image in condition_images],
        "target_pixels": image_to_model_tensor(record["solution"], image_size),
        "index": index,
    }
    if "initial" in record:
        item["initial_pixels"] = image_to_model_tensor(record["initial"], image_size)
    if "forcing" in record:
        item["forcing_pixels"] = image_to_model_tensor(record["forcing"], image_size)
    if "thermal_diffusivity" in record["params"]:
        item["thermal_diffusivity"] = torch.tensor(record["params"]["thermal_diffusivity"], dtype=torch.float32)
    if "conditioning_values" in record["params"]:
        item["conditioning_values"] = torch.tensor(record["params"]["conditioning_values"], dtype=torch.float32)
    return item


class PdePairDataset(Dataset):
    def __init__(self, records, image_size=256):
        self.records = records
        self.image_size = image_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        return record_to_item(record, image_size=self.image_size, index=idx)


class StreamingPdePairDataset(IterableDataset):
    def __init__(self, config=None, image_size=256, seed_offset=0, sim_device=None, skip_seed_ranges=None):
        self.config = config or TrainingConfig()
        self.image_size = image_size
        self.seed_offset = int(seed_offset)
        self.sim_device = sim_device
        self.skip_seed_ranges = tuple(sorted((int(a), int(b)) for a, b in (skip_seed_ranges or ()) if int(a) < int(b)))

    def _next_allowed_seed(self, seed, step=1):
        step = max(1, int(step))
        while True:
            for start, end in self.skip_seed_ranges:
                if start <= seed < end:
                    jumps = (end - seed + step - 1) // step
                    seed += jumps * step
                    break
            else:
                return seed

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            seed = self.seed_offset
            step = 1
        else:
            torch.set_num_threads(1)
            seed = self.seed_offset + worker.id
            step = worker.num_workers
        stream_chunk_size = max(1, int(getattr(self.config, "stream_chunk_size", 1)))
        while True:
            seed = self._next_allowed_seed(seed, step=step)
            if step == 1 and stream_chunk_size > 1:
                chunk_start = seed
                chunk_end = chunk_start + stream_chunk_size
                for skip_start, skip_end in self.skip_seed_ranges:
                    if chunk_start < skip_end and skip_start < chunk_end:
                        chunk_end = max(chunk_start, skip_start)
                        break
                chunk_size = max(1, chunk_end - chunk_start)
                records = make_pde_records(
                    chunk_size,
                    seed_offset=chunk_start,
                    image_size=self.image_size,
                    sim_device=self.sim_device,
                    config=self.config,
                    progress=False,
                    use_cache=False,
                )
                for offset, record in enumerate(records):
                    yield record_to_item(record, image_size=self.image_size, index=chunk_start + offset)
                seed = chunk_start + chunk_size
                continue

            record = make_pde_records(
                1,
                seed_offset=seed,
                image_size=self.image_size,
                sim_device=self.sim_device,
                config=self.config,
                progress=False,
                use_cache=False,
            )[0]
            yield record_to_item(record, image_size=self.image_size, index=seed)
            seed += step


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
