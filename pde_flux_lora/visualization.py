import random

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from .flux_lora import infer_solution
from .rendering import as_float_rgb, as_pil_image


def _record_condition_images_and_names(record, coefficient_label="", forcing_label=""):
    condition_images = record.get("condition_images")
    condition_names = record.get("condition_names")
    if condition_images is not None:
        names = list(condition_names or [f"condition {idx + 1}" for idx in range(len(condition_images))])
        return list(condition_images), names

    initial_name = "condition" if record["params"].get("pde") in {"poisson", "fourier", "airfoil"} else "initial"
    images = [as_pil_image(record["initial"])]
    names = [f"{initial_name}{coefficient_label}"]
    if "forcing" in record:
        images.append(as_pil_image(record["forcing"]))
        names.append(f"forcing{forcing_label}")
    return images, names


def show_random_inference_grid(
    pipe,
    records,
    prompt_embeds,
    text_ids,
    device,
    pde_name,
    train_image_size=256,
    num_inference_steps=4,
    n=8,
    seed=None,
):
    rng = random.Random(seed)
    chosen = rng.sample(records, k=min(n, len(records)))
    row_contexts = []
    max_conditions = 1
    for record in chosen:
        thermal_diffusivity = record["params"].get("thermal_diffusivity")
        coefficient_label = ""
        if thermal_diffusivity is not None:
            coefficient_label = f"\nalpha={thermal_diffusivity:.3e}"

        forcing_label = ""
        forcing_start_modes = record["params"].get("forcing_start_active_modes")
        forcing_end_modes = record["params"].get("forcing_end_active_modes")
        forcing_active_modes = record["params"].get("forcing_active_modes")
        if forcing_start_modes is not None and forcing_end_modes is not None:
            forcing_label = f"\nforcing modes={forcing_start_modes}->{forcing_end_modes}"
        elif forcing_active_modes is not None:
            forcing_label = f"\nforcing modes={forcing_active_modes}"

        condition_images, condition_names = _record_condition_images_and_names(
            record,
            coefficient_label=coefficient_label,
            forcing_label=forcing_label,
        )
        row_contexts.append((record, thermal_diffusivity, coefficient_label, condition_images, condition_names))
        max_conditions = max(max_conditions, len(condition_images))

    ncols = max_conditions + 3
    fig_width = max(13, 2.4 * ncols)
    fig, axes = plt.subplots(len(chosen), ncols, figsize=(fig_width, 3.2 * len(chosen)))
    if len(chosen) == 1:
        axes = np.array([axes])

    for row, (record, thermal_diffusivity, coefficient_label, condition_images, condition_names) in enumerate(
        row_contexts
    ):
        ground_truth = record["solution"]
        ground_truth_rgb = as_float_rgb(ground_truth)
        ground_truth_pil = as_pil_image(ground_truth)
        is_poisson = record["params"].get("pde") == "poisson"
        is_fourier = record["params"].get("pde") == "fourier"
        is_airfoil = record["params"].get("pde") == "airfoil"
        is_elliptic = record["params"].get("pde") == "elliptic"
        is_condition_to_image = is_poisson or is_fourier or is_airfoil or is_elliptic
        generated = infer_solution(
            pipe,
            condition_images[0],
            prompt_embeds,
            text_ids,
            device=device,
            train_image_size=train_image_size,
            num_inference_steps=num_inference_steps,
            seed=(seed or 0) + row,
            thermal_diffusivity=thermal_diffusivity,
            condition_images=condition_images,
        )

        gen_arr = as_float_rgb(generated.resize(ground_truth_pil.size))
        gt_arr = ground_truth_rgb
        abs_error = np.abs(gen_arr - gt_arr).mean(axis=2)

        col = 0
        for condition_image, condition_name in zip(condition_images, condition_names):
            axes[row, col].imshow(as_float_rgb(condition_image))
            axes[row, col].set_title(condition_name)
            col += 1
        while col < max_conditions:
            axes[row, col].axis("off")
            col += 1
        axes[row, col].imshow(generated)
        axes[row, col].set_title("inference" if is_condition_to_image else f"inference {pde_name}{coefficient_label}")
        col += 1
        axes[row, col].imshow(ground_truth_rgb)
        axes[row, col].set_title("ground truth" if is_condition_to_image else f"ground truth {pde_name}{coefficient_label}")
        col += 1
        axes[row, col].imshow(abs_error, cmap="magma", vmin=0.0, vmax=0.1)
        axes[row, col].set_title(f"abs error mean={abs_error.mean():.3f}")

        for ax in axes[row]:
            ax.axis("off")
    plt.tight_layout()
    display(fig)
    plt.close(fig)


def exponential_moving_average(values, alpha=0.08):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return np.array([], dtype=np.float32)

    smoothed = np.empty_like(values, dtype=np.float32)
    smoothed[0] = values[0]
    for idx, value in enumerate(values[1:], start=1):
        smoothed[idx] = alpha * value + (1.0 - alpha) * smoothed[idx - 1]
    return smoothed


def show_smoothed_loss(loss_history, alpha=0.08):
    if not loss_history:
        return

    steps = np.arange(len(loss_history))
    losses = np.asarray(loss_history, dtype=np.float32)
    smoothed = exponential_moving_average(losses, alpha=alpha)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(steps, losses, color="0.75", linewidth=1.0, alpha=0.55, label="raw")
    ax.plot(steps, smoothed, color="tab:blue", linewidth=2.0, label=f"EMA alpha={alpha:g}")
    ax.set_title(f"training loss through step {len(loss_history) - 1}")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("MSE loss")
    ymax = float(np.quantile(losses, 0.7))
    if np.isfinite(ymax) and ymax > 0:
        ax.set_ylim(bottom=0.0, top=ymax)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    display(fig)
    plt.close(fig)
