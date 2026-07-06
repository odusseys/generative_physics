import random

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from .flux_lora import infer_solution
from .rendering import as_pil_image


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
    show_forcing = any("forcing" in record for record in chosen)
    ncols = 5 if show_forcing else 4
    fig_width = 16 if show_forcing else 13
    fig, axes = plt.subplots(len(chosen), ncols, figsize=(fig_width, 3.2 * len(chosen)))
    if len(chosen) == 1:
        axes = np.array([axes])

    for row, record in enumerate(chosen):
        initial_image = as_pil_image(record["initial"])
        forcing_image = as_pil_image(record["forcing"]) if "forcing" in record else None
        ground_truth = as_pil_image(record["solution"])
        is_poisson = record["params"].get("pde") == "poisson"
        is_fourier = record["params"].get("pde") == "fourier"
        is_airfoil = record["params"].get("pde") == "airfoil"
        is_condition_to_image = is_poisson or is_fourier or is_airfoil
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
        generated = infer_solution(
            pipe,
            initial_image,
            prompt_embeds,
            text_ids,
            device=device,
            train_image_size=train_image_size,
            num_inference_steps=num_inference_steps,
            seed=(seed or 0) + row,
            thermal_diffusivity=thermal_diffusivity,
            forcing_image=forcing_image,
        )

        gen_arr = np.asarray(generated.resize(ground_truth.size), dtype=np.float32) / 255.0
        gt_arr = np.asarray(ground_truth, dtype=np.float32) / 255.0
        abs_error = np.abs(gen_arr - gt_arr).mean(axis=2)

        col = 0
        axes[row, col].imshow(initial_image)
        axes[row, col].set_title("condition" if is_condition_to_image else f"initial{coefficient_label}")
        col += 1
        if show_forcing:
            if forcing_image is not None:
                axes[row, col].imshow(forcing_image)
                axes[row, col].set_title(f"forcing{forcing_label}")
            col += 1
        axes[row, col].imshow(generated)
        axes[row, col].set_title("inference" if is_condition_to_image else f"inference {pde_name}{coefficient_label}")
        col += 1
        axes[row, col].imshow(ground_truth)
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
