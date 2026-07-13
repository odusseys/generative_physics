import random

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from tqdm.auto import tqdm

from .flux_lora import infer_solution
from .ks import KS_CMAP_NAME, KS_DT, KS_STEPS_PER_FRAME, flame_image_to_normalized_array
from .rendering import as_float_rgb, image_size_xy, resize_float_rgb


VARIABLE_LABELS = {
    "alpha": r"$\alpha$",
    "thermal_diffusivity": r"$\alpha$",
    "lambda": r"$\lambda$",
    "mu": r"$\mu$",
    "sigma_x": r"$\sigma_x$",
    "sigma_y": r"$\sigma_y$",
    "nu": r"$\nu$",
    "G_c": r"$G_c$",
    "epsilon_h": r"$\epsilon_h$",
    "epsilon_v": r"$\epsilon_v$",
}

COLUMN_LABELS = {
    "a20": r"$a_{20}$",
    "a11": r"$a_{11}$",
    "a02": r"$a_{02}$",
    "a10": r"$a_{10}$",
    "a01": r"$a_{01}$",
    "a00": r"$a_{00}$",
    "f": r"$f$",
    "source": r"source $\rho_0$",
    "target": r"target $\rho_1$",
    "cost": r"cost $c$",
    "initial": r"initial $u_0$",
    "forcing": r"forcing $f$",
    "condition": "condition",
}

DEBUG_IMAGE_GUTTER = 0.02


def _variable_label(name):
    return VARIABLE_LABELS.get(str(name), str(name))


def _column_label(name):
    name = str(name).split("\n", 1)[0]
    return COLUMN_LABELS.get(name, name)


def _default_condition_name(record):
    pde = record["params"].get("pde")
    if pde == "heat":
        return r"initial $u_0$"
    if pde == "poisson":
        return r"source $f$"
    if pde == "ks":
        return r"initial $u_0$"
    if pde == "airfoil":
        return "airfoil mask"
    if pde == "elasticity":
        return "hole mask"
    if pde == "eikonal":
        return r"refractive index $n$"
    if pde == "fracture":
        return "initial damage"
    return "condition"


def _format_scalar_conditioning_lines(params):
    names = params.get("conditioning_names")
    values = params.get("conditioning_values")
    if names is None or values is None:
        return []

    pairs = [(str(name), float(value)) for name, value in zip(names, values)]
    if not pairs:
        return []
    lines = []
    for name, value in pairs:
        if value != 0.0 and (abs(value) < 1e-2 or abs(value) >= 1e3):
            value_text = f"{value:.2e}"
        else:
            value_text = f"{value:.2f}"
        lines.append(f"{_variable_label(name)}={value_text}")
    return lines


def _format_row_label(record):
    params = record["params"]
    lines = []

    thermal_diffusivity = params.get("thermal_diffusivity")
    if thermal_diffusivity is not None:
        lines.append(f"{_variable_label('thermal_diffusivity')}={thermal_diffusivity:.2e}")

    lines.extend(_format_scalar_conditioning_lines(params))

    return "\n".join(lines)


def _record_condition_images_and_names(record):
    condition_images = record.get("condition_images")
    condition_names = record.get("condition_names")
    if condition_images is not None:
        fallback_names = [f"condition {idx + 1}" for idx in range(len(condition_images))]
        names = [_column_label(name) for name in (condition_names or fallback_names)]
        return list(condition_images), names

    initial_name = _default_condition_name(record)
    images = [record["initial"]]
    names = [initial_name]
    if "forcing" in record:
        images.append(record["forcing"])
        names.append(COLUMN_LABELS["forcing"])
    return images, names


def _style_image_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def _has_white_outer_background(rgb):
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or min(rgb.shape[:2]) < 2:
        return False

    border = np.concatenate(
        [
            rgb[0, :, :],
            rgb[-1, :, :],
            rgb[:, 0, :],
            rgb[:, -1, :],
        ],
        axis=0,
    )
    white_pixels = np.all(border >= 0.96, axis=-1)
    return float(np.mean(white_pixels)) >= 0.9


def _condition_display_rgb(image):
    rgb = np.array(as_float_rgb(image), dtype=np.float32, copy=True)
    if _has_white_outer_background(rgb):
        border_alpha = 0.15
        border_mask = np.zeros(rgb.shape[:2], dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        rgb[border_mask, :] *= 1.0 - border_alpha
    return rgb


def _measure_label_width_inches(labels):
    labels = [label for label in labels if label]
    if not labels:
        return 0.0

    measure_fig = plt.figure(figsize=(1.0, 1.0), dpi=100)
    texts = [measure_fig.text(0.0, 0.0, label, ha="left", va="center") for label in labels]
    measure_fig.canvas.draw()
    renderer = measure_fig.canvas.get_renderer()
    width = max(text.get_window_extent(renderer=renderer).width for text in texts) / measure_fig.dpi
    plt.close(measure_fig)
    return float(width)


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
    condition_column_titles = []
    for record in chosen:
        thermal_diffusivity = record["params"].get("thermal_diffusivity")
        conditioning_values = record["params"].get("conditioning_values")
        row_label = _format_row_label(record)

        condition_images, condition_names = _record_condition_images_and_names(record)
        while len(condition_column_titles) < len(condition_names):
            condition_column_titles.append(condition_names[len(condition_column_titles)])
        row_contexts.append((record, thermal_diffusivity, conditioning_values, row_label, condition_images))
        max_conditions = max(max_conditions, len(condition_images))

    ncols = max_conditions + 3
    nrows = len(chosen)
    column_titles = [
        condition_column_titles[idx] if idx < len(condition_column_titles) else f"condition {idx + 1}"
        for idx in range(max_conditions)
    ] + ["inference", "numerical solver", "absolute error"]

    cell_size = 2.35
    top_margin = 0.55
    label_outer_margin = 0.15
    label_area = _measure_label_width_inches([context[3] for context in row_contexts])
    label_gutter = top_margin if label_area > 0.0 else 0.0
    left_margin = (label_outer_margin + label_area + label_gutter) if label_area > 0.0 else 0.0
    fig_width = left_margin + cell_size * ncols
    fig_height = top_margin + cell_size * nrows
    image_left = left_margin / fig_width
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)
    fig.subplots_adjust(
        left=image_left,
        right=1.0,
        bottom=0.0,
        top=1.0 - top_margin / fig_height,
        wspace=DEBUG_IMAGE_GUTTER,
        hspace=DEBUG_IMAGE_GUTTER,
    )

    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, pad=8)

    for row, (
        record,
        thermal_diffusivity,
        conditioning_values,
        row_label,
        condition_images,
    ) in enumerate(row_contexts):
        ground_truth = record["solution"]
        ground_truth_rgb = as_float_rgb(ground_truth)
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
            conditioning_values=conditioning_values,
            condition_images=condition_images,
        )

        gen_arr = resize_float_rgb(as_float_rgb(generated), ground_truth_rgb.shape[:2])
        gt_arr = ground_truth_rgb
        abs_error = np.abs(gen_arr - gt_arr).mean(axis=2)

        col = 0
        for condition_image in condition_images:
            axes[row, col].imshow(_condition_display_rgb(condition_image))
            _style_image_axis(axes[row, col])
            col += 1
        while col < max_conditions:
            axes[row, col].set_axis_off()
            col += 1
        axes[row, col].imshow(as_float_rgb(generated))
        _style_image_axis(axes[row, col])
        col += 1
        axes[row, col].imshow(ground_truth_rgb)
        _style_image_axis(axes[row, col])
        col += 1
        axes[row, col].imshow(abs_error, cmap="Reds", vmin=0.0, vmax=1.0)
        _style_image_axis(axes[row, col])

        bbox = axes[row, 0].get_position()
        if row_label:
            fig.text(
                label_outer_margin / fig_width,
                0.5 * (bbox.y0 + bbox.y1),
                row_label,
                ha="left",
                va="center",
            )
    display(fig)
    plt.close(fig)


def show_ks_timewise_error(
    pipe,
    records,
    prompt_embeds,
    text_ids,
    device,
    train_image_size=256,
    num_inference_steps=4,
    n=50,
    seed=0,
):
    chosen = random.Random(seed).sample(records, k=min(n, len(records)))
    if not chosen:
        return np.array([], dtype=np.float32)

    errors = []
    for sample_index, record in enumerate(tqdm(chosen, desc="KS error samples", leave=False)):
        condition_images, _ = _record_condition_images_and_names(record)
        generated = infer_solution(
            pipe,
            condition_images[0],
            prompt_embeds,
            text_ids,
            device=device,
            train_image_size=train_image_size,
            num_inference_steps=num_inference_steps,
            seed=seed + sample_index,
            condition_images=condition_images,
        )
        ground_truth = record["solution"]
        target_size = image_size_xy(ground_truth)
        generated = resize_float_rgb(as_float_rgb(generated), (target_size[1], target_size[0]))
        generated_values = flame_image_to_normalized_array(generated, cmap_name=KS_CMAP_NAME)
        target_values = flame_image_to_normalized_array(ground_truth, cmap_name=KS_CMAP_NAME)
        errors.append(np.abs(generated_values - target_values).mean(axis=1))

    mean_error = np.stack(errors).mean(axis=0)
    dt = float(chosen[0]["params"].get("dt", KS_DT))
    steps_per_frame = int(chosen[0]["params"].get("steps_per_frame", KS_STEPS_PER_FRAME))
    times = np.arange(mean_error.size) * dt * steps_per_frame
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(times, mean_error, color="tab:red", linewidth=2.0)
    ax.set_title(f"KS decoded absolute error over {len(chosen)} samples")
    ax.set_xlabel("time")
    ax.set_ylabel("mean absolute error")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    display(fig)
    plt.close(fig)
    return mean_error


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
