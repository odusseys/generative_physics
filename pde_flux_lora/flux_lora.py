import copy

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import Flux2KleinPipeline
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps

from .rendering import image_size_xy, rgb_image_to_model_tensor


def flux2_klein_lora_targets(transformer):
    n_single = len(transformer.single_transformer_blocks)
    return ["to_k", "to_q", "to_v", "to_out.0", "to_qkv_mlp_proj"] + [
        f"single_transformer_blocks.{i}.attn.to_out" for i in range(min(24, n_single))
    ]


def trainable_parameter_count(module):
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total


def encode_flux2_latents(pipe, pixels, device):
    pixels = pixels.to(device=device, dtype=pipe.vae.dtype)
    latents = pipe.vae.encode(pixels).latent_dist.mode()
    latents = Flux2KleinPipeline._patchify_latents(latents)
    latents_bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    latents_bn_std = torch.sqrt(
        pipe.vae.bn.running_var.view(1, -1, 1, 1).to(latents.device, latents.dtype) + pipe.vae.config.batch_norm_eps
    )
    return (latents - latents_bn_mean) / latents_bn_std


_distilled_schedule_cache = {}


def distilled_timesteps_and_sigmas(scheduler, image_seq_len, device, dtype, num_inference_steps=4):
    key = (int(image_seq_len), str(device), str(dtype), int(num_inference_steps))
    if key in _distilled_schedule_cache:
        return _distilled_schedule_cache[key]

    scheduler = copy.deepcopy(scheduler)
    sigmas = None
    if not (hasattr(scheduler.config, "use_flow_sigmas") and scheduler.config.use_flow_sigmas):
        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)

    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
    timesteps, _ = retrieve_timesteps(
        scheduler,
        num_inference_steps,
        device=device,
        sigmas=sigmas,
        mu=mu,
    )
    schedule_sigmas = scheduler.sigmas[: len(timesteps)].to(device=device, dtype=dtype)
    timesteps = timesteps.to(device=device)

    _distilled_schedule_cache[key] = (timesteps, schedule_sigmas)
    return timesteps, schedule_sigmas


def sample_exact_distilled_training_step(scheduler, image_seq_len, batch_size, latent_dtype, device, num_inference_steps=4):
    timesteps_4, sigmas_4 = distilled_timesteps_and_sigmas(
        scheduler=scheduler,
        image_seq_len=image_seq_len,
        device=device,
        dtype=latent_dtype,
        num_inference_steps=num_inference_steps,
    )
    step_indices = torch.randint(0, len(timesteps_4), (batch_size,), device=device)
    timesteps = timesteps_4[step_indices]
    sigmas = sigmas_4[step_indices]
    return timesteps, sigmas, timesteps_4, sigmas_4


def pde_lora_loss(pipe, batch, prompt_embeds, text_ids, device, num_inference_steps=4, print_schedule=False):
    condition_pixel_batches = batch.get("condition_pixels")
    if condition_pixel_batches is None:
        condition_pixel_batches = [batch["initial_pixels"]]
        forcing_pixels = batch.get("forcing_pixels")
        if forcing_pixels is not None:
            condition_pixel_batches.append(forcing_pixels)
    elif torch.is_tensor(condition_pixel_batches):
        condition_pixel_batches = [condition_pixel_batches]
    else:
        condition_pixel_batches = list(condition_pixel_batches)

    target_pixels = batch["target_pixels"].to(device=device, dtype=pipe.vae.dtype)
    batch_size = target_pixels.shape[0]

    with torch.no_grad():
        target_latents = encode_flux2_latents(pipe, target_pixels, device=device)
        condition_latents = [
            encode_flux2_latents(pipe, pixels.to(device=device, dtype=pipe.vae.dtype), device=device)
            for pixels in condition_pixel_batches
        ]

    target_ids = Flux2KleinPipeline._prepare_latent_ids(target_latents).to(device)
    condition_ids = (
        Flux2KleinPipeline._prepare_image_ids([latents[:1] for latents in condition_latents])
        .to(device)
        .repeat(batch_size, 1, 1)
    )
    noise = torch.randn_like(target_latents)
    timesteps, sigmas, timesteps_4, sigmas_4 = sample_exact_distilled_training_step(
        pipe.scheduler,
        image_seq_len=target_ids.shape[1],
        batch_size=batch_size,
        latent_dtype=target_latents.dtype,
        device=device,
        num_inference_steps=num_inference_steps,
    )
    if print_schedule:
        print("exact 4-step timesteps:", [float(x) for x in timesteps_4.detach().cpu()])
        print("exact 4-step sigmas:", [float(x) for x in sigmas_4.detach().cpu()])

    while sigmas.ndim < target_latents.ndim:
        sigmas = sigmas.unsqueeze(-1)
    noisy_target_latents = (1.0 - sigmas) * target_latents + sigmas * noise

    packed_noisy_target = Flux2KleinPipeline._pack_latents(noisy_target_latents)
    packed_conditions = [Flux2KleinPipeline._pack_latents(latents) for latents in condition_latents]
    hidden_states = torch.cat([packed_noisy_target, *packed_conditions], dim=1).to(pipe.transformer.dtype)
    img_ids = torch.cat([target_ids, condition_ids], dim=1)

    batch_prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1).to(device=device, dtype=pipe.transformer.dtype)
    batch_text_ids = text_ids.repeat(batch_size, 1, 1).to(device=device)
    thermal_diffusivity = batch.get("thermal_diffusivity")
    if thermal_diffusivity is not None:
        thermal_diffusivity = thermal_diffusivity.to(device=device, dtype=torch.float32)
    conditioning_values = batch.get("conditioning_values")
    if conditioning_values is not None:
        conditioning_values = conditioning_values.to(device=device, dtype=torch.float32)
    transformer_kwargs = {}
    if conditioning_values is not None:
        transformer_kwargs["conditioning_values"] = conditioning_values
    elif thermal_diffusivity is not None:
        transformer_kwargs["thermal_diffusivity"] = thermal_diffusivity
    model_pred = pipe.transformer(
        hidden_states=hidden_states,
        timestep=timesteps / 1000,
        guidance=None,
        encoder_hidden_states=batch_prompt_embeds,
        txt_ids=batch_text_ids,
        img_ids=img_ids,
        return_dict=False,
        **transformer_kwargs,
    )[0]
    model_pred = model_pred[:, : packed_noisy_target.shape[1]]
    model_pred = Flux2KleinPipeline._unpack_latents_with_ids(model_pred, target_ids)
    target = noise - target_latents
    return F.mse_loss(model_pred.float(), target.float())


@torch.no_grad()
def infer_solution(
    pipe,
    initial_image,
    prompt_embeds,
    text_ids,
    device,
    train_image_size=256,
    num_inference_steps=4,
    seed=0,
    thermal_diffusivity=None,
    conditioning_values=None,
    forcing_image=None,
    condition_images=None,
):
    if condition_images is None:
        condition_images = [initial_image]
        if forcing_image is not None:
            condition_images.append(forcing_image)
    condition_images = [image for image in condition_images if image is not None]
    if not condition_images:
        raise ValueError("infer_solution requires at least one condition image.")

    generator = torch.Generator(device=device).manual_seed(seed)
    was_training = pipe.transformer.training
    pipe.transformer.eval()

    multiple_of = pipe.vae_scale_factor * 2
    preprocessed_condition_images = []
    for image in condition_images:
        image_width, image_height = image_size_xy(image)
        image_width = (image_width // multiple_of) * multiple_of
        image_height = (image_height // multiple_of) * multiple_of
        if image_width <= 0 or image_height <= 0:
            raise ValueError("condition images are too small for the VAE scale factor.")
        preprocessed_condition_images.append(
            rgb_image_to_model_tensor(image, image_size=(image_height, image_width)).unsqueeze(0)
        )

    prompt_embeds = prompt_embeds.to(device=device, dtype=pipe.transformer.dtype)
    text_ids = text_ids.to(device=device)
    num_channels_latents = pipe.transformer.config.in_channels // 4
    latents, latent_ids = pipe.prepare_latents(
        batch_size=1,
        num_latents_channels=num_channels_latents,
        height=train_image_size,
        width=train_image_size,
        dtype=prompt_embeds.dtype,
        device=device,
        generator=generator,
    )
    image_latents, image_latent_ids = pipe.prepare_image_latents(
        images=preprocessed_condition_images,
        batch_size=1,
        generator=generator,
        device=device,
        dtype=pipe.vae.dtype,
    )

    sigmas = None
    if not (hasattr(pipe.scheduler.config, "use_flow_sigmas") and pipe.scheduler.config.use_flow_sigmas):
        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
    mu = compute_empirical_mu(image_seq_len=latents.shape[1], num_steps=num_inference_steps)
    timesteps, _ = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps,
        device=device,
        sigmas=sigmas,
        mu=mu,
    )
    pipe.scheduler.set_begin_index(0)

    latent_model_ids = torch.cat([latent_ids, image_latent_ids], dim=1)
    for t in timesteps:
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        latent_model_input = torch.cat([latents, image_latents], dim=1).to(pipe.transformer.dtype)
        with pipe.transformer.cache_context("cond"):
            transformer_kwargs = {}
            if conditioning_values is not None:
                transformer_kwargs["conditioning_values"] = torch.as_tensor(
                    conditioning_values, device=device, dtype=torch.float32
                ).reshape(1, -1)
            elif thermal_diffusivity is not None:
                transformer_kwargs["thermal_diffusivity"] = torch.as_tensor(
                    [thermal_diffusivity], device=device, dtype=torch.float32
                )
            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=None,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_model_ids,
                return_dict=False,
                **transformer_kwargs,
            )[0]
        noise_pred = noise_pred[:, : latents.shape[1]]
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    latents = Flux2KleinPipeline._unpack_latents_with_ids(latents, latent_ids)
    latents_bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    latents_bn_std = torch.sqrt(
        pipe.vae.bn.running_var.view(1, -1, 1, 1).to(latents.device, latents.dtype) + pipe.vae.config.batch_norm_eps
    )
    latents = latents * latents_bn_std + latents_bn_mean
    latents = Flux2KleinPipeline._unpatchify_latents(latents)
    image = pipe.vae.decode(latents, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]

    if was_training:
        pipe.transformer.train()
    return image
