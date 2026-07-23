# Image Editing Models are Numerical Solvers

[![arXiv](https://img.shields.io/badge/arXiv-2607.18787-b31b1b.svg)](https://arxiv.org/abs/2607.18787)
[![Project website](https://img.shields.io/badge/Project-Website-4c8bf5.svg)](https://example.com)
<!-- Replace https://example.com with the project website URL when it is available. -->

This repository contains the training code for [*Image Editing Models are Numerical Solvers*](https://arxiv.org/abs/2607.18787). It fine-tunes FLUX.2 Klein with LoRA adapters to map rendered physical inputs to solutions of several numerical simulation problems.

## Training

### 1. Set up the environment

Clone the repository, create a virtual environment, and install the training dependencies:

```bash
git clone https://github.com/odusseys/generative_physics.git
cd generative_physics

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  "torch>=2.12" \
  "diffusers>=0.39" \
  "transformers>=5.13" \
  "accelerate>=1.14" \
  "peft>=0.19" \
  huggingface-hub numpy scipy pillow matplotlib tqdm ipython jupyterlab
```

Install the PyTorch build appropriate for your CUDA version if the default package does not match your system. Training uses `bfloat16` on CUDA and falls back to `float32` on CPU; a CUDA GPU is strongly recommended for the 4B-parameter model.

The base model is downloaded from Hugging Face on the first run. Make sure the machine has enough disk space for the model weights and training outputs.

### 2. Configure a run

Open [`train.ipynb`](train.ipynb) from the repository root:

```bash
jupyter lab train.ipynb
```

The first cell constructs a `TrainingConfig`. The checked-in example trains the Kuramoto–Sivashinsky task:

```python
from generative_physics.config import TrainingConfig
from generative_physics.run import run_training

config = TrainingConfig(
    pde_kind="ks",
    train_image_size=512,
    output_image_size=512,
    ks_grid_size=512,
    ks_condition_encoding="y_constant",
    ks_debug_num_samples=100,
    num_eval_pairs=4,
    stream_chunk_size=32,
    sim_num_workers=16,
    validate_every_n_steps=1000,
    validation_num_images=4,
    max_train_steps=10_000,
    run_initial_validation=True,
)
```

Useful settings to adjust are:

- `pde_kind`: simulation task to train.
- `train_batch_size` and `grad_accum_steps`: effective batch size and memory use.
- `learning_rate`, `lora_rank`, and `max_train_steps`: optimization settings.
- `validate_every_n_steps` and `validation_num_images`: validation cost and frequency.
- `sim_num_workers`: CPU workers used to generate simulations; lower this on smaller machines.
- `transformer_gradient_checkpointing`: reduce memory use at the cost of extra computation.
- `transformer_compile_regions`: enable or disable regional `torch.compile`.

Supported `pde_kind` values are `heat`, `cgl`, `burgers`, `poisson`, `ks`, `navier_stokes`, `navier_stokes_multiple`, `airfoil`, `elliptic`, `elasticity`, `eikonal`, `ot`, and `fracture`.

### 3. Start training

Run the second notebook cell:

```python
results = run_training(config)
```

Training data is generated online from fresh numerical simulations. A fixed validation set is created at startup, validation images are displayed at the configured interval, and the progress bar reports the current and exponential-moving-average losses.

To run the default heat-equation configuration without Jupyter:

```bash
python -m scripts.train_lora
```

### Outputs

At the end of training, the LoRA weights and any task-specific conditioning adapters are saved under:

```text
<pde_kind>_flux2_klein_4b_lora/
```

Kuramoto–Sivashinsky training with the default base model uses:

```text
ks_flux2_klein_base_4b_lora/
```

The returned `results` dictionary also contains the trained pipeline, loss history, validation records, prompt embeddings, and device information for further evaluation in the notebook.
