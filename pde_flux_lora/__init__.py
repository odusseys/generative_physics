"""Tools for generating PDE image pairs and training Flux.2 Klein LoRA adapters."""

from .config import TrainingConfig
from .data import PdePairDataset, StreamingPdePairDataset, make_pde_records

__all__ = ["PdePairDataset", "StreamingPdePairDataset", "TrainingConfig", "make_pde_records", "run_training"]


def __getattr__(name):
    if name == "run_training":
        from .run import run_training

        return run_training
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
