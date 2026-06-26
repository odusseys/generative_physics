from __future__ import annotations

# Compatibility facade for notebooks that do `import autoreg_windowed_lora as awl`.
# The implementation lives in ordered chunks under navier/model/.
from pathlib import Path as _Path

_PARTS_DIR = _Path(__file__).resolve().parent / "navier" / "model"
_PARTS = (
    "config.py",
    "recurrent_memory.py",
    "wan_runtime.py",
    "data.py",
    "losses.py",
    "profile.py",
    "training_utils.py",
    "inference_eval.py",
    "build_checkpoint.py",
    "train.py",
)

for _part in _PARTS:
    _path = _PARTS_DIR / _part
    exec(compile(_path.read_text(), str(_path), "exec"), globals(), globals())

del _part, _path, _Path, _PARTS_DIR, _PARTS
