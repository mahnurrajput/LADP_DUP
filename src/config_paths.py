"""
config_paths.py — LADP
======================
Resolve the relative paths in a LADP config against the project root, so a
script gives the same result from any working directory.

The project root is the parent of the ``configs/`` folder that holds the
config (``<root>/configs/x.yaml`` -> ``<root>``), the same rule
``train_recon.py`` and ``cache_features.py`` already use.  Absolute paths are
kept as they are.  Paths given on the command line are the caller's concern
(they are relative to the shell's working directory, as usual).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]

# (section, key) of every path-valued config entry.
PATH_KEYS = (
    (None, "output_dir"),
    (None, "checkpoints_dir"),
    ("data", "coco_train_images"),
    ("data", "vizwiz_images"),
    ("data", "vizwiz_questions"),
    ("data", "cache_dir"),
    ("data", "vizwiz_grounding_dir"),
    ("training", "fixed_val_images"),
    ("loss", "norm_calibration"),
)


def project_root_for(config_path: PathLike) -> Path:
    """``<root>/configs/x.yaml`` -> ``<root>`` (absolute)."""
    p = Path(config_path).resolve()
    return p.parent.parent


def resolve_path(value: Optional[PathLike], root: PathLike) -> Optional[str]:
    if value is None or value == "":
        return value
    p = Path(value)
    return str(p if p.is_absolute() else (Path(root) / p).resolve())


def resolve_config_paths(cfg: dict, root: PathLike) -> dict:
    """Copy of ``cfg`` with every known path key made absolute under ``root``."""
    out = copy.deepcopy(cfg)
    for section, key in PATH_KEYS:
        holder = out if section is None else out.get(section)
        if isinstance(holder, dict) and key in holder:
            holder[key] = resolve_path(holder[key], root)
    return out
