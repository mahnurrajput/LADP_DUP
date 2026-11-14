"""
onthefly_features.py
====================
Extract LADP's (E, T) pairs on the fly with only the parts of LLaVA-1.5 that
produce them: the CLIP vision tower and the multi-modal projector.  The 7B
language model is never loaded (~0.6 GB of weights in fp16 instead of ~14 GB).

E = hidden_states[vision_feature_layer] of the CLIP tower with CLS removed
    (LLaVA-1.5: layer -2, strategy "default")              → (B, 576, 1024)
T = multi_modal_projector(E)                                → (B, 576, 4096)

This is the same computation as ``LlavaAdapter`` (src/connector_adapter.py)
and the Lost in Embeddings extraction hook (projector input/output).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset


def _snapshot_dir(model_name: str, hf_home: Optional[str]) -> Path:
    from huggingface_hub import snapshot_download

    kw = {"local_files_only": True}
    if hf_home:
        kw["cache_dir"] = str(Path(hf_home) / "hub")
    return Path(snapshot_download(model_name, **kw))


class VisionProjector(torch.nn.Module):
    """CLIP vision tower + LLaVA projector, loaded straight from the checkpoint."""

    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf",
                 hf_home: Optional[str] = None,
                 device: str = "cuda", dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        from safetensors import safe_open
        from transformers import CLIPImageProcessor, CLIPVisionModel, LlavaConfig
        from transformers.models.llava.modeling_llava import LlavaMultiModalProjector

        snap = _snapshot_dir(model_name, hf_home)
        cfg = LlavaConfig.from_pretrained(snap)
        self.feature_layer = int(cfg.vision_feature_layer)
        self.select_strategy = str(cfg.vision_feature_select_strategy)
        if self.select_strategy not in ("default", "full"):
            raise ValueError(f"Unsupported vision_feature_select_strategy {self.select_strategy!r}")

        self.vision_tower = CLIPVisionModel(cfg.vision_config)
        self.projector = LlavaMultiModalProjector(cfg)

        index = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
        want = {k: f for k, f in index.items()
                if k.startswith("vision_tower.") or k.startswith("multi_modal_projector.")}
        vt, pj = {}, {}
        for shard in sorted(set(want.values())):
            with safe_open(str(snap / shard), framework="pt") as fh:
                for k in (k for k, f in want.items() if f == shard):
                    if k.startswith("vision_tower."):
                        vt[k[len("vision_tower."):]] = fh.get_tensor(k)
                    else:
                        pj[k[len("multi_modal_projector."):]] = fh.get_tensor(k)
        missing, unexpected = self.vision_tower.load_state_dict(vt, strict=False)
        # position_ids is a non-persistent buffer in recent transformers versions
        missing = [m for m in missing if not m.endswith("position_ids")]
        if missing or unexpected:
            raise RuntimeError(f"vision tower weights mismatch: missing={missing} unexpected={unexpected}")
        self.projector.load_state_dict(pj, strict=True)

        self.image_processor = CLIPImageProcessor.from_pretrained(snap)
        self.to(device=device, dtype=dtype).eval()
        for p in self.parameters():
            p.requires_grad_(False)
        self.device_, self.dtype_ = torch.device(device), dtype

    @torch.no_grad()
    def forward(self, pixel_values: Tensor) -> Tuple[Tensor, Tensor]:
        out = self.vision_tower(pixel_values.to(self.device_, self.dtype_),
                                output_hidden_states=True, return_dict=True)
        E = out.hidden_states[self.feature_layer]
        if self.select_strategy == "default":
            E = E[:, 1:]                       # drop CLS
        T = self.projector(E)
        return E, T


class ImagePathDataset(Dataset):
    """Loads and pre-processes images (CPU side; use DataLoader workers)."""

    def __init__(self, paths: List[str], image_processor) -> None:
        self.paths = list(paths)
        self.proc = image_processor

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> Tensor:
        from PIL import Image
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB")
            return self.proc(images=im, return_tensors="pt")["pixel_values"][0]
