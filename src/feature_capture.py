"""
feature_capture.py — LADP
=========================
Capture the connector's input and output while LLaVA-1.5 runs normally.

LLaVA (transformers 4.46, ``LlavaForConditionalGeneration.get_image_features``)
computes, for ``vision_feature_layer = -2`` and strategy ``"default"``::

    hs  = vision_tower(pixel_values, output_hidden_states=True).hidden_states
    E   = hs[-2][:, 1:]                 # (B, 576, 1024)  penultimate layer, CLS removed
    T   = multi_modal_projector(E)      # (B, 576, 4096)

so the projector's *input* is exactly the E that ``cache_features.py`` stores
and g_phi was trained on, and its *output* is T.  Hooking the projector
therefore captures the right pair by construction, whatever the configured
layer or strategy is.  (The earlier hook took the vision tower's
``last_hidden_state`` = layer 24 — a different representation; audit C1.)

The vision tower's ``last_hidden_state`` (final layer, CLS removed) is also
recorded as ``E_final``: it is what CLIP's ``ln_post`` + ``proj`` expect, and
the relevance scorer uses it (audit H3).  It is never used as E.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def _vision_and_projector(model):
    inner = getattr(model, "model", model)          # newer transformers nest under .model

    def find(name):
        # explicit None checks: an nn.Module with __len__ (e.g. Sequential) can be falsy
        v = getattr(model, name, None)
        return v if v is not None else getattr(inner, name, None)

    vt, pj = find("vision_tower"), find("multi_modal_projector")
    if vt is None or pj is None:
        raise RuntimeError("Model has no vision_tower / multi_modal_projector to hook "
                           "(expected a LLaVA-style model).")
    return vt, pj


class LlavaFeatureCapture:
    """Context manager: ``with LlavaFeatureCapture(model) as cap: model.generate(...)``.

    After the first image forward pass:
      cap.E        (N, dv)  projector input  — same as the cache's E
      cap.T        (N, dt)  projector output — same as the cache's T
      cap.E_final  (N, dv)  vision tower final layer, CLS removed (for relevance)
      cap.n_calls  number of projector calls seen (1 for one image + KV-cached decoding)

    Tensors are detached, moved to CPU and kept in their original dtype unless
    ``to_float`` is set.  Only the first image of the batch is supported
    (batch size 1); a larger batch raises.
    """

    def __init__(self, model, to_float: bool = False) -> None:
        self._model = model
        self._to_float = to_float
        self._hooks: List[Any] = []
        self.E: Optional[Tensor] = None
        self.T: Optional[Tensor] = None
        self.E_final: Optional[Tensor] = None
        self.n_calls = 0

    def _prep(self, x: Tensor) -> Tensor:
        if x.dim() != 3 or x.shape[0] != 1:
            raise RuntimeError(f"Expected one image (1, N, d), got shape {tuple(x.shape)}")
        x = x[0].detach().cpu()
        return x.float() if self._to_float else x

    def __enter__(self) -> "LlavaFeatureCapture":
        vt, pj = _vision_and_projector(self._model)

        def on_projector(module, inputs, output):
            self.n_calls += 1
            if self.n_calls == 1:
                self.E = self._prep(inputs[0])
                self.T = self._prep(output)

        def on_vision(module, inputs, output):
            if self.E_final is None and getattr(output, "last_hidden_state", None) is not None:
                self.E_final = self._prep(output.last_hidden_state[:, 1:])

        self._hooks.append(pj.register_forward_hook(on_projector))
        self._hooks.append(vt.register_forward_hook(on_vision))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
