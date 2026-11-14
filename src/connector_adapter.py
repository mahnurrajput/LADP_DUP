"""
connector_adapter.py — LADP project
====================================
Provides ``LlavaAdapter``, a thin wrapper around ``LlavaForConditionalGeneration``
(llava-hf/llava-1.5-7b-hf) that extracts the two feature tensors needed by the
Loss-Aware Dynamic Projection pipeline:

    E  — pre-connector CLIP patch features   shape (N, dv)  e.g. (576, 1024)
    T  — post-connector projected tokens      shape (N, dt)  e.g. (576, 4096)

The CLIP text encoding of a question (q_clip) is NOT computed here; it is
delegated to the relevance scorer module so that the question encoder can be
shared and cached independently of the vision backbone.

Architecture recap (LLaVA-1.5-7B, HuggingFace):
    model.vision_tower             — CLIPVisionModel (ViT-L/14 @ 336 × 336)
    model.multi_modal_projector    — 2-layer MLP (1024 → 4096)
    model.language_model           — LlamaForCausalLM (7 B)
    vision_feature_layer           — typically –2 (penultimate hidden state)
    vision_feature_select_strategy — "default" → remove CLS token at index 0
                                     leaving N = 576 spatial patch tokens
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import torch
import PIL.Image

from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
)

logger = logging.getLogger(__name__)


class LlavaAdapter:
    """Wraps LlavaForConditionalGeneration and exposes pre/post-connector features.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier, e.g. ``"llava-hf/llava-1.5-7b-hf"``.
    device : str | torch.device
        Target device, e.g. ``"cuda"`` or ``"cpu"``.
    precision : str
        Floating-point precision for the model weights.  Must be one of
        ``"fp16"`` or ``"fp32"``.

    Raises
    ------
    ValueError
        If ``precision`` is not ``"fp16"`` or ``"fp32"``.
    RuntimeError
        If the loaded model does not expose the expected sub-modules
        (``vision_tower``, ``multi_modal_projector``).
    """

    # Expected patch counts and channel widths for LLaVA-1.5-7B.
    _EXPECTED_N: int = 576
    _EXPECTED_DV: int = 1024
    _EXPECTED_DT: int = 4096

    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        device: str | torch.device = "cuda",
        precision: str = "fp16",
    ) -> None:
        if precision not in ("fp16", "fp32"):
            raise ValueError(
                f"precision must be 'fp16' or 'fp32', got '{precision}'."
            )

        self._device = torch.device(device)
        self._precision = precision
        self._torch_dtype = torch.float16 if precision == "fp16" else torch.float32

        logger.info(
            "Loading LLaVA model '%s' on %s with %s …",
            model_name,
            self._device,
            precision,
        )

        # ------------------------------------------------------------------
        # Load processor (handles image pre-processing + tokenisation).
        # ------------------------------------------------------------------
        self._processor = AutoProcessor.from_pretrained(model_name)

        # ------------------------------------------------------------------
        # Load model in the requested dtype.  low_cpu_mem_usage avoids a
        # full fp32 copy on CPU before the dtype cast.
        # ------------------------------------------------------------------
        self._model: LlavaForConditionalGeneration = (
            LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=self._torch_dtype,
                low_cpu_mem_usage=True,
            ).to(self._device)
        )
        self._model.eval()

        # ------------------------------------------------------------------
        # Validate that the sub-modules we depend on are present.
        # ------------------------------------------------------------------
        if not hasattr(self._model, "vision_tower"):
            raise RuntimeError(
                "Loaded model does not have a 'vision_tower' attribute.  "
                "Expected a LlavaForConditionalGeneration with a CLIPVisionModel "
                "at model.vision_tower."
            )
        if not hasattr(self._model, "multi_modal_projector"):
            raise RuntimeError(
                "Loaded model does not have a 'multi_modal_projector' attribute.  "
                "Expected a 2-layer MLP at model.multi_modal_projector."
            )

        # Cache config values used repeatedly during extraction.
        cfg = self._model.config
        # vision_feature_layer is typically –2 (index into hidden_states list).
        self._vision_feature_layer: int = int(cfg.vision_feature_layer)
        # "default" means strip the CLS token (index 0).
        self._vision_feature_select_strategy: str = cfg.vision_feature_select_strategy

        logger.info(
            "LlavaAdapter ready.  vision_feature_layer=%d  "
            "vision_feature_select_strategy='%s'",
            self._vision_feature_layer,
            self._vision_feature_select_strategy,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_features(
        self,
        image: PIL.Image.Image,
        question: Optional[str] = None,  # noqa: ARG002  (accepted, not used here)
    ) -> dict:
        """Extract pre- and post-connector visual features for a single image.

        The ``question`` parameter is accepted for API symmetry with the
        broader LADP pipeline but is **not** processed here.  CLIP text
        encoding of the question is handled by the relevance scorer so that
        the text encoder can be shared and cached across calls.

        Parameters
        ----------
        image : PIL.Image.Image
            Input image in any mode accepted by PIL (will be converted to RGB
            internally by the HuggingFace processor).
        question : str, optional
            The VQA question string.  Accepted but ignored in this method.

        Returns
        -------
        dict with keys:
            ``"E"``   — Tensor of shape ``(N, dv)`` — pre-connector CLIP patch
                        features (float32 or float16 depending on precision).
            ``"T"``   — Tensor of shape ``(N, dt)`` — post-connector projected
                        tokens.
            ``"N"``   — int, number of spatial patches (expected 576).
            ``"dv"``  — int, CLIP vision hidden dimension (expected 1024).
            ``"dt"``  — int, projected dimension / LLM hidden size (expected 4096).

        Raises
        ------
        RuntimeError
            If the vision tower does not return hidden states, or if the
            extracted tensors have unexpected shapes.
        """
        # Ensure image is RGB — some PNG/RGBA inputs break the CLIP processor.
        if image.mode != "RGB":
            image = image.convert("RGB")

        # ------------------------------------------------------------------
        # Pre-process the image using the HF processor.
        # We pass a minimal prompt so the processor produces valid pixel_values;
        # only the visual modality is used here.
        # ------------------------------------------------------------------
        inputs = self._processor(
            images=image,
            text="<image>",   # minimal placeholder; we do not run the LLM
            return_tensors="pt",
        )
        pixel_values: torch.Tensor = inputs["pixel_values"].to(
            device=self._device, dtype=self._torch_dtype
        )
        # pixel_values: (1, C, H, W)  typically (1, 3, 336, 336)

        with torch.no_grad():
            E, T = self._extract_vision_features(pixel_values)

        # ------------------------------------------------------------------
        # Shape validation.
        # ------------------------------------------------------------------
        N, dv = E.shape
        _, dt = T.shape

        if N != self._EXPECTED_N:
            warnings.warn(
                f"Expected {self._EXPECTED_N} spatial patches but got N={N}.  "
                "This may happen if the image was preprocessed at a non-standard "
                "resolution.  Downstream LADP components assume N=576.",
                UserWarning,
                stacklevel=2,
            )
        if dv != self._EXPECTED_DV:
            raise RuntimeError(
                f"Pre-connector feature dimension dv={dv} does not match the "
                f"expected value of {self._EXPECTED_DV}.  "
                "Check that you are using LLaVA-1.5-7B (ViT-L/14)."
            )
        if dt != self._EXPECTED_DT:
            raise RuntimeError(
                f"Post-connector feature dimension dt={dt} does not match the "
                f"expected value of {self._EXPECTED_DT}.  "
                "Check that you are using LLaVA-1.5-7B with a 7B LLM backbone."
            )

        return {
            "E": E,    # (N, dv)
            "T": T,    # (N, dt)
            "N": N,
            "dv": dv,
            "dt": dt,
        }

    def get_model(self) -> LlavaForConditionalGeneration:
        """Return the underlying HuggingFace model.

        Required by the LADP pipeline when it needs to run the LLM forward
        pass directly (e.g. to compute the language modelling loss).

        Returns
        -------
        LlavaForConditionalGeneration
            The frozen, eval-mode model on the configured device.
        """
        return self._model

    def get_processor(self) -> AutoProcessor:
        """Return the HuggingFace processor used for image/text pre-processing.

        Returns
        -------
        AutoProcessor
            Processor instance tied to the loaded model checkpoint.
        """
        return self._processor

    @property
    def num_patches(self) -> int:
        """Number of spatial patch tokens after CLS removal.

        For ViT-L/14 at 336 × 336 resolution this is 576 (= 24 × 24 patches).
        """
        return self._EXPECTED_N

    @property
    def d_vision(self) -> int:
        """Dimensionality of the CLIP vision hidden states (pre-connector).

        Equal to 1024 for ViT-L/14.
        """
        return self._EXPECTED_DV

    @property
    def d_projected(self) -> int:
        """Dimensionality of the projected tokens (post-connector).

        Equal to 4096 for the LLaVA-1.5-7B MLP projector → LLaMA-7B embedding
        space.
        """
        return self._EXPECTED_DT

    def smoke_test(self) -> None:
        """Run a minimal self-contained sanity check without real data.

        Creates a synthetic white 336 × 336 RGB image, runs ``extract_features``,
        and asserts that the returned tensors have the expected shapes.  Useful
        for verifying that the model loaded correctly before committing to a full
        evaluation run.

        Raises
        ------
        AssertionError
            If any output shape does not match expectations.
        RuntimeError
            Propagated from ``extract_features`` if extraction fails.
        """
        logger.info("Running LlavaAdapter smoke test …")

        dummy_image = PIL.Image.new("RGB", (336, 336), color=(255, 255, 255))

        result = self.extract_features(dummy_image, question=None)

        E: torch.Tensor = result["E"]
        T: torch.Tensor = result["T"]
        N: int = result["N"]
        dv: int = result["dv"]
        dt: int = result["dt"]

        assert E.ndim == 2, f"E should be 2-D, got shape {E.shape}"
        assert T.ndim == 2, f"T should be 2-D, got shape {T.shape}"
        assert E.shape == (N, dv), (
            f"E shape mismatch: expected ({N}, {dv}), got {E.shape}"
        )
        assert T.shape == (N, dt), (
            f"T shape mismatch: expected ({N}, {dt}), got {T.shape}"
        )
        assert dv == self._EXPECTED_DV, (
            f"dv={dv} does not equal expected {self._EXPECTED_DV}"
        )
        assert dt == self._EXPECTED_DT, (
            f"dt={dt} does not equal expected {self._EXPECTED_DT}"
        )

        logger.info(
            "Smoke test passed.  E=%s  T=%s  dtype=%s",
            tuple(E.shape),
            tuple(T.shape),
            E.dtype,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_vision_features(
        self,
        pixel_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Core extraction logic.  Called inside ``torch.no_grad()``.

        Parameters
        ----------
        pixel_values : torch.Tensor
            Pre-processed image tensor of shape ``(B, C, H, W)`` already on
            the model's device and in the correct dtype.

        Returns
        -------
        E : torch.Tensor
            Pre-connector CLIP patch features, shape ``(N, dv)``.
            CLS token has already been removed.  Batch dimension is squeezed
            (B=1 assumed at inference time).
        T : torch.Tensor
            Post-connector projected tokens, shape ``(N, dt)``.

        Raises
        ------
        RuntimeError
            If the vision tower does not return a ``hidden_states`` attribute,
            or if the indexed hidden state has an unexpected number of
            dimensions.
        """
        # ------------------------------------------------------------------
        # Step 1: Run the CLIP vision tower with all hidden states returned.
        # ------------------------------------------------------------------
        vision_outputs = self._model.vision_tower(
            pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

        if vision_outputs.hidden_states is None:
            raise RuntimeError(
                "vision_tower did not return hidden_states.  "
                "Ensure output_hidden_states=True is supported by the "
                "CLIPVisionModel in this transformers version."
            )

        # hidden_states is a tuple of tensors, one per transformer layer
        # (including the embedding layer at index 0).
        # vision_feature_layer is typically –2 → penultimate hidden layer.
        try:
            selected_hidden: torch.Tensor = vision_outputs.hidden_states[
                self._vision_feature_layer
            ]
        except IndexError as exc:
            n_layers = len(vision_outputs.hidden_states)
            raise RuntimeError(
                f"vision_feature_layer={self._vision_feature_layer} is out of "
                f"range for a hidden_states tuple of length {n_layers}."
            ) from exc

        # selected_hidden: (B, N+1, dv)  where N+1 includes the CLS token.
        if selected_hidden.ndim != 3:
            raise RuntimeError(
                f"Expected selected hidden state to be 3-D (B, N+1, dv), "
                f"got shape {tuple(selected_hidden.shape)}."
            )

        # ------------------------------------------------------------------
        # Step 2: Apply the feature select strategy (remove CLS if "default").
        # ------------------------------------------------------------------
        if self._vision_feature_select_strategy == "default":
            # Index 0 along the sequence dimension is the CLS token.
            # Slice it away → (B, N, dv)
            image_features = selected_hidden[:, 1:, :]
        elif self._vision_feature_select_strategy == "full":
            # Keep the CLS token (less common for LLaVA-1.5 but supported).
            image_features = selected_hidden
        else:
            raise RuntimeError(
                f"Unknown vision_feature_select_strategy: "
                f"'{self._vision_feature_select_strategy}'.  "
                "Expected 'default' or 'full'."
            )
        # image_features: (B, N, dv)

        # ------------------------------------------------------------------
        # Step 3: Squeeze the batch dimension (we process one image at a time).
        # ------------------------------------------------------------------
        # E: (N, dv)
        E: torch.Tensor = image_features.squeeze(0)

        # ------------------------------------------------------------------
        # Step 4: Pass E through the MLP projector → T.
        # The projector expects input of shape (B, N, dv) or (N, dv); adding
        # the batch dim back keeps it consistent with how the HF model calls it.
        # ------------------------------------------------------------------
        # projected: (B, N, dt) → squeeze → (N, dt)
        projected: torch.Tensor = self._model.multi_modal_projector(
            image_features  # (B, N, dv)
        )
        T: torch.Tensor = projected.squeeze(0)  # (N, dt)

        return E, T
