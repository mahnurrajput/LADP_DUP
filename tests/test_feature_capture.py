"""
Audit C1: the diagnostic must capture the same E (and T) as the feature cache.

* On a tiny LLaVA (real transformers code): captured E == the projector input
  == vision_tower hidden_states[-2][:, 1:]; T == projector(E); E_final is the
  last layer and differs from E.
* With real LLaVA-1.5 weights (CLIP tower + projector only, ~0.6 GB, skipped
  if the cache, the weights or CUDA are missing): captured E and T are
  bit-identical to cached features_000000.pt, and the last layer is not.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.feature_capture import LlavaFeatureCapture  # noqa: E402
from _tiny_llava import N_PATCHES, tiny_inputs, tiny_llava  # noqa: E402


def test_captures_projector_input_and_output():
    model = tiny_llava()
    ids, px = tiny_inputs()
    with LlavaFeatureCapture(model) as cap, torch.no_grad():
        model(input_ids=ids, pixel_values=px)
        hs = model.vision_tower(px, output_hidden_states=True)
    ref_E = hs.hidden_states[-2][0, 1:]
    assert cap.E.shape == (N_PATCHES, 32) and cap.T.shape == (N_PATCHES, 48)
    assert torch.equal(cap.E, ref_E)
    assert torch.allclose(cap.T, model.multi_modal_projector(ref_E[None])[0].detach())
    assert torch.equal(cap.E_final, hs.last_hidden_state[0, 1:])
    assert not torch.allclose(cap.E, cap.E_final)          # the old (wrong) layer differs
    assert cap.n_calls == 1


def test_one_capture_during_generate_with_kv_cache():
    model = tiny_llava()
    ids, px = tiny_inputs()
    with LlavaFeatureCapture(model) as cap, torch.no_grad():
        model.generate(input_ids=ids, pixel_values=px, max_new_tokens=4, do_sample=False)
    assert cap.n_calls == 1 and cap.E is not None


def test_hooks_removed_after_exit():
    model = tiny_llava()
    ids, px = tiny_inputs()
    with LlavaFeatureCapture(model) as cap:
        pass
    with torch.no_grad():
        model(input_ids=ids, pixel_values=px)
    assert cap.n_calls == 0 and cap.E is None


def test_batch_of_two_images_is_refused():
    model = tiny_llava()
    _, px = tiny_inputs()
    with LlavaFeatureCapture(model), torch.no_grad(), pytest.raises(RuntimeError, match="one image"):
        model.get_image_features(pixel_values=torch.cat([px, px]), vision_feature_layer=-2,
                                 vision_feature_select_strategy="default")


def test_model_without_projector_is_refused():
    with pytest.raises(RuntimeError, match="multi_modal_projector"):
        LlavaFeatureCapture(torch.nn.Linear(2, 2)).__enter__()


# ---------------------------------------------------------------------------
# Real weights vs the real cache (regression test for C1)
# ---------------------------------------------------------------------------

CACHE0 = ROOT / "data" / "cache" / "coco_train" / "features_000000.pt"
# Hugging Face cache from HF_HOME (no machine-specific path); the real-weights test skips without it
HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
SNAP = HF_HOME / "hub" / "models--llava-hf--llava-1.5-7b-hf"


@pytest.mark.skipif(not (CACHE0.exists() and SNAP.exists() and torch.cuda.is_available()),
                    reason="needs the COCO cache, the LLaVA-1.5 weights and CUDA")
def test_real_weights_capture_matches_cache():
    from PIL import Image
    from transformers import LlavaForConditionalGeneration
    sys.path.insert(0, str(ROOT / "src"))
    from onthefly_features import VisionProjector

    vp = VisionProjector(hf_home=str(HF_HOME), device="cuda", dtype=torch.float16)
    host = torch.nn.Module()                       # just the two modules LLaVA's code uses
    host.vision_tower, host.multi_modal_projector = vp.vision_tower, vp.projector
    cached = torch.load(CACHE0, map_location="cpu", weights_only=True)
    with Image.open(cached["image_path"]) as im:
        px = vp.image_processor(images=im.convert("RGB"), return_tensors="pt")["pixel_values"]
    with LlavaFeatureCapture(host) as cap, torch.no_grad():
        # transformers' own get_image_features, the code LLaVA runs in generate()
        LlavaForConditionalGeneration.get_image_features(
            host, pixel_values=px.to("cuda", torch.float16), vision_feature_layer=-2,
            vision_feature_select_strategy="default")
    assert torch.equal(cap.E, cached["E"])
    assert torch.equal(cap.T, cached["T"])
    assert not torch.equal(cap.E_final, cached["E"])
