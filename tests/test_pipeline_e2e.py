"""
End-to-end check of the FYP-1 pipeline:

  image -> CLIP hidden_states[-2][:, 1:]  E (576, 1024)
        -> LLaVA projector                T (576, 4096)
        -> g_phi(T_std)                   Ê (576, 1024)  ≈ E_std

Tiny-model tests (CPU, always run) check the code paths and patch ordering.
Real-weights tests (need the COCO cache, the LLaVA-1.5 weights and CUDA; only
the CLIP tower + projector are loaded, ~0.6 GB) check five cached images
bit-for-bit and the trained pilot g_phi on them.  Nothing is written.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loss_estimator import LossEstimator, NormStats, ReconstructionNet  # noqa: E402
from _tiny_llava import N_PATCHES, tiny_inputs, tiny_llava  # noqa: E402


# ---------------------------------------------------------------------------
# Code paths (tiny LLaVA, CPU)
# ---------------------------------------------------------------------------

def test_patch_order_is_row_major_tiny():
    """Changing pixel block (row r, col c) changes only token 1 + n*r + c of the embeddings."""
    model = tiny_llava()
    _, px = tiny_inputs()
    n = 56 // 14
    with torch.no_grad():
        base = model.vision_tower(px, output_hidden_states=True).hidden_states[0]
        for r, c in [(0, 0), (0, 3), (2, 1), (3, 3)]:
            p2 = px.clone()
            p2[:, :, 14 * r:14 * r + 14, 14 * c:14 * c + 14] += 1.0
            h = model.vision_tower(p2, output_hidden_states=True).hidden_states[0]
            changed = torch.nonzero((h - base).abs().amax(-1)[0] > 1e-6).flatten().tolist()
            assert changed == [1 + n * r + c]


def test_llava_adapter_formula_matches_transformers():
    """LlavaAdapter (smoke test / adapter cache engine) == LLaVA's get_image_features."""
    from src.connector_adapter import LlavaAdapter
    model = tiny_llava()
    _, px = tiny_inputs()
    ad = object.__new__(LlavaAdapter)
    ad._model, ad._vision_feature_layer, ad._vision_feature_select_strategy = model, -2, "default"
    with torch.no_grad():
        E, T = ad._extract_vision_features(px)
        ref_T = model.get_image_features(pixel_values=px, vision_feature_layer=-2,
                                         vision_feature_select_strategy="default")[0]
        ref_E = model.vision_tower(px, output_hidden_states=True).hidden_states[-2][0, 1:]
    assert E.shape == (N_PATCHES, 32) and torch.equal(E, ref_E) and torch.allclose(T, ref_T)


def test_vision_projector_formula_matches_transformers():
    """VisionProjector.forward (the cache engine) == LLaVA's get_image_features."""
    from onthefly_features import VisionProjector
    model = tiny_llava()
    _, px = tiny_inputs()
    vp = object.__new__(VisionProjector)
    torch.nn.Module.__init__(vp)
    vp.vision_tower, vp.projector = model.vision_tower, model.multi_modal_projector
    vp.feature_layer, vp.select_strategy = -2, "default"
    vp.device_, vp.dtype_ = torch.device("cpu"), torch.float32
    with torch.no_grad():
        E, T = vp(px)
        ref = model.get_image_features(pixel_values=px, vision_feature_layer=-2,
                                       vision_feature_select_strategy="default")
    assert E.shape == (1, N_PATCHES, 32) and torch.allclose(T, ref)


@pytest.mark.parametrize("depth,params", [("mlp3_small", 14_685_184), ("mlp3_paper", 27_282_432)])
def test_g_phi_maps_576x4096_to_576x1024(depth, params):
    net = ReconstructionNet(depth=depth).eval()
    assert net.param_count() == params
    with torch.no_grad():
        out = net(torch.randn(576, 4096))
    assert out.shape == (576, 1024) and torch.isfinite(out).all()


def test_loss_estimator_accepts_fp16_cache_tensors():
    net = ReconstructionNet(d_in=24, d_out=16, hidden_dim=12, depth="mlp2").eval()
    ns = NormStats(torch.randn(16), torch.rand(16) + 0.5, torch.randn(24), torch.rand(24) + 0.5, 0.1, 0.5)
    est = LossEstimator(net, ns)
    E, T = torch.randn(10, 16), torch.randn(10, 24)
    assert torch.allclose(est.compute(E.half(), T.half()), est.compute(E.half().float(), T.half().float()))


# ---------------------------------------------------------------------------
# Real weights + real cache
# ---------------------------------------------------------------------------

CACHE = ROOT / "data" / "cache" / "coco_train"
# Hugging Face cache from HF_HOME (no machine-specific path); real-weights tests skip without it
HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
SNAP = HF_HOME / "hub" / "models--llava-hf--llava-1.5-7b-hf"
PILOT = ROOT / "checkpoints" / "recon_pilot"
REAL = pytest.mark.skipif(not ((CACHE / "features_009999.pt").exists() and SNAP.exists()
                               and torch.cuda.is_available()),
                          reason="needs the 10k COCO cache, the LLaVA-1.5 weights and CUDA")
INDICES = (0, 1234, 4321, 7777, 9999)


@pytest.fixture(scope="module")
def vp():
    from onthefly_features import VisionProjector
    return VisionProjector(hf_home=str(HF_HOME), device="cuda", dtype=torch.float16)


def _pixels(vp, path):
    from PIL import Image
    with Image.open(path) as im:
        return vp.image_processor(images=im.convert("RGB"), return_tensors="pt")["pixel_values"]


@REAL
def test_real_model_configuration(vp):
    from transformers import LlavaConfig
    cfg = LlavaConfig.from_pretrained(next((SNAP / "snapshots").iterdir()))
    assert cfg.vision_feature_layer == -2 and cfg.vision_feature_select_strategy == "default"
    assert (cfg.vision_config.image_size, cfg.vision_config.patch_size, cfg.vision_config.hidden_size) == (336, 14, 1024)
    assert cfg.text_config.hidden_size == 4096
    assert vp.feature_layer == -2 and vp.select_strategy == "default"


@REAL
def test_real_preprocessing_equals_llava_processor(vp):
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", local_files_only=True)
    d = torch.load(CACHE / "features_001234.pt", map_location="cpu", weights_only=True, mmap=True)
    from PIL import Image
    with Image.open(d["image_path"]) as im:
        a = proc.image_processor(im.convert("RGB"), return_tensors="pt")["pixel_values"]
    assert torch.equal(a, _pixels(vp, d["image_path"]))


@REAL
@pytest.mark.parametrize("i", INDICES)
def test_real_features_match_cache(vp, i):
    d = torch.load(CACHE / f"features_{i:06d}.pt", map_location="cpu", weights_only=True)
    assert int(d["image_idx"]) == i
    with torch.no_grad():
        E, T = vp(_pixels(vp, d["image_path"]))
    assert E.shape == (1, 576, 1024) and T.shape == (1, 576, 4096) and E.dtype == torch.float16
    assert torch.equal(E[0].cpu(), d["E"]) and torch.equal(T[0].cpu(), d["T"])


@REAL
def test_real_patch_order_is_row_major(vp):
    d = torch.load(CACHE / "features_000000.pt", map_location="cpu", weights_only=True, mmap=True)
    px = _pixels(vp, d["image_path"]).to("cuda", torch.float16)
    with torch.no_grad():
        base = vp.vision_tower(px, output_hidden_states=True).hidden_states[0]
        for r, c in [(0, 0), (0, 23), (11, 7), (23, 23)]:
            p2 = px.clone()
            p2[:, :, 14 * r:14 * r + 14, 14 * c:14 * c + 14] += 0.5
            h = vp.vision_tower(p2, output_hidden_states=True).hidden_states[0]
            changed = torch.nonzero((h - base).float().abs().amax(-1)[0] > 1e-3).flatten().tolist()
            assert changed == [1 + 24 * r + c]


@REAL
@pytest.mark.skipif(not (PILOT / "best.pt").exists(), reason="pilot checkpoint not present")
def test_real_pilot_g_phi_reconstructs_cached_features():
    net = ReconstructionNet(depth="mlp3_small")
    net.load_state_dict(torch.load(PILOT / "best.pt", map_location="cpu", weights_only=True))
    ns = NormStats.load(str(PILOT / "norm_stats.pt"))
    est = LossEstimator(net.cuda().eval(), ns)
    for i in INDICES:
        d = torch.load(CACHE / f"features_{i:06d}.pt", map_location="cpu", weights_only=True)
        L = est.compute(d["E"].float(), d["T"].float())                     # per-patch standardised MSE
        E_std = (d["E"].float() - ns.E_mean) / (ns.E_std + 1e-8)
        trivial = float((E_std ** 2).mean())                               # predicting the mean
        assert L.shape == (576,) and torch.isfinite(L).all()
        assert float(L.mean()) < 0.2 and float(L.mean()) < 0.25 * trivial
        with torch.no_grad():
            T_std = (d["T"].float().cuda() - ns.T_mean.cuda()) / (ns.T_std.cuda() + 1e-8)
            assert net(T_std).shape == (576, 1024)


@REAL
def test_real_mlp3_paper_learns_on_cached_features():
    """A few optimiser steps on 2 real images: the 27.3M model's loss must fall (no full training)."""
    torch.manual_seed(0)
    net = ReconstructionNet(depth="mlp3_paper").cuda().train()
    ds = [torch.load(CACHE / f"features_{i:06d}.pt", map_location="cpu", weights_only=True) for i in (0, 1)]
    E = torch.cat([d["E"] for d in ds]).float().cuda()
    T = torch.cat([d["T"] for d in ds]).float().cuda()
    E = (E - E.mean(0)) / (E.std(0) + 1e-8)
    T = (T - T.mean(0)) / (T.std(0) + 1e-8)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    losses = []
    for _ in range(30):
        loss = torch.nn.functional.mse_loss(net(T), E)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]
