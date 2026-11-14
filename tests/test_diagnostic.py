"""
Tests for the Stage-1 diagnostic fixes (audit C2, H1, H2, M3, M4) and the
shared VQA evaluation module (src/vqa_eval.py).
"""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src import vqa_eval  # noqa: E402
from src.vqa_eval import answer_strings, any_match, norm, vqa_acc  # noqa: E402
import loss_estimator as le  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rd = _load("run_diagnostic_t", "scripts/run_diagnostic.py")

VIZWIZ_ANSWERS = [{"answer": a, "answer_confidence": "yes"} for a in
                  ["unanswerable", "blank screen", "unanswerable", "cannot repair this computer automatically",
                   "unanswerable", "unanswerable", "blank", "unanswerable", "unanswerable", "unanswerable"]]


def _answers(n_match, word="dog", other="cat"):
    return [{"answer": word, "answer_confidence": "yes"}] * n_match + \
           [{"answer": f"{other}{i}", "answer_confidence": "no"} for i in range(10 - n_match)]


# ---------------------------------------------------------------------------
# C2: VizWiz answer dicts
# ---------------------------------------------------------------------------

class TestAnswerExtraction:
    def test_dicts_are_unwrapped(self):
        assert answer_strings(VIZWIZ_ANSWERS)[:2] == ["unanswerable", "blank screen"]

    def test_plain_strings_still_accepted(self):
        assert answer_strings(["a", "b"]) == ["a", "b"]

    @pytest.mark.parametrize("bad", [[{"text": "x"}], [{"answer": 3}], [None], [5]])
    def test_malformed_entries_raise(self, bad):
        with pytest.raises(TypeError):
            answer_strings(bad)

    def test_score_answer_on_real_shaped_example(self):
        s = rd.score_answer("Unanswerable.", VIZWIZ_ANSWERS)
        assert s["vqa_acc"] == pytest.approx(1.0) and s["any_match"] is True
        assert s["ground_truth"][0] == "unanswerable"

    def test_real_vizwiz_annotations_score_without_error(self):
        p = ROOT / "data" / "vizwiz" / "val.json"
        if not p.exists():
            pytest.skip("VizWiz annotations not present")
        anns = rd.load_vizwiz_annotations(str(p), 20)      # tuning examples only
        for a in anns:
            s = rd.score_answer("unanswerable", a["answers"])
            assert 0.0 <= s["vqa_acc"] <= 1.0


# ---------------------------------------------------------------------------
# H2: official metric
# ---------------------------------------------------------------------------

class TestOfficialMetric:
    # leave-one-out: acc = (n*min(1,(n-1)/3) + (10-n)*min(1,n/3)) / 10
    @pytest.mark.parametrize("n,expected", [(0, 0.0), (1, 0.3), (2, 0.6), (3, 0.9), (4, 1.0), (10, 1.0)])
    def test_leave_one_out_values(self, n, expected):
        assert vqa_acc("dog", answer_strings(_answers(n))) == pytest.approx(expected)

    def test_simplified_formula_would_differ(self):
        # the old diagnostic metric gave min(3/3, 1) = 1.0 here
        assert vqa_acc("dog", answer_strings(_answers(3))) == pytest.approx(0.9)

    @pytest.mark.parametrize("pred,gt", [("Two", "2"), ("a dog.", "dog"), ("The Cat!", "cat"), ("3.5", "3.5")])
    def test_normalisation(self, pred, gt):
        assert norm(pred) == norm(gt)

    def test_any_match_is_separate_binary_label(self):
        gts = answer_strings(_answers(1))
        assert any_match("dog", gts) is True and vqa_acc("dog", gts) == pytest.approx(0.3)
        assert any_match("bird", gts) is False

    def test_baseline_and_diagnostic_share_one_implementation(self):
        rb = _load("run_baseline_t", "scripts/run_baseline.py")
        assert rb.vqa_acc is vqa_eval.vqa_acc and rb.norm is vqa_eval.norm and rb.PROMPT is vqa_eval.PROMPT
        assert rd.vqa_acc is vqa_eval.vqa_acc

    def test_metric_reproduces_stored_baseline(self):
        pred = ROOT / "results" / "baseline" / "predictions.jsonl"
        val = ROOT / "data" / "vizwiz" / "val.json"
        if not (pred.exists() and val.exists()):
            pytest.skip("baseline predictions not present")
        v = json.loads(val.read_text(encoding="utf-8"))
        recs = [json.loads(l) for l in pred.read_text(encoding="utf-8").splitlines() if l.strip()]
        tun = [r for r in recs if r["split"] == "tuning"][:500]          # tuning records only
        for r in tun:
            assert vqa_acc(r["pred"], answer_strings(v[r["idx"]]["answers"])) == pytest.approx(r["acc"], abs=0)


# ---------------------------------------------------------------------------
# H1: prompt and generation identical to the baseline
# ---------------------------------------------------------------------------

class _Proc:
    def __init__(self):
        self.seen = {}

    def __call__(self, images, text, return_tensors):
        self.seen["text"] = text

        class B(dict):
            def to(self, dev):
                return self
        return B(input_ids=torch.tensor([[1, 2, 3]]), pixel_values=torch.zeros(1, 3, 4, 4))

    def decode(self, ids, skip_special_tokens):
        self.seen["decoded"] = ids.tolist()
        return " yes "


class _Model:
    dtype = torch.float16

    def __init__(self):
        self.kw = {}

    def generate(self, **kw):
        self.kw = kw
        return torch.tensor([[1, 2, 3, 9, 8]])


def test_llava_answer_uses_baseline_prompt_and_decoding():
    proc, model = _Proc(), _Model()
    out = rd.llava_answer(model, proc, image=None, question="What is this?", device="cpu")
    assert proc.seen["text"] == vqa_eval.PROMPT.format(q="What is this?")
    assert "respond with 'Unanswerable'" in proc.seen["text"]
    assert {k: model.kw[k] for k in vqa_eval.GENERATION} == vqa_eval.GENERATION
    assert model.kw["max_new_tokens"] == 16 and model.kw["do_sample"] is False
    assert model.kw["pixel_values"].dtype == torch.float16          # cast like the baseline
    assert proc.seen["decoded"] == [9, 8] and out == "yes"           # only the new tokens


# ---------------------------------------------------------------------------
# M3: paths resolve against the project root
# ---------------------------------------------------------------------------

def test_paths_independent_of_working_directory(tmp_path, monkeypatch):
    cfg_path = str(ROOT / "configs" / "pilot_mlp3_small.yaml")
    import yaml
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    monkeypatch.chdir(tmp_path)                                         # somewhere else
    paths = rd.resolve_run_paths(rd.parse_args(["--config", cfg_path]), cfg, cfg_path)
    assert paths["recon_checkpoint"] == str((ROOT / "checkpoints" / "recon_pilot" / "best.pt").resolve())
    assert paths["norm_stats"] == str((ROOT / "checkpoints" / "recon_pilot" / "norm_stats.pt").resolve())
    assert paths["output_dir"] == str((ROOT / "results" / "pilot_mlp3_small").resolve())
    assert paths["vizwiz_questions"] == str((ROOT / "data" / "vizwiz" / "val.json").resolve())
    assert not any(str(tmp_path) in v for k, v in paths.items() if v and k != "root")


def test_cli_overrides_are_relative_to_shell(tmp_path, monkeypatch):
    cfg_path = str(ROOT / "configs" / "pilot_mlp3_small.yaml")
    monkeypatch.chdir(tmp_path)
    paths = rd.resolve_run_paths(rd.parse_args(["--config", cfg_path, "--output_dir", "mine"]),
                                 {"training": {}}, cfg_path)
    assert paths["output_dir"] == str((tmp_path / "mine").resolve())


def test_heldout_guard():
    rd.check_heldout(3800, 3800)
    with pytest.raises(SystemExit):
        rd.check_heldout(3801, 3800)


# ---------------------------------------------------------------------------
# M4: checkpoint / NormStats pairing
# ---------------------------------------------------------------------------

def _tiny_run(tmp_path):
    d = tmp_path / "cache"
    d.mkdir()
    g = torch.Generator().manual_seed(0)
    for i in range(6):
        torch.save({"E": torch.randn(8, 16, generator=g), "T": torch.randn(8, 24, generator=g)},
                   d / f"features_{i:06d}.pt")
    net = le.ReconstructionNet(d_in=24, d_out=16, hidden_dim=12, depth="mlp2")
    t = le.ReconstructionTrainer(net, {"checkpoints_dir": str(tmp_path / "ck"),
                                       "training": {"epochs": 1, "batch_size": 8, "val_fraction": 0.34,
                                                    "files_per_shard": 2, "seed": 0}})
    t.fit(str(d))
    return tmp_path / "ck" / "recon"


class TestPairing:
    CFG = {"loss": {"reconstructor": "mlp2", "hidden_dim": 12, "dropout": 0.1}}

    def test_trainer_writes_fingerprint(self, tmp_path):
        ck = _tiny_run(tmp_path)
        ns = le.NormStats.load(str(ck / "norm_stats.pt"))
        sd = torch.load(ck / "best.pt", weights_only=True)
        assert ns.meta["model_sha256"] == le.state_dict_fingerprint(sd)
        assert ns.meta["depth"] == "mlp2" and ns.meta["run_name"] == "recon"

    def test_matching_pair_loads(self, tmp_path):
        ck = _tiny_run(tmp_path)
        est = rd.load_loss_estimator(str(ck / "best.pt"), str(ck / "norm_stats.pt"), self.CFG, "cpu",
                                     require_fingerprint=True)
        assert est.compute(torch.randn(3, 16), torch.randn(3, 24)).shape == (3,)

    def test_mismatched_pair_is_refused(self, tmp_path):
        ck = _tiny_run(tmp_path)
        sd = torch.load(ck / "best.pt", weights_only=True)
        sd = {k: v + 0.01 for k, v in sd.items()}                      # "another run's" weights
        torch.save(sd, tmp_path / "other.pt")
        with pytest.raises(ValueError, match="does not belong"):
            rd.load_loss_estimator(str(tmp_path / "other.pt"), str(ck / "norm_stats.pt"), self.CFG, "cpu")

    def test_dimension_mismatch_is_refused(self):
        ns = le.NormStats(torch.zeros(8), torch.ones(8), torch.zeros(24), torch.ones(24), 0.1, 0.2)
        with pytest.raises(ValueError, match="dims"):
            le.check_norm_stats_pairing(ns, le.ReconstructionNet(d_in=24, d_out=16, hidden_dim=12, depth="mlp2"))

    def test_legacy_stats_warn_or_refuse(self, caplog):
        net = le.ReconstructionNet(d_in=24, d_out=16, hidden_dim=12, depth="mlp2")
        ns = le.NormStats(torch.zeros(16), torch.ones(16), torch.zeros(24), torch.ones(24), 0.1, 0.2)
        le.check_norm_stats_pairing(ns, net)
        assert "no fingerprint" in caplog.text
        with pytest.raises(ValueError, match="no fingerprint"):
            le.check_norm_stats_pairing(ns, net, require_meta=True)

    def test_pilot_files_still_load(self):
        ck = ROOT / "checkpoints" / "recon_pilot"
        if not (ck / "best.pt").exists():
            pytest.skip("pilot checkpoint not present")
        import yaml
        cfg = yaml.safe_load(open(ROOT / "configs" / "pilot_mlp3_small.yaml", encoding="utf-8"))
        est = rd.load_loss_estimator(str(ck / "best.pt"), str(ck / "norm_stats.pt"), cfg, "cpu")
        assert est.recon_net.param_count() == 14_685_184
        assert est.norm_stats.L_p95 == pytest.approx(0.142976, abs=1e-6)


class TestCalibration:
    def _ns(self, sha="abc"):
        return le.NormStats(torch.zeros(4), torch.ones(4), torch.zeros(6), torch.ones(6), 0.1, 0.2,
                            {"model_sha256": sha})

    def test_none_keeps_stats(self):
        ns = self._ns()
        assert le.apply_norm_calibration(ns, None) is ns

    def test_replaces_percentiles_only(self, tmp_path):
        p = tmp_path / "cal.json"
        p.write_text(json.dumps({"L_p5": 0.3, "L_p95": 0.9, "source": "vizwiz_tuning_calib", "model_sha256": "abc"}))
        out = le.apply_norm_calibration(self._ns(), str(p))
        assert (out.L_p5, out.L_p95) == (0.3, 0.9) and torch.equal(out.E_std, torch.ones(4))
        assert out.meta["percentiles_source"] == "vizwiz_tuning_calib"

    @pytest.mark.parametrize("cal,match", [({"L_p5": 0.5, "L_p95": 0.5}, "Invalid"),
                                           ({"L_p5": 0.1, "L_p95": 0.9, "model_sha256": "zzz"}, "different g_phi")])
    def test_invalid_calibration_refused(self, tmp_path, cal, match):
        p = tmp_path / "cal.json"
        p.write_text(json.dumps(cal))
        with pytest.raises(ValueError, match=match):
            le.apply_norm_calibration(self._ns(), str(p))


def test_summary_labels_metrics_correctly():
    recs = [{"vqa_acc": 0.9, "any_match": True, "mean_loss": 0.1},
            {"vqa_acc": 0.0, "any_match": False, "mean_loss": 0.3}]
    s = rd.build_summary(recs, {"pearson_r": 0.5}, {}, {"k": 10}, 1.0)
    assert "VQA accuracy (official) : 0.4500" in s and "Any-annotator match     : 50.0%" in s
    assert "not VQA accuracy" in s


def test_baseline_resume_reader(tmp_path):
    rb = _load("run_baseline_t2", "scripts/run_baseline.py")
    p = tmp_path / "predictions.jsonl"
    p.write_text(json.dumps({"idx": 0, "acc": 1.0}) + "\n\n" + json.dumps({"idx": 3, "acc": 0.0}) + "\n")
    assert sorted(rb.load_done(p)) == [0, 3]
    assert rb.load_done(tmp_path / "missing.jsonl") == {}
