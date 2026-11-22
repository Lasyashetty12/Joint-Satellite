"""Tests for Milestones 6-15: EO patches, missing-modality simulator, CNN + fusion models,
pass sampling, fusion metrics, final report helpers, run-all orchestrator, dashboard (E2E)."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

import m4_build_passes as BP  # noqa: E402
import m4_eo_patches as EP  # noqa: E402
import m4_final_report as FR  # noqa: E402
import m4_missing_modality as MM  # noqa: E402
import m4_train_fusion as TF  # noqa: E402
from m4_eo_cnn import EOCNN, encode  # noqa: E402
from m4_fusion import FusionNet, build  # noqa: E402


# ------------------------------------------------------------------ missing-modality simulator
@pytest.mark.parametrize("kind", ["cloud", "stripes", "noise", "saturation"])
def test_degradations_change_image_and_keep_shape(kind):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 64, 64)).astype(np.float32)
    before = x.copy()
    y = MM.degrade_optical(x, kind, 0.6, rng)
    assert y.shape == x.shape and not np.allclose(x, y)
    assert np.array_equal(x, before)                              # input not modified in place


def test_cloud_severity_controls_cover():
    rng = np.random.default_rng(1)
    x = np.zeros((6, 128, 128), np.float32)
    cover = [(MM.degrade_optical(x, "cloud", s, np.random.default_rng(2))[0] > 2).mean() for s in (0.25, 0.5, 0.75)]
    assert cover[0] < cover[1] < cover[2] and abs(cover[1] - 0.5) < 0.05


def test_no_degradation_is_identity():
    x = np.ones((6, 8, 8), np.float32)
    assert np.array_equal(MM.degrade_optical(x, "none", 0.9, np.random.default_rng(0)), x)


def test_masks_never_all_missing():
    m = MM.sample_masks(5000, {"tel": 0.6, "opt": 0.6, "sar": 0.6}, np.random.default_rng(3))
    assert ((m["tel"] + m["opt"] + m["sar"]) >= 1).all()


def test_drop_channels():
    w, idx = MM.drop_channels(np.zeros((120, 8)), 3, np.random.default_rng(0))
    assert len(idx) == 3 and np.isnan(w[:, idx]).all() and not np.isnan(np.delete(w, idx, axis=1)).any()


# ------------------------------------------------------------------ EO patches
def test_pool_and_label_pool():
    a = np.arange(16, dtype=np.float32).reshape(1, 4, 4); a[0, 0, 0] = np.nan
    p = EP.pool(a)
    assert p.shape == (1, 2, 2) and p[0, 0, 0] == pytest.approx((1 + 4 + 5) / 3)
    lab = np.array([[1, 1, 0, 0], [1, 0, 0, 0], [-1, -1, 1, 1], [-1, -1, 1, -1]])
    assert EP.pool_label(lab).tolist() == [[1, 0], [-1, 1]]


def test_split_patches_four_per_chip():
    parts = EP.split_patches(np.zeros((2, 512, 512)))
    assert len(parts) == 4 and parts[3][0] == (256, 256) and parts[0][1].shape == (2, 256, 256)


def test_official_split_map(tmp_path):
    pd.DataFrame([["Ghana_1_S1Hand.tif", "Ghana_1_LabelHand.tif"]]).to_csv(tmp_path / "flood_train_data.csv", header=False, index=False)
    pd.DataFrame([["Spain_2_S1Hand.tif", "Spain_2_LabelHand.tif"]]).to_csv(tmp_path / "flood_valid_data.csv", header=False, index=False)
    assert EP.chip_split_map(tmp_path) == {"Ghana_1": "train", "Spain_2": "val"}


def test_synthetic_standin_is_labelled_as_such():
    arrays, meta, stats = EP.build_synthetic(8, seed=0)
    assert (meta.dataset == EP.SYN_NAME).all()
    assert arrays["s1"].shape[1:] == (2, 128, 128) and arrays["s2"].shape[1:] == (6, 128, 128)
    assert set(meta.split) <= {"train", "val", "test", "bolivia"}
    # split by EVENT: no event appears in two splits
    assert meta.groupby("event").split.nunique().max() == 1
    assert EP.to_rgb(arrays["s2"][0], stats).shape == (128, 128, 3)


# ------------------------------------------------------------------ models
@pytest.mark.parametrize("in_ch", [2, 6, 8])
def test_eocnn_shapes(in_ch):
    m = EOCNN(in_ch).eval()
    x = np.random.default_rng(0).normal(size=(3, in_ch, 64, 64)).astype(np.float32)
    emb, prob = encode(m, x)
    assert emb.shape == (3, 512) and prob.shape == (3,) and ((prob >= 0) & (prob <= 1)).all()


def _batch(n=5):
    g = torch.Generator().manual_seed(0)
    return torch.randn(n, 120, 24, generator=g), torch.randn(n, 512, generator=g), torch.randn(n, 512, generator=g)


@pytest.mark.parametrize("name", ["tel_only", "opt_only", "sar_only", "early_fusion", "equal_weights", "gated_fusion",
                                  "gated_zero_fill", "gated_no_tel_in_gate"])
def test_fusion_output_shapes(name):
    tel, opt, sar = _batch()
    out = build(name).eval()(tel, opt, sar, torch.ones(5, 3))
    assert out["health"].shape == (5,) and out["eo"].shape == (5,) and out["alpha_eo"].shape == (5, 3)


def test_gate_alphas_sum_to_one_and_missing_get_zero():
    tel, opt, sar = _batch(4)
    m = torch.tensor([[1, 1, 1], [1, 0, 1], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
    out = FusionNet("gated").eval()(tel, opt, sar, m)
    for g in ("alpha_eo", "alpha_health"):
        a = out[g]
        assert torch.allclose(a.sum(1), torch.ones(4), atol=1e-6)
        assert (a[m == 0] == 0).all()
    assert torch.allclose(out["alpha_eo"][3], torch.tensor([0.0, 0.0, 1.0]))


def test_equal_weights_are_uniform_over_present():
    tel, opt, sar = _batch(2)
    m = torch.tensor([[1, 0, 1], [1, 1, 1]], dtype=torch.float32)
    a = FusionNet("equal").eval()(tel, opt, sar, m)["alpha_eo"]
    assert torch.allclose(a[0], torch.tensor([0.5, 0.0, 0.5])) and torch.allclose(a[1], torch.full((3,), 1 / 3))


def test_missing_input_value_is_ignored_by_gated_model():
    """With the optical mask at 0, changing the optical features must not change the output."""
    tel, opt, sar = _batch(3)
    net = FusionNet("gated").eval()
    m = torch.tensor([[1, 0, 1]] * 3, dtype=torch.float32)
    o1 = net(tel, opt, sar, m); o2 = net(tel, opt * 100 + 7, sar, m)
    assert torch.allclose(o1["eo"], o2["eo"]) and torch.allclose(o1["health"], o2["health"])


def test_gated_model_learns_toy_task():
    torch.manual_seed(0)
    tel, opt, sar = _batch(64)
    y = (sar[:, 0] > 0).float()                     # flood signal lives in SAR only
    net = FusionNet("gated"); opt_ = torch.optim.Adam(net.parameters(), 1e-2)
    for _ in range(150):
        o = net(tel, opt, sar, torch.ones(64, 3))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(o["eo"], y)
        opt_.zero_grad(); loss.backward(); opt_.step()
    acc = ((torch.sigmoid(net(tel, opt, sar, torch.ones(64, 3))["eo"]) > 0.5).float() == y).float().mean()
    assert acc > 0.9


# ------------------------------------------------------------------ pass sampling, metrics, report, orchestrator
def test_sample_starts_inside_split_and_anomaly_share():
    mask = np.zeros(10000, bool); mask[2000:6000] = True
    events = [(3000, 3100, "spike"), (8000, 8100, "drift")]
    s = BP.sample_starts(mask, events, 200, np.random.default_rng(0))
    assert (s >= 2000).all() and (s + BP.WIN - 1 <= 5999).all()
    overl = ((s <= 3100) & (s + BP.WIN - 1 >= 3000)).mean()
    assert overl >= BP.ANOM_SHARE - 0.05


def test_task_metrics_known_values():
    m = TF.task_metrics(np.array([0, 0, 1, 1.0]), np.array([0.1, 0.9, 0.8, 0.2]), 0.5)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["F1"] == 0.5


def test_report_helpers():
    t = FR.md_table(pd.DataFrame({"a": ["x"], "b": [0.12345]}))
    assert "| x | 0.123 |" in t and FR.word(None) == "**Not evaluated**"


def test_run_all_stage_names():
    import m4_run_all as RA
    a = type("A", (), {"fast": False, "synthetic_eo": True})()
    names = [s[0] for s in RA.stages(a)]
    assert len(names) == len(set(names)) and names[0] == "generator" and names[-1] == "tests"
    assert "--synthetic" in dict((s[0], s[1]) for s in RA.stages(a))["eo_patches"]


# ------------------------------------------------------------------ dashboard end-to-end (handbook E2E level)
@pytest.mark.parametrize("page", ["Mission overview", "Model accuracy", "Final conclusions", "Live demo"])
def test_dashboard_pages_render(page):
    from pathlib import Path
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    import m4_dashboard_data as D
    if page == "Live demo" and D.missing_assets():
        pytest.skip("live demo needs pipeline outputs (run m4_run_all.py)")
    app = Path(__file__).resolve().parents[1] / "dashboard" / "m4_app.py"
    at = AppTest.from_file(str(app), default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [str(e.value)[:500] for e in at.exception]


def test_all_text_io_is_explicit_utf8():
    """Windows defaults to cp1252: every read_text/write_text must say encoding='utf-8'
    (the report and dashboard contain characters such as 'α' and '⚠️')."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    bad = []
    for p in root.rglob("*.py"):
        src = p.read_text(encoding="utf-8")
        for m in re.finditer(r"\.(read_text|write_text)\(", src):
            j, depth = m.end(), 1
            while depth:
                depth += src[j] == "("; depth -= src[j] == ")"; j += 1
            if "encoding=" not in src[m.start():j]:
                bad.append(f"{p.name}:{src[:m.start()].count(chr(10)) + 1}")
    assert not bad, bad
