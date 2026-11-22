"""
m4_dashboard_data.py - data access + LIVE model inference for the Member 4 Streamlit dashboard.

Kept separate from the UI so it can be unit-tested. Everything is read from files the pipeline
produced; nothing here trains a model.

Live inference for one stored test pass (handbook section 25, final demo):
    run_pass(L, pass_id, present=(tel, opt, sar), cloud=0.0, drop=[channels], fill="zero"|"recon")
    -> per model ("gated_fusion" = proposed, "early_fusion" = simple fusion):
       health probability, flood probability, decisions, alpha weights (both gates)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MEMBER_DIR = Path(__file__).resolve().parents[1]
for _s in ("data", "models", "mlts", "evaluation", "training", "explainability"):
    p = str(MEMBER_DIR / _s)
    if p not in sys.path:
        sys.path.insert(0, p)
import m4_ts_common as C  # noqa: E402

CKPT = MEMBER_DIR / "checkpoints"
PASS_DIR = C.GEN_DIR / "passes"
FORECAST_MODELS = ["naive", "snaive", "arimax", "xgboost", "gru", "lstm"]
MODEL_COLORS = {"naive": "#2a78d6", "snaive": "#eb6834", "arimax": "#1baf7a", "xgboost": "#eda100",
                "gru": "#e87ba4", "lstm": "#008300"}          # validated categorical order, fixed per model
DEMO_MODELS = {"gated_fusion": "Proposed: reliability-gated fusion", "early_fusion": "Simple fusion (concatenation)"}
CONTEXT_BEFORE = 240                                         # 2 h of telemetry shown before the pass


# ------------------------------------------------------------------ results (static files)
def results() -> dict[str, pd.DataFrame | dict | str | None]:
    def rd(name):
        f = C.RESULTS / name
        return pd.read_csv(f) if f.exists() else None
    out = {k: rd(f) for k, f in {
        "forecast": "forecast_results.csv", "skill": "m4_forecast_skill.csv", "health": "health_metrics.csv",
        "per_type": "m4_health_per_type.csv", "opssat": "m4_opssat_results.csv", "eo": "eo_baseline_results.csv",
        "fusion": "fusion_results.csv", "alpha": "m4_alpha_summary.csv", "recon": "m4_reconstruction_results.csv",
        "passes_summary": "m4_pass_summary.csv", "test_preds": "m4_fusion_test_predictions.csv"}.items()}
    js = C.RESULTS / "m4_final_summary.json"
    out["summary"] = json.loads(js.read_text(encoding="utf-8")) if js.exists() else {}
    info = C.RESULTS / "m4_forecast_info.json"
    out["forecast_info"] = json.loads(info.read_text(encoding="utf-8")) if info.exists() else {}
    rep = C.REPORTS / "FINAL_CONCLUSIONS.md"
    out["conclusions"] = rep.read_text(encoding="utf-8") if rep.exists() else None
    return out


def eo_is_synthetic(res: dict) -> bool:
    for k, col in (("eo", "dataset"), ("fusion", "eo_dataset")):
        d = res.get(k)
        if d is not None and len(d) and col in d.columns:
            return str(d[col].iloc[0]).startswith("synthetic")
    return False


# ------------------------------------------------------------------ live assets
@dataclass
class Live:
    df: pd.DataFrame
    passes: pd.DataFrame
    tel: np.ndarray
    res: np.ndarray
    tel_nan: np.ndarray
    eo: dict
    meta: pd.DataFrame
    stats: dict
    opt_enc: object
    sar_emb: np.ndarray
    models: dict
    thr: dict
    fc_pred: np.ndarray
    fc_name: str
    fc_sd: np.ndarray
    scores: pd.DataFrame | None
    recon: np.ndarray | None
    mu: np.ndarray
    sd: np.ndarray
    labels: pd.DataFrame


def missing_assets() -> list[str]:
    need = [PASS_DIR / "m4_passes.parquet", PASS_DIR / "m4_pass_telemetry.npz", C.TELEMETRY,
            C.GEN_DIR / "eo" / "m4_eo_patches.npz", CKPT / "m4_eo_optical_only.pt",
            CKPT / "m4_fusion_gated_fusion_s42.pt", CKPT / "m4_fusion_early_fusion_s42.pt",
            CKPT / "m4_recon_SoC.json"]
    return [str(p.relative_to(MEMBER_DIR)) for p in need if not p.exists()]


def load_live() -> Live:
    import torch
    import m4_eo_patches as EP
    import m4_residual_anomaly as RA
    from m4_eo_cnn import load_encoder
    from m4_fusion import build

    passes = pd.read_parquet(PASS_DIR / "m4_passes.parquet")
    z = np.load(PASS_DIR / "m4_pass_telemetry.npz")
    eo, meta, stats = EP.load()
    df = C.load_telemetry()
    masks = C.split_masks(df)
    labels = C.load_labels()
    normal = (C.label_mask(df, labels) == 0) & ~df[C.CHANNELS].isna().any(axis=1).values
    fc_name = str(passes.forecaster.iloc[0])
    fz = np.load(C.GEN_DIR / "forecasts" / f"m4_pred_{fc_name}.npz")
    _, _, fc_sd = RA.residual_score(df, fz["pred"], list(fz["horizons"]), masks, normal)
    models, thr = {}, {}
    for name in DEMO_MODELS:
        ck = torch.load(CKPT / f"m4_fusion_{name}_s42.pt", weights_only=False)
        m = build(name); m.load_state_dict(ck["state"]); m.eval()
        models[name], thr[name] = m, ck["thr"]
    sc_f = C.GEN_DIR / "m4_anomaly_scores.csv.gz"
    scores = pd.read_csv(sc_f, parse_dates=["timestamp"]) if sc_f.exists() else None
    rec_f = C.GEN_DIR / "m4_reconstruction.npz"
    sar = np.load(EP.OUT_DIR / "m4_emb_sar_only.npz")["emb"].astype(np.float32)
    return Live(df=df, passes=passes, tel=z["tel"], res=z["res"], tel_nan=z["tel_nan"], eo=eo, meta=meta, stats=stats,
                opt_enc=load_encoder(CKPT / "m4_eo_optical_only.pt"), sar_emb=sar, models=models, thr=thr,
                fc_pred=fz["pred"][:, list(fz["horizons"]).index(1), :], fc_name=fc_name, fc_sd=fc_sd, scores=scores,
                recon=np.load(rec_f)["recon"] if rec_f.exists() else None,
                mu=df.loc[masks["train"] & normal, C.CHANNELS].mean().values,
                sd=df.loc[masks["train"] & normal, C.CHANNELS].std().values, labels=labels)


def test_pass_table(L: Live) -> pd.DataFrame:
    p = L.passes[L.passes.split == "test"].copy()
    p["tag"] = np.where(p.coupled_payload_fault == 1, "payload fault + damaged optical",
                        np.where(p.health_label == 1, "satellite anomaly: " + p.anomaly_type.str.replace(";", ", "),
                                 np.where(p.opt_degrade == "cloud", "cloudy optical", "nominal")))
    p["flood"] = np.where(p.eo_label == 1, "flood", "no flood")
    p["label"] = "pass " + p.pass_id.astype(str) + " · " + p.t_start.str[:16].str.replace("T", " ") + " · " + p.tag + " · " + p.flood
    order = {"payload fault + damaged optical": 0, "cloudy optical": 2, "nominal": 3}
    p["rank"] = p.tag.map(order).fillna(1)
    return p.sort_values(["rank", "pass_id"])


# ------------------------------------------------------------------ one-pass inputs
def optical_image(L: Live, pass_row, cloud: float = 0.0) -> np.ndarray:
    import m4_missing_modality as MM
    rng = np.random.default_rng(int(pass_row.degrade_seed))
    x = MM.degrade_optical(L.eo["s2"][int(pass_row.patch_id)], pass_row.opt_degrade, float(pass_row.opt_severity), rng)
    if cloud > 0:
        x = MM.degrade_optical(x, "cloud", cloud, np.random.default_rng(int(pass_row.degrade_seed) + 1))
    return x


def telemetry_input(L: Live, i: int, drop: list[str] | None = None, fill: str = "zero") -> tuple[np.ndarray, np.ndarray | None]:
    """Model input [1, T, 24] for pass row i; returns also the reconstructed window (raw units) if used."""
    tel, res, nan = L.tel[i].copy(), L.res[i].copy(), L.tel_nan[i].copy()
    rec_raw = None
    if drop:
        ch = [C.CHANNELS.index(c) for c in drop]
        res[:, ch] = 0.0
        if fill == "recon":
            import m4_channel_reconstruction as RC
            s = int(L.passes.start_idx.iloc[i])
            rec_raw = RC.reconstruct(L.df, np.arange(s, s + tel.shape[0]), list(drop))
            tel[:, ch] = (rec_raw - L.mu[ch]) / L.sd[ch]; nan[:, ch] = False
        else:
            tel[:, ch] = np.nan; nan[:, ch] = True
    x = np.concatenate([np.nan_to_num(tel), np.nan_to_num(res) / 5.0, nan.astype(np.float32)], axis=1)
    return x[None].astype(np.float32), rec_raw


def run_pass(L: Live, pass_id: int, present=(1, 1, 1), cloud: float = 0.0, drop: list[str] | None = None,
             fill: str = "zero") -> dict:
    import torch
    from m4_eo_cnn import encode
    i = int(np.where(L.passes.pass_id.values == pass_id)[0][0])
    row = L.passes.iloc[i]
    X, rec_raw = telemetry_input(L, i, drop, fill)
    img = optical_image(L, row, cloud)
    O, p_opt_alone = encode(L.opt_enc, img[None])
    S = L.sar_emb[int(row.patch_id)][None]
    M = np.asarray(present, np.float32)[None]
    out = {"pass_id": pass_id, "optical_image": img, "reconstructed": rec_raw, "row": row,
           "optical_only_prob": float(p_opt_alone[0])}
    with torch.no_grad():
        for name, m in L.models.items():
            o = m(torch.from_numpy(X), torch.from_numpy(O.astype(np.float32)), torch.from_numpy(S), torch.from_numpy(M))
            ph, pe = float(torch.sigmoid(o["health"])[0]), float(torch.sigmoid(o["eo"])[0])
            out[name] = {"p_health": ph, "p_flood": pe, "health_alarm": ph >= L.thr[name]["health"],
                         "flood": pe >= L.thr[name]["eo"], "thr": L.thr[name],
                         "alpha_eo": o["alpha_eo"][0].numpy().round(4).tolist(),
                         "alpha_health": o["alpha_health"][0].numpy().round(4).tolist()}
    return out


def telemetry_stream(L: Live, pass_id: int) -> pd.DataFrame:
    """Context (2 h before) + the 60-min pass: actual, 1-step forecast, ±2 sd band, anomaly score, labels."""
    i = int(np.where(L.passes.pass_id.values == pass_id)[0][0])
    s = int(L.passes.start_idx.iloc[i])
    a, b = max(1, s - CONTEXT_BEFORE), s + L.tel.shape[1]
    d = L.df.iloc[a:b][["timestamp"] + C.CHANNELS].copy().reset_index(drop=True)
    fc = L.fc_pred[a - 1:b - 1]                                     # forecast for t, made at t-1
    for j, c in enumerate(C.CHANNELS):
        d[f"{c}_fc"] = fc[:, j]; d[f"{c}_lo"] = fc[:, j] - 2 * L.fc_sd[j]; d[f"{c}_hi"] = fc[:, j] + 2 * L.fc_sd[j]
    lab = C.label_mask(L.df, L.labels)[a:b]
    d["label"] = lab
    d["in_pass"] = np.arange(a, b) >= s
    if L.scores is not None:
        sc = L.scores.set_index("timestamp")
        d["score"] = sc["score"].reindex(d.timestamp).values
        d["threshold"] = float(sc["threshold"].iloc[0])
    return d
