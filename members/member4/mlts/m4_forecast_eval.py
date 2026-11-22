"""
m4_forecast_eval.py - run the forecasting ladder and score every model identically.

Ladder: naive -> seasonal-naive -> ARIMAX -> XGBoost -> GRU -> LSTM   (simple first, handbook rule 15)

Scoring rule (same for every model):
  * origin t and target t+h must both lie in the scored split (val or test)
  * the target must be NORMAL (not inside a labelled anomaly) and observed (not NaN)
    -> forecast accuracy measures how well normal behaviour is predicted; anomalies are
       Milestone 5's job
  * metrics per channel and horizon: MAE, RMSE, sMAPE (%)
    sMAPE = 100 * mean( 2|y - y_hat| / (|y| + |y_hat| + eps) ) - unstable for channels
    near zero (batt_I, att_err); use MAE/RMSE for those.
  * model selection uses VAL; TEST numbers are reported once and never used for tuning.

Outputs:
  results/forecast_results.csv     member, model, channel, horizon, split, MAE, RMSE, sMAPE, seed, n
  results/m4_forecast_skill.csv    1 - MAE/MAE_naive per model/channel/horizon (val)
  results/m4_forecast_info.json    model details (AR orders, best iterations, GRU loss history)
  data/generated/forecasts/m4_pred_<model>.npz   predictions [N, H, C] for Milestone 5 (git-ignored)

Run from the repository root:
    python members/member4/mlts/m4_forecast_eval.py                      # all models (~10 min CPU)
    python members/member4/mlts/m4_forecast_eval.py --models naive,snaive # quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m4_ts_common as C  # noqa: E402
from m4_forecast_naive import HORIZONS, naive, seasonal_naive  # noqa: E402

PRED_DIR = C.GEN_DIR / "forecasts"
MODEL_ORDER = ["naive", "snaive", "arimax", "xgboost", "gru", "lstm"]
EPS = 1e-6


def runner(name: str):
    if name == "naive":
        return lambda df, masks, normal, **kw: (naive(df, **kw), {})
    if name == "snaive":
        return lambda df, masks, normal, **kw: (seasonal_naive(df, **kw), {})
    if name == "arimax":
        import m4_forecast_arima as m
        return m.fit_predict
    if name == "xgboost":
        import m4_forecast_xgboost as m
        return m.fit_predict
    if name in ("gru", "lstm"):
        import m4_forecast_lstm as m
        return lambda *a, **kw: m.fit_predict(*a, cell=name, **kw)
    raise ValueError(name)


def metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    e = y - yhat
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "sMAPE": float(100 * np.mean(2 * np.abs(e) / (np.abs(y) + np.abs(yhat) + EPS))), "n": int(len(y))}


def score(pred: np.ndarray, df: pd.DataFrame, masks: dict, normal: np.ndarray, model: str,
          horizons=HORIZONS, channels=C.CHANNELS, splits=("val", "test"), seed: int = 42) -> list[dict]:
    y = df[channels].values
    n = len(df)
    rows = []
    for s in splits:
        m = masks[s]
        for k, h in enumerate(horizons):
            t = np.arange(n - h)
            sel = m[t] & m[t + h] & normal[t + h]
            for j, c in enumerate(channels):
                yt, yp = y[t + h, j], pred[t, k, j]
                ok = sel & ~np.isnan(yt) & ~np.isnan(yp)
                if ok.sum() == 0:
                    continue
                rows.append({"member": "member4", "model": model, "channel": c, "horizon": h, "split": s,
                             **metrics(yt[ok], yp[ok]), "seed": seed})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Member 4 forecasting ladder + evaluation.")
    ap.add_argument("--models", default=",".join(MODEL_ORDER), help="comma list from " + ",".join(MODEL_ORDER))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reuse", action="store_true", help="load saved predictions instead of refitting")
    args = ap.parse_args(argv)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    df = C.load_telemetry()
    masks = C.split_masks(df)
    normal = (C.label_mask(df, C.load_labels()) == 0) & ~df[C.CHANNELS].isna().any(axis=1).values
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    C.RESULTS.mkdir(parents=True, exist_ok=True)

    info_path = C.RESULTS / "m4_forecast_info.json"
    all_info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    res_path = C.RESULTS / "forecast_results.csv"
    old = pd.read_csv(res_path) if res_path.exists() else pd.DataFrame()
    rows = []
    for name in models:
        f = PRED_DIR / f"m4_pred_{name}.npz"
        t0 = time.time()
        if args.reuse and f.exists():
            pred = np.load(f)["pred"]; info = all_info.get(name, {})
        else:
            print(f"[{name}] fitting ...")
            pred, info = runner(name)(df, masks, normal, horizons=HORIZONS, seed=args.seed)
            np.savez_compressed(f, pred=pred.astype(np.float32), horizons=np.array(HORIZONS),
                                channels=np.array(C.CHANNELS))
        all_info[name] = {**info, "seconds": round(time.time() - t0, 1)} if isinstance(info, dict) else info
        rows += score(pred, df, masks, normal, name, seed=args.seed)
        print(f"[{name}] done in {time.time() - t0:.0f}s")

    new = pd.DataFrame(rows)
    if len(old):
        old = old[~old["model"].isin(models)]
    res = pd.concat([old, new], ignore_index=True)
    res["model"] = pd.Categorical(res["model"], MODEL_ORDER, ordered=True)
    res = res.sort_values(["split", "channel", "horizon", "model"]).reset_index(drop=True)
    res.round(5).to_csv(res_path, index=False)
    info_path.write_text(json.dumps(all_info, indent=1, default=str), encoding="utf-8")

    val = res[res.split == "val"]
    base = val[val.model == "naive"].set_index(["channel", "horizon"])["MAE"]
    skill = val.assign(skill_vs_naive=lambda d: 1 - d["MAE"].values /
                       base.reindex(list(zip(d.channel, d.horizon))).values)
    skill[["model", "channel", "horizon", "MAE", "skill_vs_naive"]].round(4).to_csv(
        C.RESULTS / "m4_forecast_skill.csv", index=False)

    pd.set_option("display.width", 160)
    for h in HORIZONS:
        tab = val[val.horizon == h].pivot_table(index="channel", columns="model", values="MAE", observed=True)
        print(f"\nVAL MAE, horizon {h} ({h * 30} s):\n{tab.round(4).to_string()}")
    print(f"\nsaved {res_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
