"""
m4_train_fusion.py - train + evaluate all fusion models over seeds and scenarios (Milestones 8-11).

Models: tel_only, opt_only, sar_only (unimodal) | early_fusion, late_fusion, equal_weights (baselines)
        gated_fusion (MAIN) | ablations: gated_zero_fill, gated_no_tel_in_gate, gated_no_dropout,
        gated_no_residuals, gated_no_multitask
Training: train passes; modality dropout p = 0.2 per input (never all three) except unimodal models and
          gated_no_dropout; class-balanced BCE for both tasks; best epoch by validation AUPRC (mean of
          both tasks); decision thresholds = best validation F1 per task. Seeds 42, 43, 44.
Version 2 training protocol (applied to EVERY model, so the comparison stays fair):
          * telemetry sensor-loss practice: with p = 0.3 a training window loses k = 1-4 random channels,
            filled with zeros or with the cross-channel reconstruction (50/50); lost channels are always
            flagged "not measured" and their residuals set to 0 (a rebuilt value is never passed off as real)
          * cloud practice: with p = 0.3 the optical input is replaced by the same patch under 25-100 % cloud
          * every optical input carries the 5 image-quality numbers (m4_missing_modality.quality_features)
          New ablation: gated_no_quality (the gate does not read the quality numbers directly).
          Version 1 results are archived by evaluation/m4_archive_v1.py before a version 2 run.
Evaluation (TEST passes, once per scenario) - m4_missing_modality.SCENARIOS plus telemetry channel loss:
          tel_drop{k}_zero   k random channels lost, zero-filled + flagged missing
          tel_drop{k}_recon  same channels replaced by cross-channel reconstruction (RQ3); every lost
                             channel is hidden from the reconstruction models (no leak between lost channels)
                             and stays flagged "not measured" (version 2)
Metrics per task (health = pass contains a satellite fault; eo = patch is flooded):
          F1, precision, recall, AUPRC, AUROC; flood F1 on payload-fault (coupled) passes and on cloudy passes;
          mean alpha weights per subset (RQ4).

Outputs:
    results/fusion_results.csv        member, model, seed, scenario, task, F1, precision, recall, AUPRC, AUROC, ...
    results/m4_alpha_summary.csv      model, seed, scenario, subset, gate, alpha_tel, alpha_opt, alpha_sar
    results/m4_fusion_test_predictions.csv   per test pass (seed 42, all_present) for the dashboard
    results/m4_fusion_scenario_predictions.csv.gz   per test pass, every scenario and seed, for the
                                     comparison models (paired bootstrap confidence intervals)
    checkpoints/m4_fusion_<model>_s<seed>.pt   (git-ignored)

Run from the repository root (after Milestones 5-7 and m4_channel_reconstruction.py):
    python members/member4/training/m4_train_fusion.py                  # everything (~10-20 min CPU)
    python members/member4/training/m4_train_fusion.py --seeds 42 --models gated_fusion,early_fusion
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

MEMBER_DIR = Path(__file__).resolve().parents[1]
for _s in ("data", "models", "mlts", "evaluation", "training"):
    sys.path.insert(0, str(MEMBER_DIR / _s))
import m4_eo_patches as EP  # noqa: E402
import m4_missing_modality as MM  # noqa: E402
import m4_ts_common as C  # noqa: E402

PASS_DIR = C.GEN_DIR / "passes"
CKPT = MEMBER_DIR / "checkpoints"
MODELS = ["tel_only", "opt_only", "sar_only", "early_fusion", "equal_weights", "gated_fusion",
          "gated_no_quality", "gated_zero_fill", "gated_no_tel_in_gate", "gated_no_dropout", "gated_no_residuals",
          "gated_no_multitask"]
UNIMODAL = ("tel_only", "opt_only", "sar_only")
SEEDS = [42, 43, 44]
P_DROP = {"tel": 0.2, "opt": 0.2, "sar": 0.2}
EPOCHS = 40
TEL_DROP_K = (1, 2, 4)
PROTOCOL = "v2"
P_CH_AUG, P_CLOUD_AUG = 0.3, 0.3          # version 2 training practice (all models)
AUG_CLOUD = (0.25, 0.5, 0.75, 1.0)
KEEP_PRED = ("gated_fusion", "early_fusion", "equal_weights", "late_fusion", "gated_no_quality")


# ------------------------------------------------------------------ inputs
class Inputs:
    """Everything the fusion models need, with cached optical embeddings per cloud level."""

    def __init__(self):
        import torch  # noqa: F401
        from m4_eo_cnn import load_encoder
        self.passes = pd.read_parquet(PASS_DIR / "m4_passes.parquet")
        z = np.load(PASS_DIR / "m4_pass_telemetry.npz")
        self.tel, self.res, self.tel_nan = z["tel"], z["res"], z["tel_nan"]
        self.eo, self.meta, self.stats = EP.load()
        self.dataset = self.meta.dataset.iloc[0]
        self.opt_enc = load_encoder(CKPT / "m4_eo_optical_only.pt")
        sar = np.load(EP.OUT_DIR / "m4_emb_sar_only.npz")
        self.sar_emb = sar["emb"].astype(np.float32)[self.passes.patch_id.values]
        self.sar_prob = sar["prob"][self.passes.patch_id.values]
        self._opt_cache: dict = {}
        self.qref = MM.quality_reference(self.eo["s2"], self.passes, EP.OUT_DIR / "m4_quality_ref.json")
        rec = C.GEN_DIR / "m4_reconstruction.npz"
        self.recon = np.load(rec)["recon"] if rec.exists() else None
        self.split = {s: np.where(self.passes.split.values == s)[0] for s in ("train", "val", "test")}

    def optical(self, idx: np.ndarray, extra_cloud: float = 0.0, seed_offset: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Optical inputs for passes idx, after their own degradation (+ an extra cloud layer):
        [512-d embedding | 5 quality numbers] and the optical-only flood probability."""
        from m4_eo_cnn import encode
        key = (extra_cloud, seed_offset, tuple(idx[[0, -1]]), len(idx))
        if key not in self._opt_cache:
            imgs = np.empty((len(idx),) + self.eo["s2"].shape[1:], np.float32)
            for n, i in enumerate(idx):
                r = self.passes.iloc[i]
                rng = np.random.default_rng(int(r.degrade_seed))
                x = MM.degrade_optical(self.eo["s2"][int(r.patch_id)], r.opt_degrade, float(r.opt_severity), rng)
                if extra_cloud > 0:
                    x = MM.degrade_optical(x, "cloud", extra_cloud, np.random.default_rng(int(r.degrade_seed) + seed_offset))
                imgs[n] = x
            emb, prob = encode(self.opt_enc, imgs)
            q = np.stack([MM.quality_features(im, self.qref) for im in imgs])
            self._opt_cache[key] = (np.concatenate([emb.astype(np.float32), q], axis=1), prob)
        return self._opt_cache[key]

    def optical_aug(self, idx: np.ndarray) -> np.ndarray:
        """[len(AUG_CLOUD), n, 512+NQ] optical inputs of the same passes under extra cloud (training practice)."""
        return np.stack([self.optical(idx, c, seed_offset=101 + j)[0] for j, c in enumerate(AUG_CLOUD)])

    def telemetry_aug(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sensor-loss practice windows for passes idx: (zero-filled, reconstructed), same lost channels.
        Cached on disk because the reconstruction takes a few minutes."""
        f = PASS_DIR / "m4_tel_aug_v2.npz"
        ids = self.passes.pass_id.values[idx]
        if f.exists():
            z = np.load(f)
            if np.array_equal(z["pass_id"], ids):
                return z["Xz"], z["Xr"]
        rng = np.random.default_rng(7)
        Xz = np.empty((len(idx), self.tel.shape[1], 24), np.float32); Xr = np.empty_like(Xz)
        t0 = time.time()
        for n, i in enumerate(idx):
            ch = rng.choice(self.tel.shape[2], size=int(rng.integers(1, 5)), replace=False)
            Xz[n] = self._window(i, ch, "zero"); Xr[n] = self._window(i, ch, "recon")
            if n and n % 400 == 0:
                print(f"  sensor-loss practice windows {n}/{len(idx)} ({time.time() - t0:.0f}s)")
        np.savez_compressed(f, pass_id=ids, Xz=Xz, Xr=Xr)
        return Xz, Xr

    def _window(self, i: int, ch: np.ndarray, fill: str) -> np.ndarray:
        """One pass window [T, 24] with channels ch lost: values 0 (zero) or rebuilt (recon), flagged missing."""
        tel = self.tel[i].copy(); res = self.res[i].copy(); nan = self.tel_nan[i].copy()
        res[:, ch] = 0.0; nan[:, ch] = True
        if fill == "recon":
            import m4_channel_reconstruction as RC
            mu, sd = self._std()
            s = int(self.passes.start_idx.iloc[i])
            rec = RC.reconstruct(self._df(), np.arange(s, s + tel.shape[0]), [C.CHANNELS[c] for c in ch])
            tel[:, ch] = (rec - mu[ch]) / sd[ch]
        else:
            tel[:, ch] = 0.0
        return np.concatenate([np.nan_to_num(tel, nan=0.0), np.nan_to_num(res, nan=0.0) / 5.0,
                               nan.astype(np.float32)], axis=1).astype(np.float32)

    def telemetry(self, idx: np.ndarray, drop_k: int = 0, fill: str = "zero", seed: int = 0) -> np.ndarray:
        tel = self.tel[idx].copy(); res = self.res[idx].copy(); nan = self.tel_nan[idx].copy()
        if drop_k:
            rng = np.random.default_rng(seed)
            out = np.empty((len(idx), tel.shape[1], 24), np.float32)
            for n, i in enumerate(idx):
                ch = rng.choice(tel.shape[2], size=drop_k, replace=False)
                out[n] = self._window(i, ch, fill)
            return out
        x = np.concatenate([np.nan_to_num(tel, nan=0.0), np.nan_to_num(res, nan=0.0) / 5.0, nan.astype(np.float32)], axis=2)
        return x.astype(np.float32)

    def _df(self):
        if not hasattr(self, "_tel_df"):
            self._tel_df = C.load_telemetry()
        return self._tel_df

    def _std(self):
        if not hasattr(self, "_mu"):
            df = self._df(); masks = C.split_masks(df)
            normal = (C.label_mask(df, C.load_labels()) == 0) & ~df[C.CHANNELS].isna().any(axis=1).values
            self._mu = df.loc[masks["train"] & normal, C.CHANNELS].mean().values
            self._sd = df.loc[masks["train"] & normal, C.CHANNELS].std().values
        return self._mu, self._sd

    def labels(self, idx):
        return self.passes.health_label.values[idx].astype(np.float32), self.passes.eo_label.values[idx].astype(np.float32)


# ------------------------------------------------------------------ metrics
def task_metrics(y, p, thr) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    pred = p >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    two = len(np.unique(y)) == 2
    return {"F1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "precision": prec, "recall": rec,
            "accuracy": float((pred == (y == 1)).mean()),
            "AUPRC": float(average_precision_score(y, p)) if two else np.nan,
            "AUROC": float(roc_auc_score(y, p)) if two else np.nan, "n": int(len(y))}


def best_thr(y, p) -> float:
    c = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97)))
    return float(c[int(np.argmax([task_metrics(y, p, t)["F1"] for t in c]))])


# ------------------------------------------------------------------ train / predict
def predict(model, tel, opt, sar, m, batch: int = 512) -> dict:
    import torch
    model.eval(); outs = {"health": [], "eo": [], "alpha_health": [], "alpha_eo": []}
    with torch.no_grad():
        for i in range(0, len(tel), batch):
            o = model(torch.from_numpy(tel[i:i + batch]), torch.from_numpy(opt[i:i + batch]),
                      torch.from_numpy(sar[i:i + batch]), torch.from_numpy(m[i:i + batch]))
            outs["health"].append(torch.sigmoid(o["health"]).numpy()); outs["eo"].append(torch.sigmoid(o["eo"]).numpy())
            outs["alpha_health"].append(o["alpha_health"].numpy()); outs["alpha_eo"].append(o["alpha_eo"].numpy())
    return {k: np.concatenate(v) for k, v in outs.items()}


def masks_array(n, tri) -> np.ndarray:
    return np.tile(np.asarray(tri, np.float32), (n, 1))


def no_res(X: np.ndarray) -> np.ndarray:
    X = X.copy(); X[:, :, 8:16] = 0.0
    return X


def train_model(name: str, I: Inputs, seed: int, epochs: int = EPOCHS, verbose: bool = False,
                task_w=(1.0, 1.0), tag: str | None = None):
    import torch
    from m4_fusion import build
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = build(name)
    tr, va = I.split["train"], I.split["val"]
    Xt, (Ot, _), St = I.telemetry(tr), I.optical(tr), I.sar_emb[tr]
    Xv, (Ov, _), Sv = I.telemetry(va), I.optical(va), I.sar_emb[va]
    Xz, Xr = I.telemetry_aug(tr)                       # version 2 practice data (same for every model)
    Oa = I.optical_aug(tr)
    if name == "gated_no_residuals":
        Xt, Xv, Xz, Xr = no_res(Xt), no_res(Xv), no_res(Xz), no_res(Xr)
    yh, ye = I.labels(tr); vh, ve = I.labels(va)
    pw_h = torch.tensor((len(yh) - yh.sum()) / max(1, yh.sum())); pw_e = torch.tensor((len(ye) - ye.sum()) / max(1, ye.sum()))
    lf_h = torch.nn.BCEWithLogitsLoss(pos_weight=pw_h); lf_e = torch.nn.BCEWithLogitsLoss(pos_weight=pw_e)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    use_drop = name not in UNIMODAL and name != "gated_no_dropout"
    best, best_state, hist = -1, None, []
    mv = masks_array(len(va), (1, 1, 1))
    for ep in range(epochs):
        model.train(); perm = rng.permutation(len(tr))
        for b in range(0, len(perm), 128):
            ib = perm[b:b + 128]
            if use_drop:
                mk = MM.sample_masks(len(ib), P_DROP, rng)
                m = np.stack([mk["tel"], mk["opt"], mk["sar"]], 1).astype(np.float32)
            else:
                m = masks_array(len(ib), (1, 1, 1))
            xb, ob = Xt[ib].copy(), Ot[ib].copy()
            lose = rng.random(len(ib)) < P_CH_AUG                      # sensor-loss practice
            rec = rng.random(len(ib)) < 0.5
            xb[lose & ~rec], xb[lose & rec] = Xz[ib[lose & ~rec]], Xr[ib[lose & rec]]
            cl = np.where(rng.random(len(ib)) < P_CLOUD_AUG)[0]          # cloud practice
            if len(cl):
                ob[cl] = Oa[rng.integers(0, len(AUG_CLOUD), len(cl)), ib[cl]]
            o = model(torch.from_numpy(xb), torch.from_numpy(ob), torch.from_numpy(St[ib]), torch.from_numpy(m))
            loss = task_w[0] * lf_h(o["health"], torch.from_numpy(yh[ib])) + task_w[1] * lf_e(o["eo"], torch.from_numpy(ye[ib]))
            opt.zero_grad(); loss.backward(); opt.step()
        pv = predict(model, Xv, Ov, Sv, mv)
        aps = [task_metrics(vh, pv["health"], .5)["AUPRC"], task_metrics(ve, pv["eo"], .5)["AUPRC"]]
        score = float(np.nansum([w * a for w, a in zip(task_w, aps)]) / sum(task_w))
        hist.append(float(score))
        if score > best:
            best, best_state = score, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    pv = predict(model, Xv, Ov, Sv, mv)
    thr = {"health": best_thr(vh, pv["health"]), "eo": best_thr(ve, pv["eo"])}
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state, "name": name, "seed": seed, "thr": thr, "dataset": I.dataset, "protocol": PROTOCOL},
               CKPT / f"m4_fusion_{tag or name}_s{seed}.pt")
    return model, thr, hist


class Predictor:
    """Uniform call interface over one model, or two single-task models (no-multitask ablation)."""

    def __init__(self, name, I, seed, epochs):
        self.name = name
        if name == "gated_no_multitask":
            self.mh, th, h1 = train_model("gated_no_multitask", I, seed, epochs, task_w=(1.0, 0.0), tag="gated_health_only")
            self.me, te_, h2 = train_model("gated_no_multitask", I, seed, epochs, task_w=(0.0, 1.0), tag="gated_eo_only")
            self.thr, self.hist = {"health": th["health"], "eo": te_["eo"]}, {"health_only": h1, "eo_only": h2}
        else:
            self.m, self.thr, self.hist = train_model(name, I, seed, epochs)

    def __call__(self, X, O, S, M):
        if self.name == "gated_no_residuals":
            X = no_res(X)
        if self.name == "gated_no_multitask":
            ph, pe = predict(self.mh, X, O, S, M), predict(self.me, X, O, S, M)
            return {"health": ph["health"], "alpha_health": ph["alpha_health"], "eo": pe["eo"], "alpha_eo": pe["alpha_eo"]}
        return predict(self.m, X, O, S, M)


# ------------------------------------------------------------------ scenarios
def scenario_inputs(I: Inputs, scen: str, seed: int):
    te = I.split["test"]
    if scen.startswith("tel_drop"):
        k = int(scen.split("_")[1].replace("drop", "")); fill = scen.split("_")[2]
        X = I.telemetry(te, drop_k=k, fill=fill, seed=1000 + k)
        (O, Op), tri = I.optical(te), (1, 1, 1)
    else:
        cfg = MM.SCENARIOS[scen]
        X = I.telemetry(te)
        O, Op = I.optical(te, extra_cloud=cfg.get("cloud", 0.0))
        tri = cfg["masks"]
    return X, O, Op, I.sar_emb[te], I.sar_prob[te], masks_array(len(te), tri)


def evaluate(name, preds, thr, I, scen, seed) -> tuple[list, list]:
    te = I.split["test"]; yh, ye = I.labels(te)
    p = I.passes.iloc[te]
    coupled = p.coupled_payload_fault.values == 1
    cloudy = (p.opt_degrade.values == "cloud") | scen.startswith("cloud")
    rows = []
    for task, y in (("health", yh), ("eo", ye)):
        r = {"member": "member4", "model": name, "seed": seed, "scenario": scen, "task": task,
             "eo_dataset": I.dataset, "protocol": PROTOCOL, **task_metrics(y, preds[task], thr[task]), "threshold": thr[task]}
        if task == "eo":
            r["F1_coupled"] = task_metrics(y[coupled], preds["eo"][coupled], thr["eo"])["F1"] if coupled.any() else np.nan
            r["F1_cloudy"] = task_metrics(y[cloudy], preds["eo"][cloudy], thr["eo"])["F1"] if cloudy.any() else np.nan
            clean = ~coupled & ~cloudy
            r["F1_clean"] = task_metrics(y[clean], preds["eo"][clean], thr["eo"])["F1"] if clean.any() else np.nan
        rows.append(r)
    arows = []
    if "alpha_eo" in preds:
        for subset, sel in (("all", np.ones(len(te), bool)), ("coupled_payload_fault", coupled),
                            ("cloudy", cloudy & ~coupled), ("clean", ~coupled & ~cloudy)):
            if sel.any():
                for g in ("eo", "health"):
                    a = preds[f"alpha_{g}"][sel].mean(0)
                    arows.append({"model": name, "seed": seed, "scenario": scen, "subset": subset, "gate": g,
                                  "alpha_tel": a[0], "alpha_opt": a[1], "alpha_sar": a[2], "n": int(sel.sum())})
    return rows, arows


def scenarios() -> list[str]:
    return list(MM.SCENARIOS) + [f"tel_drop{k}_{f}" for k in TEL_DROP_K for f in ("zero", "recon")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Member 4 fusion: train all models x seeds, evaluate all scenarios.")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args(argv)
    I = Inputs()
    if I.dataset == EP.SYN_NAME:
        print("WARNING: EO inputs are the SYNTHETIC stand-in (pipeline check). Fusion numbers are not results.")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]
    rows, arows, info, preds_keep, scen_pred = [], [], {}, {}, []
    te_ids = I.passes.pass_id.values[I.split["test"]]
    te_y = I.labels(I.split["test"])

    def keep_scen(name, pr, thr, scen, seed):
        if name in KEEP_PRED:
            scen_pred.append(pd.DataFrame({"model": name, "seed": seed, "scenario": scen, "pass_id": te_ids,
                                           "y_health": te_y[0].astype(int), "y_eo": te_y[1].astype(int),
                                           "p_health": np.round(pr["health"], 4), "p_eo": np.round(pr["eo"], 4),
                                           "thr_health": thr["health"], "thr_eo": thr["eo"]}))
    for seed in seeds:
        uni_preds: dict = {}
        for name in models:
            t0 = time.time()
            model = Predictor(name, I, seed, args.epochs)
            thr, hist = model.thr, model.hist
            info[f"{name}_s{seed}"] = {"thr": thr, "val_curve": hist, "seconds": round(time.time() - t0, 1)}
            for scen in scenarios():
                X, O, _, S, _, M = scenario_inputs(I, scen, seed)
                pr = model(X, O, S, M)
                if name in UNIMODAL and M[0, UNIMODAL.index(name)] == 0:
                    r, a = [], []          # its only input is missing in this scenario -> not applicable
                else:
                    r, a = evaluate(name, pr, thr, I, scen, seed)
                rows += r; arows += a
                keep_scen(name, pr, thr, scen, seed)
                if name in UNIMODAL:
                    uni_preds.setdefault(scen, {})[name] = (pr, M)
                if seed == seeds[0] and scen == "all_present":
                    preds_keep[name] = pr
            print(f"seed {seed}  {name:<22} done ({time.time() - t0:.0f}s)")
        # late fusion = average of the available unimodal probabilities (thresholds from validation)
        if all(u in models for u in UNIMODAL):
            va = I.split["val"]; vh, ve = I.labels(va)
            vp = {}
            for u in UNIMODAL:
                ck = __import__("torch").load(CKPT / f"m4_fusion_{u}_s{seed}.pt", weights_only=False)
                from m4_fusion import build
                mdl = build(u); mdl.load_state_dict(ck["state"])
                vp[u] = predict(mdl, I.telemetry(va), I.optical(va)[0], I.sar_emb[va], masks_array(len(va), (1, 1, 1)))
            late_val = {t: np.mean([vp[u][t] for u in UNIMODAL], axis=0) for t in ("health", "eo")}
            thr = {"health": best_thr(vh, late_val["health"]), "eo": best_thr(ve, late_val["eo"])}
            for scen, d in uni_preds.items():
                M = d["tel_only"][1]
                w = M / M.sum(1, keepdims=True)
                pr = {t: sum(w[:, k] * d[u][0][t] for k, u in enumerate(UNIMODAL)) for t in ("health", "eo")}
                r, _ = evaluate("late_fusion", pr, thr, I, scen, seed); rows += r
                keep_scen("late_fusion", pr, thr, scen, seed)
                if seed == seeds[0] and scen == "all_present":
                    preds_keep["late_fusion"] = pr
            print(f"seed {seed}  late_fusion            done")

    C.RESULTS.mkdir(parents=True, exist_ok=True)
    rerun = set(models) | ({"late_fusion"} if all(u in models for u in UNIMODAL) else set())

    def merge(path, new):
        if path.exists() and len(new):
            old = pd.read_csv(path)
            old = old[~(old.model.isin(rerun) & old.seed.isin(seeds))]
            if "eo_dataset" in old.columns and "eo_dataset" in new.columns:
                old = old[old.eo_dataset == I.dataset]           # never mix synthetic and real EO rows
            if "protocol" in new.columns:                         # never mix version 1 and version 2 rows
                old = old[old["protocol"] == PROTOCOL] if "protocol" in old.columns else old.iloc[0:0]
            new = pd.concat([old, new], ignore_index=True)
        new.round(4).to_csv(path, index=False)
        return new

    res = merge(C.RESULTS / "fusion_results.csv", pd.DataFrame(rows))
    if scen_pred:
        sp = pd.concat(scen_pred, ignore_index=True); sp["eo_dataset"] = I.dataset; sp["protocol"] = PROTOCOL
        merge(C.RESULTS / "m4_fusion_scenario_predictions.csv.gz", sp)
    amerge = pd.DataFrame(arows); amerge["eo_dataset"] = I.dataset; amerge["protocol"] = PROTOCOL
    merge(C.RESULTS / "m4_alpha_summary.csv", amerge)
    ip = C.RESULTS / "m4_fusion_train_info.json"
    old_info = json.loads(ip.read_text(encoding="utf-8")) if ip.exists() else {}
    ip.write_text(json.dumps({**old_info, **info}, indent=1, default=str), encoding="utf-8")
    if "gated_fusion" not in preds_keep:
        print("\n(test-prediction file kept: gated_fusion was not part of this run)")
        return 0
    te = I.split["test"]
    keep = I.passes.iloc[te][["pass_id", "health_label", "anomaly_type", "eo_label", "patch_id", "opt_degrade",
                              "coupled_payload_fault"]].reset_index(drop=True)
    for name, pr in preds_keep.items():
        keep[f"p_health_{name}"] = np.round(pr["health"], 4); keep[f"p_eo_{name}"] = np.round(pr["eo"], 4)
        if name == "gated_fusion":
            for k, mname in enumerate(("tel", "opt", "sar")):
                keep[f"alpha_eo_{mname}"] = np.round(pr["alpha_eo"][:, k], 4)
                keep[f"alpha_health_{mname}"] = np.round(pr["alpha_health"][:, k], 4)
    keep["eo_dataset"] = I.dataset
    keep.to_csv(C.RESULTS / "m4_fusion_test_predictions.csv", index=False)

    s = res[res.scenario == "all_present"].groupby(["model", "task"])[["F1", "AUPRC"]].agg(["mean", "std"]).round(3)
    print("\nTEST, all inputs present (mean ± std over seeds):\n" + s.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
