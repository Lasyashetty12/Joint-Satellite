"""
m4_train_eo.py - train the Earth-observation baselines (Milestone 6).

    optical_only      CNN on the 6 Sentinel-2 bands
    sar_only          CNN on Sentinel-1 VV + VH
    optical_sar_cnn   CNN on all 8 bands stacked (image fusion WITHOUT telemetry; handbook baseline)

Task: patch-level flood detection (eo_label = water fraction >= 5 %).
Training: train split only; flips / 90-degree rotations as augmentation; class-balanced BCE;
best epoch = highest VALIDATION AUPRC; decision threshold = best VALIDATION F1. Test and the
unseen Bolivia event are scored once.

Outputs:
    results/eo_baseline_results.csv   member, model, dataset, split, accuracy, precision, recall, F1, AUROC, AUPRC, n, threshold, seed
    checkpoints/m4_eo_<model>.pt      encoder weights (git-ignored)
    data/generated/eo/m4_emb_<model>.npz   512-d embeddings + flood probability for EVERY patch (for fusion)

Run from the repository root (GPU used automatically if available; CPU works):
    python members/member4/training/m4_train_eo.py                     # all three, 12 epochs
    python members/member4/training/m4_train_eo.py --models sar_only --epochs 5
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
for _s in ("data", "models", "mlts", "evaluation"):
    sys.path.insert(0, str(MEMBER_DIR / _s))
import m4_eo_patches as EP  # noqa: E402

RESULTS = MEMBER_DIR / "results"
CKPT = MEMBER_DIR / "checkpoints"
MODELS = ["optical_only", "sar_only", "optical_sar_cnn"]
SCORE_SPLITS = ("val", "test", "bolivia")


def inputs(arrays: dict, model: str) -> np.ndarray:
    if model == "optical_only":
        return arrays["s2"]
    if model == "sar_only":
        return arrays["s1"]
    return np.concatenate([arrays["s2"], arrays["s1"]], axis=1)


def binary_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    both = len(np.unique(y)) == 2
    return {"accuracy": float((pred == y).mean()), "precision": prec, "recall": rec,
            "F1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "AUROC": float(roc_auc_score(y, p)) if both else np.nan,
            "AUPRC": float(average_precision_score(y, p)) if both else np.nan, "n": int(len(y))}


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    cands = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97)))
    f1 = [binary_metrics(y, p, t)["F1"] for t in cands]
    return float(cands[int(np.argmax(f1))])


def augment(xb):
    import torch
    if torch.rand(1) < 0.5:
        xb = xb.flip(-1)
    if torch.rand(1) < 0.5:
        xb = xb.flip(-2)
    k = int(torch.randint(0, 4, (1,)))
    return torch.rot90(xb, k, dims=(-2, -1))


def train_one(model_name: str, arrays: dict, meta: pd.DataFrame, epochs: int, seed: int, pretrained: bool,
              batch: int, lr: float, verbose: bool = True) -> tuple[list[dict], dict]:
    import torch
    from m4_eo_cnn import EOCNN, encode

    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = inputs(arrays, model_name)
    y = meta["eo_label"].values
    ok = y >= 0
    tr = np.where((meta.split == "train").values & ok)[0]
    va = np.where((meta.split == "val").values & ok)[0]
    model = EOCNN(X.shape[1], pretrained=pretrained).to(dev)
    pos = max(1, int(y[tr].sum())); neg = max(1, len(tr) - pos)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos, device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(seed)

    def predict(idx):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), 64):
                xb = torch.from_numpy(X[idx[i:i + 64]].astype(np.float32)).to(dev)
                out.append(torch.sigmoid(model(xb)).cpu().numpy())
        return np.concatenate(out)

    best, best_state, hist = -1.0, None, []
    for ep in range(epochs):
        t0 = time.time(); model.train(); tot = 0.0
        perm = rng.permutation(tr)
        for b in range(0, len(perm), batch):
            ib = perm[b:b + batch]
            xb = augment(torch.from_numpy(X[ib].astype(np.float32))).to(dev)
            yb = torch.from_numpy(y[ib].astype(np.float32)).to(dev)
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
            tot += loss.item() * len(ib)
        sched.step()
        pv = predict(va)
        ap = binary_metrics(y[va], pv, 0.5)["AUPRC"]
        hist.append({"epoch": ep + 1, "train_loss": tot / len(tr), "val_AUPRC": ap})
        if verbose:
            print(f"  {model_name:<16} epoch {ep + 1}/{epochs}  loss {tot / len(tr):.4f}  val AUPRC {ap:.3f}  ({time.time() - t0:.0f}s)")
        if np.nan_to_num(ap, nan=0) > best:
            best = np.nan_to_num(ap, nan=0); best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.to("cpu").eval()
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state, "in_ch": X.shape[1], "model": model_name, "seed": seed,
                "dataset": meta.dataset.iloc[0]}, CKPT / f"m4_eo_{model_name}.pt")

    emb, prob = encode(model, X)
    EP.OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EP.OUT_DIR / f"m4_emb_{model_name}.npz", emb=emb.astype(np.float16), prob=prob.astype(np.float32))
    thr = best_f1_threshold(y[va], prob[va])
    rows = []
    for s in SCORE_SPLITS:
        idx = np.where((meta.split == s).values & ok)[0]
        if len(idx):
            rows.append({"member": "member4", "model": model_name, "dataset": meta.dataset.iloc[0], "split": s,
                         **binary_metrics(y[idx], prob[idx], thr), "threshold": thr, "seed": seed})
    return rows, {"history": hist, "best_val_AUPRC": best, "threshold": thr}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Member 4 EO baselines: optical-only, SAR-only, optical+SAR CNN.")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pretrained", action="store_true", help="ImageNet initialisation (downloads weights once)")
    args = ap.parse_args(argv)

    arrays, meta, _ = EP.load()
    if meta.dataset.iloc[0] == EP.SYN_NAME:
        print("WARNING: training on the SYNTHETIC stand-in (pipeline check). Numbers are not results.")
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "eo_baseline_results.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    info_path = RESULTS / "m4_eo_train_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    rows = []
    for m in [x.strip() for x in args.models.split(",") if x.strip()]:
        t0 = time.time()
        r, inf = train_one(m, arrays, meta, args.epochs, args.seed, args.pretrained, args.batch, args.lr)
        rows += r; info[m] = {**inf, "seconds": round(time.time() - t0), "dataset": meta.dataset.iloc[0]}
        print(f"[{m}] done in {time.time() - t0:.0f}s")
    new = pd.DataFrame(rows)
    if len(old):
        old = old[~(old.model.isin(new.model) & (old.dataset == meta.dataset.iloc[0]))]
        old = old[old.dataset == meta.dataset.iloc[0]]          # never mix synthetic and real rows
    res = pd.concat([old, new], ignore_index=True)
    res.round(4).to_csv(path, index=False)
    info_path.write_text(json.dumps(info, indent=1, default=float), encoding="utf-8")
    print(res.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
