"""
m4_fusion.py - multimodal fusion models (Milestones 8-9).

Branches (all fusion models share them):
    telemetry : GRU(64) over the 60-min window; per step = 8 standardised values, 8 forecast-residual
                z-scores (/5) and 8 missing-value flags  -> h_tel (64-d)
    optical   : frozen Milestone-6 CNN embedding (512-d) -> Linear -> h_opt (64-d)
    SAR       : frozen Milestone-6 CNN embedding (512-d) -> Linear -> h_sar (64-d)

Models (mode):
    tel_only / opt_only / sar_only   one branch only (unimodal baselines)
    early                            concatenate [h_tel, h_opt, h_sar] (missing -> zeros) -> heads
    equal                            mask-aware average of the available branches (all alpha equal)
    gated   (MAIN METHOD)            reliability gate:
                                       missing branch -> learned MISSING TOKEN e_m
                                       alpha = masked softmax(gate([h_tel, h_opt, h_sar, m_tel, m_opt, m_sar]))
                                       z = sum_m alpha_m h_m      (alphas of missing inputs are exactly 0)
                                     one gate per task (health, flood): telemetry should dominate the
                                     health decision and images the flood decision, so the two tasks
                                     must be free to trust the inputs differently.
Ablations: gated with zero-fill instead of missing tokens and an unmasked softmax (mask_token=False);
           gated whose gate cannot see telemetry (tel_in_gate=False); trained without modality dropout;
           without forecast-residual inputs; without the multi-task loss (one net per task).
Late fusion (average of unimodal probabilities) is computed in training/m4_train_fusion.py.

Outputs per pass: health logit, flood (EO) logit, and the alpha weights of both gates.
"""

from __future__ import annotations

import torch
import torch.nn as nn

D = 64
MODES = ("tel_only", "opt_only", "sar_only", "early", "equal", "gated")


class FusionNet(nn.Module):
    def __init__(self, mode: str = "gated", d: int = D, tel_in: int = 24, emb_in: int = 512,
                 mask_token: bool = True, tel_in_gate: bool = True):
        super().__init__()
        assert mode in MODES, mode
        self.mode, self.d, self.mask_token, self.tel_in_gate = mode, d, mask_token, tel_in_gate
        self.tel_rnn = nn.GRU(tel_in, d, batch_first=True)
        self.tel_proj = nn.Sequential(nn.Linear(d, d), nn.ReLU())
        self.opt_proj = nn.Sequential(nn.Linear(emb_in, d), nn.ReLU(), nn.Dropout(0.2))
        self.sar_proj = nn.Sequential(nn.Linear(emb_in, d), nn.ReLU(), nn.Dropout(0.2))
        self.missing = nn.Parameter(torch.zeros(3, d))                  # learned missing tokens
        gate_in = 3 * d + 3
        self.gate = nn.ModuleDict({t: nn.Sequential(nn.Linear(gate_in, 32), nn.ReLU(), nn.Linear(32, 3))
                                   for t in ("health", "eo")})
        head_in = 3 * d if mode == "early" else d
        self.heads = nn.ModuleDict({t: nn.Sequential(nn.Linear(head_in, 32), nn.ReLU(), nn.Linear(32, 1))
                                    for t in ("health", "eo")})

    def branches(self, tel, opt, sar):
        _, h = self.tel_rnn(tel)
        return [self.tel_proj(h[-1]), self.opt_proj(opt), self.sar_proj(sar)]

    def forward(self, tel, opt, sar, m):
        """tel [B,T,24], opt/sar [B,512], m [B,3] float (1 = present). Returns dict."""
        hs = self.branches(tel, opt, sar)
        mm = m.unsqueeze(-1)
        if self.mode in ("tel_only", "opt_only", "sar_only"):
            k = ("tel_only", "opt_only", "sar_only").index(self.mode)
            z = hs[k]
            out = {t: self.heads[t](z).squeeze(1) for t in ("health", "eo")}
            a = torch.zeros(len(m), 3, device=m.device); a[:, k] = 1
            return {**out, "alpha_health": a, "alpha_eo": a}
        H = torch.stack(hs, dim=1)                                      # [B,3,d]
        if self.mode == "early":
            Hm = H * mm                                                 # missing -> zeros
            z = Hm.flatten(1)
            out = {t: self.heads[t](z).squeeze(1) for t in ("health", "eo")}
            a = m / m.sum(1, keepdim=True)
            return {**out, "alpha_health": a, "alpha_eo": a}
        if self.mask_token:
            H = H * mm + self.missing.unsqueeze(0) * (1 - mm)
        else:
            H = H * mm
        out = {}
        for t in ("health", "eo"):
            if self.mode == "equal":
                a = m / m.sum(1, keepdim=True)
            else:
                g_in = H.clone()
                if not self.tel_in_gate:
                    g_in = torch.cat([torch.zeros_like(g_in[:, :1]), g_in[:, 1:]], dim=1)
                logits = self.gate[t](torch.cat([g_in.flatten(1), m], dim=1))
                if self.mask_token:
                    logits = logits.masked_fill(m < 0.5, float("-inf"))
                a = torch.softmax(logits, dim=1)
            z = (a.unsqueeze(-1) * H).sum(1)
            out[t] = self.heads[t](z).squeeze(1)
            out[f"alpha_{t}"] = a
        return out


def build(name: str) -> FusionNet:
    """Named model variants used in the experiments."""
    table = {
        "tel_only": dict(mode="tel_only"), "opt_only": dict(mode="opt_only"), "sar_only": dict(mode="sar_only"),
        "early_fusion": dict(mode="early"), "equal_weights": dict(mode="equal"),
        "gated_fusion": dict(mode="gated"),
        "gated_zero_fill": dict(mode="gated", mask_token=False),
        "gated_no_tel_in_gate": dict(mode="gated", tel_in_gate=False),
        "gated_no_dropout": dict(mode="gated"),          # same net, trained without modality dropout
        "gated_no_residuals": dict(mode="gated"),        # same net, forecast-residual inputs set to 0
        "gated_no_multitask": dict(mode="gated"),        # two nets, each trained on ONE task's loss
    }
    return FusionNet(**table[name])
