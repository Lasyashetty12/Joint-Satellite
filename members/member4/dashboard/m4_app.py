"""
m4_app.py - Member 4 Streamlit dashboard: satellite health + flood detection with missing inputs.

Run from the repository root:
    streamlit run members/member4/dashboard/m4_app.py

Pages
  1. Mission overview     what the system does, architecture, headline results, data status
  2. Live demo            handbook section 25: replay a stored test pass as a live telemetry stream
                          (actual vs forecast + residual band + anomaly score), show the optical/SAR
                          patches and flood label, run the integrated model, show alpha weights,
                          toggle missing inputs / cloud / telemetry channel loss, compare simple vs
                          proposed fusion on the same pass
  3. Model accuracy       every measured result: forecasting (incl. GRU and LSTM), health detection
                          (simulated + real OPSSAT-AD), EO baselines, fusion, robustness, ablations
  4. Final conclusions    reports/FINAL_CONCLUSIONS.md (generated from the measured results)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m4_dashboard_data as D  # noqa: E402
import m4_ts_common as C  # noqa: E402

st.set_page_config(page_title="Satellite Health & Flood Monitor", page_icon="🛰️", layout="wide")

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
GREY, RED = "#8a8984", "#e34948"
MOD_COL = {"tel": ORANGE, "opt": BLUE, "sar": AQUA}
MOD_NAME = {"tel": "telemetry", "opt": "optical", "sar": "SAR"}
NICE = {"tel_only": "telemetry only", "opt_only": "optical only", "sar_only": "SAR only", "early_fusion": "early fusion",
        "late_fusion": "late fusion", "equal_weights": "equal weights", "gated_fusion": "gated fusion (ours)",
        "gated_zero_fill": "− missing token", "gated_no_tel_in_gate": "− telemetry in gate",
        "gated_no_dropout": "− modality dropout", "gated_no_residuals": "− forecast residuals",
        "gated_no_multitask": "− multi-task loss"}
SCEN_LABEL = {"all_present": "all inputs", "missing_optical": "− optical", "missing_sar": "− SAR",
              "missing_telemetry": "− telemetry", "only_optical": "optical only", "only_sar": "SAR only",
              "only_telemetry": "telemetry only", "cloud_25": "+25% cloud", "cloud_50": "+50% cloud",
              "cloud_75": "+75% cloud", "cloud_100": "+100% cloud"}
LAYOUT = dict(margin=dict(l=10, r=10, t=80, b=10), hovermode="x unified",
              legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, traceorder="normal"))


@st.cache_data(show_spinner=False)
def get_results():
    return D.results()


@st.cache_resource(show_spinner="Loading models and stored passes ...")
def get_live():
    return D.load_live()


RES = get_results()
SYN = D.eo_is_synthetic(RES)

# ------------------------------------------------------------------ sidebar
st.sidebar.title("🛰️ Satellite Health & Flood Monitor")
st.sidebar.caption("Member 4 · forecasting-based health monitoring + missing-modality-aware fusion")
page = st.sidebar.radio("Page", ["Mission overview", "Live demo", "Model accuracy", "Final conclusions"])
if SYN:
    st.sidebar.error("Image data: SYNTHETIC stand-in (pipeline check). Image and fusion numbers are not results. "
                     "Run the pipeline on Sen1Floods11 to replace them.")
else:
    st.sidebar.success("Image data: Sen1Floods11 (real Sentinel-1/2)")
st.sidebar.caption("Telemetry: simulated generator v1 + real OPSSAT-AD benchmark. "
                   "Telemetry–image pairing is a simulation assumption (virtual passes).")


def synth_banner():
    if SYN:
        st.warning("⚠️ Earth-observation inputs are the **synthetic stand-in**: image and fusion numbers on this page "
                   "only prove the pipeline works. Telemetry results are measured.", icon="⚠️")


def metric_or_dash(v, fmt="{:.3f}"):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt.format(v)


# ================================================================== 1. overview
def page_overview():
    st.title("Satellite health & flood detection that keeps working when inputs go missing")
    st.markdown(
        "A satellite sends two kinds of data: **telemetry** (its own health: battery, temperatures, voltage, "
        "pointing) and **Earth images** (optical Sentinel-2 and radar Sentinel-1). This system watches both. "
        "It **forecasts** every telemetry channel and raises an alarm when reality departs from the forecast, "
        "and it detects **floods** in the images. A **reliability gate** decides, pass by pass, how much to trust "
        "each input, so the system keeps working when clouds hide the ground, the camera is damaged, or data is lost.")
    s = RES["summary"]
    c = st.columns(4)
    c[0].metric("Best forecaster (5 min ahead)", str(s.get("forecast_best_model", "–")).upper(),
                f"skill {metric_or_dash(s.get('forecast_best_skill_h10'))} vs naive")
    c[1].metric("Health detection, simulated test", metric_or_dash(s.get("health_best_test_F1")),
                f"event-F1 · {s.get('health_best_detector', '')}")
    c[2].metric("Health detection, REAL OPSSAT-AD", metric_or_dash(s.get("opssat_best_F1")),
                f"F1 · MCC {metric_or_dash(s.get('opssat_best_MCC'))}")
    if SYN:
        c[3].metric("Flood F1 kept when inputs missing", "pending", "needs the real Sen1Floods11 run", delta_color="off")
    else:
        c[3].metric("Flood F1 kept when inputs missing", metric_or_dash(s.get("fusion_flood_F1_kept"), "{:.0%}"),
                    "gated fusion, test, 5 missing-input scenarios", delta_color="off")
    st.subheader("How it works")
    st.graphviz_chart("""
    digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#f4f3ef", color="#c9c8c3", fontname="Helvetica", fontsize=11];
      tel [label="Telemetry window\\n60 min × 8 channels"]; fc [label="Forecaster\\n(GRU / LSTM / XGBoost / ARIMAX)"];
      res [label="Forecast residuals\\n= surprise per channel"]; te [label="Telemetry encoder\\nGRU"];
      opt [label="Optical patch\\nSentinel-2"]; sar [label="SAR patch\\nSentinel-1"];
      oe [label="CNN encoder\\nResNet-18"]; se [label="CNN encoder\\nResNet-18"];
      gate [label="Reliability gate\\nmissing tokens + α weights", fillcolor="#dfeafb", color="#2a78d6"];
      h [label="Health score\\n(satellite fault?)"]; f [label="Flood score\\n(Earth anomaly?)"];
      tel -> fc -> res -> te; tel -> te; opt -> oe; sar -> se; te -> gate; oe -> gate; se -> gate; gate -> h; gate -> f; }""")
    st.subheader("Research questions")
    rq = {"Primary": "Does missing-modality-aware fusion keep health and flood detection reliable when inputs are missing?",
          "RQ2": "Does forecast-residual detection beat static limit checks and Isolation Forest?",
          "RQ3": "Can missing telemetry channels be reconstructed so detection degrades gracefully?",
          "RQ4": "Do the learned trust weights shift sensibly when an input is degraded or removed?"}
    key = {"Primary": "primary", "RQ2": "RQ2", "RQ3": "RQ3", "RQ4": "RQ4"}
    for k, q in rq.items():
        v = s.get(key[k])
        badge = {True: "✅ supported", False: "❌ not supported", "partial": "🟡 partially supported",
                 None: "⏳ withheld (needs real images)"}.get(v, "–")
        st.markdown(f"**{k}** — {q}  \n&nbsp;&nbsp;&nbsp;{badge}")


# ================================================================== 2. live demo
def stream_figure(d: pd.DataFrame, upto: int, channels: list[str]) -> go.Figure:
    n = len(channels) + 1
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[1] * len(channels) + [1.1])
    x = d.timestamp
    pass_start = d.timestamp[d.in_pass.values.argmax()]
    shown = d.iloc[:upto]
    for r, c in enumerate(channels, start=1):
        fig.add_trace(go.Scatter(x=x[:upto], y=shown[f"{c}_hi"], line=dict(width=0), showlegend=False, hoverinfo="skip"), r, 1)
        fig.add_trace(go.Scatter(x=x[:upto], y=shown[f"{c}_lo"], fill="tonexty", fillcolor="rgba(235,104,52,0.15)",
                                 line=dict(width=0), name="forecast ± 2 sd", showlegend=r == 1, hoverinfo="skip"), r, 1)
        fig.add_trace(go.Scatter(x=x[:upto], y=shown[f"{c}_fc"], line=dict(color=ORANGE, width=1.2, dash="dot"),
                                 name="forecast", showlegend=r == 1), r, 1)
        fig.add_trace(go.Scatter(x=x[:upto], y=shown[c], line=dict(color=BLUE, width=1.6), name="actual",
                                 showlegend=r == 1), r, 1)
        fig.update_yaxes(title_text=f"{c} [{C.UNITS[c]}]", title_font_size=10, row=r, col=1)
    if "score" in d:
        fig.add_trace(go.Scatter(x=x[:upto], y=shown.score, line=dict(color=MAGENTA, width=1.4), name="anomaly score"), n, 1)
        fig.add_hline(y=float(d.threshold.iloc[0]), line=dict(color=RED, dash="dash", width=1), row=n, col=1,
                      annotation_text="alarm threshold", annotation_font_size=10)
    fig.update_yaxes(title_text="score", title_font_size=10, row=n, col=1)
    for r in range(1, n + 1):
        fig.add_vrect(x0=pass_start, x1=x.iloc[-1], fillcolor="#2a78d6", opacity=0.06, line_width=0, row=r, col=1)
    lab = d.label.values.astype(bool)
    if lab.any():
        idx = np.flatnonzero(lab)
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        for run in runs:
            for r in range(1, n + 1):
                fig.add_vrect(x0=x.iloc[run[0]], x1=x.iloc[run[-1]], fillcolor=RED, opacity=0.10, line_width=0, row=r, col=1)
    fig.update_xaxes(range=[x.iloc[0], x.iloc[-1]])
    fig.update_layout(height=150 * n + 90, hovermode="x unified", margin=dict(l=10, r=10, t=80, b=10),
                      legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0),
                      title=dict(text="Telemetry stream · blue band = the satellite pass · red = labelled anomaly",
                                 font_size=13, y=0.995, yanchor="top"))
    return fig


def alpha_chart(results: dict, baseline: dict | None) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Flood decision gate", "Health decision gate"), shared_yaxes=True)
    for col, gate in ((1, "alpha_eo"), (2, "alpha_health")):
        rows = []
        if baseline is not None:
            rows.append(("proposed · all inputs", baseline["gated_fusion"][gate]))
        rows.append(("proposed · this scenario", results["gated_fusion"][gate]))
        for k, mod in enumerate(("tel", "opt", "sar")):
            fig.add_trace(go.Bar(y=[r[0] for r in rows], x=[r[1][k] for r in rows], orientation="h",
                                 name=f"α {MOD_NAME[mod]}", marker_color=MOD_COL[mod], showlegend=col == 1,
                                 text=[f"{r[1][k]:.2f}" if r[1][k] >= 0.06 else "" for r in rows],
                                 textposition="inside", insidetextanchor="middle",
                                 marker_line=dict(color="white", width=1)), 1, col)
    fig.update_layout(barmode="stack", height=240, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.25, traceorder="normal"),
                      xaxis=dict(range=[0, 1]), xaxis2=dict(range=[0, 1]), yaxis=dict(autorange="reversed"))
    return fig


def page_live():
    st.title("Live demo: one satellite pass, end to end")
    synth_banner()
    miss = D.missing_assets()
    if miss:
        st.error("The live demo needs files that the pipeline creates (they are not stored in Git):\n\n"
                 + "\n".join(f"- `{m}`" for m in miss)
                 + "\n\nRun `python members/member4/m4_run_all.py` once, then reload this page.")
        return
    L = get_live()
    tab = D.test_pass_table(L)
    st.markdown("**Step 1 — choose a stored test pass.** Passes with a payload fault come first: in those, the "
                "telemetry fault and a damaged optical image arrive together.")
    choice = st.selectbox("Test pass", tab.label.tolist(), index=0)
    pid = int(tab[tab.label == choice].pass_id.iloc[0])
    row = tab[tab.pass_id == pid].iloc[0]
    if st.session_state.get("pid") != pid:
        st.session_state.update(pid=pid, cursor=60, playing=False)

    st.markdown("**Step 2 — replay the telemetry as a live stream** (actual vs forecast, residual band, anomaly score).")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
    if c1.button("▶ Play", width="stretch"):
        st.session_state.playing = True
    if c2.button("⏸ Pause", width="stretch"):
        st.session_state.playing = False
    if c3.button("⏭ Jump to end", width="stretch"):
        st.session_state.update(cursor=10 ** 6, playing=False)
    speed = c4.slider("Replay speed (samples per tick; 1 sample = 30 s)", 2, 30, 8)
    channels = st.multiselect("Channels on screen", C.CHANNELS,
                              default=["SoC", "bus_V", "batt_I", "batt_temp", "payload_P"])
    stream = D.telemetry_stream(L, pid)

    @st.fragment(run_every=0.5 if st.session_state.get("playing") else None)
    def live_stream():
        n = len(stream)
        if st.session_state.get("playing"):
            st.session_state.cursor = min(n, st.session_state.cursor + speed)
            if st.session_state.cursor >= n:
                st.session_state.playing = False
                st.rerun()
        cur = min(n, st.session_state.cursor)
        t = stream.timestamp.iloc[cur - 1]
        alarm = bool("score" in stream and np.nan_to_num(stream.score.iloc[cur - 1]) > stream.threshold.iloc[0])
        k1, k2, k3 = st.columns(3)
        k1.metric("Satellite clock (UTC)", t.strftime("%Y-%m-%d %H:%M"))
        k2.metric("Health monitor", "🔴 ALARM" if alarm else "🟢 nominal")
        k3.metric("Replay progress", f"{cur / n:.0%}")
        st.plotly_chart(stream_figure(stream, cur, channels or ["SoC"]), width="stretch",
                        key=f"stream_{pid}")

    live_stream()

    st.markdown("**Step 3 — the Earth images taken during this pass.**")
    import m4_eo_patches as EP
    tg = st.columns([1, 1, 1, 1.2])
    st.session_state.setdefault("cloud", 0.0)
    img = D.optical_image(L, row, st.session_state.get("cloud", 0.0))
    tg[0].image(EP.to_rgb(img, L.stats), caption=f"Optical (Sentinel-2) · {row.opt_degrade}"
                + (f" {row.opt_severity:.2f}" if row.opt_degrade != "none" else ""), width="stretch")
    vv = L.eo["s1"][int(row.patch_id)][0].astype(np.float32)
    vv = np.clip((vv - np.percentile(vv, 2)) / (np.percentile(vv, 98) - np.percentile(vv, 2) + 1e-6), 0, 1)
    tg[1].image((vv * 255).astype(np.uint8), caption="SAR VV (Sentinel-1) · water looks dark", width="stretch")
    mask = L.eo["mask"][int(row.patch_id)]
    rgb = np.stack([np.where(mask == 1, 42, np.where(mask == 0, 242, 190)),
                    np.where(mask == 1, 120, np.where(mask == 0, 230, 190)),
                    np.where(mask == 1, 214, np.where(mask == 0, 201, 190))], -1).astype(np.uint8)
    tg[2].image(rgb, caption="Ground-truth label · blue = water", width="stretch")
    tg[3].markdown(f"**Ground truth for this pass**  \n"
                   f"- Satellite health: {'🔴 fault — ' + row.anomaly_type.replace(';', ', ') if row.health_label else '🟢 nominal'}  \n"
                   f"- Flood in the patch: {'🌊 yes' if row.eo_label else 'no'}  \n"
                   f"- Optical image: {'damaged by the payload fault' if row.coupled_payload_fault else row.opt_degrade}  \n"
                   f"- Patch {int(row.patch_id)} · {L.meta.event.iloc[int(row.patch_id)]}")

    st.markdown("**Steps 4–7 — run the integrated model, then toggle what is available and run again.**")
    t1, t2, t3, t4 = st.columns([1.1, 1, 1.4, 1])
    with t1:
        st.caption("Inputs available")
        use_tel = st.checkbox("Telemetry", True); use_opt = st.checkbox("Optical", True); use_sar = st.checkbox("SAR", True)
    cloud = t2.slider("Extra cloud on optical", 0.0, 1.0, 0.0, 0.25)
    st.session_state.cloud = cloud
    drop = t3.multiselect("Telemetry channels lost", C.CHANNELS, default=[])
    fill = t4.radio("Lost channels are", ["zero-filled", "reconstructed"], index=1)
    if not (use_tel or use_opt or use_sar):
        st.error("At least one input must be available.")
        return
    base = D.run_pass(L, pid)
    out = D.run_pass(L, pid, present=(int(use_tel), int(use_opt), int(use_sar)), cloud=cloud, drop=drop,
                     fill="recon" if fill == "reconstructed" else "zero")
    g, e = out["gated_fusion"], out["early_fusion"]
    st.markdown("**Step 8 — simple fusion vs the proposed model on the same pass.**")
    cc = st.columns(2)
    for col, (name, r) in zip(cc, (("gated_fusion", g), ("early_fusion", e))):
        with col:
            st.markdown(f"##### {D.DEMO_MODELS[name]}")
            m1, m2 = st.columns(2)
            m1.metric("Satellite fault probability", f"{r['p_health']:.2f}",
                      f"{r['p_health'] - base[name]['p_health']:+.2f} vs all inputs")
            m2.metric("Flood probability", f"{r['p_flood']:.2f}", f"{r['p_flood'] - base[name]['p_flood']:+.2f} vs all inputs")
            ok_h = r["health_alarm"] == bool(row.health_label); ok_f = r["flood"] == bool(row.eo_label)
            st.markdown(f"Health: **{'🔴 FAULT' if r['health_alarm'] else '🟢 nominal'}** {'✅' if ok_h else '❌'} · "
                        f"Flood: **{'🌊 FLOOD' if r['flood'] else 'no flood'}** {'✅' if ok_f else '❌'}  \n"
                        f"<span style='color:#8a8984;font-size:0.85em'>✅/❌ = matches ground truth · thresholds from validation</span>",
                        unsafe_allow_html=True)
    st.markdown("**Trust weights (α) of the proposed model** — how much each input counts; missing inputs get exactly 0.")
    st.plotly_chart(alpha_chart(out, base), width="stretch")
    a0, a1 = base["gated_fusion"]["alpha_eo"], g["alpha_eo"]
    shift = ", ".join(f"α {MOD_NAME[m]} {a0[k]:.2f} → {a1[k]:.2f}" for k, m in enumerate(("tel", "opt", "sar")) if abs(a1[k] - a0[k]) > 0.01)
    st.caption("Flood-gate shift vs all inputs: " + (shift or "no change"))
    if drop:
        st.markdown("**Lost telemetry channels during the pass**" + (" — reconstructed from the other channels" if out["reconstructed"] is not None else " — zero-filled"))
        win = D.telemetry_stream(L, pid)
        win = win[win.in_pass].reset_index(drop=True)
        fig = make_subplots(rows=1, cols=len(drop), subplot_titles=drop)
        for k, ch in enumerate(drop, start=1):
            fig.add_trace(go.Scatter(x=win.timestamp, y=win[ch], name="actual (hidden from the model)", line=dict(color=BLUE),
                                     showlegend=k == 1), 1, k)
            if out["reconstructed"] is not None:
                fig.add_trace(go.Scatter(x=win.timestamp, y=out["reconstructed"][:, k - 1], name="reconstructed",
                                         line=dict(color=ORANGE, dash="dot"), showlegend=k == 1), 1, k)
        fig.update_layout(height=260, **LAYOUT)
        st.plotly_chart(fig, width="stretch")
    status = []
    status.append("🔴 **Satellite fault detected**" if g["health_alarm"] else "🟢 Satellite healthy")
    status.append("🌊 **Flood detected**" if g["flood"] else "No flood detected")
    if g["alpha_eo"][1] < 0.2 and use_opt:
        status.append("optical image distrusted")
    st.info("**Final status (proposed model):** " + " · ".join(status))


# ================================================================== 3. accuracy
def page_accuracy():
    st.title("Model accuracy — every number here was measured")
    tabs = st.tabs(["Forecasting (incl. GRU & LSTM)", "Health anomaly detection", "Earth observation", "Fusion & robustness",
                    "Channel reconstruction"])
    with tabs[0]:
        fr = RES["forecast"]
        if fr is None:
            st.info("Run mlts/m4_forecast_eval.py"); return
        models = [m for m in D.FORECAST_MODELS if m in set(fr.model)]
        c1, c2, c3 = st.columns(3)
        split = c1.radio("Split", ["val", "test"], horizontal=True)
        h = c2.radio("Horizon", [1, 5, 10], index=2, horizontal=True,
                     format_func=lambda v: {1: "30 s ahead", 5: "2.5 min ahead", 10: "5 min ahead"}[v])
        ch = c3.selectbox("Channel for the horizon chart", C.CHANNELS, index=C.CHANNELS.index("SoC"))
        d = fr[(fr.split == split) & (fr.horizon == h)]
        base = d[d.model == "naive"].set_index("channel").MAE
        fig = go.Figure()
        for m in models:
            v = d[d.model == m].set_index("channel").reindex(C.CHANNELS)
            fig.add_bar(x=C.CHANNELS, y=1 - v.MAE / base.reindex(C.CHANNELS), name=m.upper(), marker_color=D.MODEL_COLORS[m],
                        customdata=v.MAE, hovertemplate="%{x}<br>skill %{y:.2f}<br>MAE %{customdata:.4f}<extra>" + m.upper() + "</extra>")
        fig.update_layout(barmode="group", height=380, yaxis_title="skill vs naive (1 − MAE/MAE naive) · higher is better",
                          yaxis_range=[-1, 1], title=f"Forecast skill per channel ({split}, h = {h}) · naive = 0 by definition", **{k: v for k, v in LAYOUT.items() if k != 'hovermode'})
        st.plotly_chart(fig, width="stretch")
        g2 = go.Figure()
        for m in models:
            v = fr[(fr.split == split) & (fr.model == m) & (fr.channel == ch)].sort_values("horizon")
            g2.add_scatter(x=v.horizon * 0.5, y=v.MAE, mode="lines+markers", name=m.upper(), line=dict(color=D.MODEL_COLORS[m], width=2),
                           marker=dict(size=8))
        g2.update_layout(height=320, xaxis_title="minutes ahead", yaxis_title=f"MAE [{C.UNITS[ch]}]",
                         title=f"{ch}: error grows with the horizon", **LAYOUT)
        st.plotly_chart(g2, width="stretch")
        info = RES["forecast_info"]
        cells = [c for c in ("gru", "lstm") if info.get(c, {}).get("history")]
        if cells:
            g3 = go.Figure()
            for c in cells:
                hst = pd.DataFrame(info[c]["history"])
                g3.add_scatter(x=hst.epoch, y=hst.val_loss, name=f"{c.upper()} validation loss", mode="lines+markers",
                               line=dict(color=D.MODEL_COLORS[c], width=2))
            g3.update_layout(height=280, xaxis_title="epoch", yaxis_title="MSE (standardised)", title="Deep forecasters: GRU vs LSTM training", **LAYOUT)
            st.plotly_chart(g3, width="stretch")
        with st.expander("Table: MAE per channel and model"):
            st.dataframe(d.pivot_table(index="channel", columns="model", values="MAE", observed=True)[models].style.format("{:.4f}")
                         .highlight_min(axis=1, color="#dfeafb"), width="stretch")
    with tabs[1]:
        hm = RES["health"]
        if hm is not None:
            sim = hm[(hm.dataset == "simulated_v1")]
            order = ["limit_check", "isolation_forest", "residual_naive", "residual_snaive", "residual_arimax",
                     "residual_xgboost", "residual_gru", "residual_lstm", "residual_ensemble"]
            order = [m for m in order if m in set(sim.model)]
            fig = go.Figure()
            for split, col in (("val", BLUE), ("test", ORANGE)):
                v = sim[sim.split == split].set_index("model").reindex(order)
                fig.add_bar(x=[m.replace("residual_", "residual · ") for m in order], y=v.event_F1, name=split, marker_color=col,
                            customdata=np.stack([v.false_alarms_per_day, v.mean_delay_min], -1),
                            hovertemplate="%{x}<br>event-F1 %{y:.3f}<br>false alarms/day %{customdata[0]:.2f}<br>delay %{customdata[1]:.0f} min<extra>" + split + "</extra>")
            fig.update_layout(barmode="group", height=380, yaxis_range=[0, 1.05], yaxis_title="event-F1",
                              title="Simulated telemetry: forecast-residual detectors vs limit checks and Isolation Forest",
                              **{k: v for k, v in LAYOUT.items() if k != 'hovermode'})
            st.plotly_chart(fig, width="stretch")
            st.dataframe(sim[sim.split == "test"][["model", "event_precision", "event_recall", "event_F1", "AUPRC", "mean_delay_min",
                                                    "false_alarms_per_day"]].set_index("model").style.format("{:.3f}"), width="stretch")
        op = RES["opssat"]
        if op is not None:
            st.subheader("Real satellite telemetry: OPSSAT-AD (ESA OPS-SAT), 529 test segments")
            fig = go.Figure()
            for k, met in enumerate(("F1", "MCC", "AUC_ROC")):
                fig.add_bar(x=op.model, y=op[met], name=met.replace("_", "-"), marker_color=[BLUE, ORANGE, AQUA][k])
            fig.update_layout(barmode="group", height=340, yaxis_range=[0, 1.05], **{k: v for k, v in LAYOUT.items() if k != 'hovermode'})
            st.plotly_chart(fig, width="stretch")
        pt = RES["per_type"]
        if pt is not None:
            pt = pt.set_index("anomaly_type")
            fig = go.Figure(go.Heatmap(z=pt.values, x=[c.replace("residual_", "res·") for c in pt.columns], y=pt.index,
                                       colorscale=[[0, "#eef4fc"], [0.5, "#2a78d6"], [1, "#0d3a73"]], zmin=0, zmax=1,
                                       text=pt.values.round(2), texttemplate="%{text}", colorbar=dict(title="recall")))
            fig.update_layout(height=380, title="Which detector catches which fault (all 59 events)", margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, width="stretch")
    with tabs[2]:
        synth_banner()
        eo = RES["eo"]
        if eo is None:
            st.info("Run training/m4_train_eo.py")
        else:
            fig = go.Figure()
            for k, m in enumerate(("optical_only", "sar_only", "optical_sar_cnn")):
                v = eo[eo.model == m].set_index("split").reindex(["val", "test", "bolivia"])
                fig.add_bar(x=["validation", "test", "Bolivia (unseen event)"], y=v.F1, name=m.replace("_", " "), marker_color=[BLUE, ORANGE, AQUA][k])
            fig.update_layout(barmode="group", height=360, yaxis_range=[0, 1.05], yaxis_title="flood F1 (patch level)",
                              **{k: v for k, v in LAYOUT.items() if k != 'hovermode'})
            st.plotly_chart(fig, width="stretch")
            st.dataframe(eo.drop(columns=["member"]), width="stretch")
    with tabs[3]:
        synth_banner()
        fu = RES["fusion"]
        if fu is None:
            st.info("Run training/m4_train_fusion.py")
        else:
            task = st.radio("Task", ["eo", "health"], format_func=lambda t: "Flood detection" if t == "eo" else "Satellite-health detection", horizontal=True)
            a = fu[(fu.scenario == "all_present") & (fu.task == task)].groupby("model").F1.agg(["mean", "std"])
            order = [m for m in ["tel_only", "opt_only", "sar_only", "early_fusion", "late_fusion", "equal_weights", "gated_fusion"] if m in a.index]
            fig = go.Figure(go.Bar(y=[NICE[m] for m in order], x=a.loc[order, "mean"], orientation="h",
                                   error_x=dict(array=a.loc[order, "std"].fillna(0)),
                                   marker_color=[BLUE if m == "gated_fusion" else "#c9c8c3" for m in order],
                                   text=a.loc[order, "mean"].round(3), textposition="outside"))
            fig.update_layout(height=330, xaxis_range=[0, 1.1], xaxis_title="test F1 (mean ± std, 3 seeds)",
                              title="All inputs present", margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, width="stretch")
            scen = ["all_present", "missing_optical", "missing_sar", "missing_telemetry", "only_optical", "only_sar", "only_telemetry",
                    "cloud_25", "cloud_50", "cloud_75", "cloud_100"]
            fig = go.Figure()
            for m, col in (("early_fusion", ORANGE), ("late_fusion", AQUA), ("gated_fusion", BLUE)):
                v = fu[(fu.model == m) & (fu.task == task)].groupby("scenario").F1.agg(["mean", "std"]).reindex(scen)
                fig.add_bar(x=[SCEN_LABEL[x] for x in scen], y=v["mean"], error_y=dict(array=v["std"].fillna(0)),
                            name=NICE[m], marker_color=col)
            fig.update_layout(barmode="group", height=380, yaxis_range=[0, 1.05], yaxis_title="test F1",
                              title="Robustness: inputs missing or clouded", **{k: v for k, v in LAYOUT.items() if k != 'hovermode'})
            st.plotly_chart(fig, width="stretch")
            al = RES["alpha"]
            if al is not None:
                e = al[(al.model == "gated_fusion") & (al.gate == ("eo" if task == "eo" else "health"))]
                rows = [("all_present", "clean", "clear"), ("all_present", "cloudy", "cloudy"), ("all_present", "coupled_payload_fault", "payload fault"),
                        ("cloud_100", "all", "+100% cloud"), ("missing_optical", "all", "optical missing"), ("missing_sar", "all", "SAR missing"),
                        ("missing_telemetry", "all", "telemetry missing")]
                vals = [(lab, e[(e.scenario == s) & (e.subset == sub)][["alpha_tel", "alpha_opt", "alpha_sar"]].mean().values) for s, sub, lab in rows]
                fig = go.Figure()
                for k, mod in enumerate(("tel", "opt", "sar")):
                    fig.add_bar(x=[v[0] for v in vals], y=[v[1][k] for v in vals], name=f"α {MOD_NAME[mod]}", marker_color=MOD_COL[mod],
                                marker_line=dict(color="white", width=1))
                fig.update_layout(barmode="stack", height=340, yaxis_range=[0, 1.02], yaxis_title="mean trust weight",
                                  title="RQ4: trust weights of the proposed model by situation", **{k: v for k, v in LAYOUT.items() if k != 'hovermode'})
                st.plotly_chart(fig, width="stretch")
            ab = [m for m in ["gated_fusion", "gated_zero_fill", "gated_no_tel_in_gate", "gated_no_dropout", "gated_no_residuals", "gated_no_multitask"] if m in set(fu.model)]
            miss = ["missing_optical", "missing_sar", "missing_telemetry", "only_optical", "only_sar"]
            tbl = pd.DataFrame([{"variant": NICE[m],
                                 "F1 all inputs": fu[(fu.model == m) & (fu.task == task) & (fu.scenario == "all_present")].F1.mean(),
                                 "F1 inputs missing (avg)": fu[(fu.model == m) & (fu.task == task) & fu.scenario.isin(miss)].F1.mean()} for m in ab]).set_index("variant")
            st.markdown("**Ablations**"); st.dataframe(tbl.style.format("{:.3f}"), width="stretch")
    with tabs[4]:
        rc = RES["recon"]
        if rc is not None:
            p = rc.pivot(index="channel", columns="method", values="MAE").reindex(C.CHANNELS)
            fig = go.Figure()
            for m, col, lab in (("mean_fill", GREY, "train mean"), ("ffill", ORANGE, "last value"), ("cross_channel", BLUE, "cross-channel (ours)")):
                fig.add_bar(x=p.index, y=p[m] / p["mean_fill"], name=lab, marker_color=col, customdata=p[m],
                            hovertemplate="%{x}<br>relative %{y:.3f}<br>MAE %{customdata:.4f}<extra>" + lab + "</extra>")
            fig.update_layout(barmode="group", height=360, yaxis_type="log", yaxis_title="MAE relative to train-mean fill (log)",
                              title="RQ3: reconstructing a lost telemetry channel (test, 60-min gaps)", **{k: v for k, v in LAYOUT.items() if k != 'hovermode'})
            st.plotly_chart(fig, width="stretch")
        fu = RES["fusion"]
        if fu is not None:
            hg = fu[(fu.model == "gated_fusion") & (fu.task == "health")]
            fig = go.Figure()
            for fill, col, lab in (("zero", ORANGE, "zero-filled"), ("recon", BLUE, "reconstructed")):
                v = [hg[hg.scenario == f"tel_drop{k}_{fill}"].F1.mean() for k in (1, 2, 4)]
                fig.add_scatter(x=[1, 2, 4], y=v, name=lab, mode="lines+markers", line=dict(color=col, width=2), marker=dict(size=8))
            fig.add_hline(y=hg[hg.scenario == "all_present"].F1.mean(), line=dict(color=GREY, dash="dot"), annotation_text="no channel lost")
            fig.update_layout(height=320, xaxis_title="telemetry channels lost per pass", yaxis_title="health F1 (gated fusion, test)", **LAYOUT)
            st.plotly_chart(fig, width="stretch")


# ================================================================== 4. conclusions
def page_conclusions():
    if RES["conclusions"] is None:
        st.info("Run `python members/member4/evaluation/m4_final_report.py` to generate the conclusions.")
        return
    st.markdown(RES["conclusions"])


{"Mission overview": page_overview, "Live demo": page_live, "Model accuracy": page_accuracy,
 "Final conclusions": page_conclusions}[page]()
