"""
m4_theme.py - visual identity of the Member 4 dashboard (ground-station console, light, projector-friendly).

Tokens
    paper  #F2F4F6   app background (cool, not cream)
    panel  #FFFFFF   raised surfaces
    ink    #172231   text and the heavy top rule of readouts
    slate  #5A6573   secondary text
    rule   #D3D9E0   hairlines, grid
    accent #1F5FB8   Sentinel blue: links, primary buttons, focus
    modality colours (validated categorical palette): telemetry #eb6834, optical #2a78d6, SAR #1baf7a
    status: nominal #2E7D4F (circle), fault #C8322F (triangle) - always shape + word, never colour alone

Type: Archivo only. Headings use its wide cut (font-stretch 115 %), numbers are tabular.
One bold element: the trust strip (alpha weights as a segmented bar). Everything else stays quiet.

The Streamlit theme itself (fonts, base colours) is set in .streamlit/config.toml at the repository root;
the CSS below refines it and keeps the look acceptable even if that file is missing.
"""

from __future__ import annotations

import html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

PAPER, PANEL, INK, SLATE, RULE, ACCENT = "#F2F4F6", "#FFFFFF", "#172231", "#5A6573", "#D3D9E0", "#1F5FB8"
TEL, OPT, SAR = "#eb6834", "#2a78d6", "#1baf7a"
OK, BAD, WARN = "#2E7D4F", "#C8322F", "#9A6700"
FONT = "Archivo, 'Helvetica Neue', Arial, sans-serif"
MOD = {"tel": ("telemetry", TEL), "opt": ("optical", OPT), "sar": ("SAR", SAR)}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,300..800&display=swap');
:root {{ --ink:{INK}; --slate:{SLATE}; --rule:{RULE}; --paper:{PAPER}; --panel:{PANEL}; --accent:{ACCENT};
         --ok:{OK}; --bad:{BAD}; --warn:{WARN}; }}
.stApp {{ background:var(--paper); color:var(--ink); }}
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 {{ font-family:{FONT}; }}
.block-container {{ padding-top:2.2rem; max-width:1320px; }}

.stApp h1 {{ font-stretch:115%; font-weight:650; font-size:2.3rem; line-height:1.08;
            letter-spacing:-0.015em; max-width:26ch; margin-bottom:.4rem; }}
.stApp h2, .stApp h3 {{ font-stretch:112%; font-weight:620; letter-spacing:-0.01em; }}
.stApp h3 {{ font-size:1.35rem; margin-top:1.6rem; }}
.stApp h5 {{ font-stretch:110%; font-weight:620; font-size:1.05rem; }}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{ max-width:74ch; line-height:1.55; }}
.m4-lede {{ font-size:1.12rem; line-height:1.55; color:var(--ink); max-width:68ch; margin:.2rem 0 1.6rem; }}
.m4-note {{ color:var(--slate); font-size:.88rem; max-width:74ch; }}
[data-testid="stMetricValue"], .m4-num {{ font-variant-numeric:tabular-nums; font-stretch:110%; }}
:focus-visible {{ outline:2px solid var(--accent) !important; outline-offset:2px; }}

/* sidebar: plain, ruled */
section[data-testid="stSidebar"] {{ background:var(--panel); border-right:1px solid var(--rule); }}
.m4-brand {{ font-stretch:118%; font-weight:680; font-size:1.18rem; line-height:1.15; color:var(--ink); margin:0 0 .3rem; }}
.m4-brand-sub {{ color:var(--slate); font-size:.86rem; line-height:1.4; margin-bottom:1rem; }}
.m4-data {{ border-top:1px solid var(--rule); padding-top:.8rem; margin-top:1rem; font-size:.86rem; color:var(--slate); line-height:1.45; }}
.m4-data b {{ color:var(--ink); font-weight:600; }}

/* readout strip: instrument panel row, heavy top rule, hairline dividers */
.m4-readout {{ display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); border-top:2px solid var(--ink);
               border-bottom:1px solid var(--rule); margin:.4rem 0 1.8rem; }}
.m4-readout > div {{ padding:.9rem 1.1rem 1rem 0; }}
.m4-readout > div + div {{ border-left:1px solid var(--rule); padding-left:1.1rem; }}
.m4-readout .k {{ font-size:.9rem; color:var(--slate); line-height:1.3; min-height:2.4em; }}
.m4-readout .v {{ font-size:2.05rem; font-weight:620; font-stretch:112%; font-variant-numeric:tabular-nums;
                  line-height:1.1; margin:.25rem 0 .2rem; color:var(--ink); }}
.m4-readout .v.pending {{ color:var(--slate); font-weight:500; }}
.m4-readout .n {{ font-size:.82rem; color:var(--slate); line-height:1.35; }}
@media (max-width: 860px) {{
  .m4-readout {{ grid-template-columns:repeat(2, minmax(0,1fr)); }}
  .m4-readout > div:nth-child(3) {{ border-left:none; padding-left:0; }}
  .m4-readout > div:nth-child(n+3) {{ border-top:1px solid var(--rule); }}
  .stApp h1 {{ font-size:1.8rem; }}
}}

/* status pills: shape + word */
.m4-pill {{ display:inline-flex; align-items:center; gap:.4em; padding:.12em .6em; border-radius:999px;
            font-size:.88rem; font-weight:600; border:1px solid currentColor; white-space:nowrap; background:var(--panel); }}
.m4-pill.ok {{ color:var(--ok); }} .m4-pill.bad {{ color:var(--bad); }} .m4-pill.warn {{ color:var(--warn); }}
.m4-pill.idle {{ color:var(--slate); }}

/* research questions: ruled list */
.m4-rq {{ border-top:1px solid var(--rule); }}
.m4-rq > div {{ display:grid; grid-template-columns:5.5rem 1fr auto; gap:1rem; align-items:baseline;
                padding:.75rem 0; border-bottom:1px solid var(--rule); }}
.m4-rq .id {{ font-weight:650; font-stretch:112%; }}
.m4-rq .q {{ max-width:70ch; line-height:1.45; }}
@media (max-width: 760px) {{ .m4-rq > div {{ grid-template-columns:1fr; gap:.3rem; }} }}

/* steps of the demo (a real sequence, so numbered) */
.m4-step {{ display:flex; gap:.9rem; align-items:flex-start; margin:2rem 0 .7rem; padding-top:1rem; border-top:1px solid var(--rule); }}
.m4-step .n {{ flex:none; width:1.9rem; height:1.9rem; border-radius:50%; background:var(--ink); color:var(--panel);
               display:flex; align-items:center; justify-content:center; font-weight:650; font-size:.95rem;
               font-variant-numeric:tabular-nums; }}
.m4-step .t {{ font-stretch:112%; font-weight:640; font-size:1.12rem; line-height:1.3; }}
.m4-step .d {{ color:var(--slate); font-size:.93rem; line-height:1.45; max-width:74ch; margin-top:.1rem; }}

/* console line above the live stream */
.m4-console {{ display:flex; flex-wrap:wrap; gap:2.2rem; align-items:center; padding:.75rem 1rem;
               background:var(--panel); border:1px solid var(--rule); border-left:4px solid var(--ink); margin:.6rem 0 .4rem; }}
.m4-console .k {{ font-size:.8rem; color:var(--slate); }}
.m4-console .v {{ font-size:1.25rem; font-weight:620; font-stretch:110%; font-variant-numeric:tabular-nums; }}
.m4-progress {{ flex:1 1 180px; min-width:160px; }}
.m4-progress .track {{ height:6px; background:var(--rule); border-radius:3px; overflow:hidden; margin-top:.35rem; }}
.m4-progress .fill {{ height:100%; background:var(--ink); }}

/* the trust strip - the one bold element */
.m4-trust {{ background:var(--panel); border:1px solid var(--rule); padding:1.1rem 1.2rem 1rem; margin:.4rem 0 .6rem; }}
.m4-gate {{ margin-bottom:1rem; }}
.m4-gate:last-of-type {{ margin-bottom:.4rem; }}
.m4-gate .gname {{ font-stretch:112%; font-weight:640; font-size:1.02rem; margin-bottom:.35rem; }}
.m4-row {{ display:grid; grid-template-columns:8.5rem 1fr; gap:.8rem; align-items:center; margin:.3rem 0; }}
.m4-row .lab {{ color:var(--slate); font-size:.86rem; }}
.m4-bar {{ display:flex; gap:2px; height:30px; background:var(--panel); }}
.m4-bar.thin {{ height:12px; opacity:.55; }}
.m4-bar span {{ display:block; height:100%; }}
.m4-bar span:first-child {{ border-radius:4px 0 0 4px; }}
.m4-bar span:last-child {{ border-radius:0 4px 4px 0; }}
.m4-bar span:only-child {{ border-radius:4px; }}
.m4-key {{ display:flex; flex-wrap:wrap; gap:1.4rem; font-size:.9rem; margin-top:.2rem; padding-left:9.3rem; }}
.m4-key i {{ display:inline-block; width:.8rem; height:.8rem; border-radius:2px; margin-right:.4rem; vertical-align:-1px; }}
.m4-key .gone {{ color:var(--slate); text-decoration:line-through; }}
@media (max-width: 760px) {{ .m4-row {{ grid-template-columns:1fr; gap:.2rem; }} .m4-key {{ padding-left:0; }} }}

/* verdict line under the model comparison */
.m4-verdict {{ display:flex; flex-wrap:wrap; gap:.6rem; align-items:center; margin:.4rem 0 .2rem; }}
.m4-final {{ border-top:2px solid var(--ink); padding:.9rem 0 .4rem; margin-top:1.6rem; display:flex; flex-wrap:wrap;
             gap:.7rem; align-items:center; }}
.m4-final .k {{ font-stretch:112%; font-weight:640; margin-right:.4rem; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition:none !important; animation:none !important; }} }}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    _register_plotly()


def _register_plotly() -> None:
    axis = dict(gridcolor="#E3E7EC", linecolor=RULE, zerolinecolor=RULE, ticks="", title_font=dict(size=12, color=SLATE),
                tickfont=dict(size=11, color=SLATE))
    pio.templates["m4"] = go.layout.Template(layout=dict(
        font=dict(family=FONT, color=INK, size=13), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PANEL,
        title=dict(font=dict(size=15, color=INK), x=0, xanchor="left"), xaxis=axis, yaxis=axis,
        colorway=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
        hoverlabel=dict(font=dict(family=FONT, size=12), bgcolor=PANEL, bordercolor=RULE),
        legend=dict(font=dict(size=12)), bargap=0.32, bargroupgap=0.08, barcornerradius=3))
    pio.templates.default = "plotly_white+m4"


def chart(fig: go.Figure, key: str | None = None) -> None:
    """Plotly chart in the dashboard style (our template, not Streamlit's chart theme)."""
    st.plotly_chart(fig, theme=None, width="stretch", key=key, config={"displaylogo": False})


def esc(s) -> str:
    return html.escape(str(s))


def pill(text: str, kind: str = "idle") -> str:
    shape = {"ok": "●", "bad": "▲", "warn": "◆", "idle": "○"}[kind]
    return f"<span class='m4-pill {kind}'><span aria-hidden='true'>{shape}</span>{esc(text)}</span>"


def readout(items: list[tuple[str, str, str, bool]]) -> str:
    """items: (label, value, note, pending)."""
    cells = "".join(f"<div><div class='k'>{esc(k)}</div><div class='v{' pending' if p else ''}'>{esc(v)}</div>"
                    f"<div class='n'>{esc(n)}</div></div>" for k, v, n, p in items)
    return f"<div class='m4-readout'>{cells}</div>"


def step(n: int, title: str, detail: str = "") -> None:
    d = f"<div class='d'>{esc(detail)}</div>" if detail else ""
    st.markdown(f"<div class='m4-step'><div class='n'>{n}</div><div><div class='t'>{esc(title)}</div>{d}</div></div>",
                unsafe_allow_html=True)


def _bar(alpha, thin: bool = False) -> str:
    segs = "".join(f"<span style='flex:{a:.4f};background:{MOD[m][1]}' title='{MOD[m][0]} {a:.2f}'></span>"
                   for m, a in zip(("tel", "opt", "sar"), alpha) if a > 0.004)
    return f"<div class='m4-bar{' thin' if thin else ''}'>{segs}</div>"


def trust_strip(gates: list[tuple[str, list[float], list[float] | None]], present=(1, 1, 1)) -> str:
    """gates: (gate name, alpha now, alpha with all inputs or None). Widths = trust weights."""
    rows = []
    for name, now, base in gates:
        r = f"<div class='m4-gate'><div class='gname'>{esc(name)}</div>"
        r += f"<div class='m4-row'><div class='lab'>This scenario</div>{_bar(now)}</div>"
        if base is not None:
            r += f"<div class='m4-row'><div class='lab'>All inputs</div>{_bar(base, thin=True)}</div>"
        key = "".join(
            f"<span class='{'gone' if not present[k] else ''}'><i style='background:{MOD[m][1]}'></i>{MOD[m][0]} "
            f"<span class='m4-num'>{now[k]:.2f}</span>{' (missing)' if not present[k] else ''}</span>"
            for k, m in enumerate(("tel", "opt", "sar")))
        rows.append(r + f"<div class='m4-key'>{key}</div></div>")
    return "<div class='m4-trust'>" + "".join(rows) + "</div>"
