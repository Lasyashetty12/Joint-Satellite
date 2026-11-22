# Member 4 — Satellite health + flood detection with missing inputs

| Item | Value |
|---|---|
| Branch | `member4/full-pipeline` |
| Workspace | `members/member4/` · file prefix `m4_` (tests `test_m4_*`) |
| Diversity focus | Visualisation, explainability (α weights), dashboard, integration |
| Final results | `reports/FINAL_CONCLUSIONS.md` (generated from the measured result files) |

**Primary research question.** Can a lightweight missing-modality-aware fusion model, built on time-series forecasting of telemetry and CNN features of optical/SAR imagery, keep satellite-health and Earth-observation (flood) anomaly detection reliable when one or more inputs are unavailable?

## Architecture

```
telemetry window (60 min × 8 ch) ──► forecaster (GRU · LSTM · XGBoost · ARIMAX) ──► residual z-scores ─┐
                                 └──────────────────────────────────────────────► GRU encoder ── h_tel ┤
Sentinel-2 optical patch ──► ResNet-18 ── h_opt ─────────────────────────────────────────────────────┤──► reliability gate ──► health score
Sentinel-1 SAR patch     ──► ResNet-18 ── h_sar ─────────────────────────────────────────────────────┘   (missing tokens,      flood score
                                                                                                          α_tel+α_opt+α_sar=1)  α weights
```

## Setup (once, from the repository root)

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/macOS:  source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r members/member4/requirements_member4.txt
```
Data (kept outside Git, in `<repo>/data_external/`):
- Sen1Floods11 hand-labelled subset (≈1.75 GB) is downloaded automatically by the run-all script (stage `eo_download`).
- OPSSAT-AD (optional, real telemetry, 18.5 MB): put `segments.csv` and `dataset.csv` from https://zenodo.org/records/12588359 in `data_external/opssat_ad/`.

## Reproduce everything (one command)

```bash
python members/member4/m4_run_all.py            # full run, real data, ~1.5–3 h on a laptop CPU (GPU faster)
python members/member4/m4_run_all.py --fast     # quick check: fewer epochs, one seed
python members/member4/m4_run_all.py --from eo_patches          # resume from a stage
streamlit run members/member4/dashboard/m4_app.py               # the dashboard
```
`--synthetic-eo` replaces Sen1Floods11 with a synthetic stand-in **only to test the pipeline** when the data cannot be downloaded; every output is then labelled `synthetic_eo_devcheck` and the final report withholds all image-dependent conclusions.

| Stage | Script | Milestone |
|---|---|---|
| generator, splits | `data/m4_telemetry_generator.py`, `data/m4_make_splits.py` | M2–M3 |
| eda, stationarity | `mlts/m4_ts_eda.py`, `mlts/m4_stationarity_decomposition.py` | M3 |
| forecast (+plots) | `mlts/m4_forecast_eval.py` — naive, seasonal-naive, ARIMAX, XGBoost, **GRU, LSTM** | M4 |
| health, opssat (+plots) | `mlts/m4_residual_anomaly.py`, `mlts/m4_opssat_benchmark.py` | M5 |
| eo_download, eo_patches, eo_train | `data/m4_download_eo_sample.py`, `data/m4_eo_patches.py`, `training/m4_train_eo.py` | M6 |
| reconstruction | `mlts/m4_channel_reconstruction.py` | M10 (RQ3) |
| passes | `data/m4_build_passes.py` + `data/m4_missing_modality.py` | M7 |
| fusion (+plots) | `models/m4_fusion.py`, `training/m4_train_fusion.py`, `explainability/m4_plot_fusion.py` | M8–M11 |
| report | `evaluation/m4_final_report.py` → `reports/FINAL_CONCLUSIONS.md` | M14 |
| dashboard | `dashboard/m4_app.py` (+ `m4_dashboard_data.py`) | M13 |

## Dashboard (handbook section 25 demo)

`streamlit run members/member4/dashboard/m4_app.py`
1. **Mission overview** — what the system does, architecture, headline numbers, research-question status.
2. **Live demo** — pick a stored test pass → replay its telemetry as a live stream (actual vs forecast, ±2 sd band, anomaly score, alarms) → optical + SAR patches and the flood label → run the proposed model → α bars → toggle telemetry/optical/SAR, cloud, lost telemetry channels (zero-filled or reconstructed) → see α shift, the reconstructed channel and updated scores → simple fusion vs proposed on the same pass.
3. **Model accuracy** — every measured result, interactive: forecasting (incl. GRU vs LSTM), health detection (simulated + real OPSSAT-AD, per-fault recall), EO baselines, fusion, robustness, α behaviour, ablations, reconstruction.
4. **Final conclusions** — the generated report.

## Tests

`python -m pytest members/member4/tests -q` — environment, generator, features/splits (no look-ahead), forecasting, metrics, EO/missing-modality/fusion models (α sum to 1, missing input gets α = 0 and cannot affect the output), orchestrator, and an end-to-end dashboard test of all pages.

## Reports

`reports/GENERATOR_CARD.md`, `reports/DATASET_CARD.md`, `reports/M3_TIMESERIES_EDA.md`, `reports/M4_FORECASTING.md`, `reports/M5_HEALTH_ANOMALY.md`, `reports/FINAL_CONCLUSIONS.md`; figures in `reports/figures/`; all numbers in `results/`.

## Milestone log

| Milestone | Status |
|---|---|
| P1-M1 Environment + Git + Colab | done (laptop) |
| P1-M2 Generator + dataset validation | done |
| P1-M3 EDA, decomposition, features, split | done |
| P1-M4 Forecasting ladder (incl. GRU + LSTM) | done |
| P1-M5 Health anomaly detection (+ OPSSAT-AD) | done |
| P1-M6 EO patches + optical/SAR baselines | code done; real-data run via `m4_run_all.py` |
| P2-M7 Virtual passes + missing-modality simulator | done |
| P2-M8/M9 Simple fusion + reliability-gated fusion | done |
| P2-M10/M11 Missing-input experiments, reconstruction, ablations, 3 seeds | done |
| P3-M13 Dashboard + E2E test | done |
| P3-M14/M15 Final report generator + run-all reproduction | done |

## Limitations

Simulated telemetry (real-telemetry check on OPSSAT-AD, segment level); telemetry–image pairing is a simulation assumption; 9 test events in the simulated split (one event ≈ ±0.1 F1); 128×128 patches with 6 optical bands for CPU training.
