# Member 4 workspace

| Item | Value |
|---|---|
| Branch | `member4/full-pipeline` |
| Workspace | `members/member4/` |
| File prefix | `m4_` (tests: `test_m4_*`) |
| Diversity focus | Visualization, explainability (α weights, SHAP for XGBoost), dashboard, integration cleanliness |

Member 4 still implements the **complete** pipeline (generator, MLTS, EO baselines, fusion, evaluation, dashboard).

## Setup (local, run from repository root)

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/macOS:  source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r members/member4/requirements_member4.txt
python members/member4/m4_env_check.py --save-report
python -m pytest members/member4/tests -q
```

## Colab smoke test

```python
!git clone <REPOSITORY_URL>
%cd multimodal-satellite-ai
!git checkout member4/full-pipeline
!pip install -q -r members/member4/requirements_member4.txt
!python members/member4/m4_env_check.py
!python -m pytest members/member4/tests -q
```
Colab: Runtime → Change runtime type → T4 GPU (if available) to see `cuda_available: True`.

## Pipeline run order (from the repository root, laptop CPU)

| Step | Command | Time |
|---|---|---|
| 1 | `python members/member4/data/m4_telemetry_generator.py --plot` | ~5 s |
| 2 | `python members/member4/data/m4_make_splits.py` | ~3 s |
| 3 | `python members/member4/mlts/m4_ts_eda.py` | ~5 s |
| 4 | `python members/member4/mlts/m4_stationarity_decomposition.py` | ~2.5 min |
| 5 | `python members/member4/mlts/m4_forecast_eval.py` | ~6 min |
| 6 | `python members/member4/explainability/m4_plot_forecasts.py` | ~10 s |
| 7 | `python members/member4/mlts/m4_residual_anomaly.py` | ~10 s |
| 8 | `python members/member4/mlts/m4_opssat_benchmark.py` (needs data_external/opssat_ad) | ~1 min |
| 9 | `python members/member4/explainability/m4_plot_health.py` | ~5 s |
| — | `python -m pytest members/member4/tests -q` | ~25 s |

Step 4 must run before step 5 (ARIMAX reads `results/m4_arima_orders.json`).

## Milestone log

| Milestone | Status | Report |
|---|---|---|
| P1-M1 Environment + Git + Colab smoke test | done locally; Colab run pending | `reports/WEEKLY_STATUS.md` |
| P1-M2 Telemetry generator + dataset validation | done | `reports/GENERATOR_CARD.md`, `reports/DATASET_CARD.md` |
| P1-M3 Time-series EDA, decomposition, features, split | done | `reports/M3_TIMESERIES_EDA.md` |
| P1-M4 Forecasting ladder | done | `reports/M4_FORECASTING.md` |
| P1-M5 Health anomaly detection | done on simulated data; OPSSAT-AD run pending (laptop) | `reports/M5_HEALTH_ANOMALY.md` |
| P1-M6 EO preprocessing + optical-only / SAR-only baselines | next (15–18 Oct, Colab GPU) | |
