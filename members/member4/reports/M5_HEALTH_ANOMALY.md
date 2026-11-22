# Milestone 5 — Health anomaly detection (Member 4)

Run (after Milestone 4):
```
python members/member4/mlts/m4_residual_anomaly.py        # simulated telemetry, ~10 s
python members/member4/mlts/m4_opssat_benchmark.py        # real telemetry, OPSSAT-AD
python members/member4/explainability/m4_plot_health.py
```
Results: `results/health_metrics.csv`, `results/m4_health_per_type.csv`, `results/m4_opssat_results.csv`.
Figures: `health_score_timeline.png`, `health_per_type_recall.png`, `health_detector_f1.png`.

## Method (handbook section 4)
- **Forecast-residual score:** z_c[t] = |y_c[t] − ŷ_c[t]| / sd_c, where ŷ is the 1-step forecast made at t−1 and sd_c comes from normal **train** residuals. score = max over channels, smoothed with EWMA (span 10).
- **Threshold τ** chosen on **validation** only (maximum event-F1, ties broken by fewer false alarms). Test is reported once.
- **Events:** alarms within 5 min of each other are merged. An alarm up to 60 min after a labelled event still counts (recovery tail found in Milestone 3).
- **Baselines:** `limit_check` (fixed train-normal 0.05/99.95 percentile limits per channel, no tuning) and `isolation_forest` (values, 5-min rolling std, differences, sunlit and orbit phase; fitted on normal train rows).
- `residual_ensemble` takes the larger of the XGBoost and ARIMAX scores, each scaled by its own val 99.9th percentile.

## Results on simulated telemetry v1 (12 val events, 9 test events)
| Detector | val F1 | test precision | test recall | **test F1** | test AUPRC | test delay (min) | false alarms/day |
|---|---|---|---|---|---|---|---|
| limit_check | 0.45 | 0.25 | 0.89 | 0.38 | 0.40 | 22.8 | **10.5** |
| isolation_forest | 0.77 | 0.97 | 0.56 | 0.71 | 0.34 | 41.9 | 0.06 |
| residual_naive | 0.96 | 1.00 | 0.56 | 0.71 | 0.21 | 135.7 | 0 |
| residual_snaive | 0.74 | 0.74 | 0.67 | 0.70 | 0.34 | 20.6 | 0.29 |
| residual_arimax | 0.96 | 1.00 | 0.78 | **0.88** | 0.43 | 35.4 | 0 |
| residual_xgboost | 0.89 | 0.78 | 0.78 | 0.78 | **0.55** | 34.7 | 0.47 |
| **residual_gru** (selected on val) | **0.98** | 1.00 | 0.78 | **0.88** | 0.40 | 57.4 | 0 |
| residual_ensemble | 0.94 | 0.86 | 0.67 | 0.75 | 0.45 | 33.0 | 0.12 |

**Selected detector:** `residual_gru` (best validation event-F1), test F1 0.875 (7/9 events, no false alarms). Its scores are in `data/generated/m4_anomaly_scores.csv.gz` (contract columns). Note: GRU training is not bit-for-bit deterministic across machines, so your numbers may differ slightly.

## Recall per anomaly type (all 59 events; diagnostic) — `health_per_type_recall.png`
| Type | limit | iForest | res. arimax | res. xgboost | res. gru |
|---|---|---|---|---|---|
| spike | 0.83 | 0.00 | 1.00 | 1.00 | 1.00 |
| stuck | 0.62 | 0.00 | 0.50 | 0.62 | 0.62 |
| drift | 1.00 | 0.11 | 0.78 | 0.89 | 1.00 |
| noise_burst | 1.00 | 0.30 | 0.90 | 0.80 | 0.90 |
| v_drop | 1.00 | 0.80 | 1.00 | 1.00 | 1.00 |
| batt_fade | 1.00 | 1.00 | 0.00 | 1.00 | 0.67 |
| thermal_runaway | 0.80 | 1.00 | 1.00 | **0.00** | 0.60 |
| payload_fault | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## Findings (answers handbook RQ2 on simulated data)
1. **Forecast-residual detection beats static limit checks.** Limits catch most faults but raise about 10 false alarms per day (237 alarms in 17 test days), which is unusable for an operator. The best residual detectors reach F1 0.88 with zero false alarms.
2. **Forecast-residual detection beats plain Isolation Forest** (test F1 0.71). Isolation Forest misses spikes, stuck sensors and drifts because it scores each moment without knowing what *should* happen next.
3. **A better forecaster is not automatically a better detector.** XGBoost, the most accurate forecaster, misses every thermal runaway: the ramp is gradual, so it follows it and the error stays small ("explaining away"). ARIMAX models only the normal orbit pattern, so it catches the runaways but misses battery fade. Combining them (ensemble) caught thermal runaways but lost battery fades, so it was not better overall.
4. **Hardest faults:** stuck sensors (best recall 0.62). A frozen value stays close to its forecast for a while. A dedicated "zero variance" check is a cheap addition for Milestone 11's error analysis.
5. **Detection delay** is dominated by slow faults: battery fade lasts 1–3 days and is often found hours in, which inflates the mean delay.

## Caveats (state in the paper)
- **Small event counts:** 12 val / 9 test events, so one event ≈ ±0.1 F1. Differences between ARIMAX, GRU and XGBoost are within noise. Three seeds plus a second generator seed are planned for Milestone 14.
- The validation block has no `drift` or `thermal_runaway` events, so thresholds never saw those types.
- **Event precision counts alarm segments:** one long fault can produce several alarms, all counted as correct (see `n_pred_events`). Report alarms per day next to precision.
- Simulated data: the real-telemetry check below is required before any claim.

## Real telemetry: OPSSAT-AD (segment-level)
`m4_opssat_benchmark.py` applies the same ideas per segment: naive and AR(5) residuals, limits, and Isolation Forest on the 18 provided features. The threshold comes from the official train segments; the 529 test segments are scored once with precision, recall, F1, MCC, AUC-ROC and AUC-PR.
**Status:** code and tests are ready. Results must be produced on the laptop (the dataset cannot be downloaded from the cloud workspace) → `results/m4_opssat_results.csv`, then copy the table here.

| Detector | precision | recall | F1 | MCC | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|
| residual_naive | | | | | | |
| residual_ar | | | | | | |
| limit_check | | | | | | |
| isolation_forest | | | | | | |

Not done (optional per handbook): LSTM autoencoder.
