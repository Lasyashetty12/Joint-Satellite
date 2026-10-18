# Milestone 4 — Forecasting ladder (Member 4)

Run: `python members/member4/mlts/m4_forecast_eval.py` (about 6 min on CPU), then `python members/member4/explainability/m4_plot_forecasts.py`.
Results: `results/forecast_results.csv` (all channels × horizons × val/test), `results/m4_forecast_skill.csv`, `results/m4_forecast_info.json`.
Figures: `forecast_skill_heatmap.png`, `forecast_actual_vs_pred.png`, `forecast_gru_loss.png`.

## Set-up (identical for every model)
- Horizons: h = 1, 5, 10 samples (30 s, 2.5 min, 5 min ahead). Output shape `[N, H, C]`.
- **Trained on normal behaviour:** training samples whose input window or target touches a labelled train anomaly or a missing value are dropped.
- **Scored on normal targets only:** origin and target in the same split; target not inside a labelled anomaly and not missing. Forecasting measures how well normal behaviour is predicted; detecting anomalies is Milestone 5.
- **Known-future covariates:** the eclipse flag and orbit phase at the target time come from the orbit geometry (the ephemeris), so ARIMAX, XGBoost and the GRU all receive them. This is not look-ahead.
- Model choices use **validation** only. Test numbers are reported once, below, and were not used for any decision.

## Models
| Model | What it does | Fit time (CPU) |
|---|---|---|
| naive | ŷ[t+h] = y[t] | <1 s |
| seasonal-naive | ŷ[t+h] = y[t+h−190] (one orbit earlier) | <1 s |
| ARIMAX | Regression on [1, Fourier K=3, sunlit] with AR(p ≤ 6) errors, d = 0 (from Milestone 3); statsmodels SARIMAX MLE on the last 7 train days | ~2 min |
| XGBoost | Direct strategy: one model per channel × horizon predicts the change y[t+h] − y[t] from ~190 no-look-ahead features plus known-future orbit covariates; early stopping on val | ~2 min |
| GRU | 60-step window of 12 standardised inputs; GRU(64) final state + future sunlit flags → MLP → changes for h = 1..10, all channels; 15 epochs, best epoch on val | ~2 min |

## Results — mean skill vs naive (1 − MAE/MAE_naive, validation, averaged over 8 channels)
| Model | h = 1 | h = 5 | h = 10 |
|---|---|---|---|
| seasonal-naive | −1.22 | −0.11 | +0.25 |
| ARIMAX | +0.19 | +0.31 | +0.43 |
| GRU | +0.16 | +0.49 | +0.60 |
| **XGBoost** | **+0.40** | **+0.58** | **+0.64** |

Validation MAE at h = 10 (5 min ahead):

| channel | naive | snaive | arimax | xgboost | gru |
|:--|--:|--:|--:|--:|--:|
| SoC [%] | 1.149 | 0.362 | 0.508 | **0.107** | 0.134 |
| batt_I [A] | 0.386 | 0.075 | 0.253 | **0.043** | 0.098 |
| batt_temp [°C] | 0.157 | 0.137 | 0.102 | **0.084** | 0.085 |
| bus_V [V] | 0.100 | 0.041 | 0.050 | **0.020** | 0.026 |
| panel_temp [°C] | 9.753 | 0.718 | 2.271 | **0.549** | 0.671 |
| solar_I [A] | 0.382 | 0.034 | 0.040 | **0.030** | 0.067 |
| att_err [deg] | 0.0124 | 0.0276 | **0.0117** | 0.0118 | 0.0118 |
| payload_P [W] | 1.121 | 2.059 | 1.174 | **0.942** | 0.953 |

**Test** (reported once) agrees with validation: XGBoost h = 10 test MAE is SoC 0.106, bus_V 0.020, panel_temp 0.569. No sign of overfitting to the validation days.

## Findings
1. **XGBoost is the best forecaster** on every channel at every horizon except att_err, where all models tie. It is the default for Milestone 5's residual scoring; the other models are kept for comparison.
2. Seasonal-naive is a strong 5-minute baseline for orbit-driven channels, but poor at 30 s. Any model that does not beat it at h = 10 is not learning the orbit.
3. Adding the known eclipse flag improved ARIMAX sharply on step-like channels: solar_I h = 1 MAE dropped from 0.114 to 0.029. Smooth Fourier terms alone cannot draw the sunrise/sunset steps.
4. **att_err and payload_P are essentially unpredictable** (skill ≈ 0). Attitude error is random AR noise, and payload power follows an imaging schedule the models do not receive. Residual-based detection on these channels will rely on large deviations only.
5. The GRU was still improving at epoch 15 (`forecast_gru_loss.png`). With more epochs or a GPU it may approach XGBoost. Deep models are Member 2's diversity focus; this run is the fair baseline.

## Limitations
- Single chronological fold (no walk-forward re-fitting); seed 42 only. Three-seed runs come in Milestone 14 (final experiments).
- sMAPE is unstable for channels near zero (batt_I, att_err); use MAE/RMSE for those.
- Simulated telemetry; real-telemetry validation is in Milestone 5 (OPSSAT-AD).
