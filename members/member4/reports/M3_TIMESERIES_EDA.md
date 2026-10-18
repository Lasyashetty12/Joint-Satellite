# Milestone 3 — Time-series EDA, decomposition, features, split (Member 4)

Data: generator v1 (seed 42, 90 days, 30 s cadence, 259,200 rows). All numbers below come from the scripts listed; rerun them to reproduce.

## 1. Chronological split (`data/m4_make_splits.py` → `data/generated/m4_splits.json`)
| Split | Days | Rows | Anomaly events |
|---|---|---|---|
| train | 0–53 (54 d) | 155,520 | 38 |
| gap | 54 | 2,880 | — |
| val | 55–71 (17 d) | 48,960 | 12 |
| gap | 72 | 2,880 | — |
| test | 73–89 (17 d) | 48,960 | 9 |

No event crosses a boundary (38 + 12 + 9 = 59). **Note:** by chance the validation block has no `drift` or `thermal_runaway` events. Thresholds tuned on val have never seen those two types; report this alongside test results.

## 2. EDA (`mlts/m4_ts_eda.py`)
- Figures: `eda_overview`, `eda_orbit_fold`, `eda_histograms`, `eda_missing_map`, `corr_channels`, `corr_crosslag`.
- Time components: day index 0–89, 1,365 orbits, 190 samples per orbit, measured eclipse fraction 0.3473.
- Histograms are two-peaked (sunlight vs eclipse), so a single global mean/limit is a poor normality model.
- Cross-correlation (train, normal points). Peaks match the physics built into the generator:

| Pair | Peak lag | r |
|---|---|---|
| solar_I → panel_temp | +7.5 min | 0.91 |
| solar_I → SoC | +9.5 min | 0.88 |
| batt_I → bus_V | +0.5 min | 0.75 |
| batt_I → batt_temp | +9.5 min | 0.42 |

## 3. Decomposition & stationarity (`mlts/m4_stationarity_decomposition.py`)
- Window: the cleanest 7-day block of train (days 8–14); 1,046 anomalous or missing samples removed and interpolated.
- STL seasonal strength (period 190): solar_I 0.95, panel_temp 0.95, bus_V 0.93, batt_I 0.93, SoC 0.85, batt_temp 0.84. att_err (0.17) and payload_P (0.02) have **no orbit pattern**: attitude is AR noise and payload follows the imaging schedule.
- ADF/KPSS (`results/stationarity_report.csv`): after removing the orbit pattern (STL remainder or lag-190 differencing), every channel tests stationary. **Decision:** model seasonality explicitly (Fourier orbit terms / seasonal lag 190) and use d = 0. Ordinary first differencing is not needed.
- PACF of the STL remainder suggests AR orders 3–10 (`results/m4_arima_orders.json`). ARIMAX caps p at 6 for speed.
- Observation: physical faults (`v_drop`, `batt_fade`) leave a **recovery tail** of a few hours after the labelled interval ends. Event metrics in Milestone 5 therefore allow a tolerance after each event.

## 4. Features (`mlts/m4_ts_features.py`)
Per channel: current value; lags 1, 2, 3, 5, 10, 190; first difference; rolling mean/std over 10, 60, 190 samples (window ends at t); rolling min/max over 10. Also sunlit, payload_on, and Fourier orbit terms K = 3 (phase origin = generator start time, identical for every split). Scaler: `TrainScaler`, fitted on train rows only, saved as JSON.

## 5. Tests (`tests/test_m4_ts_features.py`, 14 tests)
No look-ahead (changing the future leaves past features identical), split order and gaps, train-only scaler, Fourier phase consistency, cross-correlation and stationarity helpers on known toy series.

## Known issue (generator v1, left unchanged for team reproducibility)
Sensor-drift size scales with the channel's total standard deviation. For panel_temp (large orbit swing) one test-block drift reaches about −200 °C. It is still a valid anomaly but physically extreme. Planned v2 fix: scale sensor anomalies by measurement noise / seasonal residual instead.
