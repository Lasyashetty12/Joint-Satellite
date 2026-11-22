# Final conclusions — Member 4

_Generated 2026-11-22 by `evaluation/m4_final_report.py` from the measured result files. Earth-observation data: `synthetic_eo_devcheck`._

> ⚠️ **The Earth-observation inputs in this run are the SYNTHETIC stand-in** (`synthetic_eo_devcheck`). Image and fusion numbers below only prove the pipeline runs; they are **not results**, and every image-dependent conclusion is withheld. Run the full pipeline on Sen1Floods11 (`python members/member4/m4_run_all.py`) to fill them in.

## Answers to the research questions

**Primary — can a lightweight missing-modality-aware fusion model keep satellite-health and flood detection reliable when inputs are missing?** **Withheld: needs the real Sen1Floods11 run**

**RQ2 — does forecast-residual detection beat static limits and Isolation Forest?** **Supported**. On simulated telemetry the best residual detector (`residual_gru`) reaches test event-F1 0.875 vs 0.384 for limit checks (which raise 10.53 false alarms/day) and 0.706 for Isolation Forest. On **real** OPSSAT-AD telemetry the AR forecast-residual detector reaches F1 0.862 (MCC 0.828, AUC-ROC 0.954) vs 0.444 (Isolation Forest) and 0.323 (limits).

**RQ3 — can missing telemetry channels be reconstructed so detection degrades gracefully?** **Partially supported**. Cross-channel reconstruction cuts the error on the six physically linked channels by 88% on average versus mean-filling; attitude error and payload power cannot be reconstructed (no physical link). In the fused model, reconstructing lost channels instead of zero-filling them raises health F1 from 0.73 to 0.792 (1 channel lost) and from 0.697 to 0.745 (2 lost), but not when half the channels are lost (0.674 zero-filled vs 0.625 reconstructed, 4 of 8): degradation is graceful only while most channels remain. Image-dependent part provisional (synthetic EO).

**RQ4 — do the learned trust weights shift sensibly when an input is degraded or removed?** **Withheld: needs the real Sen1Floods11 run**

Decision rules are stated in `evaluation/m4_final_report.py` and applied mechanically.

## 1. Forecasting ladder (MLTS)

**Validation MAE, h = 1 (30 s ahead)** — lower is better

| channel | naive | snaive | arimax | xgboost | gru | lstm |
|---|---|---|---|---|---|---|
| SoC | 0.1786 | 0.3622 | 0.1231 | 0.0880 | 0.1145 | 0.1146 |
| att_err | 0.0046 | 0.0276 | 0.0046 | 0.0046 | 0.0046 | 0.0046 |
| batt_I | 0.0693 | 0.0748 | 0.0771 | 0.0314 | 0.0693 | 0.0701 |
| batt_temp | 0.1133 | 0.1369 | 0.0879 | 0.0827 | 0.0843 | 0.1050 |
| bus_V | 0.0285 | 0.0413 | 0.0256 | 0.0178 | 0.0233 | 0.0244 |
| panel_temp | 1.2046 | 0.7183 | 0.6542 | 0.4596 | 0.5836 | 0.5874 |
| payload_P | 0.4183 | 2.0581 | 0.4436 | 0.3276 | 0.4458 | 0.4459 |
| solar_I | 0.0686 | 0.0340 | 0.0286 | 0.0264 | 0.0675 | 0.0688 |

**Validation MAE, h = 10 (300 s ahead)** — lower is better

| channel | naive | snaive | arimax | xgboost | gru | lstm |
|---|---|---|---|---|---|---|
| SoC | 1.1491 | 0.3623 | 0.5085 | 0.1069 | 0.1343 | 0.1313 |
| att_err | 0.0123 | 0.0277 | 0.0117 | 0.0118 | 0.0118 | 0.0118 |
| batt_I | 0.3856 | 0.0748 | 0.2532 | 0.0432 | 0.0984 | 0.0935 |
| batt_temp | 0.1570 | 0.1369 | 0.1016 | 0.0839 | 0.0846 | 0.1025 |
| bus_V | 0.0999 | 0.0413 | 0.0495 | 0.0205 | 0.0260 | 0.0262 |
| panel_temp | 9.7528 | 0.7183 | 2.2707 | 0.5350 | 0.6709 | 0.6619 |
| payload_P | 1.1213 | 2.0585 | 1.1740 | 0.9611 | 0.9532 | 0.9235 |
| solar_I | 0.3819 | 0.0340 | 0.0396 | 0.0305 | 0.0671 | 0.0701 |

**Mean skill vs naive (1 − MAE/MAE_naive), validation**

| model | skill h=1 | skill h=5 | skill h=10 |
|---|---|---|---|
| naive | 0.000 | 0.000 | 0.000 |
| snaive | -1.223 | -0.109 | 0.246 |
| arimax | 0.189 | 0.312 | 0.428 |
| xgboost | 0.395 | 0.580 | 0.638 |
| gru | 0.158 | 0.494 | 0.598 |
| lstm | 0.126 | 0.480 | 0.587 |

The deep models: GRU mean skill at 5 min = 0.598, LSTM = 0.587; best overall = `xgboost` (0.638).

## 2. Satellite-health anomaly detection

**Simulated telemetry, test split (thresholds chosen on validation)**

| model | event_precision | event_recall | event_F1 | AUPRC | mean_delay_min | false_alarms_per_day |
|---|---|---|---|---|---|---|
| residual_naive | 1.000 | 0.556 | 0.714 | 0.209 | 135.700 | 0.000 |
| residual_snaive | 0.737 | 0.667 | 0.700 | 0.344 | 20.583 | 0.294 |
| residual_arimax | 1.000 | 0.778 | 0.875 | 0.434 | 35.357 | 0.000 |
| residual_xgboost | 0.868 | 0.778 | 0.821 | 0.544 | 60.643 | 0.294 |
| residual_gru | 1.000 | 0.778 | 0.875 | 0.401 | 57.357 | 0.000 |
| residual_lstm | 1.000 | 0.778 | 0.875 | 0.392 | 70.643 | 0.000 |
| residual_ensemble | 1.000 | 0.667 | 0.800 | 0.447 | 41.667 | 0.000 |
| limit_check | 0.245 | 0.889 | 0.384 | 0.400 | 22.750 | 10.529 |
| isolation_forest | 0.969 | 0.556 | 0.706 | 0.336 | 41.900 | 0.059 |

**Real telemetry: OPSSAT-AD (ESA OPS-SAT CubeSat), 529 test segments**

| model | precision | recall | F1 | MCC | AUC_ROC | AUC_PR |
|---|---|---|---|---|---|---|
| residual_naive | 0.871 | 0.655 | 0.748 | 0.701 | 0.879 | 0.811 |
| residual_ar | 0.895 | 0.832 | 0.862 | 0.828 | 0.954 | 0.924 |
| limit_check | 0.239 | 0.496 | 0.323 | 0.056 | 0.588 | 0.365 |
| isolation_forest | 0.311 | 0.779 | 0.444 | 0.255 | 0.716 | 0.432 |

## 3. Earth-observation baselines (flood)

> Synthetic stand-in: pipeline check only.

| model | split | accuracy | precision | recall | F1 | AUROC | AUPRC | n |
|---|---|---|---|---|---|---|---|---|
| optical_only | test | 0.938 | 0.922 | 0.991 | 0.955 | 0.996 | 0.998 | 160 |
| optical_only | bolivia | 0.900 | 0.788 | 0.963 | 0.867 | 0.983 | 0.976 | 80 |
| sar_only | test | 0.944 | 0.938 | 0.982 | 0.959 | 0.989 | 0.995 | 160 |
| sar_only | bolivia | 0.925 | 0.818 | 1.000 | 0.900 | 0.990 | 0.981 | 80 |
| optical_sar_cnn | test | 0.938 | 0.922 | 0.991 | 0.955 | 0.991 | 0.996 | 160 |
| optical_sar_cnn | bolivia | 0.925 | 0.839 | 0.963 | 0.897 | 0.981 | 0.965 | 80 |

## 4. Fusion, missing inputs and ablations

> Synthetic EO stand-in: pipeline check only.

**All inputs present, test (mean ± std over seeds)**

| model | health F1 | flood F1 |
|---|---|---|
| telemetry only | 0.839 ± 0.045 | 0.732 ± 0.005 |
| optical only | 0.677 ± 0.004 | 0.918 ± 0.004 |
| SAR only | 0.677 ± 0.006 | 0.955 ± 0.002 |
| early fusion | 0.833 ± 0.033 | 0.955 ± 0.004 |
| late fusion | 0.833 ± 0.050 | 0.961 ± 0.002 |
| equal weights | 0.808 ± 0.071 | 0.953 ± 0.002 |
| **gated fusion (ours)** | 0.822 ± 0.017 | 0.952 ± 0.000 |
| − missing token | 0.817 ± 0.047 | 0.955 ± 0.003 |
| − telemetry in gate | 0.852 ± 0.012 | 0.950 ± 0.006 |
| − modality dropout | 0.802 ± 0.021 | 0.954 ± 0.003 |
| − forecast residuals | 0.690 ± 0.047 | 0.953 ± 0.001 |
| − multi-task loss | 0.808 ± 0.025 | 0.952 ± 0.000 |

**Flood F1 under missing / degraded inputs (test, mean over seeds)**

| scenario | early_fusion | gated_fusion | late_fusion |
|---|---|---|---|
| all_present | 0.955 | 0.952 | 0.961 |
| missing_optical | 0.950 | 0.952 | 0.953 |
| missing_sar | 0.911 | 0.924 | 0.922 |
| missing_telemetry | 0.955 | 0.952 | 0.961 |
| only_optical | 0.910 | 0.924 | 0.923 |
| only_sar | 0.951 | 0.952 | 0.950 |
| only_telemetry | 0.429 | 0.740 | 0.733 |
| cloud_25 | 0.957 | 0.952 | 0.958 |
| cloud_50 | 0.956 | 0.952 | 0.946 |
| cloud_75 | 0.955 | 0.951 | 0.939 |
| cloud_100 | 0.952 | 0.951 | 0.951 |

**Telemetry channels lost (gated fusion, health task)**

| channels lost | health F1 zero-filled | health F1 reconstructed |
|---|---|---|
| 1.000 | 0.730 | 0.792 |
| 2.000 | 0.697 | 0.745 |
| 4.000 | 0.674 | 0.625 |

**Ablations (test, mean over seeds)**

| variant | flood F1 (all) | flood F1 (missing avg) | health F1 (all) | health F1 (missing avg) |
|---|---|---|---|---|
| **gated fusion (ours)** | 0.952 | 0.941 | 0.822 | 0.591 |
| − missing token | 0.955 | 0.935 | 0.817 | 0.737 |
| − telemetry in gate | 0.950 | 0.939 | 0.852 | 0.674 |
| − modality dropout | 0.954 | 0.917 | 0.802 | 0.723 |
| − forecast residuals | 0.953 | 0.938 | 0.690 | 0.604 |
| − multi-task loss | 0.952 | 0.936 | 0.808 | 0.733 |

**RQ4: optical trust weight by situation (gated fusion, test)**

| situation | mean α_opt (flood gate) |
|---|---|
| clear | 0.137 |
| cloudy | 0.093 |
| payload fault | 0.014 |
| +100 % cloud | 0.041 |
| optical missing | 0.000 |

## 5. Telemetry channel reconstruction (test, 60-min gaps, MAE)

| channel | cross_channel | cross_channel_4_lost | ffill | mean_fill |
|---|---|---|---|---|
| batt_temp | 0.1351 | 0.1306 | 0.3924 | 0.2919 |
| panel_temp | 0.4204 | 0.4363 | 41.1830 | 29.9603 |
| bus_V | 0.0176 | 0.0209 | 0.2566 | 0.1801 |
| batt_I | 0.0301 | 0.0450 | 1.1010 | 0.6950 |
| solar_I | 0.0746 | 0.0865 | 1.7333 | 1.4742 |
| SoC | 0.1294 | 0.1432 | 4.0304 | 3.0585 |
| att_err | 0.0200 | 0.0200 | 0.0219 | 0.0200 |
| payload_P | 1.1915 | 1.1850 | 1.2849 | 1.2229 |

## Limitations (state these in the paper)

- Telemetry is **simulated** (generator v1); real-telemetry evidence comes from OPSSAT-AD, segment-level only.
- Telemetry–image pairing is a **simulation assumption** (virtual passes): payload faults are coupled to damaged optical patches by design.
- Small event counts in the simulated test split (9 events): one event ≈ ±0.1 F1.
- Patches are 128×128 (2× downsampled) with 6 optical bands for CPU training; flood = water fraction ≥ 5 %, which includes permanent water.
- Single generator seed; fusion results averaged over 3 training seeds.

## Future work

- Real mission data with co-located imagery; thermal modality; on-board deployment of the lightweight gate; a second generator seed and a walk-forward evaluation.
