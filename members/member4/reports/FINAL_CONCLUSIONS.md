# Final conclusions — Member 4

_Generated 2026-11-23 by `evaluation/m4_final_report.py` from the measured result files. Earth-observation data: `sen1floods11_handlabeled`._

## Answers to the research questions

**Primary — can a lightweight missing-modality-aware fusion model keep satellite-health and flood detection reliable when inputs are missing?** **Supported**. Averaged over the five missing-input scenarios, gated fusion keeps 94% of its all-inputs flood F1 (0.834 vs 0.886); early fusion averages 0.835 and late fusion 0.830 in the same scenarios.

**RQ2 — does forecast-residual detection beat static limits and Isolation Forest?** **Supported**. On simulated telemetry the best residual detector (`residual_gru`) reaches test event-F1 0.875 vs 0.384 for limit checks (which raise 10.53 false alarms/day) and 0.706 for Isolation Forest. On **real** OPSSAT-AD telemetry the AR forecast-residual detector reaches F1 0.862 (MCC 0.828, AUC-ROC 0.954) vs 0.444 (Isolation Forest) and 0.323 (limits).

**RQ3 — can missing telemetry channels be reconstructed so detection degrades gracefully?** **Supported**. Cross-channel reconstruction cuts the error on the six physically linked channels by 88% on average versus mean-filling; attitude error and payload power cannot be reconstructed (no physical link). In the fused model, health F1 when lost channels are rebuilt instead of zero-filled: 0.776 → 0.811 with 1 lost; 0.737 → 0.766 with 2 lost; 0.665 → 0.698 with 4 lost. Rebuilding is at least as good as zero-filling in every case.

**RQ4 — do the learned trust weights shift sensibly when an input is degraded or removed?** **Supported**. Mean α_opt for the flood decision: clear 0.499, cloudy 0.424, payload-fault-damaged 0.236, optical missing 0.000.

Decision rules are stated in `evaluation/m4_final_report.py` and applied mechanically.

## Version 1 and version 2

Version 2 changes the training and the model, not the rules: every model practises with lost telemetry sensors and with extra cloud, rebuilt channels stay flagged as not measured, and every model receives five image-quality numbers that the reliability gate also reads directly. Both versions are judged by the same decision rules on the same test passes.

| question | version 1 | version 2 |
|---|---|---|
| Primary | Not supported | Supported |
| RQ2 | Supported | Supported |
| RQ3 | Partly supported | Supported |
| RQ4 | Not supported | Supported |

| measure | version 1 | version 2 |
|---|---|---|
| gated flood F1, all inputs | 0.876 | 0.886 |
| gated flood F1, inputs missing | 0.830 | 0.834 |
| early fusion flood F1, inputs missing | 0.836 | 0.835 |
| late fusion flood F1, inputs missing | 0.835 | 0.830 |
| gated health F1, all inputs | 0.821 | 0.831 |
| health F1, 4 lost, zero-filled | 0.677 | 0.665 |
| health F1, 4 lost, rebuilt | 0.624 | 0.698 |
| optical trust, clear | 0.485 | 0.499 |
| optical trust, cloudy | 0.496 | 0.424 |
| optical trust, 100 % cloud | 0.491 | 0.408 |
| optical trust, payload fault | 0.190 | 0.236 |

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

**Mean skill vs naive on the TEST split (final comparison; higher is better)**

| model | skill h=1 | skill h=5 | skill h=10 |
|---|---|---|---|
| Naive | 0.000 | 0.000 | 0.000 |
| Seasonal naive | -1.139 | -0.082 | 0.258 |
| ARIMAX | 0.191 | 0.316 | 0.435 |
| XGBoost | 0.386 | 0.556 | 0.618 |
| GRU | 0.156 | 0.496 | 0.600 |
| LSTM | 0.127 | 0.482 | 0.592 |

Ranking at 5 min ahead on test: XGBoost 0.618, GRU 0.600, LSTM 0.592, ARIMAX 0.435, Seasonal naive 0.258, Naive 0.000. `xgboost` was chosen on validation.

On the test split at 5 min ahead: GRU skill 0.600, LSTM 0.592; the validation choice `xgboost` scores 0.618.

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

| model | split | accuracy | precision | recall | F1 | AUROC | AUPRC | n |
|---|---|---|---|---|---|---|---|---|
| optical_only | test | 0.923 | 0.895 | 0.872 | 0.883 | 0.950 | 0.945 | 352 |
| optical_only | bolivia | 0.862 | 0.781 | 0.962 | 0.862 | 0.983 | 0.984 | 58 |
| sar_only | test | 0.892 | 0.832 | 0.846 | 0.839 | 0.919 | 0.904 | 352 |
| sar_only | bolivia | 0.759 | 0.731 | 0.731 | 0.731 | 0.865 | 0.842 | 58 |
| optical_sar_cnn | test | 0.932 | 0.904 | 0.889 | 0.897 | 0.958 | 0.957 | 352 |
| optical_sar_cnn | bolivia | 0.948 | 0.926 | 0.962 | 0.943 | 0.983 | 0.981 | 58 |

## 4. Fusion, missing inputs and ablations

**All inputs present, test (mean ± std over seeds)**

| model | health F1 | flood F1 |
|---|---|---|
| telemetry only | 0.857 ± 0.006 | 0.556 ± 0.007 |
| optical only | 0.684 ± 0.000 | 0.844 ± 0.003 |
| SAR only | 0.674 ± 0.010 | 0.812 ± 0.004 |
| early fusion | 0.836 ± 0.022 | 0.874 ± 0.010 |
| late fusion | 0.856 ± 0.004 | 0.868 ± 0.014 |
| equal weights | 0.812 ± 0.024 | 0.878 ± 0.005 |
| **gated fusion (ours)** | 0.831 ± 0.039 | 0.886 ± 0.005 |
| − missing token | 0.851 ± 0.012 | 0.871 ± 0.003 |
| − telemetry in gate | 0.795 ± 0.010 | 0.877 ± 0.010 |
| − modality dropout | 0.827 ± 0.027 | 0.893 ± 0.009 |
| − forecast residuals | 0.629 ± 0.048 | 0.890 ± 0.002 |
| − multi-task loss | 0.850 ± 0.015 | 0.884 ± 0.012 |
| − quality check in gate | 0.832 ± 0.041 | 0.883 ± 0.009 |

**Flood F1 under missing / degraded inputs (test, mean over seeds)**

| scenario | early_fusion | gated_fusion | late_fusion |
|---|---|---|---|
| all_present | 0.874 | 0.886 | 0.868 |
| missing_optical | 0.807 | 0.814 | 0.804 |
| missing_sar | 0.848 | 0.827 | 0.838 |
| missing_telemetry | 0.879 | 0.877 | 0.862 |
| only_optical | 0.840 | 0.838 | 0.833 |
| only_sar | 0.802 | 0.816 | 0.810 |
| only_telemetry | 0.000 | 0.103 | 0.024 |
| cloud_25 | 0.865 | 0.872 | 0.856 |
| cloud_50 | 0.855 | 0.857 | 0.843 |
| cloud_75 | 0.831 | 0.828 | 0.821 |
| cloud_100 | 0.796 | 0.803 | 0.784 |

**Telemetry channels lost (gated fusion, health task)**

| channels lost | health F1 zero-filled | health F1 reconstructed |
|---|---|---|
| 1.000 | 0.776 | 0.811 |
| 2.000 | 0.737 | 0.766 |
| 4.000 | 0.665 | 0.698 |

**Ablations (test, mean over seeds)**

| variant | flood F1 (all) | flood F1 (missing avg) | health F1 (all) | health F1 (missing avg) |
|---|---|---|---|---|
| **gated fusion (ours)** | 0.886 | 0.834 | 0.831 | 0.575 |
| − quality check in gate | 0.883 | 0.830 | 0.832 | 0.677 |
| − missing token | 0.871 | 0.826 | 0.851 | 0.613 |
| − telemetry in gate | 0.877 | 0.817 | 0.795 | 0.639 |
| − modality dropout | 0.893 | 0.839 | 0.827 | 0.702 |
| − forecast residuals | 0.890 | 0.833 | 0.629 | 0.561 |
| − multi-task loss | 0.884 | 0.802 | 0.850 | 0.724 |

**Every model compared (test, F1 mean ± std over seeds)**

| model | flood F1, all inputs | flood F1, inputs missing (5) | flood F1, cloud 25-100 % | flood F1, sensors lost (1-4) | health F1, all inputs | health F1, inputs missing (5) | health F1, cloud 25-100 % | health F1, sensors lost (1-4) |
|---|---|---|---|---|---|---|---|---|
| telemetry only | 0.556 ± 0.007 | 0.556 ± 0.007 | 0.556 ± 0.007 | 0.559 ± 0.003 | 0.857 ± 0.006 | 0.857 ± 0.006 | 0.857 ± 0.006 | 0.751 ± 0.030 |
| optical only | 0.844 ± 0.003 | 0.844 ± 0.003 | 0.582 ± 0.007 | 0.844 ± 0.003 | 0.684 ± 0.000 | 0.684 ± 0.000 | 0.684 ± 0.000 | 0.684 ± 0.000 |
| SAR only | 0.812 ± 0.004 | 0.812 ± 0.004 | 0.812 ± 0.004 | 0.812 ± 0.004 | 0.674 ± 0.010 | 0.674 ± 0.010 | 0.674 ± 0.010 | 0.674 ± 0.010 |
| early fusion | 0.874 ± 0.010 | 0.835 ± 0.007 | 0.837 ± 0.004 | 0.873 ± 0.008 | 0.836 ± 0.022 | 0.739 ± 0.013 | 0.838 ± 0.024 | 0.764 ± 0.025 |
| late fusion | 0.868 ± 0.014 | 0.830 ± 0.008 | 0.826 ± 0.007 | 0.866 ± 0.011 | 0.856 ± 0.004 | 0.717 ± 0.023 | 0.856 ± 0.007 | 0.758 ± 0.012 |
| equal weights | 0.878 ± 0.005 | 0.833 ± 0.008 | 0.841 ± 0.002 | 0.878 ± 0.004 | 0.812 ± 0.024 | 0.664 ± 0.080 | 0.808 ± 0.023 | 0.739 ± 0.033 |
| **gated fusion (ours)** | 0.886 ± 0.005 | 0.834 ± 0.010 | 0.840 ± 0.010 | 0.886 ± 0.005 | 0.831 ± 0.039 | 0.575 ± 0.187 | 0.831 ± 0.039 | 0.742 ± 0.057 |
| − missing token | 0.871 ± 0.003 | 0.826 ± 0.003 | 0.833 ± 0.003 | 0.873 ± 0.002 | 0.851 ± 0.012 | 0.613 ± 0.234 | 0.851 ± 0.012 | 0.751 ± 0.008 |
| − telemetry in gate | 0.877 ± 0.010 | 0.817 ± 0.011 | 0.840 ± 0.009 | 0.877 ± 0.010 | 0.795 ± 0.010 | 0.639 ± 0.077 | 0.794 ± 0.010 | 0.712 ± 0.056 |
| − modality dropout | 0.893 ± 0.009 | 0.839 ± 0.007 | 0.839 ± 0.005 | 0.893 ± 0.008 | 0.827 ± 0.027 | 0.702 ± 0.050 | 0.828 ± 0.028 | 0.723 ± 0.039 |
| − forecast residuals | 0.890 ± 0.002 | 0.833 ± 0.006 | 0.844 ± 0.006 | 0.889 ± 0.002 | 0.629 ± 0.048 | 0.561 ± 0.182 | 0.629 ± 0.046 | 0.598 ± 0.051 |
| − multi-task loss | 0.884 ± 0.012 | 0.802 ± 0.037 | 0.833 ± 0.007 | 0.883 ± 0.012 | 0.850 ± 0.015 | 0.724 ± 0.023 | 0.850 ± 0.015 | 0.778 ± 0.045 |
| − quality check in gate | 0.883 ± 0.009 | 0.830 ± 0.007 | 0.838 ± 0.003 | 0.881 ± 0.011 | 0.832 ± 0.041 | 0.677 ± 0.013 | 0.831 ± 0.041 | 0.759 ± 0.032 |

_Each cell averages the scenarios of its group per seed, then over seeds. A dash means the model has no input in that condition._

**Is gated fusion really different from simple fusion? Paired bootstrap, 2000 resamples of the test passes**

| task | condition | gated minus | difference in F1 | 95 % CI | reading |
|---|---|---|---|---|---|
| flood | all inputs | early fusion | 0.012 | [-0.000, +0.024] | no clear difference |
| flood | all inputs | late fusion | 0.018 | [+0.007, +0.030] | gated better |
| flood | inputs missing (5) | early fusion | -0.001 | [-0.011, +0.009] | no clear difference |
| flood | inputs missing (5) | late fusion | 0.005 | [-0.004, +0.014] | no clear difference |
| flood | cloud 25-100 % | early fusion | 0.003 | [-0.004, +0.012] | no clear difference |
| flood | cloud 25-100 % | late fusion | 0.013 | [+0.002, +0.025] | gated better |
| health | all inputs | early fusion | -0.005 | [-0.026, +0.016] | no clear difference |
| health | all inputs | late fusion | -0.025 | [-0.046, -0.004] | gated worse |
| health | inputs missing (5) | early fusion | -0.164 | [-0.184, -0.144] | gated worse |
| health | inputs missing (5) | late fusion | -0.141 | [-0.162, -0.121] | gated worse |
| health | cloud 25-100 % | early fusion | -0.007 | [-0.028, +0.014] | no clear difference |
| health | cloud 25-100 % | late fusion | -0.025 | [-0.046, -0.004] | gated worse |

_If the 95 % interval contains 0, the two models cannot be told apart on this test set. Descriptive only: the decision rules are unchanged._

**RQ4: optical trust weight by situation (gated fusion, test)**

| situation | mean α_opt (flood gate) |
|---|---|
| clear | 0.500 |
| cloudy | 0.424 |
| payload fault | 0.236 |
| +100 % cloud | 0.408 |
| optical missing | 0.000 |

## 5. Telemetry channel reconstruction (test, 60-min gaps, MAE)

| channel | cross_channel | cross_channel_4_lost | ffill | mean_fill |
|---|---|---|---|---|
| batt_temp | 0.1354 | 0.1298 | 0.3924 | 0.2919 |
| panel_temp | 0.4117 | 0.4179 | 41.1830 | 29.9603 |
| bus_V | 0.0176 | 0.0209 | 0.2566 | 0.1801 |
| batt_I | 0.0293 | 0.0436 | 1.1010 | 0.6950 |
| solar_I | 0.0741 | 0.0860 | 1.7333 | 1.4742 |
| SoC | 0.1287 | 0.1395 | 4.0304 | 3.0585 |
| att_err | 0.0200 | 0.0200 | 0.0219 | 0.0200 |
| payload_P | 1.2239 | 1.2277 | 1.2849 | 1.2229 |

## Limitations

- Telemetry is **simulated** (generator v1); real-telemetry evidence comes from OPSSAT-AD, segment-level only.
- Telemetry–image pairing is a **simulation assumption** (virtual passes): payload faults are coupled to damaged optical patches by design.
- Small event counts in the simulated test split (9 events): one event ≈ ±0.1 F1.
- Patches are 128×128 (2× downsampled) with 6 optical bands for CPU training; flood = water fraction ≥ 5 %, which includes permanent water.
- Single generator seed; fusion results averaged over 3 training seeds.
- The image-quality check is fitted on clean training patches and tested on SIMULATED damage (clouds, stripes, noise, saturation); real clouds in Sen1Floods11 are not labelled, so its behaviour on real clouds is not measured.

## Future work

- Real mission data with co-located imagery; thermal modality; on-board deployment of the lightweight gate; a second generator seed and a walk-forward evaluation.
