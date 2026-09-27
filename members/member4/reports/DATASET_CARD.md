# DATASET_CARD — Member 4

Status: **v1 candidate — measured 27 Sep 2026.** Every value below was measured by `m4_dataset_inventory.py` / `m4_download_eo_sample.py` or read from an official page (source given). Items marked **OPEN** still need a team decision or an answer from the dataset authors.

## 0. Datasets in scope

| Role | Dataset | Decision |
|---|---|---|
| Primary telemetry | Own generator v1 (`GENERATOR_CARD.md`) | fixed |
| Real telemetry validation (ONE only) | OPSSAT-AD | Member 4 proposal; **OPEN: team confirmation** |
| Earth observation | Sen1Floods11, hand-labelled subset | fixed |

---

## 1. Sen1Floods11 (hand-labelled subset)

### 1.1 Source, license, citation
| Item | Value | How verified |
|---|---|---|
| Repository | https://github.com/cloudtostreet/Sen1Floods11 | read 27 Sep 2026 |
| Data location | `gs://sen1floods11/v1.1/data/flood_events/HandLabeled/` (public, HTTPS) | bucket listing, 27 Sep 2026 |
| Folders present | JRCWaterHand, LabelHand, S1Hand, S1OtsuLabelHand, S2Hand | bucket listing |
| Used | S1Hand, S2Hand, LabelHand | decision |
| Official splits | `v1.1/splits/flood_handlabeled/flood_{train,valid,test,bolivia}_data.csv` | bucket listing |
| License | **OPEN** — not stated in the README; check the repo LICENSE file / ask the authors before publication | — |
| Citation | Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. "Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1." CVPR Workshops 2020 | README |

### 1.2 Full hand-labelled set (measured with `--list-only`)
| Layer | Files | Size |
|---|---|---|
| S1Hand | 446 | 729.5 MB |
| S2Hand | 446 | 1,018.6 MB |
| LabelHand | 446 | 2.6 MB |
| **Total** | **446 chips × 3** | **≈ 1.75 GB** |

Chips per event: Bolivia 15, Ghana 53, India 68, Mekong 30, Nigeria 18, Pakistan 28, Paraguay 67, Somalia 26, Spain 30, Sri-Lanka 42, USA 69 (**11 events**). All three layers have identical per-event counts.

### 1.3 Sample measured (Bolivia, 15 chips)
| Property | Measured value |
|---|---|
| Chips with all 3 layers | 15 / 15 |
| Aligned (same CRS, shape, pixel size, offset < 0.5 px) | **15 / 15**, max offset **0.0 px** |
| CRS | EPSG:4326 for all layers |
| Shape | 512 × 512 |
| S1 | 2 bands (VV, VH) · float32 · no-data = NaN · range **−103.88 … 8.28 dB** |
| S2 | 13 bands · int16 · no-data = 0 · range 0 … 9,925 (reflectance × 10,000) |
| Label | int16 · metadata no-data = None, but **−1 is used as no-data** in the pixel values · values seen {−1, 0, 1} |
| Flood fraction (valid pixels), mean | 0.170 |
| eo_label at τ_flood = 0.05 | 1: **9**, 0: **6** |
| Official split | all 15 in `flood_bolivia_data.csv` (unseen-event test set) |
| Visual check (`figures/m4_eo_sample_Bolivia_129334.png`) | ✅ Label water matches dark SAR VV areas; ✅ S2 bands (4, 3, 2) give a natural colour image → band order B1, B2, B3, B4… confirmed for RGB |

### 1.4 Preprocessing decisions this card implies (implemented at 15–18 Oct)
1. **S1:** replace NaN; clip extreme dB values (min seen −103.9 dB, i.e. outliers). The clip range will be chosen from training-set percentiles and recorded here.
2. **S2:** treat 0 as no-data; scale by 1/10,000; normalise with training-set statistics only.
3. **Label:** treat −1 as no-data explicitly (the file metadata does not declare it).
4. **Water ≠ flood:** label 1 = any water, including permanent water. v1 definition: `eo_label = 1` if water fraction of valid pixels ≥ 0.05. This is a documented simplification (JRCWaterHand could remove permanent water later).
5. **Patches:** each 512 × 512 chip → four 256 × 256 patches (≈ 1,784 patches from 446 chips).
6. **Split:** official hand-labelled split CSVs (by event); Bolivia is an unseen-event test set. Never split by random patch.

---

## 2. OPSSAT-AD (proposed real telemetry benchmark)

### 2.1 Source, license, citation
| Item | Value | How verified |
|---|---|---|
| Record | https://zenodo.org/records/12588359 (DOI 10.5281/zenodo.12588359) | read 27 Sep 2026 |
| License | CC-BY-4.0 | Zenodo page |
| Dataset citation | Ruszczak, B., Kotowski, K., Nalepa, J., Evans, D. "OPSSAT-AD – anomaly detection dataset for satellite telemetry." Zenodo, 2024 | Zenodo page |
| Paper citation | Ruszczak, B., Kotowski, K., Evans, D., Nalepa, J. "The OPS-SAT benchmark for detecting anomalies in satellite telemetry." *Scientific Data* 12, 710 (2025) | journal page |
| Code | https://github.com/kplabs-pl/OPS-SAT-AD | GitHub |

### 2.2 Measured properties
| Property | segments.csv (raw telemetry) | dataset.csv (per-segment features) |
|---|---|---|
| Size / rows | 17.97 MB / 303,493 readings | 0.51 MB / 2,123 segments |
| Columns | channel, timestamp, value, label, sampling, anomaly, segment, train | segment, anomaly, train, channel, sampling + 18 hand-crafted features (mean, var, std, kurtosis, skew, peaks, …) |
| Missing values | none | none |
| Time range | 2022-01-04 20:00:50 → 2022-06-02 15:10:42 UTC | — |
| Channels | 9 (CADC0872, 0873, 0874, 0884, 0886, 0888, 0890, 0892, 0894) | same 9 |
| Sampling | `sampling` ∈ {1, 5}; measured median gap within a segment = **1 s** and **5 s** respectively ✅ | 793 segments at 1 s, 1,330 at 5 s |
| Irregular gaps | gaps within segments: 1 s (187,304), 5 s (63,680), 2 s (26,941), **0 s duplicates (22,115)**, 6 s (693) | — |
| Anomaly label (`anomaly`) | 1: 100,264 readings · 0: 203,229 | **1: 434 (20.4 %) · 0: 1,689** |
| `label` text column | `anomaly` 100,264 (= exactly the `anomaly`=1 rows ✅), `a2` 75,272, `a3` 106,930, `a4` 21,027 | — |
| Train / test (`train`) | 225,178 / 78,315 readings | **1,594 / 529 segments** |

Physical channels (from the paper): 3 magnetometer channels (I_B_FB_MM_0…2) and 6 photodiode channels (I_PD1…6_THETA). **OPEN:** the mapping from CADC codes to these names is not given in the files.

**OPEN:** the meaning of `a2`, `a3`, `a4` is not documented on the Zenodo page, the paper summary or the GitHub README. We use **only the binary `anomaly` column**.

### 2.3 How we use it
- Task is **segment-level** anomaly detection (each segment is normal or anomalous). It is not event timing, so we report segment-level metrics as the benchmark recommends (precision, recall, F1, MCC, AUC-ROC, AUC-PR).
- Follow the official split: `train` = 1 → fit/threshold, `train` = 0 → test once.
- Loader must: drop duplicate timestamps within a segment; keep channels separate; never mix 1 s and 5 s segments without resampling.
- Telemetry-only validation of forecast-residual scoring, Isolation Forest and LSTM autoencoder. **Never paired with Sen1Floods11 images.**

---

## 3. Selection gate checklist (handbook section 6)
- [x] 1. EO task = flood; telemetry = own generator + ONE real benchmark (OPSSAT-AD proposed; team to confirm)
- [x] 2. Source, license, citation recorded (Sen1Floods11 license **OPEN**)
- [x] 3. Sen1Floods11 sample: aligned optical + SAR + labels — measured 15/15, 0.0 px offset
- [x] 4. OPSSAT-AD: channels, cadence, label format, split recorded
- [x] 5. Tiny sample downloaded and visualised (EO figure checked by eye; telemetry day plots checked)
- [x] 6. Sizes, bands, no-data values, mask classes, telemetry gaps checked
- [x] 7. Telemetry v1 generated; one day plotted and looks physical
- [x] 8. Storage estimated: full hand-labelled set ≈ 1.75 GB; OPSSAT-AD ≈ 18.5 MB; own telemetry a few MB
- [x] 9. Splits planned: telemetry chronological by day blocks (Milestone 3); EO by official event split
- [ ] 10. Freeze benchmark v1 — **after the team agrees on the real telemetry dataset and compares cards**

## 4. Open items
| # | Item | Owner | Due |
|---|---|---|---|
| 1 | Team confirms OPSSAT-AD as the single real telemetry benchmark | team | before 3 Oct |
| 2 | Sen1Floods11 license | Member 4 | before paper draft |
| 3 | Meaning of OPSSAT-AD `a2/a3/a4` labels and CADC → channel-name mapping (ask authors / read full paper) | Member 4 | optional |
