# DATASET_CARD — Member 4

Status: **DRAFT**. Fill every `____` from YOUR inventory output. The team freezes benchmark v1 only after all four members' cards agree.
Rule: write only what you measured or read on the official page. Never guess.

## 0. Datasets in scope

| Role | Dataset | Decision |
|---|---|---|
| Primary telemetry | Own generator v1 (see `GENERATOR_CARD.md`) | fixed |
| Real telemetry validation (ONE only) | OPSSAT-AD (recommended) / ESA-ADB / SMAP-MSL | team decision: ____ |
| Earth observation | Sen1Floods11, hand-labelled subset | fixed |

---

## 1. Sen1Floods11 (hand-labelled subset)

### 1.1 Source, license, citation
| Item | Value | How verified |
|---|---|---|
| Repository | https://github.com/cloudtostreet/Sen1Floods11 | checked 27 Sep 2026 |
| Data location | `gs://sen1floods11/v1.1/data/flood_events/HandLabeled/` (public; HTTPS via storage.googleapis.com) | bucket listing, 27 Sep 2026 |
| Folders present | JRCWaterHand, LabelHand, S1Hand, S1OtsuLabelHand, S2Hand | bucket listing, 27 Sep 2026 |
| We use | **S1Hand, S2Hand, LabelHand** only | decision |
| Official splits | `v1.1/splits/flood_handlabeled/flood_{train,valid,test,bolivia}_data.csv` | bucket listing, 27 Sep 2026 |
| License | Not stated in README → ____ (check the repo LICENSE file or contact the authors) | ____ |
| Citation | Bonafilia et al., "Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1", CVPR Workshops 2020 | README |

### 1.2 Documented properties (from README)
- GeoTIFF chips, 512 × 512 px, 10 m resolution. Files are named `EVENT_CHIPID_LAYER.tif`.
- S1: 2 bands (VV, VH). S2: 13 bands.
- Label: −1 = no data, 0 = not water, 1 = water.

### 1.3 Measured properties (from `m4_eo_inventory_summary.json`)
| Property | Value |
|---|---|
| Total hand-labelled chips in bucket (`--list-only`) | ____ (expected ≈ 446, 11 events) |
| Total size of S1Hand + S2Hand + LabelHand | ____ MB |
| Chips downloaded / event | ____ / ____ |
| Chips with all 3 layers | ____ |
| Chips aligned (same CRS, transform, size) | ____ |
| Shape | ____ |
| S1 bands / dtype / nodata / value range | ____ / ____ / ____ / ____ |
| S2 bands / dtype / nodata / value range | ____ / ____ / ____ / ____ |
| S1 NaN fraction (max) | ____ |
| Label values seen | ____ |
| Flood fraction mean | ____ |
| eo_label counts (τ_flood = 0.05) | 0: ____ 1: ____ |
| S2 band order confirmed (B1…B12 incl. B8A?) | ____ |
| Sample figure checked by eye (optical, SAR, label overlap) | yes / no |

### 1.4 How we use it
- **Water ≠ flood.** Label 1 = water, which includes permanent water. For v1 we define `eo_label = 1` if the water fraction of valid pixels is ≥ 0.05. This is a documented simplification; we may later subtract JRC permanent water (JRCWaterHand).
- Patches: each 512 × 512 chip is split into four 256 × 256 patches.
- Split: by event/location, using the official hand-labelled split CSVs (Bolivia is an unseen-event test set). Never split by random patch.

---

## 2. Real telemetry benchmark: ____ (OPSSAT-AD recommended)

### 2.1 Source, license, citation
| Item | Value | How verified |
|---|---|---|
| Record | https://zenodo.org/records/12588359 | checked 27 Sep 2026 |
| Files | `dataset.csv` (0.5 MB), `segments.csv` (18.0 MB) | Zenodo page, 27 Sep 2026 |
| License | CC-BY-4.0 | Zenodo page, 27 Sep 2026 |
| Citation | ____ (copy "Cite as" from the Zenodo page) | ____ |

### 2.2 Measured properties (from `m4_opssat_inventory.json`)
| Property | segments.csv | dataset.csv |
|---|---|---|
| Rows | ____ | ____ |
| Columns | ____ | ____ |
| Channels (number and names) | ____ | |
| Sampling cadence | ____ | |
| Label column and meaning | ____ | ____ |
| Anomaly vs normal counts | ____ | ____ |
| Train/test indicator column | ____ | ____ |
| Missing values | ____ | ____ |

### 2.3 How we use it
- Telemetry-only validation of the forecast-residual detector, Isolation Forest and the LSTM autoencoder.
- **Never paired with Sen1Floods11 images.**
- Follow the dataset's own train/test protocol: ____

---

## 3. Selection gate checklist (handbook section 6)
- [ ] 1. EO task = flood; telemetry = own generator + ONE real benchmark
- [ ] 2. Source, license and citation recorded for each dataset
- [ ] 3. Sen1Floods11 sample has aligned optical + SAR + labels (measured, not assumed)
- [ ] 4. Real benchmark: channels, cadence, label format and protocol recorded
- [ ] 5. Tiny sample downloaded; every modality, channel and label visualised
- [ ] 6. Sizes, bands, no-data values, mask classes and telemetry gaps checked
- [ ] 7. Telemetry v1 generated; one day plotted and looks physical ✅ (27 Sep 2026)
- [ ] 8. Storage estimated before the full download
- [ ] 9. Splits planned: telemetry chronological by day blocks; EO by event
- [ ] 10. DATASET_CARD.md + GENERATOR_CARD.md written → benchmark v1 frozen
