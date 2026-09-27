# GENERATOR_CARD — Member 4 virtual telemetry generator v1

| Item | Value |
|---|---|
| Code | `data/m4_telemetry_generator.py`, `data/m4_anomaly_injector.py` |
| Config | `configs/m4_generator_config.json` (version `m4_v1`) |
| Seed | 42 (separate random streams for schedule / physics / noise / anomalies / dropouts) |
| Output | `data/generated/m4_telemetry_v1.parquet` (+ labels, dropouts, summary) |
| Size (90 days) | 259,200 rows × 8 channels at 30 s cadence |
| Reproduce | `python members/member4/data/m4_telemetry_generator.py --plot` |

## 1. Purpose
No public dataset pairs a satellite's health telemetry with the Earth images taken by that same satellite. This generator produces **physically linked** LEO telemetry with **exact anomaly labels**. It is the project's primary telemetry source. Results are also validated on one real public telemetry benchmark.

## 2. Orbit and time base
- Orbit period 95 min (190 samples). Eclipse is the last 35 % of every orbit, so `sunlit` = 1 for orbit phase < 0.65.
- Measured eclipse fraction over 90 days: **0.3473**. It isn't exactly 0.35 because 190 samples don't split evenly.
- Sun angle β varies as 25°·sin(2π·day/60). This gives a slow seasonal change in solar current.
- Imaging passes: on average 4 per day, 8 min each, **sunlit only** (optical camera).

## 3. Channel physics (true state, before measurement noise)

| Channel | Model |
|---|---|
| `solar_I` [A] | 3.5 · sunlit · cos β · (1 − 0.0002·day) |
| `payload_P` [W] | 30 W while imaging, 2 W idle |
| load current | (25 W + payload_P + extra_load) / 28 V |
| `batt_I` [A] | solar_I − load_I (positive = charging). Set to 0 when SoC = 100 % and surplus power is shunted. |
| `SoC` [%] | SoC += batt_I·Δt / capacity · 100, clipped to 0–100. Capacity = 5 Ah · (1 − 0.0003·day). |
| `bus_V` [V] | 24 + 0.04·SoC + 0.15·batt_I + offset |
| `panel_temp` [°C] | First-order lag (τ = 12 min) toward 60 °C in sun and −40 °C in eclipse |
| `batt_temp` [°C] | First-order lag (τ = 60 min) toward 15 + 3·sunlit + 2·\|batt_I\| + extra_heat |
| `att_err` [deg] | AR(1): a[t] = 0.98·a[t−1] + N(0, 0.005) |

Gaussian measurement noise is then added (standard deviation per channel is in the config).

## 4. Injected anomalies (ground truth)
Events never overlap, and at least 60 min separates any two. Rate is 0.6 per day.

| Type | Family | Injection | Labelled channels |
|---|---|---|---|
| spike | sensor | 1–3 samples, ±6–10 × channel std | 1 random channel |
| stuck | sensor | channel frozen at last good value for 30–180 min | 1 random channel |
| drift | sensor | linear offset growing to ±3–6 × std over 3–12 h | 1 random channel |
| noise_burst | sensor | extra noise 5–10 × measurement noise for 20–120 min | 1 random channel |
| v_drop | physics | bus_V −0.8…−2.0 V plus 5–15 W extra load for 30–120 min | bus_V, batt_I |
| batt_fade | physics | capacity × 0.5–0.7 for 1–3 days, so eclipse discharge is deeper | SoC, bus_V |
| thermal_runaway | physics | battery heat target ramps up to +15…25 °C over 1–4 h | batt_temp |
| payload_fault | physics | payload power 1.6–2.2 × nominal, erratic, 20–90 min | payload_P, batt_I |

**Physics** anomalies change the simulated satellite, so every linked channel reacts consistently. **Sensor** anomalies change only what one sensor reports.

`coupled_pass_id` is empty for now. It is filled in Phase 2, when payload faults are paired with degraded optical patches.

Seed-42, 90-day realisation: **59 events**. By type: payload_fault 10, noise_burst 10, drift 9, stuck 8, spike 6, batt_fade 6, thermal_runaway 5, v_drop 5.

## 5. Dropouts (not anomalies)
Rate 0.3 per day, 5–60 min each. Half of them blank all channels (downlink loss); the rest blank 1–3 random channels. Values become NaN. They are recorded in `m4_telemetry_dropouts.csv` and tested separately (missing-data experiments).

Seed-42, 90 days: 33 dropouts, 0.61 % of values missing.

## 6. Sanity checks (automated in `tests/test_m4_generator.py`)
- Same seed gives identical output; a different seed gives different output.
- Timestamps are monotonic with exactly 30 s cadence.
- True SoC stays within 0–100. Measured eclipse fraction is within ±0.01 of the config.
- solar_I ≈ 0 in eclipse; the battery discharges in eclipse; panels are more than 20 °C hotter in sun; the healthy payload runs only in sunlight.
- Every labelled interval actually changes its labelled channel(s). Stuck segments are constant. Label timestamps match injection indices.
- NaNs appear only inside recorded dropout windows.

## 7. Limitations (state these in the paper)
- Simplified physics: fixed eclipse fraction (not β-dependent), linear bus-voltage model, first-order thermal models, no albedo or attitude-power coupling.
- Anomaly shapes are hand-designed and may be easier to detect than real faults. That is why we validate on one real telemetry benchmark.
- Payload fault ↔ optical degradation coupling is a **simulation assumption**. Telemetry and imagery do not come from the same real satellite.
- Parameters are plausible for a small LEO satellite but are not calibrated to any specific mission.

## 8. Peer review
Reviewed by: ______ Date: ______ Comments: ______
