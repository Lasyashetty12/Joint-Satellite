"""Generate the same deterministic synthetic telemetry for Member 2."""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "artifacts"
SEED = 42

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    cadence_seconds, days = 30, 90
    n = int(days * 24 * 60 * 60 / cadence_seconds)
    t = np.arange(n)
    ts = pd.date_range("2026-01-01", periods=n, freq=f"{cadence_seconds}s", tz="UTC")
    orbit_samples = int(95 * 60 / cadence_seconds)
    phase = 2 * np.pi * (t % orbit_samples) / orbit_samples
    eclipse = (np.sin(phase) < -0.3).astype(float)
    solar_i = np.clip(np.maximum(0, 5*np.sin(phase))*(1-eclipse)+rng.normal(0,.08,n), 0, None)
    soc = np.clip(78+12*np.sin(phase+.8)-10*eclipse+rng.normal(0,1,n), 5, 100)
    batt_temp = 24+5*eclipse+1.5*np.sin(phase)+rng.normal(0,.4,n)
    panel_temp = 20+13*np.sin(phase)+rng.normal(0,.8,n)
    bus_v = 12.2+.15*np.sin(phase)+.02*(soc-75)+rng.normal(0,.035,n)
    batt_i = 1.8+.7*eclipse+.15*np.sin(phase+1)+rng.normal(0,.08,n)
    att_err = np.abs(rng.normal(.15,.06,n))
    payload_p = 8+1*(1-eclipse)+rng.normal(0,.25,n)
    df = pd.DataFrame({"timestamp":ts,"batt_temp":batt_temp,"panel_temp":panel_temp,"bus_V":bus_v,
                       "batt_I":batt_i,"solar_I":solar_i,"SoC":soc,"att_err":att_err,
                       "payload_P":payload_p,"eclipse":eclipse,"health_anomaly":np.zeros(n,dtype=int)})
    events=[(int(n*.25),int(n*.25)+40,"voltage"),(int(n*.50),int(n*.50)+50,"thermal"),
            (int(n*.70),int(n*.70)+60,"stuck"),(int(n*.82),int(n*.82)+80,"drift")]
    for start,end,kind in events:
        end=min(end,n); df.loc[start:end-1,"health_anomaly"]=1
        if kind=="voltage": df.loc[start:end-1,"bus_V"]-=1.8
        elif kind=="thermal": df.loc[start:end-1,"batt_temp"]+=12
        elif kind=="stuck": df.loc[start:end-1,"att_err"]=df.loc[start-1,"att_err"]
        elif kind=="drift": df.loc[start:end-1,"payload_P"]+=np.linspace(.1,4.0,end-start)
    path=OUT/"m2_telemetry.csv"; df.to_csv(path,index=False)
    print(f"Saved {len(df):,} rows to {path}")
    print("NOTE: synthetic telemetry; anomaly labels represent injected events.")

if __name__=="__main__": main()
