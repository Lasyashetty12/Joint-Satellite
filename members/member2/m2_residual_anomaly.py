"""Residual-based anomaly score for Member 2 LSTM/GRU forecasts."""
from pathlib import Path
import numpy as np
import pandas as pd

ART=Path(__file__).resolve().parent/"artifacts"

def main():
    found=0
    for kind in ("lstm","gru"):
        path=ART/f"m2_{kind}_forecast_results.csv"
        if not path.exists(): continue
        df=pd.read_csv(path); residual=np.abs(df.actual_bus_V-df.predicted_bus_V)
        split=max(1,len(df)//2); scale=max(float(residual.iloc[:split].std(ddof=0)),1e-8)
        score=residual/scale; threshold=float(score.iloc[:split].quantile(.99))
        df["abs_residual"]=residual; df["normalized_score"]=score
        df["ewma_score"]=score.ewm(span=10,adjust=False).mean()
        df["threshold_demo"]=threshold; df["predicted_anomaly_demo"]=(df.ewma_score>threshold).astype(int)
        df.to_csv(ART/f"m2_{kind}_anomaly_scores.csv",index=False); found+=1
        print(f"{kind.upper()}: saved anomaly scores; demo threshold={threshold:.4f}")
    if not found: raise SystemExit("Run m2_sequence_forecast.py first.")
    print("For final research, calibrate threshold on a separate validation set and evaluate event-wise metrics.")

if __name__=="__main__": main()
