"""Small PyTorch LSTM/GRU one-step forecasting baselines for Member 2."""
from pathlib import Path
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE = Path(__file__).resolve().parent
ART = BASE / "artifacts"
DATA = ART / "m2_telemetry.csv"
FEATURES = ["bus_V", "batt_temp", "batt_I", "solar_I", "SoC", "eclipse"]
TARGET_IDX = FEATURES.index("bus_V")
WINDOW = 20
SEED = 42

def make_sequences(values, window):
    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i-window:i])
        y.append(values[i, TARGET_IDX])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

class SeqModel(nn.Module):
    def __init__(self, kind, input_size, hidden=32):
        super().__init__()
        layer = nn.LSTM if kind == "LSTM" else nn.GRU
        self.rnn = layer(input_size, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)

def smape(y,p):
    return float(np.mean(2*np.abs(y-p)/np.maximum(np.abs(y)+np.abs(p),1e-8))*100)

def main():
    if not DATA.exists(): raise SystemExit("Telemetry CSV missing. Run m2_generate_telemetry.py first.")
    torch.manual_seed(SEED); np.random.seed(SEED)
    df=pd.read_csv(DATA)
    values=df[FEATURES].to_numpy(dtype=np.float32)
    n=len(values); tr_end=int(n*.60); val_end=int(n*.80)
    scaler=StandardScaler().fit(values[:tr_end])
    scaled=scaler.transform(values).astype(np.float32)
    X,y=make_sequences(scaled,WINDOW)
    # Sequence target corresponds to original row WINDOW onward; split chronologically.
    target_rows=np.arange(WINDOW,n)
    train_mask=target_rows<tr_end
    val_mask=(target_rows>=tr_end)&(target_rows<val_end)
    test_mask=target_rows>=val_end
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results=[]
    for kind in ("LSTM","GRU"):
        model=SeqModel(kind,len(FEATURES)).to(device)
        opt=torch.optim.Adam(model.parameters(),lr=1e-3)
        loss_fn=nn.MSELoss()
        train_ds=TensorDataset(torch.from_numpy(X[train_mask]), torch.from_numpy(y[train_mask]))
        val_ds=TensorDataset(torch.from_numpy(X[val_mask]), torch.from_numpy(y[val_mask]))
        train_loader=DataLoader(train_ds, batch_size=512, shuffle=False)
        val_loader=DataLoader(val_ds, batch_size=1024, shuffle=False)
        best_state=copy.deepcopy(model.state_dict()); best_val=float("inf"); patience=0
        for epoch in range(8):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(); loss=loss_fn(model(xb),yb); loss.backward(); opt.step()
            model.eval(); val_sum=0.0; val_count=0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    batch_loss=loss_fn(model(xb),yb)
                    val_sum += float(batch_loss.item()) * len(yb); val_count += len(yb)
            val=val_sum/max(val_count,1)
            if val<best_val:
                best_val=val; best_state=copy.deepcopy(model.state_dict()); patience=0
            else: patience+=1
            if patience>=2: break
        model.load_state_dict(best_state); model.eval()
        test_ds=TensorDataset(torch.from_numpy(X[test_mask]))
        test_loader=DataLoader(test_ds, batch_size=1024, shuffle=False)
        pred_parts=[]
        with torch.no_grad():
            for (xb,) in test_loader:
                pred_parts.append(model(xb.to(device)).cpu().numpy())
        pred_scaled=np.concatenate(pred_parts)
        # Invert only the target feature's standardization.
        pred=pred_scaled*scaler.scale_[TARGET_IDX]+scaler.mean_[TARGET_IDX]
        actual=y[test_mask]*scaler.scale_[TARGET_IDX]+scaler.mean_[TARGET_IDX]
        results.append({"model":kind,"target":"bus_V","MAE":float(mean_absolute_error(actual,pred)),
                        "RMSE":float(np.sqrt(mean_squared_error(actual,pred))),"sMAPE_percent":smape(actual,pred),
                        "test_rows":len(actual),"device":str(device)})
        pd.DataFrame({"row":target_rows[test_mask],"actual_bus_V":actual,"predicted_bus_V":pred}).to_csv(ART/f"m2_{kind.lower()}_forecast_results.csv",index=False)
    pd.DataFrame(results).to_csv(ART/"m2_sequence_forecast_metrics.csv",index=False)
    print(pd.DataFrame(results).to_string(index=False))
    print(f"Saved forecast outputs in {ART}")

if __name__=="__main__": main()
