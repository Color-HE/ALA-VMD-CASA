# -*- coding: utf-8 -*-
"""
UNIFIED ROBUSTNESS SCRIPT
= ALA-VMD-CASA (Proposed) + Baselines (Transformer / VMD-GRU / VMD-Informer)

Added outputs:
1) R2_errors_long.csv : per-point errors for boxplots (multi-horizon)
2) CASA_attention_heatmap.csv/.npy : attention weights heatmap data (representative)

"""

import os
import math
import random
from typing import Dict, Tuple, Callable, List, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from vmdpy import VMD as VMD_PY

# your CASA + time features
from models import CASA
from utils.timefeatures import time_features


# ======================
# Global Config
# ======================
CSV_PATH = r"C:\改进itransformer\VMD-CASA\data\brent单独分解数据集.csv"
TARGET_COL = "brent"
DATE_COL_CANDIDATES = ["date", "time", "Date", "Time"]
CSV_ENCODING = "gbk"

T0 = 2500
H = 60
S = 60
HORIZONS = [1, 5, 10, 20]
SEEDS = [2024, 2025, 7, 11, 19, 23, 29, 31, 37, 41]

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Common utils
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_date(x) -> str:
    try:
        return pd.to_datetime(x).strftime("%Y-%m-%d")
    except Exception:
        return str(x)


def load_df(csv_path: str) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(csv_path, encoding=CSV_ENCODING)
    if TARGET_COL not in df.columns:
        raise ValueError(f"CSV must contain column '{TARGET_COL}'. Got columns: {df.columns.tolist()}")

    date_col = ""
    for c in DATE_COL_CANDIDATES:
        if c in df.columns:
            date_col = c
            df[c] = pd.to_datetime(df[c], errors="coerce")
            df = df.sort_values(c).reset_index(drop=True)
            break

    return df.reset_index(drop=True), date_col


def get_date_range(df: pd.DataFrame, date_col: str, start_idx: int, end_idx_exclusive: int) -> Tuple[str, str]:
    if not date_col:
        return f"idx{start_idx}", f"idx{end_idx_exclusive-1}"
    s = df.loc[start_idx, date_col]
    e = df.loc[end_idx_exclusive - 1, date_col]
    return format_date(s), format_date(e)


def cal_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100.0)
    r2 = float(r2_score(y_true, y_pred))
    return {"R2": r2, "MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE(%)": mape}


def summarize_metrics_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    def _summ(g: pd.DataFrame) -> pd.Series:
        out = {}
        for col in ["R2", "MSE", "RMSE", "MAE", "MAPE(%)"]:
            out[f"{col}_Mean"] = g[col].mean()
            out[f"{col}_Std"] = g[col].std(ddof=1)
            out[f"{col}_Median"] = g[col].median()
            out[f"{col}_Q1"] = g[col].quantile(0.25)
            out[f"{col}_Q3"] = g[col].quantile(0.75)
        return pd.Series(out)

    return df.groupby(group_col, as_index=False).apply(_summ).reset_index(drop=True)


def winrate_by_min_mse(df_windows: pd.DataFrame, window_col="Window", model_col="Model") -> pd.DataFrame:
    winners = []
    for w in sorted(df_windows[window_col].unique()):
        sub = df_windows[df_windows[window_col] == w]
        best = sub.loc[sub["MSE"].idxmin(), model_col]
        winners.append({window_col: w, "BestModel_by_MSE": best})
    df_winners = pd.DataFrame(winners)

    wr = df_winners["BestModel_by_MSE"].value_counts(normalize=True).reset_index()
    wr.columns = ["Model", "WinRate_by_MSE"]
    return wr


# ============================================================
# Part A — Baselines
# ============================================================
class SlidingWindowDatasetH(Dataset):
    def __init__(self, series_scaled_1d: np.ndarray, window: int, horizon: int):
        xs, ys = [], []
        T = len(series_scaled_1d)
        for i in range(T - window - horizon + 1):
            xs.append(series_scaled_1d[i:i + window])
            ys.append(series_scaled_1d[i + window + horizon - 1])
        xs = np.asarray(xs, dtype=np.float32)
        ys = np.asarray(ys, dtype=np.float32)
        self.x = torch.from_numpy(xs).unsqueeze(-1)
        self.y = torch.from_numpy(ys).unsqueeze(-1)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


BASE_WINDOW = 12
BASE_HIDDEN = 100
BASE_BATCH = 64
BASE_EPOCHS = 100
BASE_LR = 1e-4

N_HEADS = 4
N_LAYERS = 2
FF_DIM = 256
DROPOUT = 0.1
PROBSPARSE_FACTOR = 5

VMD_K_BASE = 6
VMD_ALPHA_BASE = 2000
VMD_TAU_BASE = 0
VMD_DC_BASE = 0
VMD_INIT_BASE = 1
VMD_TOL_BASE = 1e-7


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerRegressor(nn.Module):
    def __init__(self, d_model=BASE_HIDDEN, n_heads=N_HEADS, n_layers=N_LAYERS, ff_dim=FF_DIM, dropout=DROPOUT):
        super().__init__()
        self.in_proj = nn.Linear(1, d_model)
        self.pos = PositionalEncoding(d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):
        z = self.pos(self.in_proj(x))
        z = self.encoder(z)
        return self.head(z[:, -1, :])


class GRUBlock(nn.Module):
    def __init__(self, hidden=BASE_HIDDEN):
        super().__init__()
        self.rnn = nn.GRU(input_size=1, hidden_size=hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.fc(o[:, -1, :])


class ProbSparseAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, factor: int = 5):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.factor = factor
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x):
        B, L, D = x.shape
        return x.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

    def _merge(self, x):
        B, Hh, L, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, Hh * Dh)

    def forward(self, x):
        B, L, _ = x.shape
        Q = self._split(self.w_q(x))
        K = self._split(self.w_k(x))
        V = self._split(self.w_v(x))

        k = int(self.factor * math.log(L + 1))
        k = max(1, min(L, k))

        score_q = torch.norm(Q, dim=-1)
        topk_idx = torch.topk(score_q, k=k, dim=-1).indices

        V_mean = V.mean(dim=2, keepdim=True).expand(-1, -1, L, -1)
        out = V_mean.clone()

        Q_sel = torch.gather(Q, 2, topk_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_head))
        attn = torch.matmul(Q_sel, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out_sel = torch.matmul(attn, V)

        out = out.scatter(2, topk_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_head), out_sel)
        out = self._merge(out)
        return self.w_o(out)


class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float, factor: int):
        super().__init__()
        self.attn = ProbSparseAttention(d_model, n_heads, dropout, factor=factor)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class InformerRegressor(nn.Module):
    def __init__(self, d_model=BASE_HIDDEN, n_heads=N_HEADS, n_layers=N_LAYERS,
                 ff_dim=FF_DIM, dropout=DROPOUT, factor=PROBSPARSE_FACTOR):
        super().__init__()
        self.in_proj = nn.Linear(1, d_model)
        self.pos = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([
            InformerEncoderLayer(d_model, n_heads, ff_dim, dropout, factor=factor)
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):
        z = self.pos(self.in_proj(x))
        for layer in self.layers:
            z = layer(z)
        return self.head(z[:, -1, :])


def train_predict_one_series_direct_baseline(model: nn.Module, train_1d: np.ndarray, test_1d: np.ndarray,
                                            horizon: int) -> np.ndarray:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_1d.reshape(-1, 1)).reshape(-1)
    test_scaled = scaler.transform(test_1d.reshape(-1, 1)).reshape(-1)

    train_ds = SlidingWindowDatasetH(train_scaled, BASE_WINDOW, horizon)
    test_ds = SlidingWindowDatasetH(test_scaled, BASE_WINDOW, horizon)

    train_loader = DataLoader(train_ds, batch_size=BASE_BATCH, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=BASE_BATCH, shuffle=False, drop_last=False)

    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=BASE_LR)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(BASE_EPOCHS):
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

    model.eval()
    preds_scaled = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(DEVICE)
            preds_scaled.append(model(xb).cpu().numpy())
    preds_scaled = np.vstack(preds_scaled)
    preds = scaler.inverse_transform(preds_scaled).reshape(-1)
    return preds


def decompose_vmd_baseline(x: np.ndarray, K=VMD_K_BASE) -> np.ndarray:
    u, _, _ = VMD_PY(x, alpha=VMD_ALPHA_BASE, tau=VMD_TAU_BASE, K=K, DC=VMD_DC_BASE,
                    init=VMD_INIT_BASE, tol=VMD_TOL_BASE)
    return u


def run_vmd_plus_model_direct_baseline(train_seg: np.ndarray, test_seg: np.ndarray,
                                      base_model_factory: Callable[[], nn.Module],
                                      horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    imfs_train = decompose_vmd_baseline(train_seg)
    imfs_test = decompose_vmd_baseline(test_seg)

    n_imf = min(imfs_train.shape[0], imfs_test.shape[0])
    imfs_train = imfs_train[:n_imf]
    imfs_test = imfs_test[:n_imf]

    y_true = test_seg[BASE_WINDOW + horizon - 1:]
    pred_sum = np.zeros_like(y_true, dtype=np.float64)

    for k in range(n_imf):
        pred_k = train_predict_one_series_direct_baseline(base_model_factory(), imfs_train[k], imfs_test[k], horizon=horizon)
        pred_sum += pred_k

    return y_true, pred_sum


def run_kept_baselines_on_window(train_seg: np.ndarray, test_seg: np.ndarray, horizon: int) -> Dict[str, Dict[str, float]]:
    out = {}
    y_true = test_seg[BASE_WINDOW + horizon - 1:]

    pred = train_predict_one_series_direct_baseline(TransformerRegressor(), train_seg, test_seg, horizon=horizon)
    out["Transformer"] = cal_metrics(y_true, pred)

    yt, yp = run_vmd_plus_model_direct_baseline(train_seg, test_seg, base_model_factory=lambda: GRUBlock(), horizon=horizon)
    out["VMD-GRU"] = cal_metrics(yt, yp)

    yt, yp = run_vmd_plus_model_direct_baseline(train_seg, test_seg, base_model_factory=lambda: InformerRegressor(), horizon=horizon)
    out["VMD-Informer"] = cal_metrics(yt, yp)

    return out


def run_kept_baselines_on_window_with_preds(train_seg: np.ndarray, test_seg: np.ndarray, horizon: int):
    out = {}
    y_true = test_seg[BASE_WINDOW + horizon - 1:]

    pred = train_predict_one_series_direct_baseline(TransformerRegressor(), train_seg, test_seg, horizon=horizon)
    out["Transformer"] = (y_true.copy(), pred.copy())

    yt, yp = run_vmd_plus_model_direct_baseline(train_seg, test_seg, base_model_factory=lambda: GRUBlock(), horizon=horizon)
    out["VMD-GRU"] = (yt.copy(), yp.copy())

    yt, yp = run_vmd_plus_model_direct_baseline(train_seg, test_seg, base_model_factory=lambda: InformerRegressor(), horizon=horizon)
    out["VMD-Informer"] = (yt.copy(), yp.copy())

    return out


def baselines_R1(df: pd.DataFrame, date_col: str, series: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    win = 0
    train_end = T0
    N = len(series)

    while train_end + H <= N:
        test_end = train_end + H
        win += 1

        train_seg = series[:train_end]
        test_seg = series[train_end:test_end]
        sdate, edate = get_date_range(df, date_col, train_end, test_end)

        print(f"\n[Baselines R1] Window {win}: train=[0,{train_end}) test=[{train_end},{test_end}) horizon=1")
        res = run_kept_baselines_on_window(train_seg, test_seg, horizon=1)

        for model_name, m in res.items():
            rows.append({
                "Window": win,
                "Model": model_name,
                "TrainEndIdx": train_end,
                "TestStartIdx": train_end,
                "TestEndIdx": test_end - 1,
                "TestStartDate": sdate,
                "TestEndDate": edate,
                **m
            })

        train_end += S

    df_windows = pd.DataFrame(rows)
    df_summary = summarize_metrics_table(df_windows, group_col="Model")
    df_winrate = winrate_by_min_mse(df_windows, window_col="Window", model_col="Model")
    return df_windows, df_summary, df_winrate


def baselines_R2(df: pd.DataFrame, date_col: str, series: np.ndarray, train_end: int, test_end: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_seg = series[:train_end]
    test_seg = series[train_end:test_end]
    sdate, edate = get_date_range(df, date_col, train_end, test_end)

    if date_col:
        test_dates = pd.to_datetime(df.loc[train_end:test_end-1, date_col]).reset_index(drop=True)
    else:
        test_dates = pd.Series(np.arange(train_end, test_end))

    rows_metrics = []
    rows_err = []

    for h in HORIZONS:
        print(f"\n[Baselines R2] horizon={h} train=[0,{train_end}) test=[{train_end},{test_end})")

        res_m = run_kept_baselines_on_window(train_seg, test_seg, horizon=h)
        for model_name, m in res_m.items():
            rows_metrics.append({
                "Horizon": h,
                "Model": model_name,
                "TestStartDate": sdate,
                "TestEndDate": edate,
                **m
            })

        preds_dict = run_kept_baselines_on_window_with_preds(train_seg, test_seg, horizon=h)
        offset = BASE_WINDOW + h - 1
        idx_global = np.arange(train_end + offset, test_end)
        date_slice = test_dates.iloc[offset:].reset_index(drop=True)

        for model_name, (y_true, y_pred) in preds_dict.items():
            err = (y_true.reshape(-1) - y_pred.reshape(-1))
            L = min(len(err), len(idx_global), len(date_slice))
            for i in range(L):
                rows_err.append({
                    "Horizon": int(h),
                    "Model": model_name,
                    "Index": int(idx_global[i]),
                    "Date": format_date(date_slice.iloc[i]) if date_col else str(idx_global[i]),
                    "y_true": float(y_true[i]),
                    "y_pred": float(y_pred[i]),
                    "error": float(err[i]),
                })

    return pd.DataFrame(rows_metrics), pd.DataFrame(rows_err)


def baselines_R3(df: pd.DataFrame, date_col: str, series: np.ndarray, train_end: int, test_end: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_seg = series[:train_end]
    test_seg = series[train_end:test_end]
    sdate, edate = get_date_range(df, date_col, train_end, test_end)

    rows = []
    for sd in SEEDS:
        print(f"\n[Baselines R3] seed={sd} train=[0,{train_end}) test=[{train_end},{test_end}) horizon=1")
        set_seed(sd)
        res = run_kept_baselines_on_window(train_seg, test_seg, horizon=1)
        for model_name, m in res.items():
            rows.append({
                "Seed": sd,
                "Model": model_name,
                "TestStartDate": sdate,
                "TestEndDate": edate,
                **m
            })

    df_runs = pd.DataFrame(rows)
    df_stats = (
        df_runs
        .groupby("Model")[["R2", "MSE", "RMSE", "MAE", "MAPE(%)"]]
        .agg(["mean", "std"])
    )
    df_stats.columns = [f"{a}_{b}" for a, b in df_stats.columns]
    df_stats = df_stats.reset_index()
    return df_runs, df_stats


# ============================================================
# Part B — Proposed: ALA-VMD-CASA
# ============================================================
def VMD_np(signal, alpha, tau, K, DC=0, init=1, tol=1e-7, N_iter=500):
    f = np.asarray(signal, dtype=float)
    N = len(f)
    f_mirror = np.concatenate([f[:N//2][::-1], f, f[-N//2:][::-1]])
    T = len(f_mirror)

    freqs = np.fft.rfftfreq(T, d=1.0)
    f_hat = np.fft.rfft(f_mirror)

    u_hat = np.zeros((K, len(freqs)), dtype=np.complex128)
    omega = np.zeros(K, dtype=float)
    lam = np.zeros(len(freqs), dtype=np.complex128)

    if init == 1:
        omega = 0.5 * np.arange(K) / K
    elif init == 2:
        omega = np.sort(np.random.rand(K) * 0.5)
    else:
        omega = np.zeros(K)

    if DC:
        omega[0] = 0.0

    uDiff = tol + 1.0
    n = 0

    while (uDiff > tol) and (n < N_iter):
        u_hat_prev = u_hat.copy()

        for k in range(K):
            sum_others = np.sum(u_hat, axis=0) - u_hat[k]
            residual = f_hat - sum_others - lam / 2.0

            denom = 1.0 + 2.0 * alpha * (freqs - omega[k])**2
            u_hat[k] = residual / denom

            if not (DC and k == 0):
                power = np.abs(u_hat[k])**2
                omega[k] = np.sum(freqs * power) / (np.sum(power) + 1e-12)

        lam = lam + tau * (np.sum(u_hat, axis=0) - f_hat)
        uDiff = np.mean([np.linalg.norm(u_hat[k] - u_hat_prev[k])**2 for k in range(K)])
        n += 1

    u = np.zeros((K, T), dtype=float)
    for k in range(K):
        u[k] = np.fft.irfft(u_hat[k], n=T)

    u = u[:, N//2: N//2 + N]
    return u


def seed_everything(seed: int = 2024):
    set_seed(seed)


def build_loader_univariate_with_index(window, pred_len, batch_size, data, data_mark, global_start_index, shuffle):
    seq_len = window
    label_len = int(window / 2)

    data = np.asarray(data)
    data_mark = np.asarray(data_mark)

    M = len(data)
    num = M - (seq_len + pred_len) + 1
    if num <= 0:
        raise ValueError("样本长度不足，请减小 window/pred_len 或增加数据量。")

    xs, ys, xms, yms, idxs = [], [], [], [], []
    for i in range(num):
        x_seq = data[i: i + seq_len]
        x_mark = data_mark[i: i + seq_len]

        y_start = i + seq_len - label_len
        y_end = i + seq_len + pred_len
        y_seq = data[y_start: y_end]
        y_mark = data_mark[y_start: y_end]

        pred_global_idx = global_start_index + (i + seq_len + pred_len - 1)

        xs.append(x_seq)
        ys.append(y_seq)
        xms.append(x_mark)
        yms.append(y_mark)
        idxs.append(pred_global_idx)

    x_temp = torch.tensor(np.array(xs), dtype=torch.float32)
    y_temp = torch.tensor(np.array(ys), dtype=torch.float32)
    x_temp_mark = torch.tensor(np.array(xms), dtype=torch.float32)
    y_temp_mark = torch.tensor(np.array(yms), dtype=torch.float32)
    idx_temp = torch.tensor(np.array(idxs), dtype=torch.long)

    ds = TensorDataset(x_temp, y_temp, x_temp_mark, y_temp_mark, idx_temp)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return loader


def make_decoder_input(y, label_len, device):
    dec_inp = torch.zeros_like(y).to(device)
    dec_inp[:, :label_len, :] = y[:, :label_len, :].to(device)
    return dec_inp


def model_train(net, train_loader, pred_len, optimizer, criterion, num_epochs, device, print_train=False):
    train_loss = []
    print_frequency = max(1, num_epochs // 10)

    for epoch in range(num_epochs):
        net.train()
        total = 0.0

        for x, y, x_mark, y_mark, _idx in train_loader:
            x = x.to(device)
            y = y.to(device)
            x_mark = x_mark.to(device)
            y_mark = y_mark.to(device)

            optimizer.zero_grad()
            out = net(x, x_mark, y, y_mark, None)  # teacher forcing

            preds = out[0] if (isinstance(out, (tuple, list))) else out
            preds = preds[:, -pred_len:, :].squeeze()
            y_true = y[:, -pred_len:, :].squeeze()

            loss = criterion(preds, y_true)
            loss.backward()
            optimizer.step()
            total += loss.item()

        avg = total / len(train_loader)
        train_loss.append(avg)

        if print_train and ((epoch + 1) % print_frequency == 0 or (epoch + 1) == num_epochs):
            print(f"Epoch: {epoch + 1}, Train Loss: {avg:.6f}")

    return net, train_loss


@torch.no_grad()
def model_predict_batch(net, x, x_mark, y, y_mark, label_len, pred_len, device, return_attn=False):
    net.eval()
    x = x.to(device)
    x_mark = x_mark.to(device)
    y = y.to(device)
    y_mark = y_mark.to(device)

    dec_inp = make_decoder_input(y, label_len, device)
    out = net(x, x_mark, dec_inp, y_mark, None)

    attn_list = None
    if isinstance(out, (tuple, list)) and len(out) == 2:
        pred, attn_list = out[0], out[1]
    else:
        pred = out

    pred = pred[:, -pred_len:, :].detach().cpu().numpy()
    true = y[:, -pred_len:, :].detach().cpu().numpy()

    if return_attn:
        return true, pred, attn_list
    return true, pred


def cal_eval_proposed(y_real, y_pred):
    y_real = np.array(y_real).ravel()
    y_pred = np.array(y_pred).ravel()
    r2 = r2_score(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_real, y_pred)
    mape = np.mean(np.abs((y_real - y_pred) / (y_real + 1e-8))) * 100
    return {"R2": float(r2), "MSE": float(mse), "RMSE": float(rmse), "MAE": float(mae), "MAPE(%)": float(mape)}


class CASAConfig:
    def __init__(self, window, pred_len, batch_size, num_epochs, lr, data_dim):
        self.seq_len = window
        self.label_len = int(window / 2)
        self.pred_len = pred_len
        self.freq = 'b'

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = lr
        self.stop_ratio = 0.2

        self.dec_in = data_dim
        self.enc_in = data_dim
        self.c_out = 1

        self.d_model = 64
        self.n_heads = 8
        self.dropout = 0.1
        self.e_layers = 2
        self.d_layers = 1
        self.d_ff = 64
        self.factor = 5
        self.activation = 'gelu'
        self.channel_independence = 0

        self.patch_len = 16
        self.stride = 8
        self.top_k = 5
        self.num_kernels = 6

        self.embed = 'timeF'
        # ✅ IMPORTANT: now CASA.py supports returning attention list when this=1
        self.output_attention = 1
        self.distil = 1
        self.task_name = 'short_term_forecast'

        self.kernel = 3


def run_ala_vmd_casa_one_window(series, data_stamp, dates,
                               K, alpha, tau, DC,
                               window, pred_len,
                               train_end, test_end,
                               batch_size, num_epochs, learning_rate,
                               device,
                               vmd_cache=None,
                               print_train=False):
    N = len(series)
    assert 0 < train_end < test_end <= N

    if vmd_cache is None:
        imfs = VMD_np(series, alpha=alpha, tau=tau, K=K, DC=DC, init=1, tol=1e-7, N_iter=1000)
    else:
        imfs = vmd_cache

    true_idx = np.arange(train_end, test_end, dtype=int)
    brent_true = series[true_idx].reshape(-1, 1)

    seg_start = train_end - window - pred_len + 1
    if seg_start < 0:
        seg_start = 0
    seg_end = test_end

    eval_mark = data_stamp[seg_start:seg_end]
    pred_sum = np.zeros_like(brent_true, dtype=float)

    # store one representative attention heatmap (N x d_model)
    attn_heatmap_saved: Optional[np.ndarray] = None

    for k in range(K):
        sig = imfs[k].reshape(-1, 1)

        scaler = MinMaxScaler()
        sig_scaled = sig.copy()
        sig_scaled[:train_end] = scaler.fit_transform(sig[:train_end])
        sig_scaled[train_end:] = scaler.transform(sig[train_end:])

        train_seg = sig_scaled[:train_end]
        train_mark = data_stamp[:train_end]
        train_loader = build_loader_univariate_with_index(
            window, pred_len, batch_size,
            train_seg, train_mark,
            global_start_index=0,
            shuffle=True
        )

        eval_seg = sig_scaled[seg_start:seg_end]
        eval_loader = build_loader_univariate_with_index(
            window, pred_len, batch_size,
            eval_seg, eval_mark,
            global_start_index=seg_start,
            shuffle=False
        )

        config = CASAConfig(window, pred_len, batch_size, num_epochs, learning_rate, data_dim=1)
        net = CASA.Model(config).to(device)
        criterion = nn.MSELoss().to(device)
        optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)

        net, _ = model_train(net, train_loader, pred_len, optimizer, criterion,
                             num_epochs=config.num_epochs, device=device, print_train=print_train)

        pred_pairs = []
        for bx, by, bxm, bym, idx in eval_loader:
            if attn_heatmap_saved is None:
                _, p, attn_list = model_predict_batch(net, bx, bxm, by, bym, config.label_len, pred_len, device, return_attn=True)

                # attn_list: list of tensors, each (B, N, d_model) after your CASA structure
                if attn_list is not None and isinstance(attn_list, list) and len(attn_list) > 0:
                    try:
                        last_attn = attn_list[-1]                 # (B,N,d_model)
                        a = last_attn[0].detach().cpu().numpy()   # take first sample -> (N,d_model)
                        attn_heatmap_saved = a
                    except Exception:
                        attn_heatmap_saved = None
            else:
                _, p = model_predict_batch(net, bx, bxm, by, bym, config.label_len, pred_len, device, return_attn=False)

            p_last = p[:, -1, 0].reshape(-1, 1)
            pred_pairs.append((idx.numpy().astype(int), p_last))

        idx_all = np.concatenate([a for a, _ in pred_pairs], axis=0)
        pred_all = np.concatenate([b for _, b in pred_pairs], axis=0).reshape(-1, 1)

        mask = (idx_all >= train_end) & (idx_all < test_end)
        idx_keep = idx_all[mask]
        pred_keep = pred_all[mask]

        order = np.argsort(idx_keep)
        idx_keep = idx_keep[order]
        pred_keep = pred_keep[order]

        if not np.array_equal(idx_keep, true_idx):
            mapping = {int(i): float(v) for i, v in zip(idx_keep, pred_keep.ravel())}
            pred_keep_aligned = np.array([mapping[int(i)] for i in true_idx], dtype=float).reshape(-1, 1)
        else:
            pred_keep_aligned = pred_keep

        y_pred_inv = scaler.inverse_transform(pred_keep_aligned)
        pred_sum += y_pred_inv

        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    m = cal_eval_proposed(brent_true, pred_sum)
    date_slice = pd.to_datetime(dates.iloc[true_idx]).reset_index(drop=True)
    return m, brent_true, pred_sum, date_slice, imfs, attn_heatmap_saved


CASA_WINDOW = 10
CASA_PREDLEN_MAIN = 1
CASA_BATCH = 64
CASA_EPOCHS = 20
CASA_LR = 0.001

VMD_K_PROP = 3
VMD_ALPHA_PROP = 500
VMD_TAU_PROP = 0.0
VMD_DC_PROP = 0


def proposed_R2(series, data_stamp, dates, train_end: int, test_end: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows_metrics = []
    rows_err = []

    imfs_cache = VMD_np(series, alpha=VMD_ALPHA_PROP, tau=VMD_TAU_PROP, K=VMD_K_PROP, DC=VMD_DC_PROP,
                       init=1, tol=1e-7, N_iter=1000)

    for h in HORIZONS:
        print(f"\n[Proposed R2] pred_len={h} train=[0,{train_end}) test=[{train_end},{test_end})")
        m, y_true, y_pred, d_slice, _, _ = run_ala_vmd_casa_one_window(
            series, data_stamp, dates,
            VMD_K_PROP, VMD_ALPHA_PROP, VMD_TAU_PROP, VMD_DC_PROP,
            CASA_WINDOW, pred_len=h,
            train_end=train_end, test_end=test_end,
            batch_size=CASA_BATCH, num_epochs=CASA_EPOCHS, learning_rate=CASA_LR,
            device=DEVICE,
            vmd_cache=imfs_cache,
            print_train=False
        )

        rows_metrics.append({
            "Horizon": h,
            "Model": "ALA-VMD-CASA",
            "TestStartDate": format_date(d_slice.iloc[0]),
            "TestEndDate": format_date(d_slice.iloc[-1]),
            **m
        })

        err = (y_true.reshape(-1) - y_pred.reshape(-1))
        idx_global = np.arange(train_end, test_end)
        L = min(len(err), len(idx_global), len(d_slice))
        for i in range(L):
            rows_err.append({
                "Horizon": int(h),
                "Model": "ALA-VMD-CASA",
                "Index": int(idx_global[i]),
                "Date": format_date(d_slice.iloc[i]),
                "y_true": float(y_true[i]),
                "y_pred": float(y_pred[i]),
                "error": float(err[i]),
            })

    return pd.DataFrame(rows_metrics), pd.DataFrame(rows_err)


# ============================================================
# MAIN
# ============================================================
def main():
    set_seed(SEED)
    print("Device:", DEVICE)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df, date_col = load_df(CSV_PATH)
    series = df[TARGET_COL].astype(float).values
    N = len(series)

    fit_size = int(N * 0.8)
    train_end_common = fit_size
    test_end_common = N

    if date_col:
        df_stamp = df[[date_col]].copy()
        df_stamp[date_col] = pd.to_datetime(df_stamp[date_col])
    else:
        df_stamp = pd.DataFrame({"date": pd.date_range("2000-01-01", periods=N, freq="B")})
        date_col = "date"

    data_stamp = time_features(df_stamp, timeenc=1, freq="B")
    dates = df[date_col] if date_col in df.columns else df_stamp[date_col]

    out_root = "robustness_outputs_unified"
    out_base = os.path.join(out_root, "baselines_kept")
    out_prop = os.path.join(out_root, "proposed_ala_vmd_casa")
    out_comb = os.path.join(out_root, "combined")
    os.makedirs(out_base, exist_ok=True)
    os.makedirs(out_prop, exist_ok=True)
    os.makedirs(out_comb, exist_ok=True)

    # =========================
    # R2 — Multi-horizon (common split)
    # =========================
    df_r2_base, df_r2_base_err = baselines_R2(df, date_col, series, train_end_common, test_end_common)
    df_r2_prop, df_r2_prop_err = proposed_R2(series, data_stamp, dates, train_end_common, test_end_common)
    df_r2_all = pd.concat([df_r2_base, df_r2_prop], ignore_index=True)

    df_r2_base.to_csv(os.path.join(out_base, "R2_multi_horizon_baselines.csv"), index=False, encoding="utf-8-sig")
    df_r2_prop.to_csv(os.path.join(out_prop, "R2_multi_horizon_proposed.csv"), index=False, encoding="utf-8-sig")
    df_r2_all.to_csv(os.path.join(out_comb, "R2_multi_horizon_ALL.csv"), index=False, encoding="utf-8-sig")

    # ✅ boxplot data
    df_r2_err_all = pd.concat([df_r2_base_err, df_r2_prop_err], ignore_index=True)
    df_r2_err_all.to_csv(os.path.join(out_comb, "R2_errors_long.csv"), index=False, encoding="utf-8-sig")
    print(f"[OK] Boxplot error data saved: {os.path.join(out_comb, 'R2_errors_long.csv')}")

    # =========================
    # EXTRA — Export CASA attention heatmap (representative)
    # =========================
    print("\n[Export] CASA attention heatmap (representative)")
    m, y_true, y_pred, d_slice, imfs, attn_heatmap = run_ala_vmd_casa_one_window(
        series, data_stamp, dates,
        VMD_K_PROP, VMD_ALPHA_PROP, VMD_TAU_PROP, VMD_DC_PROP,
        CASA_WINDOW, pred_len=1,
        train_end=train_end_common, test_end=test_end_common,
        batch_size=CASA_BATCH, num_epochs=CASA_EPOCHS, learning_rate=CASA_LR,
        device=DEVICE,
        vmd_cache=None,
        print_train=False
    )

    if attn_heatmap is not None and isinstance(attn_heatmap, np.ndarray) and attn_heatmap.ndim == 2:
        # rows: channel index, cols: latent position (d_model)
        attn_csv = os.path.join(out_comb, "CASA_attention_heatmap.csv")
        attn_npy = os.path.join(out_comb, "CASA_attention_heatmap.npy")
        pd.DataFrame(attn_heatmap).to_csv(attn_csv, index=False, encoding="utf-8-sig")
        np.save(attn_npy, attn_heatmap)
        print(f"[OK] Attention heatmap saved: {attn_csv}")
    else:
        print("[WARN] Attention heatmap not available. Please confirm CASAConfig.output_attention=1 and updated CASA.py is used.")

    print(f"\nAll outputs saved to: {out_root}")
    print(f"Boxplot data: {os.path.join(out_comb, 'R2_errors_long.csv')}")
    print(f"Attention data: {os.path.join(out_comb, 'CASA_attention_heatmap.csv')}")


if __name__ == "__main__":
    main()
