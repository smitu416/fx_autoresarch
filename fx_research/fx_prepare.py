"""
One-time data preparation for FX autoresearch experiments.
Downloads USD/JPY daily OHLCV data from Yahoo Finance (2015-present),
computes features, normalizes, and provides fixed evaluation metric.

Fixed constants and evaluation metric — do not modify.

Usage:
    python fx_prepare.py
"""

import os
import math

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

LOOKBACK   = 60         # input sequence length (trading days)
TIME_BUDGET = 300       # training time budget in seconds (5 minutes)
TICKER     = "USDJPY=X"
DATA_START = "2015-01-01"
VAL_START  = "2023-01-01"   # held-out validation period

N_FEATURES = 10  # number of features per timestep (see compute_features)

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "fx_autoresearch")
DATA_FILE  = os.path.join(CACHE_DIR, "usdjpy.parquet")

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_data():
    """Download USD/JPY daily data from Yahoo Finance and cache as parquet."""
    import yfinance as yf
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(DATA_FILE):
        print(f"Data: already cached at {DATA_FILE}")
        return
    print(f"Data: downloading {TICKER} from {DATA_START}...")
    df = yf.download(TICKER, start=DATA_START, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"Failed to download data for {TICKER}")
    df.to_parquet(DATA_FILE)
    print(f"Data: saved {len(df)} rows to {DATA_FILE}")

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame):
    """
    Compute 10 features from OHLC DataFrame (Volume is ignored for FX).

    Features:
        0  log_ret    — daily log return
        1  hl_range   — log(high/low) intraday range
        2  co_change  — log(close/open)
        3  ma5_dev    — close / 5-day MA - 1
        4  ma20_dev   — close / 20-day MA - 1
        5  ma60_dev   — close / 60-day MA - 1
        6  rvol       — 20-day realized volatility of log returns
        7  rsi_norm   — RSI(14) normalized to [-1, 1]
        8  dow_sin    — day-of-week cyclical sine
        9  dow_cos    — day-of-week cyclical cosine

    Target: next-day log return (shifted -1).

    Returns:
        feat   : np.ndarray (T, N_FEATURES) float32
        target : np.ndarray (T,)            float32 — next-day log return
        dates  : pd.DatetimeIndex           (T,)
    """
    # Flatten MultiIndex columns that yfinance sometimes returns
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].astype(float)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    open_ = df["Open"].astype(float)

    log_ret   = np.log(close / close.shift(1))
    hl_range  = np.log(high / low)
    co_change = np.log(close / open_)

    ma5_dev  = close / close.rolling(5).mean()  - 1.0
    ma20_dev = close / close.rolling(20).mean() - 1.0
    ma60_dev = close / close.rolling(60).mean() - 1.0

    rvol = log_ret.rolling(20).std()

    # RSI(14): normalize to [-1, 1] (raw RSI is in [0, 1])
    gain = log_ret.clip(lower=0).rolling(14).mean()
    loss = (-log_ret.clip(upper=0)).rolling(14).mean()
    rs       = gain / loss.replace(0.0, np.nan)
    rsi_raw  = (rs / (1 + rs)).fillna(1.0)  # all-gain windows → RSI=1
    rsi_norm = 2.0 * rsi_raw - 1.0

    dow     = pd.Series(df.index.dayofweek, index=df.index, dtype=float)
    dow_sin = np.sin(2.0 * np.pi * dow / 5.0)
    dow_cos = np.cos(2.0 * np.pi * dow / 5.0)

    features = pd.DataFrame({
        "log_ret":   log_ret,
        "hl_range":  hl_range,
        "co_change": co_change,
        "ma5_dev":   ma5_dev,
        "ma20_dev":  ma20_dev,
        "ma60_dev":  ma60_dev,
        "rvol":      rvol,
        "rsi_norm":  rsi_norm,
        "dow_sin":   dow_sin,
        "dow_cos":   dow_cos,
    })

    # Target: next-day log return
    target = log_ret.shift(-1).rename("target")

    combined = features.join(target).dropna()

    feat_arr   = combined[features.columns].values.astype(np.float32)
    target_arr = combined["target"].values.astype(np.float32)
    dates      = combined.index

    return feat_arr, target_arr, dates

# ---------------------------------------------------------------------------
# Data loading (called by train.py)
# ---------------------------------------------------------------------------

def load_data():
    """
    Load, engineer features, normalize (train stats), and build sequences.

    Normalization uses training-set mean/std only (no data leakage).

    Returns:
        x_train : torch.Tensor (N_train, LOOKBACK, N_FEATURES)
        y_train : torch.Tensor (N_train,)
        x_val   : torch.Tensor (N_val,   LOOKBACK, N_FEATURES)
        y_val   : torch.Tensor (N_val,)
    """
    df = pd.read_parquet(DATA_FILE)

    feat_arr, target_arr, dates = compute_features(df)

    val_mask = dates >= pd.Timestamp(VAL_START)
    train_feat = feat_arr[~val_mask]
    train_ret  = target_arr[~val_mask]
    val_feat   = feat_arr[val_mask]
    val_ret    = target_arr[val_mask]

    # Normalize using train statistics only
    feat_mean = train_feat.mean(axis=0)
    feat_std  = train_feat.std(axis=0) + 1e-8
    train_feat_norm = (train_feat - feat_mean) / feat_std
    val_feat_norm   = (val_feat   - feat_mean) / feat_std

    def make_sequences(feat, ret):
        T = len(feat)
        xs = np.stack([feat[i - LOOKBACK:i] for i in range(LOOKBACK, T)])
        ys = ret[LOOKBACK:]
        return torch.from_numpy(xs), torch.from_numpy(ys)

    x_train, y_train = make_sequences(train_feat_norm, train_ret)
    x_val,   y_val   = make_sequences(val_feat_norm,   val_ret)

    return x_train, y_train, x_val, y_val

# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------

def make_dataloader(x: torch.Tensor, y: torch.Tensor, batch_size: int):
    """
    Infinite random-batch iterator over (x, y) tensors.

    Yields:
        x_batch : (batch_size, LOOKBACK, N_FEATURES)
        y_batch : (batch_size,)
    """
    N = len(x)
    indices = torch.randperm(N)
    pos = 0
    while True:
        if pos + batch_size > N:
            indices = torch.randperm(N)
            pos = 0
        idx = indices[pos:pos + batch_size]
        pos += batch_size
        yield x[idx], y[idx]

# ---------------------------------------------------------------------------
# Evaluation metric (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_sharpe(model, x_val: torch.Tensor, y_val: torch.Tensor,
                    batch_size: int = 512) -> float:
    """
    Annualized Sharpe ratio on the validation set.

    Strategy:
        signal[t] = model(x_val[t])       — scalar, any magnitude
        daily_pnl[t] = signal[t] * y_val[t]   — proportional P&L

    Annualized Sharpe = mean(daily_pnl) / std(daily_pnl) * sqrt(252)
    Higher is better. Returns 0.0 if std is near zero.
    """
    model.eval()
    all_signals = []
    for start in range(0, len(x_val), batch_size):
        xb = x_val[start:start + batch_size]
        sig = model(xb).squeeze(-1)
        all_signals.append(sig.cpu().float())

    signals = torch.cat(all_signals)          # (N_val,)
    returns = y_val.cpu().float()             # (N_val,)

    daily_pnl = signals * returns

    mean_pnl = daily_pnl.mean().item()
    std_pnl  = daily_pnl.std().item()

    if std_pnl < 1e-10:
        return 0.0

    return mean_pnl / std_pnl * math.sqrt(252)

# ---------------------------------------------------------------------------
# Main (one-time setup)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    download_data()

    print("Verifying data and features...")
    x_train, y_train, x_val, y_val = load_data()

    print(f"Train sequences : {len(x_train):,}  ({len(x_train)} trading days)")
    print(f"Val   sequences : {len(x_val):,}   (from {VAL_START})")
    print(f"Sequence shape  : {tuple(x_train.shape[1:])}  (lookback={LOOKBACK}, features={N_FEATURES})")
    print(f"Val return mean : {y_val.mean():.6f},  std: {y_val.std():.6f}")
    print()
    print("Done! Ready to train.")
