"""
Data preparation for FX autoresearch — 4-hour swing trading.
Downloads USD/JPY 1-hour data from Yahoo Finance, resamples to 4h OHLCV.
Fixed constants and TP/SL backtest evaluation metric — do not modify.

Trading sessions (entry/exit only during these windows):
  JST AM  6:00–9:00  = UTC 21:00–00:00  → 4h bar starting UTC 20:00
  JST PM 22:00–00:00 = UTC 13:00–15:00  → 4h bar starting UTC 12:00

Usage: python fx_prepare.py
"""

import os
import math
from collections import namedtuple

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

LOOKBACK    = 30    # input window: 30 × 4h ≈ 5 trading days
TIME_BUDGET = 300   # training time budget in seconds (5 minutes)
N_FEATURES  = 12    # features per timestep
MAX_HOLD    = 12    # max candles to hold trade before force-close (12 × 4h = 48h)

# UTC bar-open hours that correspond to the JST trading windows
# JST 6–9 AM  → UTC 21–24 (prev day)  → 4h bar opening at UTC 20:00
# JST 22–24   → UTC 13–15             → 4h bar opening at UTC 12:00
TRADING_BAR_UTC = {20, 12}   # only bars whose UTC hour is in this set are tradeable

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "fx_autoresearch")
DATA_FILE  = os.path.join(CACHE_DIR, "usdjpy_4h.parquet")
VAL_FRAC   = 0.2   # last 20 % of data held out for validation

# Bundle of validation data returned by load_data()
ValData = namedtuple("ValData", [
    "x",            # (N, LOOKBACK, N_FEATURES)  float32 — feature sequences
    "y",            # (N,)                        float32 — next-candle log return
    "close",        # (N,)                        float32 — entry close price
    "atr",          # (N,)                        float32 — ATR(14) at entry candle
    "future_ohlc",  # (N, MAX_HOLD, 3)            float32 — [high, low, close] of next candles
    "tradeable",    # (N,)                        bool    — True if candle in trading window
])

# ---------------------------------------------------------------------------
# Download and resample
# ---------------------------------------------------------------------------

def download_data():
    """
    Download USD/JPY 1h OHLCV from Yahoo Finance (max available ≈ 730 days),
    resample to 4h, and cache as parquet.
    """
    import yfinance as yf
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(DATA_FILE):
        print(f"Data: already cached at {DATA_FILE}")
        return

    print("Data: downloading USDJPY=X 1h (max available ~730 days)...")
    df = yf.download("USDJPY=X", period="max", interval="1h",
                     auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError("Failed to download USDJPY=X data from Yahoo Finance")

    # Flatten MultiIndex columns that yfinance sometimes returns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Resample 1h → 4h (UTC-anchored at midnight: bars at 0,4,8,12,16,20 UTC)
    df_4h = (
        df.resample("4h")
        .agg({"Open": "first", "High": "max", "Low": "min",
              "Close": "last", "Volume": "sum"})
        .dropna(subset=["Open", "High", "Low", "Close"])
    )

    df_4h.to_parquet(DATA_FILE)
    print(f"Data: saved {len(df_4h)} 4h candles  "
          f"({df_4h.index[0].date()} – {df_4h.index[-1].date()})")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _compute_features_full(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 12 features + target + raw ATR aligned to df's index.
    Rows with insufficient history will contain NaN (filtered later).

    Features (N_FEATURES = 12):
        0  log_ret    daily 4h log return
        1  hl_range   log(high/low) intraday range
        2  co_change  log(close/open)
        3  atr_norm   ATR(14) / close  (normalized ATR)
        4  ma10_dev   close / MA(10) − 1  (≈ 1.7 trading days)
        5  ma30_dev   close / MA(30) − 1  (≈ 5 days)
        6  ma60_dev   close / MA(60) − 1  (≈ 10 days)
        7  rvol       20-period realized volatility of log returns
        8  rsi_norm   RSI(14) normalized to [−1, 1]
        9  hour_sin   UTC hour cyclical sine
        10 hour_cos   UTC hour cyclical cosine
        11 dow_sin    day-of-week cyclical sine

    Extra columns (not features, used in simulation):
        target      next-candle log return
        atr         raw ATR(14) for TP/SL sizing
    """
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

    # ATR(14) — average true range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr14    = tr.rolling(14).mean()
    atr_norm = atr14 / close

    ma10_dev = close / close.rolling(10).mean() - 1.0
    ma30_dev = close / close.rolling(30).mean() - 1.0
    ma60_dev = close / close.rolling(60).mean() - 1.0

    rvol = log_ret.rolling(20).std()

    gain     = log_ret.clip(lower=0).rolling(14).mean()
    loss     = (-log_ret.clip(upper=0)).rolling(14).mean()
    rs       = gain / loss.replace(0.0, np.nan)
    rsi      = (rs / (1 + rs)).fillna(1.0)
    rsi_norm = 2.0 * rsi - 1.0

    # Cyclical time encodings
    hour = pd.Series(df.index.hour,      index=df.index, dtype=float)
    dow  = pd.Series(df.index.dayofweek, index=df.index, dtype=float)
    hour_sin = np.sin(2.0 * np.pi * hour / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hour / 24.0)
    dow_sin  = np.sin(2.0 * np.pi * dow  / 5.0)

    result = pd.DataFrame({
        "log_ret":   log_ret,
        "hl_range":  hl_range,
        "co_change": co_change,
        "atr_norm":  atr_norm,
        "ma10_dev":  ma10_dev,
        "ma30_dev":  ma30_dev,
        "ma60_dev":  ma60_dev,
        "rvol":      rvol,
        "rsi_norm":  rsi_norm,
        "hour_sin":  hour_sin,
        "hour_cos":  hour_cos,
        "dow_sin":   dow_sin,
        # extras
        "target":    log_ret.shift(-1),
        "atr":       atr14,
    }, index=df.index)

    return result

FEATURE_COLS = [
    "log_ret", "hl_range", "co_change", "atr_norm",
    "ma10_dev", "ma30_dev", "ma60_dev", "rvol",
    "rsi_norm", "hour_sin", "hour_cos", "dow_sin",
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """
    Load 4h data, compute features, normalize (train stats only),
    build sliding-window sequences, and bundle validation data.

    Returns:
        x_train  (N_train, LOOKBACK, N_FEATURES)  float32
        y_train  (N_train,)                        float32
        val_data  ValData namedtuple
    """
    df = pd.read_parquet(DATA_FILE)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Remove timezone info to avoid comparison issues
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    full = _compute_features_full(df)

    # Valid rows: no NaN in any column
    valid_mask = ~full.isnull().any(axis=1)
    valid_idx  = np.where(valid_mask.values)[0]   # positions in original df

    feat_df = full.loc[valid_mask]
    feat_arr   = feat_df[FEATURE_COLS].values.astype(np.float32)  # (T, 12)
    target_arr = feat_df["target"].values.astype(np.float32)      # (T,)
    atr_arr    = feat_df["atr"].values.astype(np.float32)         # (T,)
    close_arr  = df["Close"].values.astype(np.float32)[valid_idx] # (T,)

    T      = len(feat_arr)
    T_full = len(df)

    # Tradeable flag: bar open-hour in TRADING_BAR_UTC
    tradeable = np.array(
        [feat_df.index[i].hour in TRADING_BAR_UTC for i in range(T)],
        dtype=bool,
    )

    # Future OHLC for TP/SL simulation:
    # future_ohlc[i, k] = [high, low, close] of the (k+1)-th candle after valid_idx[i]
    hlc_full    = df[["High", "Low", "Close"]].values.astype(np.float32)
    future_ohlc = np.zeros((T, MAX_HOLD, 3), dtype=np.float32)
    for i, pos in enumerate(valid_idx):
        end = min(pos + 1 + MAX_HOLD, T_full)
        n   = end - (pos + 1)
        if n > 0:
            future_ohlc[i, :n] = hlc_full[pos + 1 : end]
            if n < MAX_HOLD:
                future_ohlc[i, n:] = hlc_full[end - 1]  # pad with last available

    # Train / validation split
    n_val   = max(1, int(T * VAL_FRAC))
    n_train = T - n_val

    # Normalize using training-set statistics only (no data leakage)
    feat_mean = feat_arr[:n_train].mean(axis=0)
    feat_std  = feat_arr[:n_train].std(axis=0) + 1e-8
    feat_norm = (feat_arr - feat_mean) / feat_std

    # Build sliding-window sequences
    def _make_sequences(start, end):
        indices = [t for t in range(start, end) if t >= LOOKBACK]
        if not indices:
            empty = np.zeros((0, LOOKBACK, N_FEATURES), dtype=np.float32)
            return empty, np.zeros(0, dtype=np.float32), indices
        xs = np.stack([feat_norm[t - LOOKBACK : t] for t in indices])
        ys = target_arr[indices]
        return xs, ys, indices

    x_tr_np, y_tr_np, _         = _make_sequences(LOOKBACK, n_train)
    x_val_np, y_val_np, val_seq = _make_sequences(n_train, T)

    x_train = torch.from_numpy(x_tr_np)
    y_train = torch.from_numpy(y_tr_np)

    val_data = ValData(
        x           = torch.from_numpy(x_val_np),
        y           = torch.from_numpy(y_val_np),
        close       = torch.from_numpy(close_arr[val_seq]),
        atr         = torch.from_numpy(atr_arr[val_seq]),
        future_ohlc = torch.from_numpy(future_ohlc[val_seq]),
        tradeable   = torch.from_numpy(tradeable[val_seq]),
    )

    return x_train, y_train, val_data


# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------

def make_dataloader(x: torch.Tensor, y: torch.Tensor, batch_size: int):
    """Infinite random-batch iterator."""
    N = len(x)
    idx = torch.randperm(N)
    pos = 0
    while True:
        if pos + batch_size > N:
            idx = torch.randperm(N)
            pos = 0
        batch = idx[pos : pos + batch_size]
        pos  += batch_size
        yield x[batch], y[batch]


# ---------------------------------------------------------------------------
# Evaluation metric — DO NOT MODIFY
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_sharpe(
    model,
    val_data: ValData,
    tp_mult: float = 2.0,
    sl_mult: float = 1.0,
    batch_size: int = 512,
) -> float:
    """
    TP/SL backtest Sharpe ratio on the validation set.

    Only candles whose UTC open-hour is in TRADING_BAR_UTC are traded:
      JST AM 6–9   → UTC bar at 20:00
      JST PM 22–24 → UTC bar at 12:00

    For each tradeable candle t:
      signal = model(x[t])
        > 0 → LONG  : buy at close[t], TP = entry + ATR*tp_mult, SL = entry − ATR*sl_mult
        < 0 → SHORT : sell at close[t], TP = entry − ATR*tp_mult, SL = entry + ATR*sl_mult

      Scan future_ohlc[t, 0..MAX_HOLD-1]:
        First candle where high ≥ TP (long) or low ≤ TP (short) → TP hit
        First candle where low  ≤ SL (long) or high ≥ SL (short) → SL hit
        If neither hit within MAX_HOLD candles → close at future_ohlc[t, -1, 2]

      trade_pnl = direction × (exit_price − entry_price) / entry_price

    Annualized Sharpe = mean(trade_pnl) / std(trade_pnl) × sqrt(N_annual)
    where N_annual ≈ 2 sessions/day × 252 days = 504.
    Returns 0.0 if fewer than 2 trades were executed.
    """
    model.eval()

    # Get model signals for all validation sequences
    signals = []
    for start in range(0, len(val_data.x), batch_size):
        xb  = val_data.x[start : start + batch_size]
        sig = model(xb).squeeze(-1).cpu().float()
        signals.append(sig)
    signals = torch.cat(signals)   # (N_val,)

    trade_pnl = []

    for t in range(len(signals)):
        if not val_data.tradeable[t].item():
            continue

        sig   = signals[t].item()
        entry = val_data.close[t].item()
        atr   = val_data.atr[t].item()

        if entry <= 0 or atr <= 0:
            continue

        direction = 1 if sig >= 0 else -1  # +1 = long, -1 = short
        tp_price  = entry + direction * atr * tp_mult
        sl_price  = entry - direction * atr * sl_mult

        # Default exit: close of last available candle
        exit_price = val_data.future_ohlc[t, -1, 2].item()

        for k in range(MAX_HOLD):
            h = val_data.future_ohlc[t, k, 0].item()
            l = val_data.future_ohlc[t, k, 1].item()
            c = val_data.future_ohlc[t, k, 2].item()

            if h == 0.0 and l == 0.0:
                exit_price = c
                break

            if direction == 1:   # long
                if h >= tp_price:
                    exit_price = tp_price
                    break
                if l <= sl_price:
                    exit_price = sl_price
                    break
            else:                # short
                if l <= tp_price:
                    exit_price = tp_price
                    break
                if h >= sl_price:
                    exit_price = sl_price
                    break

        pnl = direction * (exit_price - entry) / entry
        trade_pnl.append(pnl)

    if len(trade_pnl) < 2:
        return 0.0

    arr      = np.array(trade_pnl, dtype=np.float64)
    mean_pnl = arr.mean()
    std_pnl  = arr.std()

    if std_pnl < 1e-10:
        return 0.0

    # Annualized Sharpe: ~2 trading sessions × 252 trading days
    ann_factor = math.sqrt(2 * 252)
    return float(mean_pnl / std_pnl * ann_factor)


# ---------------------------------------------------------------------------
# Main (one-time setup)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    download_data()

    print("\nVerifying data pipeline...")
    x_train, y_train, val_data = load_data()

    n_tradeable = int(val_data.tradeable.sum().item())
    print(f"Train sequences : {len(x_train):,}")
    print(f"Val   sequences : {len(val_data.x):,}  (tradeable: {n_tradeable:,})")
    print(f"Sequence shape  : {tuple(val_data.x.shape[1:])}  "
          f"(lookback={LOOKBACK}, features={N_FEATURES})")
    print(f"Val return mean : {val_data.y.mean():.6f},  "
          f"std: {val_data.y.std():.6f}")
    print(f"ATR mean        : {val_data.atr.mean():.4f} JPY")
    print()
    print("Done! Ready to train.")
