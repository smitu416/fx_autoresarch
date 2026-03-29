"""
FX autoresearch training script — 4h swing trading, CPU-only.
Agent modifies this file to maximize val_sharpe.

Strategy: model outputs a scalar signal per 4h candle.
  signal > 0 → LONG  (buy entry, exit at TP or SL)
  signal < 0 → SHORT (sell entry, exit at TP or SL)
Trades are only executed during JST AM 6–9 and PM 22–24 sessions.

Usage: python fx_train.py
"""

import math
import time

import torch
import torch.nn as nn

from fx_prepare import (
    LOOKBACK, TIME_BUDGET, N_FEATURES, ValData,
    load_data, make_dataloader, evaluate_sharpe,
)

# ---------------------------------------------------------------------------
# Hyperparameters (edit freely — this is the only file you modify)
# ---------------------------------------------------------------------------

# Model architecture
HIDDEN_DIM = 128    # hidden layer width
N_LAYERS   = 3      # number of hidden layers

# Optimization
BATCH_SIZE   = 128
LR           = 3e-4
WEIGHT_DECAY = 1e-4

# Trade management (used in evaluate_sharpe and displayed in summary)
TP_MULT = 2.0       # take-profit  = entry ± ATR × TP_MULT
SL_MULT = 1.0       # stop-loss    = entry ∓ ATR × SL_MULT

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FXModel(nn.Module):
    """
    MLP that maps a (LOOKBACK × N_FEATURES) window to a scalar trading signal.
    Positive → long USD/JPY.  Negative → short USD/JPY.
    """

    def __init__(self):
        super().__init__()
        in_dim = LOOKBACK * N_FEATURES
        layers = []
        d = in_dim
        for _ in range(N_LAYERS):
            layers += [nn.Linear(d, HIDDEN_DIM), nn.GELU()]
            d = HIDDEN_DIM
        layers.append(nn.Linear(HIDDEN_DIM, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, LOOKBACK, N_FEATURES)
        return self.net(x.view(x.size(0), -1))   # (B, 1)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)

print("Loading data...")
x_train, y_train, val_data = load_data()
n_tradeable = int(val_data.tradeable.sum().item())
print(f"Train: {len(x_train):,}  |  Val: {len(val_data.x):,}  "
      f"(tradeable sessions: {n_tradeable:,})")

model     = FXModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loader    = make_dataloader(x_train, y_train, BATCH_SIZE)

n_params = sum(p.numel() for p in model.parameters())
print(f"Params: {n_params:,}  |  Budget: {TIME_BUDGET}s")
print(f"TP: {TP_MULT}× ATR  |  SL: {SL_MULT}× ATR")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def sharpe_loss(signals: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """
    Differentiable negative Sharpe ratio (surrogate training loss).
    P&L = signal × next-candle log return.
    Minimizing this loss maximizes the Sharpe-like objective.
    """
    pnl  = signals * returns
    mean = pnl.mean()
    std  = pnl.std() + 1e-6
    return -(mean / std) * math.sqrt(252)


t0_train   = time.time()
train_time = 0.0
smooth     = 0.0
step       = 0

while True:
    t0 = time.time()

    x_b, y_b = next(loader)

    model.train()
    sig  = model(x_b).squeeze(-1)   # (B,)
    loss = sharpe_loss(sig, y_b)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    dt = time.time() - t0
    if step > 5:
        train_time += dt

    progress  = min(train_time / TIME_BUDGET, 1.0)
    smooth    = 0.95 * smooth + 0.05 * loss.item()
    debiased  = smooth / (1 - 0.95 ** (step + 1))
    remaining = max(0.0, TIME_BUDGET - train_time)

    print(
        f"\rstep {step:05d} ({100 * progress:.1f}%) | "
        f"loss: {debiased:+.4f} | remaining: {remaining:.0f}s    ",
        end="", flush=True,
    )
    step += 1

    if step > 5 and train_time >= TIME_BUDGET:
        break

print()

# ---------------------------------------------------------------------------
# Final evaluation — TP/SL backtest on held-out validation set
# ---------------------------------------------------------------------------

val_sharpe = evaluate_sharpe(model, val_data, tp_mult=TP_MULT, sl_mult=SL_MULT)

t_end = time.time()
print("---")
print(f"val_sharpe:       {val_sharpe:.6f}")
print(f"tp_mult:          {TP_MULT}")
print(f"sl_mult:          {SL_MULT}")
print(f"training_seconds: {train_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"num_steps:        {step}")
print(f"num_params:       {n_params:,}")
print(f"hidden_dim:       {HIDDEN_DIM}")
print(f"n_layers:         {N_LAYERS}")
