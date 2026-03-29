"""
FX autoresearch training script. CPU-only, single-file.
Predicts USD/JPY direction; evaluated by annualized Sharpe ratio.

Usage: python fx_train.py
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from fx_prepare import (
    LOOKBACK, TIME_BUDGET, N_FEATURES,
    load_data, make_dataloader, evaluate_sharpe,
)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
HIDDEN_DIM = 128       # hidden layer width
N_LAYERS   = 3         # number of hidden layers

# Optimization
BATCH_SIZE   = 128     # training batch size
LR           = 3e-4    # AdamW learning rate
WEIGHT_DECAY = 1e-4    # AdamW weight decay

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FXModel(nn.Module):
    """
    MLP that maps a (LOOKBACK, N_FEATURES) window to a scalar trading signal.
    Positive signal → long USD/JPY, negative → short.
    """

    def __init__(self):
        super().__init__()
        input_dim = LOOKBACK * N_FEATURES
        layers = []
        in_dim = input_dim
        for _ in range(N_LAYERS):
            layers.append(nn.Linear(in_dim, HIDDEN_DIM))
            layers.append(nn.GELU())
            in_dim = HIDDEN_DIM
        layers.append(nn.Linear(HIDDEN_DIM, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, LOOKBACK, N_FEATURES)
        return self.net(x.view(x.size(0), -1))  # (B, 1)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)

print("Loading data...")
x_train, y_train, x_val, y_val = load_data()
print(f"Train: {len(x_train):,} sequences  |  Val: {len(x_val):,} sequences")

model = FXModel()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

train_loader = make_dataloader(x_train, y_train, BATCH_SIZE)

print(f"Time budget: {TIME_BUDGET}s")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def sharpe_loss(signals: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """
    Differentiable negative Sharpe ratio.
    Minimizing this maximizes the annualized Sharpe.
    """
    pnl = signals * returns
    return -(pnl.mean() / (pnl.std() + 1e-6)) * math.sqrt(252)

t_start_training = time.time()
total_training_time = 0.0
smooth_loss = 0.0
step = 0

while True:
    t0 = time.time()

    x_batch, y_batch = next(train_loader)

    model.train()
    signals = model(x_batch).squeeze(-1)  # (B,)
    loss = sharpe_loss(signals, y_batch)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    dt = time.time() - t0
    if step > 5:
        total_training_time += dt

    progress = min(total_training_time / TIME_BUDGET, 1.0)
    ema = 0.95
    smooth_loss = ema * smooth_loss + (1 - ema) * loss.item()
    debiased = smooth_loss / (1 - ema ** (step + 1))
    remaining = max(0.0, TIME_BUDGET - total_training_time)

    print(
        f"\rstep {step:05d} ({100 * progress:.1f}%) | "
        f"loss: {debiased:+.4f} | "
        f"remaining: {remaining:.0f}s    ",
        end="", flush=True,
    )

    step += 1

    if step > 5 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r log

# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

val_sharpe = evaluate_sharpe(model, x_val, y_val)

t_end = time.time()
print("---")
print(f"val_sharpe:       {val_sharpe:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"num_steps:        {step}")
print(f"num_params:       {sum(p.numel() for p in model.parameters()):,}")
print(f"hidden_dim:       {HIDDEN_DIM}")
print(f"n_layers:         {N_LAYERS}")
