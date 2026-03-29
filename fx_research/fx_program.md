# fx_autoresearch — 4h Swing Trading

Autonomous experiment loop for USD/JPY FX trading strategy research.
**Goal**: maximize the annualized Sharpe ratio of a TP/SL swing-trading strategy on
the held-out validation set using 4-hour candles.

## Trading rules (fixed — enforced in fx_prepare.py)

| Rule | Details |
|------|---------|
| Timeframe | 4-hour OHLCV candles |
| Instrument | USD/JPY |
| Direction | Long or Short |
| Entry sessions | JST AM 6:00–9:00 (UTC 20:00 bar) and JST PM 22:00–00:00 (UTC 12:00 bar) |
| Take-profit | entry ± ATR(14) × `TP_MULT` (default 2.0) |
| Stop-loss | entry ∓ ATR(14) × `SL_MULT` (default 1.0) |
| Max hold time | 12 candles (48h) — force-closed at last candle close if TP/SL not hit |
| Long P&L | sell_price − buy_price (positive = profit) |
| Short P&L | buy_price − sell_price (positive = profit, i.e. sold high, bought low) |

## Setup

1. **Agree on a run tag**: e.g. `mar30`. Branch `fx/autoresearch/<tag>` must not exist.
2. **Create branch**: `git checkout -b fx/autoresearch/<tag>`
3. **Read the in-scope files**:
   - `fx_prepare.py` — fixed: data download, features, TP/SL backtest, evaluation.
   - `fx_train.py` — the file you modify: model, loss, optimizer, TP/SL multipliers.
4. **Verify data**: check that `~/.cache/fx_autoresearch/usdjpy_4h.parquet` exists.
   If not, run `python fx_prepare.py`.
5. **Initialize results.tsv** with just the header row.
6. **Confirm and start the loop**.

## Experimentation

Run experiments with: `python fx_train.py > run.log 2>&1`

**What you CAN modify in `fx_train.py`:**
- Model architecture: MLP, LSTM, GRU, 1D-CNN, Transformer, hybrid, etc.
- Loss function: differentiable Sharpe, Sortino, MSE on return, ranking loss, etc.
- Optimizer and schedule: AdamW, SGD, cosine LR, warmup/warmdown, etc.
- Batch size, hidden size, depth, dropout, normalization.
- `TP_MULT` and `SL_MULT`: take-profit and stop-loss ATR multipliers.
- Any additional preprocessing of input sequences (e.g. extra normalization, masking).

**What you CANNOT do:**
- Modify `fx_prepare.py` — it is read-only (fixed features, fixed evaluation, fixed data split).
- Change the validation split or look into the future (data leakage).
- Install packages outside `pyproject.toml`.
- Use GPU/CUDA — this project is CPU-only.

## Output format

```
---
val_sharpe:       0.823456
tp_mult:          2.0
sl_mult:          1.0
training_seconds: 300.1
total_seconds:    303.4
num_steps:        14230
num_params:       52,609
hidden_dim:       128
n_layers:         3
```

Extract key metric: `grep "^val_sharpe:" run.log`

## Logging results

Log to `results.tsv` (tab-separated). Do NOT git-commit this file (leave untracked).

```
commit	val_sharpe	tp_mult	sl_mult	num_params	status	description
a1b2c3d	0.823456	2.0	1.0	52609	keep	baseline MLP 128×3
b2c3d4e	0.901200	2.5	0.8	52609	keep	wider TP, tighter SL
c3d4e5f	0.740000	2.0	1.0	210433	discard	MLP 512×4 overfit
d4e5f6g	0.000000	2.0	1.0	0	crash	LSTM compile error
```

## The experiment loop

LOOP FOREVER:

1. Check git state (current branch/commit).
2. Edit `fx_train.py` with a new idea.
3. `git commit`
4. `python fx_train.py > run.log 2>&1`
5. `grep "^val_sharpe:\|^tp_mult:\|^sl_mult:\|^num_params:" run.log`
6. If empty → crash. Run `tail -n 50 run.log`, fix if trivial, else skip.
7. Log to `results.tsv`.
8. val_sharpe improved (higher) → keep commit.
9. val_sharpe same or worse → `git reset --hard HEAD~1`.

Timeout: kill runs exceeding 10 minutes. Treat as crash.

**NEVER STOP** — loop until manually interrupted.

## Ideas to explore

**Architecture**
- LSTM / GRU: capture sequential momentum and mean-reversion patterns
- 1D-CNN: detect local candlestick patterns (doji, engulfing, hammer)
- Transformer with causal mask: attend to key past candles
- Ensemble: multiple weak signals averaged or gated

**Loss function**
- Pure Sharpe loss (current baseline)
- Sortino ratio (penalizes only downside volatility)
- Win-rate maximization + profit factor combination
- Direct TP/SL hit-rate classification

**TP/SL tuning**
- Wider TP + tighter SL (trend-following)
- Tighter TP + wider SL (mean-reversion)
- Asymmetric: `TP_MULT=3.0, SL_MULT=1.0` (risk-reward 3:1)
- Volatility-regime-dependent TP/SL

**Signal quality**
- Confidence threshold: only trade when |signal| > threshold
- Session-aware features: add separate embeddings for JST AM vs PM sessions
- Trend filter: only trade in direction of 60-candle MA
