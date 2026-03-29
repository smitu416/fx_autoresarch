# fx_autoresearch

This is an autonomous experiment loop for FX (USD/JPY) trading strategy research.
The goal is to find the model and training configuration that maximizes the annualized
Sharpe ratio on the held-out validation set (2023–present).

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar30`). The branch `fx/autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b fx/autoresearch/<tag>` from current master.
3. **Read the in-scope files**: Read these files for full context:
   - `fx_prepare.py` — fixed constants, data loading, feature engineering, evaluation. Do not modify.
   - `fx_train.py` — the file you modify. Model architecture, optimizer, training loop, loss function.
4. **Verify data exists**: Check that `~/.cache/fx_autoresearch/usdjpy.parquet` exists. If not, tell the human to run `python fx_prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on **CPU only** (no CUDA). The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup). You launch it as: `python fx_train.py`.

**What you CAN do:**
- Modify `fx_train.py` — this is the only file you edit. Everything is fair game:
  - Model architecture (MLP, RNN, Transformer, CNN, etc.)
  - Loss function (differentiable Sharpe, MSE on returns, ranking loss, etc.)
  - Optimizer (AdamW, SGD, RMSProp, etc.) and hyperparameters
  - Batch size, learning rate, weight decay, gradient clipping
  - Any preprocessing on the input sequences (e.g. further normalization, masking, etc.)

**What you CANNOT do:**
- Modify `fx_prepare.py`. It is read-only. It contains the fixed evaluation, data loading, feature engineering, and training constants (time budget, lookback, feature definitions, train/val split).
- Install new packages or add dependencies beyond `pyproject.toml`.
- Change the validation split or look ahead into the future (data leakage).
- Modify the `evaluate_sharpe` function. It is the ground truth metric.

**The goal is simple: get the highest val_sharpe.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the loss function, the batch size. The only constraint is that the code runs without crashing and finishes within the time budget.

**Simplicity criterion**: All else being equal, simpler is better. A tiny improvement that adds ugly complexity is not worth it. Removing code and getting equal or better results is a great outcome. Weigh complexity cost against improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so run the training script as-is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_sharpe:       0.823456
training_seconds: 300.1
total_seconds:    302.5
num_steps:        12450
num_params:       52,609
hidden_dim:       128
n_layers:         3
```

Extract the key metric:
```
grep "^val_sharpe:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	val_sharpe	num_params	status	description
```

1. git commit hash (short, 7 chars)
2. val_sharpe achieved (e.g. 0.823456) — use 0.000000 for crashes
3. num_params (e.g. 52609) — use 0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_sharpe	num_params	status	description
a1b2c3d	0.823456	52609	keep	baseline MLP (128 hidden, 3 layers)
b2c3d4e	0.891200	52609	keep	switch to GELU + sharpe loss warmup
c3d4e5f	0.750000	210433	discard	larger MLP (512 hidden) — overfit
d4e5f6g	0.000000	0	crash	Transformer (compile error)
```

## The experiment loop

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Tune `fx_train.py` with an experimental idea by directly hacking the code.
3. `git commit`
4. Run: `python fx_train.py > run.log 2>&1`
5. Read results: `grep "^val_sharpe:\|^num_params:" run.log`
6. If empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and fix. Give up after a few attempts.
7. Record in `results.tsv` (do NOT commit this file — leave it untracked).
8. If val_sharpe improved (higher), keep the commit.
9. If val_sharpe is equal or worse, `git reset --hard HEAD~1`.

**Timeout**: Each experiment should take ~5 minutes. If a run exceeds 10 minutes, kill it and treat as failure.

**Crashes**: If it's a quick fix (typo, missing import), fix and re-run. If the idea is broken, skip it, log "crash", and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The loop runs until the human interrupts you, period.

## Ideas to explore

- Architecture: LSTM/GRU (sequential inductive bias), 1D-CNN (local pattern detection), Transformer (global attention), ensemble of weak learners
- Loss: pure Sharpe loss, Sortino ratio, Calmar ratio, MSE + Sharpe combo, ranking loss
- Input: different normalization strategies (per-window vs. global), feature selection/ablation, adding lag features
- Regularization: dropout, batch norm, layer norm, L1/L2
- Optimizer: learning rate schedule (cosine, linear warmup/warmdown), gradient clipping, different betas
- Position sizing: tanh output (continuous position), sign (binary long/short), soft threshold
