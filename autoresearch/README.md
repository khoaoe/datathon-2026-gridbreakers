# autoresearch (datathon fork)

A fork of the [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
pattern adapted for this revenue-forecasting datathon. Same core idea: the
agent iterates on a single file, a frozen harness judges each experiment, and
`results.tsv` logs everything.

## Mapping from karpathy's repo

| karpathy/autoresearch   | this fork                                  |
|-------------------------|--------------------------------------------|
| `prepare.py` (frozen)   | `autoresearch/prepare.py`                  |
| `train.py` (mutable)    | `autoresearch/train.py`                    |
| `program.md`            | `autoresearch/program.md`                  |
| target: `val_bpb`       | target: `val_mae_rev`                      |
| dataset: FineWebEdu     | dataset: sales.csv + 13 aux tables         |
| budget: 5 min wall-time | budget: 15 min wall-time (configurable)    |
| runs on: single NVIDIA GPU | runs on: CPU (LightGBM), conda env `datathon` |

## Files

```
autoresearch/
├── prepare.py          # FROZEN harness: split, metric, submission writer, results tsv
├── train.py            # mutable experiment script (agent edits this)
├── program.md          # agent instructions
├── run_experiment.sh   # ergonomic launcher (timeout + log capture)
├── results.tsv         # experiment scoreboard (untracked by git, git-ignored)
└── README.md           # this file
```

## Quick start

```bash
# 1. Verify harness loads data correctly
./autoresearch/run_experiment.sh --dry-run

# 2. Run the current train.py (baseline first)
./autoresearch/run_experiment.sh

# 3. Inspect scoreboard
cat autoresearch/results.tsv | column -t -s$'\t'
```

## Metric

Primary: `val_mae_rev` (Mean Absolute Error of Revenue on the fixed
2022-10-01 → 2022-12-31 holdout). Lower is better. The harness also emits
`val_mae_cogs`, `val_rmse_rev`, `val_mape_rev`, `val_r2_rev`.

## Submission output

Every successful run writes `output/submissions/autoresearch.csv` (548 rows,
3 cols: `Date`, `Revenue`, `COGS`). The write guards against null
predictions and negative values.

## Environment

Everything runs inside the conda env `datathon`. Override with
`AUTORESEARCH_ENV=<name>`. Override timeout with
`AUTORESEARCH_TIMEOUT=<secs>` (default 900).

## Running the agent

Follow `autoresearch/program.md`. Spin up your coding agent (Cursor / Claude
Code / Codex CLI / etc.), point it at `program.md`, and it will loop:

```
edit train.py → commit → run → grep metric → keep or revert → repeat
```

Leave it running. When you come back, check `results.tsv` for the scoreboard
and pick the best commit on the `autoresearch/<tag>` branch.
