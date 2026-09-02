# Pair 1 on Modal: nested-bottleneck convnet vs transformer at ~2M params

A self-contained [Modal](https://modal.com) pipeline that trains two small KataGo
networks on the **same fixed pool of kata1 self-play data** and compares them on
held-out loss and fixed-visit Elo. Everything else about the two runs is held
constant; the only difference is the inner block type:

| Run | `-model-kind` | Inner block | Params (approx) |
|---|---|---|---|
| `conv` | `b5c192nbt-fson-mish-rvglr-bnh` | two 3x3 conv residual blocks at width 96 | ~2.0M |
| `tf` | `b5c192h3nbttfrs-fson-silu-rsnh` | 3-head RoPE attention + gated FFN at width 96 | ~1.4M |

Both share the nested-bottleneck wrapper, trunk width 192, inner width 96, five
blocks, and roughly equal compute per evaluation. The suffixes mirror the
conventions of the deployed nets of each family (`b28c512nbt-fson-mish-rvglr-bnh`
and `b11c768h12nbt3tflrs-fson-silu`), see `docs/NetworkArchitectures.md`.

Nothing here modifies the normal self-play loop. The only change outside this
directory is an optional `-seed` argument added to `python/train.py`.

## Prerequisites

```bash
pip install modal
modal token new          # once, authenticates the CLI
```

Run every command below from the repository root. The first invocation builds
the image (CUDA toolchain, torch, and a multi-architecture CUDA build of the C++
engine), which takes a while but is cached afterwards.

## Quick start

```bash
# 1. End-to-end check on the 1024-row test file: benchmark, tiny train, export,
#    8-game match, Elo, report. A few minutes on an L4, a few cents.
modal run experiments/modal_pair1/app.py --stage smoke

# 2. Full pipeline, detached so it keeps running after you close the terminal.
modal run --detach experiments/modal_pair1/app.py --stage all

# 3. Fetch the report.
modal volume get katago-pair1 eval/pair1-s1-v200/report.md ./report.md
```

Stages can also be run one at a time and re-run safely:

| Stage | What it does | Hardware | Skips / resumes |
|---|---|---|---|
| `data` | Downloads daily kata1 archives from katagoarchive.org, extracts on local disk, runs `shuffle.py` into `/data/shuffled/<dataset>/{train,val}` | 16 CPU, 64 GB RAM | Skips if the dataset exists (`--force-data` rebuilds) |
| `train` | Runs `train.py` for both models in parallel | 1x H100 each | Resumes from the last checkpoint in `runs/<run>/train` |
| `export` | Converts torch checkpoints to `.bin.gz` with `export_model_pytorch.py -use-swa` | CPU | Skips already-exported checkpoints |
| `eval` | Picks log-spaced checkpoints from each run, plays a round-robin with `katago match` at fixed visits, computes Bayes Elo with `summarize_sgfs.py` | 1x L4 | Overwrites `eval/<eval_name>` |
| `report` | Joins per-epoch validation metrics with checkpoint Elos into `report.md` / `report.json` | CPU | |
| `all` | `data` then `train` then `export` then `eval` then `report` | | |

The orchestration itself runs in a small remote container, so `--detach` is
enough to survive a closed laptop.

## What the defaults do

- **Data:** the 30 daily archives ending 2025-12-04, about 40 GB compressed.
  5% of files by hash become validation (the same split rule as
  `python/selfplay/shuffle.sh`). 50M rows are kept for training and 2M for
  validation, so 200M samples means each row is seen about four times, matching
  the reuse ratio of KataGo's normal training.
- **Training:** batch 256, 200M samples, 2M samples per epoch, an export every
  5 epochs (20 checkpoints per run), 500K validation rows scored after every
  epoch, seed 1, SGD, and an explicit LR schedule
  `(0,8.0),(100M,4.0),(140M,2.0),(170M,1.0),(190M,0.5)`. The schedule is a
  multiplier on `train.py`'s built-in per-sample LR; the main run holds 8.0 for
  its first 550M samples, so this compresses that shape into 200M with a cooldown
  at the end. Because both runs use the same `-seed`, they also see the same
  file order.
- **Evaluation:** 7 log-spaced checkpoints per run (14 bots), 200 visits per
  move, 19x19, area scoring, positional ko, komi 7, fp32 inference, about 400
  games per bot. Any `.bin.gz` files placed under `/anchors` on the volume are
  added as secondary bots (they play the checkpoints but not each other), which
  pins the Elo scale to known nets:

  ```bash
  modal volume put katago-pair1 ./g170-b6c96-s175395328-d26788732.bin.gz /anchors/g170-b6c96.bin.gz
  ```

## Knobs

Every default above is a flag on the local entrypoint; `modal run
experiments/modal_pair1/app.py --help` lists them. The ones you are most likely
to change:

| Flag | Default | Notes |
|---|---|---|
| `--seed` | 1 | Also names the runs (`pair1-conv-s1`, `pair1-tf-s1`). Use different seeds for replication runs. |
| `--optimizer` | `sgd` | `sgd`, `adamw`, or `muon`. The repo does not record which optimizer the published transformer nets used; Muon is the likely candidate. Whatever you pick, use it for both runs. |
| `--lr-schedule` | see above | `(samples,scale)` points, `K/M/B` suffixes accepted. For an LR sweep, change the scales and give each run a different `--run-tag`. |
| `--train-extra-args` | empty | Passed through to `train.py`, e.g. `"-use-tf32-matmul"` or `"-use-bf16"`. |
| `--conv-kind`, `--tf-kind` | see above | Any name in `python/katago/train/modelconfigs.py`. |
| `--days`, `--end-date`, `--keep-rows` | 30, 2025-12-04, 50M | The archive index at https://katagoarchive.org/kata1/trainingdata/ shows which dates exist. |
| `--visits`, `--games-per-bot`, `--checkpoints-per-run` | 200, 400, 7 | Elo standard error is roughly 350/sqrt(games) per bot. |
| `--max-val-samples` | 500K | Validation rows scored per epoch. Lower it if validation time dominates. |

## Cost and time (rough)

Modal bills per second. With the defaults, on the September 2026 price list:

| Step | Wall-clock | Cost |
|---|---|---|
| Image build (once) | 20 to 40 min | ~$1 |
| `data` | 1 to 2 h | ~$3 |
| `train` (both, parallel, H100) | 1 to 3 h | $10 to $25 |
| `export` | minutes | < $1 |
| `eval` (L4) | ~1 h | ~$1 |
| **Total** | **~3 to 5 h** | **$15 to $30** |

These are FLOP-based estimates within about a factor of two. The `smoke` stage
prints measured training throughput for both models on the GPU it ran on; hours
for the real run are `200M / (samples per second)` scaled by H100 versus L4 speed.
The full protocol of three seeds plus a three-point LR sweep is roughly 5 to 7x
the base cost and the same wall-clock, since Modal runs the jobs in parallel.

## Where things land on the volume

```
/data/shuffled/<dataset>/{train,val,train.json,dataset.json}
/data/runs/<run>/train/          train.py checkpoints, metrics_train.json, metrics_val.json
/data/runs/<run>/torchmodels/    exported torch checkpoints (<run>-s<samples>-d<rows>)
/data/runs/<run>/exported/       .bin.gz files for the C++ engine
/data/eval/<eval_name>/          match.cfg, sgfs/, elo.txt, elo.json, report.md, report.json
/data/anchors/                   optional anchor nets you upload
/data/smoke/<timestamp>/         artifacts of each smoke run
```

Browse with `modal volume ls katago-pair1 <path>` and download with
`modal volume get katago-pair1 <remote> <local>`.

## Caveats

- The two runs share a seed and data order but GPU training is not bit-for-bit
  deterministic, so treat differences within a few Elo as noise and use extra
  seeds before drawing conclusions.
- Held-out loss compares well within a family; across families the loss-to-Elo
  mapping can shift, so the Elo table is the primary result.
- The L4 used for evaluation is not in the C++ build's native architecture list
  for this CUDA version, so the first engine start JIT-compiles from PTX. It is a
  one-time delay per container.
- This is a supervised proxy. It says nothing about how well either net would
  generate self-play data.
