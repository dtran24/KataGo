# Pair 1 on Modal: nested-bottleneck convnet vs transformer at ~2M params

A self-contained [Modal](https://modal.com) pipeline that trains two small KataGo
networks on the **same fixed pool of kata1 self-play data** and compares them on
held-out loss and fixed-visit Elo. Everything else about the two runs is held
constant; the only difference is the inner block type:

| Run | `-model-kind` | Inner block | Params (measured) |
|---|---|---|---|
| `conv` | `b5c192nbt-fson-mish-rvglr-bnh` | two 3x3 conv residual blocks at width 96 | 1,935,298 |
| `tf` | `b5c192h3nbttfrs-fson-silu-rsnh` | 3-head RoPE attention + gated FFN at width 96 | 1,376,301 |

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
engine); that took about 12 minutes and is cached afterwards.

## Quick start

```bash
# 1. End-to-end check on the 1024-row test file: benchmark, tiny train, export,
#    8-game match, Elo, report. About 15 minutes on an L4, a few cents.
modal run experiments/modal_pair1/app.py --stage smoke

# 2. Full pipeline, detached so it keeps running after you close the terminal.
modal run --detach experiments/modal_pair1/app.py --stage all

# 3. Fetch the report.
modal volume get katago-pair1 eval/pair1-s1-v200/report.md ./report.md
```

Stages can also be run one at a time and re-run safely:

| Stage | What it does | Hardware | Skips / resumes |
|---|---|---|---|
| `smoke` | Benchmarks both models, trains each for `--smoke-samples` (default 8192) on the 1024-row test file, exports, plays an 8-game match, computes Elo, writes the report. Artifacts land under `smoke/<timestamp>` on the volume. | 1x L4 | |
| `diag` | Runs `benchmark_fresh_model.py` variants (`--diag-names`, comma separated; empty = all) and prints gradient-norm health and throughput per variant. `--diag-gpu h100` runs them on an H100. | L4 or H100 | |
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
  epoch, seed 1, Muon, `-no-compile -use-bf16` (see the known issue below), and an explicit
  LR schedule `(0,8.0),(100M,4.0),(140M,2.0),(170M,1.0),(190M,0.5)`. The
  schedule is a multiplier on `train.py`'s built-in per-sample LR; the main run
  holds 8.0 for its first 550M samples, so this compresses that shape into 200M
  with a cooldown at the end. Because both runs use the same `-seed`, they also
  see the same file order.
- **Evaluation:** 7 log-spaced checkpoints per run (14 bots), 200 visits per
  move, 19x19, area scoring, positional ko, komi 7, fp32 inference, about 400
  games per bot. Any `.bin.gz` files placed under `/anchors` on the volume are
  added as secondary bots (they play the checkpoints but not each other), which
  pins the Elo scale to known nets:

  ```bash
  modal volume put katago-pair1 ./g170-b6c96-s175395328-d26788732.bin.gz /anchors/g170-b6c96.bin.gz
  ```

## Known issue: torch.compile corrupts the transformer's gradients

With torch 2.8.0+cu128, the compiled transformer produces non-finite gradient
norms on every batch, on both the L4 and the H100, in fp32, bf16, and tf32. The
convnet compiles cleanly. `train.py`'s gradient watcher catches it: a compiled
transformer run halted at batch 88 with 32 non-finite and 23 extreme norms.
Without compile every variant is clean, gradient norms shrink normally, and
losses fall. Results from `--stage diag` (batch 256, Muon):

| Variant | L4 | H100 |
|---|---|---|
| transformer, compile, fp32 | **8/8 non-finite**, 439/s | **8/8 non-finite**, 1,599/s |
| transformer, compile, bf16 | | **8/8 non-finite**, 8,182/s |
| transformer, compile, tf32 | | **8/8 non-finite**, 4,451/s |
| transformer, no compile, fp32 | clean, 350/s | clean, 1,445/s |
| transformer, no compile, bf16 | clean, 729/s | clean, 3,680/s |
| transformer, no compile, SGD / env toggles / base config | clean, ~350/s | |
| convnet, compile, fp32 | clean, ~1,400/s | clean, 9,488/s |
| convnet, compile, bf16 | | clean, 8,108/s |
| convnet, no compile, fp32 | clean, 877/s | clean, 4,522/s |

So the pipeline passes `-no-compile` to `train.py` by default for both runs
(`--train-extra-args`). On the H100 that costs the transformer about 10% and the
convnet about 2x, which bf16 autocast more than wins back. The shared flag applies to both runs; to compile only the
convnet, use `--train-extra-args="" --tf-extra-args=-no-compile`. The root cause
is somewhere in the inductor graph for the attention path and is worth a look
upstream, but it is out of scope here.

## Knobs

Every default above is a flag on the local entrypoint; `modal run
experiments/modal_pair1/app.py --help` lists them. The ones you are most likely
to change:

| Flag | Default | Notes |
|---|---|---|
| `--seed` | 1 | Also names the runs (`pair1-conv-s1`, `pair1-tf-s1`). Use different seeds for replication runs. |
| `--optimizer` | `muon` | `sgd`, `adamw`, or `muon`. Per the [July 2026 symmetry study](https://lightvector.github.io/katagostudies/202607-symmetry/), the released transformers (`b10c512h8nbt3tflrs`, `b11c768h12nbt3tflrs`) were Muon-trained from the start, the main kata1 `b28c512nbt` line was SGD with a Muon-finetuned fork about 50 Elo stronger, and the current main net `b40c768nbt` went SGD early then Muon for most of its run. Muon for both runs is the like-for-like choice; `train.py` rescales the LR for Muon internally, so the same `--lr-schedule` applies. Use `sgd` only if you want the historical convnet recipe, and use the same optimizer for both runs either way. |
| `--train-extra-args` | `-no-compile -use-bf16` | Passed to `train.py` for both runs. `-no-compile` is required for the transformer (known issue below). bf16 autocast is on because a 131K-sample smoke gave the same validation losses as fp32 to three decimals (transformer value loss 0.9778 vs 0.9774, convnet 1.1308 vs 1.1307) at 1.75x to 2.5x the speed. Set `--train-extra-args=-no-compile` for fp32. |
| `--conv-extra-args`, `--tf-extra-args` | empty | Appended to `train.py` args for one run only. |
| `--lr-schedule` | see above | `(samples,scale)` points, `K/M/B` suffixes accepted. For an LR sweep, change the scales and give each run a different `--run-tag`. |
| `--conv-kind`, `--tf-kind` | see above | Any name in `python/katago/train/modelconfigs.py`. |
| `--days`, `--end-date`, `--keep-rows` | 30, 2025-12-04, 50M | The archive index at https://katagoarchive.org/kata1/trainingdata/ shows which dates exist. |
| `--visits`, `--games-per-bot`, `--checkpoints-per-run` | 200, 400, 7 | Elo standard error is roughly 350/sqrt(games) per bot. |
| `--max-val-samples` | 500K | Validation rows scored per epoch. Lower it if validation time dominates. |
| `--smoke-samples` | 8192 | Longer smoke runs (e.g. 131072) show whether both models learn and give `train.py` enough batches to log gradient norms. |

## Cost and time (measured)

Throughput is from `--stage diag` on Modal's H100 at batch 256 with Muon and
`-no-compile`; the training-loop numbers include optimizer and metrics overhead.
Prices are Modal's September 2026 list (H100 $3.95/h, L4 $0.80/h).

| Configuration | conv samples/s | tf samples/s | Wall-clock for 200M (parallel) | H100 cost, both runs |
|---|---|---|---|---|
| bf16, `-no-compile` (default) | ~8,000 (est. from compiled bf16 at 8,108) | 3,680 | ~15 h (tf is the long pole) | ~$85 |
| fp32, `-no-compile` (`--train-extra-args=-no-compile`) | 4,522 | 1,445 | ~38 h | ~$200 |

Add about $3 and 1 to 2 hours for `data`, about $1 for `eval` on an L4, and
about $1 for the one-time image build. Multiply by 5 to 7x for the full protocol
of three seeds plus a three-point LR sweep; wall-clock stays the same because
Modal runs the jobs in parallel. The transformer is the cost driver: on the same
GPU it trains about 3x slower per sample than the convnet in fp32.

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
- `benchmark_fresh_model.py` always compiles, so the transformer benchmark inside
  `smoke` will keep reporting non-finite gradient norms even though the training
  run right after it is fine. Read the training metrics, not the benchmark line.
- The L4 used for evaluation is not in the C++ build's native architecture list
  for this CUDA version, so the first engine start JIT-compiles from PTX. It is a
  one-time delay per container.
- This is a supervised proxy. It says nothing about how well either net would
  generate self-play data.
