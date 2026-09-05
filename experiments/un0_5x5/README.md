# Un-0-inspired 5x5 Go pilot

First question: can a board-conditioned Kuramoto trunk learn the existing
teacher pool under the pair-1 training recipe? **One seed only (1).**
This is an exploratory supervised-learning study, not a playing-strength,
generalization, energy-efficiency, or architecture-superiority result.

The [completed seed-1 results](results/seed1/README.md) include fixed validation,
an overlap-filtered secondary comparison, dynamics diagnostics, and learning curves.

## Architecture

Reference: [Un-0](https://github.com/unconv-ai/Un-0/tree/75243cab846c092527734925c2433b556f5e5ee7),
particularly `un0/model.py`, pinned to
`75243cab846c092527734925c2433b556f5e5ee7` (MIT).
Un-0 is a class-conditional image generator. This experiment adapts its
conditional Kuramoto dynamics; it does not use released image weights or the
image-generation loss.

`un0-b5c192-n1250-e10` has **1,720,987 parameters**, versus 1,935,298 for
the historical CNN and 1,376,301 for the transformer. These are measured counts,
not a parameter or compute match.

- Keep the CNN's 22 spatial / 19 global input features, 192-channel stem,
  final normalization, main and auxiliary heads, and complete KataGo losses.
- Replace all five trunk blocks with one residual oscillator module.
  There are 50 oscillators per point, flattened into 1,250 oscillators with a
  learned, directed, dense coupling matrix. Its diagonal is removed.
- A 1x1 convolution of the stem produces board-dependent drive strengths.
  One independent driver evolves analytically as `phi(t) = phi(0) + omega_d*t`.
  Main dynamics are
  `dtheta_i/dt = omega_i + sum_j K_ij sin(theta_j-theta_i) + drive_i(x) sin(phi-theta_i)`.
- Integrate to time 1 with 10 Euler steps and shared parameters across steps.
  Read out sine/cosine of phases relative to oscillator 0, reshape to the board,
  and project through a 1x1 convolution. The stem residual bypass is retained.
- Unlike the generator's per-example random phases, initial main/driver phases
  are sampled once at initialization and saved as buffers. Predictions are
  deterministic. This is an intentional discriminative adaptation.
- Use ordinary `1/sqrt(N)` coupling initialization; no muP claims. Phase state
  and trigonometry stay fp32; dense products use bf16 autocast during training.

Dense coupling and fixed phase buffers have no built-in D4 equivariance. Training
uses KataGo's usual symmetry augmentation; validation averages losses over all
eight transformations. The model accepts only full 5x5 boards. Python checkpoints
work; C++ export/match does not support this block.

## Frozen first-pass protocol

Use the existing `teacher-b5-1000k` pool on Modal volume `katago-pair1`:
985,600 training rows and 63,000 validation rows. Do not regenerate it.

| Setting | First pass |
|---|---|
| Seed | 1 |
| Samples | 20M nominal; report actual counts (epoch boundaries can overshoot) |
| Batch | 2048 |
| Optimizer | Muon, unchanged KataGo implementation and scaling |
| Precision | `-no-compile -use-bf16` |
| Epoch | 500K nominal samples |
| Torch export | Every epoch, to capture early learning |
| LR multipliers | `(0,8.0),(10M,4.0),(14M,2.0),(17M,1.0),(19M,0.5)` |
| Training-loop validation | 50K rows/epoch, original randomized preprocessing |
| Primary endpoint | Final raw checkpoint, same fixed validation for all models |

The primary comparisons are differences in policy cross entropy
and value loss against **re-evaluated** historical CNN/transformer final raw
checkpoints for seed 1. Do not treat positions or repeated
symmetries as independent training replicates. One seed cannot measure training
variance or establish statistical significance or equivalence. Endpoints are not selected by minimum
validation loss. Plot full curves as exploratory evidence and report actual
sample counts, not checkpoint indices.

The new evaluator uses every validation row, including partial batches, full
available history, fp32, and all eight D4 transforms. It averages per-transform
losses, not predictions. It retains global weights, policy weights, the value
weight `1-target[35]`, KataGo's 1.2 value-loss factor, and the global-weight
denominator. Report policy KL to the teacher and target entropy separately.
Entropy is not evidence that excess cross entropy is irreducible teacher noise.
`pacc1` is top-1 teacher agreement, not perfect-play accuracy.

Why re-evaluate the old checkpoints: the old `-seed` controls initialization and
file order, but the loader seeds symmetry draws from `os.urandom`, and history
dropout consumes the torch RNG after architecture-dependent initialization.
Historical training metrics therefore do not guarantee paired examples and
augmentations. This pilot leaves that training recipe intact and fixes the
comparison at evaluation. It cannot isolate every effect of architecture from
normalization, optimizer suitability, or LR choice.

The pool was split by output file. The `audit` stage checks the preserved 128-bit
game IDs and hashes all full-history model inputs, including D4 equivalence.
The first audit found 115 shared games (849 validation rows), 14,081 validation
rows with an exact training input, and 17,853 with a D4-equivalent training input.
The evaluator additionally reports a secondary subset excluding both shared games
and shared D4 inputs. This subset was specified after the audit, changes the
position distribution by excluding repeated openings, and is not a new untouched
test set. Treat the primary benchmark as performance on the existing teacher
distribution. A confirmatory study needs a new,
game-disjoint test set that remains untouched during tuning, duplicate/symmetry
overlap auditing, and an equal tuning budget per architecture. A perfect-solver
audit would answer a different question about move quality.

## Gates and diagnostics

1. Run local numerical checks: pairwise-equation values and gradients, analytic
   zero-coupling case, integrator refinement, full-model gradient flow, parameter
   registration, checkpoint round trip, and batch-independent inference.
2. Run a short H100 benchmark with real teacher data and the full Muon update.
   Record forward/backward/update throughput and gradient health for Un-0, CNN,
   and transformer. Size cost from measured throughput before starting training.
3. Train seed 1 only. Abort or investigate non-finite gradients or persistent
   training failure; do not count failed runs as evidence against the architecture.
4. Re-evaluate the three seed-1 final checkpoints with the fixed evaluator. On the first
   2,048 validation rows, also measure doubled integration steps and coupling
   disabled for Un-0. These are inference interventions, **not trained controls**.
   A strong step-size effect would weaken the continuous-dynamics interpretation.

If promising, the next study should train a no-coupling and a tied-weight
non-oscillatory control, tune integration steps and LR with an equal budget,
and report both equal-sample and equal-compute comparisons. A future shared
Python search harness must use the same rules, features, legal-move handling,
search budget, openings, and color pairing for every architecture before any Elo
comparison. The historical C++ Elo values cannot be combined with a different
search implementation.

## Run

The volume belongs to the `armoredmeatball` Modal profile on this machine;
the currently selected `xxdavidtran` profile has a different workload. Select the
profile per command so other tasks retain their configuration.

```bash
PYTHONPATH=python python -m pytest -q experiments/un0_5x5
MODAL_PROFILE=armoredmeatball modal run experiments/un0_5x5/app.py --stage diag
MODAL_PROFILE=armoredmeatball modal run --detach experiments/un0_5x5/app.py --stage train
MODAL_PROFILE=armoredmeatball modal run experiments/un0_5x5/app.py --stage audit
MODAL_PROFILE=armoredmeatball modal run experiments/un0_5x5/app.py --stage evaluate
MODAL_PROFILE=armoredmeatball modal volume get katago-pair1 studies/un0-pilot-v1 ./un0-results
```

Training lives at `runs/un0-pilot-v1-s1`; diagnostics and fixed evaluation
live at `studies/un0-pilot-v1`. Original runs are read only. Each new run saves
config, dataset manifest, and Python source hashes, and rejects reuse with a
different specification. Modal retries are disabled. Explicit restarts resume
the training checkpoint, but RNG continuity across preemption is not guaranteed
by the inherited trainer and any restart must be recorded.

The GPU functions have timeouts; these are failure limits, not dollar budgets.
On [Modal's pricing page](https://modal.com/pricing) checked 2026-09-05, an H100
is $0.001097/s; 8 physical CPU cores and 32 GiB add up to $0.00017584/s at full
allocation. Thus one 20M-sample training run at measured throughput `r` costs
approximately `20_000_000/r * 0.00127284` dollars before validation, startup,
checkpoint I/O, and diagnostics. Report estimates separately from billing.
