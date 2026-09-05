# Fourier neural operator 5x5 pilot

One seed (1), one architecture, 20M nominal samples. This exploratory study asks
whether a Fourier neural operator trunk learns the existing 5x5 Go teacher pool
competitively with the CNN, transformer, and Un-0 adaptation. No hyperparameter
sweep, new data generation, or additional training seeds are included.

## Connection to Anandkumar's work

The reference is Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart, and
Anandkumar, [Fourier Neural Operator for Parametric Partial Differential Equations,
ICLR 2021, arXiv v3](https://arxiv.org/abs/2010.08895v3). The authors'
[implementation guide](https://neuraloperator.github.io/dev/theory_guide/fno.html)
explains the spectral layer plus pointwise linear path. Our small implementation
uses that mathematical construction, with no external model weights or neuralop
package dependency. It is a residual bottleneck adaptation for Go, not a
reproduction of a PDE benchmark or the canonical end-to-end FNO network.

FNO's spectral layer learns global convolution in frequency space. Here the input
is a field of Go features, the policy is a field over legal board coordinates plus
pass, and the existing KataGo head produces a game value. Training retains the
existing soft teacher-policy and game-result targets, including all auxiliary
losses. Neural-operator resolution transfer is **not** tested: 5x5 and 9x9 Go are
different games, not two discretizations of the same continuous physical domain.
On 5x5 the CNN already covers the whole board, so global communication alone is
not evidence for an advantage. Discrete stones and boundaries also differ from
smooth physical fields where FNO is commonly evaluated.

## Architecture fixed before training

`fno-b5c192-w112-m2-p1`: **1,976,626 trainable real parameters**, 2.14% more than
the 1,935,298-parameter CNN. The transformer has 1,376,301 and Un-0 has 1,720,987.
These are not equal-compute or exact equal-parameter comparisons.

- Preserve the CNN's 22 spatial / 19 global features, 192-channel 3x3 stem,
  configured final normalization, both policy/value heads, and full training loss.
- Five residual bottlenecks, each with 112 internal channels. The outer residual
  addition and depth scaling remain in KataGo's existing model code.
- Each block: existing fixed-scale norm/bias and Mish; 1x1 lift to 112 channels;
  spectral convolution plus a learned 1x1 pointwise path, scaled by `1/sqrt(2)`;
  Mish; 1x1 projection back to 192 channels.
- Spectral path: symmetrically zero-pad 5x5 to 7x7, orthonormal real FFT, learn
  full channel mixing on the square `kx,ky in [-2,2]`, inverse FFT, crop to 5x5.
  This retains 25 of 49 spatial-frequency degrees of freedom. One-cell padding
  moves the periodic seam outside the board; it does not make the layer a
  nonperiodic mathematical operator. The local path and stem preserve fine detail.
- Store one real DC matrix and 12 conjugate pairs of real/imaginary matrices.
  Enforce Hermitian symmetry, including the `ky=0` axis. No imaginary DC or unused
  conjugate parameters are trained. Real and imaginary matrices are distinct 2D
  Muon parameters, grouped by the existing batched optimizer implementation.
- Initialize DC with standard deviation `1/sqrt(112)`, each real/imaginary pair
  part with `1/sqrt(224)`, and the pointwise projections with Xavier normal.
  FFTs and complex products run fp32; other operations use the inherited bf16
  autocast. Numerical tests also support double precision.
- Full 5x5 boards only. D4 symmetry is learned through the original augmentation;
  no exact rotation/reflection equivariance is imposed. Python checkpoints only;
  C++ export/search cannot run this trunk, so this experiment produces no Elo.

The width was chosen to put the parameter count near the CNN before examining
any trained FNO validation metrics. Padding, retained modes, depth, initialization,
and the endpoint are fixed for this first run. An inference-only spectral-off
check on the first 2,048 validation rows tests whether the trained model uses its
spectral path. It is not a separately trained control or evidence of superiority.

## Training and evaluation protocol

Reuse `teacher-b5-1000k` on the `katago-pair1` Modal volume: 985,600 training rows,
63,000 validation rows. Reuse the exact one-seed [Un-0 training recipe](../un0_5x5/README.md):
batch 2048, 500K samples per epoch, 20M nominal total, exports every epoch,
Muon, `-no-compile -use-bf16`, LR multipliers
`(0,8.0),(10M,4.0),(14M,2.0),(17M,1.0),(19M,0.5)`, 50K randomized validation rows
per epoch. Report actual samples, because the final epoch overshoots 20M.

Primary endpoint: final raw checkpoint. Re-evaluate all four seed-1 models with
the unchanged shared evaluator: fp32, full history, all 63,000 rows including tails,
all eight D4 symmetries, mean of per-transform losses rather than an ensemble.
Use KataGo's exact weights and denominator. Report policy CE, policy KL, value
loss (1.2 times CE), and top-1 teacher agreement. Curves are exploratory and use
the original randomized epoch validation. No minimum-loss checkpoint selection.

The existing audit found 115 shared games and 17,853 validation rows with a
D4-equivalent training input. Reuse its frozen 44,514-row secondary subset without
shared games or D4-equivalent full inputs. The shared evaluator verifies the
validation-file digest against the audit. This subset was specified before the
FNO run but after prior experiments; it is not an untouched test set, and removing
repeated openings changes its distribution. The audit and masks remain in the
Un-0 results directory and on the volume. Do not overwrite earlier study results.

One seed cannot estimate training variance, significance, or architectural
superiority. The inherited seed does not fully control random symmetries/history
augmentation. Same optimizer/LR is a controlled first recipe, not evidence each
architecture is optimally tuned. Separate real/imaginary Muon updates are another
optimizer parameterization choice. Target entropy does not establish an
irreducible excess-loss floor. These metrics cannot be converted into Elo.

## Execution and provenance

First run numerical/gradient/checkpoint/optimizer tests. Then benchmark the full
H100 training step on real data, including backward and Muon, before launching
training. Inspect nonfinite/extreme gradients and cost from measured throughput.
Only seed 1 is accepted. Training and evaluation have timeouts and zero automatic
retries. Record any interruption or restart rather than silently replacing a run.
Source digests, dataset manifest, configuration, and this protocol are saved at
launch. Existing run names reject a changed specification. The previous trainer
and datasets are reused; historical models are not retrained.

```bash
PYTHONPATH=python python -m pytest -q experiments/fno_5x5 experiments/un0_5x5
MODAL_PROFILE=armoredmeatball modal run experiments/fno_5x5/app.py --stage diag
MODAL_PROFILE=armoredmeatball modal run --detach experiments/fno_5x5/app.py --stage train
MODAL_PROFILE=armoredmeatball modal run experiments/fno_5x5/app.py --stage evaluate
```

Training is stored at `runs/fno-pilot-v1-s1`; diagnostic and evaluation artifacts
at `studies/fno-pilot-v1`. Use the Modal profile per command; do not switch the
user's default profile or stop unrelated workloads. Poll saved checkpoints and
verify this study's apps are stopped at completion. Training runtime estimates
exclude startup, checkpoint I/O, validation, and diagnostic/evaluation work unless
explicitly included; cost estimates are not invoices.
