"""Rebuild the four-model comparison from recorded results; requires matplotlib."""
import json
from pathlib import Path

NAMES = ['CNN', 'Transformer', 'Un-0 adaptation', 'FNO adaptation']


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root = Path(__file__).resolve().parent
    out = root / 'results/seed1'
    report = json.loads((out / 'report.json').read_text())
    if not report.get('complete'):
        raise ValueError('Evaluation incomplete')
    records = {}
    for row in report['models']:
        path = row['checkpoint']
        name = ('CNN' if 'pair1-b5-conv' in path else 'Transformer' if 'pair1-b5-tf' in path
                else 'Un-0 adaptation' if 'un0-pilot' in path else 'FNO adaptation' if 'fno-pilot' in path else None)
        if name is None or name in records:
            raise ValueError(f'Unexpected or duplicate checkpoint: {path}')
        records[name] = row
    if set(records) != set(NAMES) or len({r['samples'] for r in records.values()}) != 1:
        raise ValueError('Expected four models at the same training sample count')
    prior = root.parent / 'modal_pair1/results/teacher-b5-1000k/s1'
    curves = {}
    for name, path in zip(NAMES, [prior / 'conv.metrics_val.json', prior / 'tf.metrics_val.json',
                                 root.parent / 'un0_5x5/results/seed1/un0.metrics_val.json',
                                 out / 'fno.metrics_val.json']):
        curves[name] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    colors = ['#247BA0', '#D17A22', '#7546A3', '#198754']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, metric, label in zip(axes, ['p0loss', 'vloss', 'pacc1'],
                                 ['Policy cross entropy (nats)', 'Value loss (1.2 × CE)', 'Top-1 teacher agreement']):
        for name, color in zip(NAMES, colors):
            rows = curves[name]
            ax.plot([r['nsamp_train'] / 1e6 for r in rows], [r[metric] for r in rows],
                    label=name, color=color, linewidth=1.8)
        ax.set(xlabel='Training samples (millions)', ylabel=label)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=0.15)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('5×5 teacher learning — seed 1\nOriginal randomized epoch validation; exploratory curves', fontsize=12)
    fig.savefig(out / 'curves.png', dpi=180)
    plt.close(fig)

    fno = records['FNO adaptation']
    lines = ['# Fourier neural operator 5x5 pilot: seed 1', '',
             f"One architecture, one seed, {fno['samples']:,} training samples. "
             'The final checkpoint was fixed in advance; no hyperparameter sweep or endpoint selection.', '',
             '## Final fixed validation', '',
             'All four final raw checkpoints were evaluated together: fp32, full history, all 63,000 rows, '
             'mean loss over eight D4 transformations. Lower losses are better; top-1 measures agreement '
             'with the teacher, not perfect play.', '',
             '| Model | Parameters | Policy CE ↓ | Value loss ↓ | Top-1 ↑ | Policy KL ↓ |',
             '|---|---:|---:|---:|---:|---:|']
    for name in NAMES:
        r = records[name]
        lines.append(f"| {name} | {r['parameters']:,} | {r['p0loss']:.5f} | {r['vloss']:.5f} | {r['pacc1']:.5f} | {r['policy_kl']:.5f} |")
    lines += ['', 'FNO minus reference (positive loss differences are worse):', '']
    for name in NAMES[:-1]:
        r = records[name]
        lines.append(f"- {name}: policy CE {fno['p0loss']-r['p0loss']:+.5f}; value loss {fno['vloss']-r['vloss']:+.5f}; "
                     f"top-1 {100*(fno['pacc1']-r['pacc1']):+.2f} percentage points.")
    audit = report['audit']
    lines += ['', '## Secondary subset without overlap', '',
              f"The existing audit found {audit['shared_games']} shared games and "
              f"{audit['validation_rows_with_D4_equivalent_train_input']:,} validation rows matching a training input under D4. "
              f"This subset retains {audit['validation_rows_with_unseen_game_and_D4_input']:,} rows without shared games or full inputs. "
              'It was fixed before this run, but after prior studies, and is not an untouched test set. '
              'Removing repeated openings changes its distribution.', '',
              '| Model | Policy CE ↓ | Value loss ↓ | Top-1 ↑ | Policy KL ↓ |', '|---|---:|---:|---:|---:|']
    for name in NAMES:
        r = records[name]['unseen_game_and_D4_input']
        lines.append(f"| {name} | {r['p0loss']:.5f} | {r['vloss']:.5f} | {r['pacc1']:.5f} | {r['policy_kl']:.5f} |")
    diag = json.loads((out / 'diagnostics.json').read_text())
    lines += ['', '## Spectral-path intervention', '',
              'First 2,048 validation rows, all eight symmetries. Disabling all five spectral paths at inference '
              'tests their contribution to this trained model; this is not a trained pointwise control.', '',
              '| Condition | Policy CE | Value loss | Top-1 |', '|---|---:|---:|---:|']
    for name in ['base', 'spectral_off']:
        r = diag[name]
        lines.append(f"| {name} | {r['p0loss']:.5f} | {r['vloss']:.5f} | {r['pacc1']:.5f} |")
    summary = json.loads((out / 'fno.summary.json').read_text())
    seconds = summary['elapsed_sec']
    lines += ['', '## Learning curves and runtime', '', '![Learning curves](curves.png)', '',
              'Curves use actual sample counts and the original randomized validation preprocessing. '
              'Use the fixed tables above for the endpoint comparison.', '',
              f"FNO training took {seconds / 60:.1f} minutes. At full allocation of H100 + 8 CPU + 32 GiB, "
              f"the training subprocess costs approximately ${seconds * 0.00127284:.2f} using "
              '[Modal list rates](https://modal.com/pricing) checked September 5, 2026. '
              'This includes training-loop validation and checkpoint writing, but excludes container startup, '
              'the short diagnostic, and final evaluation. This is an estimate, not a bill.', '',
              '## Interpretation', '',
              'This is a residual bottleneck FNO adaptation following '
              '[Li et al., including Anima Anandkumar](https://arxiv.org/abs/2010.08895v3). '
              'Its 1.98M parameters are close to the CNN but compute is not matched. One seed with one inherited '
              'Muon/LR recipe cannot establish training variance, significance, or a general architectural ranking. '
              'Resolution transfer, PDE-solving claims, and Go playing strength are not tested. '
              'There is no FNO Elo: the C++ search engine does not support this block.', '',
              'See [frozen protocol](../../README.md), [raw evaluation](report.json), '
              '[launch specification](study.json), [execution record](launch.json), '
              'and [H100 diagnostic](diag/fno.txt).', '']
    (out / 'README.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
