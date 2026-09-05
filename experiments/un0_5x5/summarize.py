"""Rebuild the seed-1 report and learning-curve plot from saved artifacts.

Requires matplotlib for the plot. Run from any directory:
  python experiments/un0_5x5/summarize.py
"""

import json
from pathlib import Path


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(__file__).resolve().parent
    out = root / "results/seed1"
    old = root.parent / "modal_pair1/results/teacher-b5-1000k/s1"
    report = json.loads((out / "report.json").read_text())
    if not report.get("complete"):
        raise ValueError("Fixed evaluation has not completed")
    records = {}
    for row in report["models"]:
        path = row["checkpoint"]
        name = "CNN" if "pair1-b5-conv" in path else "Transformer" if "pair1-b5-tf" in path else "Un-0 adaptation"
        if name in records:
            raise ValueError(f"Duplicate model {name}")
        records[name] = row
    if set(records) != {"CNN", "Transformer", "Un-0 adaptation"}:
        raise ValueError("Expected exactly the three seed-1 models")

    curves = {}
    for name, path in [("CNN", old / "conv.metrics_val.json"),
                       ("Transformer", old / "tf.metrics_val.json"),
                       ("Un-0 adaptation", out / "un0.metrics_val.json")]:
        curves[name] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    colors = {"CNN": "#247BA0", "Transformer": "#D17A22", "Un-0 adaptation": "#7546A3"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    for ax, metric, label in zip(axes, ["p0loss", "vloss", "pacc1"],
                                 ["Policy cross entropy (nats)", "Value loss (1.2 × cross entropy)", "Top-1 teacher agreement"]):
        for name, rows in curves.items():
            ax.plot([r["nsamp_train"] / 1e6 for r in rows], [r[metric] for r in rows],
                    label=name, color=colors[name], linewidth=1.8)
        ax.set(xlabel="Training samples (millions)", ylabel=label)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("5×5 teacher learning — seed 1\nOriginal randomized per-epoch validation; exploratory curves", fontsize=12)
    fig.savefig(out / "curves.png", dpi=180)
    plt.close(fig)

    un0 = records["Un-0 adaptation"]
    behind = all(un0[metric] > records[name][metric] and
                 un0["unseen_game_and_D4_input"][metric] > records[name]["unseen_game_and_D4_input"][metric]
                 for name in ("CNN", "Transformer") for metric in ("p0loss", "vloss"))
    finding = ("This configuration trails both references on final policy and value loss, "
               "including the subset without game/input overlap. " if behind else
               "The fixed tables below compare this configuration with both references. ")
    finding += "One exploratory run does not establish an architectural advantage or a general limit of oscillator models."
    text = ["# Un-0-inspired 5x5 pilot: seed 1", "",
            "One exploratory run, 20M nominal samples, existing teacher pool. "
            "Seed 2 was cancelled on user request shortly after launch and is excluded.", "",
            finding, "",
            "## Fixed final validation", "",
            "Final raw weights, fp32, full available history, all 63,000 rows and "
            "eight board symmetries. Losses are averaged across transformed examples, "
            "not across ensemble predictions. All three checkpoints were re-evaluated together.", "",
            "| Model | Parameters | Samples | Policy CE ↓ | Value loss ↓ | Top-1 ↑ | Policy KL ↓ |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("CNN", "Transformer", "Un-0 adaptation"):
        r = records[name]
        text.append(f"| {name} | {r['parameters']:,} | {r['samples']:,} | {r['p0loss']:.5f} | "
                    f"{r['vloss']:.5f} | {r['pacc1']:.5f} | {r['policy_kl']:.5f} |")
    text += ["", "Un-0 minus baseline (positive loss differences are worse):", ""]
    for name in ("CNN", "Transformer"):
        ref = records[name]
        text.append(f"- {name}: policy CE {un0['p0loss']-ref['p0loss']:+.5f}; "
                    f"value loss {un0['vloss']-ref['vloss']:+.5f}; "
                    f"top-1 {100*(un0['pacc1']-ref['pacc1']):+.2f} percentage points.")
    audit = report["audit"]
    text += ["", "## Dataset audit and secondary subset", "",
             f"There are {audit['shared_games']:,} games represented in both splits "
             f"({audit['validation_rows_from_shared_games']:,} validation rows). "
             f"{audit['validation_rows_with_exact_train_input']:,} validation rows have an exact training input; "
             f"{audit['validation_rows_with_D4_equivalent_train_input']:,} match after board symmetry.", "",
             f"The following subset retains {audit['validation_rows_with_unseen_game_and_D4_input']:,} rows "
             "whose game and full input (including all eight symmetries) are absent from training. "
             "It was defined after the overlap audit and is secondary. Excluding repeated openings "
             "changes the position distribution; this is not a fresh untouched test set.", "",
             "| Model | Policy CE ↓ | Value loss ↓ | Top-1 ↑ | Policy KL ↓ |",
             "|---|---:|---:|---:|---:|"]
    for name in ("CNN", "Transformer", "Un-0 adaptation"):
        r = records[name]["unseen_game_and_D4_input"]
        text.append(f"| {name} | {r['p0loss']:.5f} | {r['vloss']:.5f} | {r['pacc1']:.5f} | {r['policy_kl']:.5f} |")
    text += ["", "## Dynamics diagnostics", "",
             "First 2,048 validation rows, all eight symmetries. These interventions "
             "were applied at inference to the trained model; they are not separately trained controls.", "",
             "| Intervention | Policy CE | Value loss | Top-1 |", "|---|---:|---:|---:|"]
    for name, r in un0["diagnostics_first_2048"].items():
        text.append(f"| {name} | {r['p0loss']:.5f} | {r['vloss']:.5f} | {r['pacc1']:.5f} |")
    summary = json.loads((out / "un0.summary.json").read_text())
    seconds = summary["elapsed_sec"]
    text += ["", "## Learning curves and runtime", "", "![Learning curves](curves.png)", "",
             "The curves retain the original randomized validation preprocessing; use "
             "the fixed table above for the final comparison. Curves use actual sample counts, not checkpoint indices.", "",
             f"Seed 1 training took {seconds/60:.1f} minutes. At the September 5, 2026 "
             f"H100 + 8 CPU + 32 GiB list rates, this is approximately ${seconds*0.00127284:.2f} "
             "at full resource allocation, excluding startup, diagnostic/evaluation jobs and the "
             "cancelled seed. This is an estimate, not an invoice.", "",
             "## Interpretation limits", "",
             "One seed measures neither training variance nor statistical significance. "
             "The adaptation has 1.72M parameters and a different trunk; it is not matched "
             "on compute, and no LR or solver sweep was run. The audited old pool has "
             "game/position overlap across its file split. These are teacher "
             "prediction metrics, not perfect-play accuracy or Elo. The next controls "
             "are trained no-coupling and tied-weight non-oscillatory models, followed "
             "by a genuinely held-out test and shared search evaluation.", "",
             "See [protocol](../../README.md), [raw fixed evaluation](report.json), "
             "[launch record](launch.json), and [H100 diagnostic](diag/un0.txt).", ""]
    (out / "README.md").write_text("\n".join(text))


if __name__ == "__main__":
    main()
