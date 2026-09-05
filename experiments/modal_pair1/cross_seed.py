#!/usr/bin/env python3
"""Compare the pair-1 convnet and transformer runs across seeds.

Usage: python cross_seed.py <results_dir> <seed> [<seed> ...]

Expects <results_dir>/s<seed>/report.json for every seed, as written by the report
stage (fetch it with `modal volume get katago-pair1 eval/<eval_name>/report.json`).
Each report.json maps run names to {"curve": [...], "checkpoints": [...]}; the run
whose name contains "-conv-" is taken as the convnet and the one containing "-tf-"
as the transformer.

Prints Markdown tables: the per-checkpoint head-to-head (tf minus conv) for every
seed with the mean and SD over seeds, the final validation metrics per seed, the Elo
of every checkpoint inside each seed's own round-robin, and where each run's curves
flatten. Elo scales are per eval, so only the within-seed differences are comparable
across seeds. Checkpoints are matched by index because the log-spaced picker lands on
slightly different sample counts when the seed changes the file order.
"""
import json
import math
import sys


def mean_sd(xs):
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else float("nan")
    return m, sd


def load_runs(results_dir, seeds):
    runs = {}
    for s in seeds:
        d = json.load(open(f"{results_dir}/s{s}/report.json"))
        conv = next(k for k in d if "-conv-" in k)
        tf = next(k for k in d if "-tf-" in k)
        runs[s] = {"conv": d[conv], "tf": d[tf]}
    return runs


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    results_dir, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]
    runs = load_runs(results_dir, seeds)
    n_ckpt = min(len(runs[s][m]["checkpoints"]) for s in seeds for m in ("conv", "tf"))

    def label(i):
        xs = [runs[s]["conv"]["checkpoints"][i]["samples"] for s in seeds]
        lo, hi = min(xs) / 1e6, max(xs) / 1e6
        return f"ckpt {i + 1} ({lo:.1f}M)" if hi - lo < 0.05 else f"ckpt {i + 1} ({lo:.1f}-{hi:.1f}M)"

    print("## Head to head per checkpoint, tf minus conv "
          "(+/- is each seed's match standard error; the mean column shows the SD over seeds)\n")
    hdr = ("| checkpoint | " + " | ".join(f"s{s} Elo" for s in seeds) + " | mean Elo diff (SD) | "
           + " | ".join(f"s{s} p0 diff" for s in seeds) + " | mean p0 diff | "
           + " | ".join(f"s{s} v diff" for s in seeds) + " | mean v diff |")
    print(hdr)
    print("|" + "---|" * (hdr.count("|") - 1))
    for i in range(n_ckpt):
        elos, p0s, vs, cells = [], [], [], []
        for s in seeds:
            c = runs[s]["conv"]["checkpoints"][i]
            t = runs[s]["tf"]["checkpoints"][i]
            de = t["elo"] - c["elo"]
            se = math.sqrt(c["stderr"] ** 2 + t["stderr"] ** 2)
            elos.append(de)
            p0s.append(t["p0loss"] - c["p0loss"])
            vs.append(t["vloss"] - c["vloss"])
            cells.append(f"{de:+.0f} ± {se:.0f}")
        me, sde = mean_sd(elos)
        mp, _ = mean_sd(p0s)
        mv, _ = mean_sd(vs)
        print(f"| {label(i)} | " + " | ".join(cells) + f" | {me:+.0f} ({sde:.0f}) | "
              + " | ".join(f"{x:+.4f}" for x in p0s) + f" | {mp:+.4f} | "
              + " | ".join(f"{x:+.4f}" for x in vs) + f" | {mv:+.4f} |")

    print("\n## Final validation (last epoch) per seed\n")
    print("| run | " + " | ".join(f"s{s}" for s in seeds) + " | mean (SD) |")
    print("|---|" + "---|" * (len(seeds) + 1))
    for m in ("conv", "tf"):
        for k in ("p0loss", "vloss", "pacc1"):
            xs = [runs[s][m]["curve"][-1][k] for s in seeds]
            mu, sd = mean_sd(xs)
            print(f"| {m} {k} | " + " | ".join(f"{x:.4f}" for x in xs) + f" | {mu:.4f} ({sd:.4f}) |")

    print("\n## Elo per checkpoint within each seed's eval (not comparable across seeds)\n")
    print("| checkpoint | " + " | ".join(f"s{s} conv | s{s} tf" for s in seeds) + " |")
    print("|---|" + "---|" * (2 * len(seeds)))
    for i in range(n_ckpt):
        print(f"| {label(i)} | " + " | ".join(
            f"{runs[s]['conv']['checkpoints'][i]['elo']:+.0f} | {runs[s]['tf']['checkpoints'][i]['elo']:+.0f}"
            for s in seeds) + " |")

    print("\n## Flattening: last checkpoint whose Elo gain over the previous one exceeds one standard error; "
          "epoch of minimum validation p0loss\n")
    for s in seeds:
        for m in ("conv", "tf"):
            cps = runs[s][m]["checkpoints"]
            last = None
            for a, b in zip(cps, cps[1:]):
                if b["elo"] - a["elo"] > math.sqrt(a["stderr"] ** 2 + b["stderr"] ** 2):
                    last = b["samples"]
            cur = runs[s][m]["curve"]
            jmin = min(range(len(cur)), key=lambda j: cur[j]["p0loss"])
            print(f"- s{s} {m}: last significant Elo gain at {last:,} samples; "
                  f"min val p0loss {cur[jmin]['p0loss']:.4f} at {cur[jmin]['samples']:,} (final {cur[-1]['p0loss']:.4f})")


if __name__ == "__main__":
    main()
