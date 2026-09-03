"""Modal pipeline for the KataGo "pair 1" architecture comparison.

Trains two ~2M-parameter networks that share the nested-bottleneck wrapper,
trunk width (192), inner width (96) and depth (5 blocks) but differ in the
inner block type:

  * conv: b5c192nbt-fson-mish-rvglr-bnh    (two 3x3 convolution residual blocks)
  * tf:   b5c192h3nbttfrs-fson-silu-rsnh   (3-head RoPE attention + gated FFN)

Both are trained supervised on the same fixed pool of kata1 self-play data with
identical batch size, LR schedule, seed and data order, then compared on
held-out loss and on fixed-visit Elo from round-robin matches between exported
checkpoints.

Usage (from the repo root):

  modal run experiments/modal_pair1/app.py --stage smoke        # minutes, cents
  modal run --detach experiments/modal_pair1/app.py --stage all # full pipeline
  modal run experiments/modal_pair1/app.py --stage data|train|export|eval|report
  modal run experiments/modal_pair1/app.py --stage diag --diag-gpu h100   # benchmark variants
  modal run experiments/modal_pair1/app.py --stage gen --board-size 5     # teacher self-play data
  modal run --detach experiments/modal_pair1/app.py --stage gen --board-size 5 --gen-rows 1000000 \
      --gen-visits 200 --gen-cheap-visits 50 --gen-cheap-prob 0.5 --gen-shards 4   # ~30 min on 4 L4s

Both runs pass "-no-compile -use-bf16" to train.py by default: with torch 2.8 the compiled
transformer produces non-finite gradients (see README, "Known issue"), and bf16 autocast
matched fp32 losses in the long smoke while running 1.75x to 2.5x faster. Use --stage diag to re-check.

See README.md next to this file for details, costs and knobs.
"""

import datetime as _dt
import glob
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import modal

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

APP_NAME = "katago-pair1"
VOLUME_NAME = "katago-pair1"
DATA = "/data"  # volume mount point inside containers

REMOTE_REPO = "/root/KataGo"


def _find_repo_root() -> Path:
    """Locate the KataGo checkout locally (for image building). Inside a Modal container this
    module is mounted at /root/app.py, where no checkout exists; only the remote paths matter there."""
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "python" / "train.py").exists():
            return candidate
    return Path(REMOTE_REPO)


REPO_ROOT = _find_repo_root()
PY_DIR = f"{REMOTE_REPO}/python"
KATAGO_BIN = f"{REMOTE_REPO}/cpp/build/katago"

ARCHIVE_URL = "https://katagoarchive.org/kata1/trainingdata/{date}npzs.tgz"

DEFAULT_CONV_KIND = "b5c192nbt-fson-mish-rvglr-bnh"
DEFAULT_TF_KIND = "b5c192h3nbttfrs-fson-silu-rsnh"
# Piecewise-constant multiplier on train.py's hardcoded per-sample LR. The main
# run uses 8.0 early on; we step down through a cooldown so every checkpoint
# after ~100M has seen at least one drop, and the final one is fully annealed.
DEFAULT_LR_SCHEDULE = "(0,8.0),(100M,4.0),(140M,2.0),(170M,1.0),(190M,0.5)"

OPTIMIZER_FLAGS = {
    "sgd": [],
    "adamw": ["-use-adamw"],
    "muon": ["-use-muon"],
}

# Teacher for generated small-board data: the strongest released b18c384nbt (fast, and far stronger
# than anything a 2M-param net will reach). Any katagotraining.org .bin.gz URL works.
DEFAULT_TEACHER_URL = (
    "https://media.katagotraining.org/uploaded/networks/models/kata1/"
    "kata1-b18c384nbt-s9996604416-d4316597426.bin.gz"
)
# Self-play config template; board size, visits and threads are overridden on the command line.
SELFPLAY_TEMPLATE_CFG = f"{REMOTE_REPO}/cpp/configs/training/selfplay1_maxsize9.cfg"
# Fair komi under area scoring for solved boards (5x5: Black wins by 25; 7x7: by 9), else 7.
DEFAULT_KOMI_BY_SIZE = {5: 25.0, 7: 9.0}
# Modal fixes a function's resources when it is defined, so the generation container exists at these
# CPU sizes (physical cores; Modal gives 2 vCPUs per core) and --gen-cpu picks one. See GEN_SHARD_FUNCTIONS.
GEN_CPU_SIZES = (8, 16, 32)


def default_komi(board_size: int) -> float:
    return DEFAULT_KOMI_BY_SIZE.get(int(board_size), 7.0)

VAL_METRIC_KEYS = ("nsamp_train", "p0loss", "vloss", "pacc1", "loss")

# ----------------------------------------------------------------------------
# Image: CUDA toolchain for the C++ engine (match games) + torch for training.
# ----------------------------------------------------------------------------

CUDA_IMAGE = "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04"
CPP_IGNORE = ["build", "build/**", "**/__pycache__/**", "**/*.o", "**/*.a", "katago", "tests/scratch/**"]
PY_IGNORE = ["**/__pycache__/**", "**/*.pyc", ".pytest_cache/**"]

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.11")
    .apt_install("git", "cmake", "build-essential", "zlib1g-dev", "libzip-dev", "wget", "ca-certificates", "pigz")
    .pip_install("torch==2.8.0", extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("numpy", "scipy", "psutil", "packaging", "sgfmill")
    # The C++ engine is baked into the image so the (slow, multi-arch) CUDA build is cached.
    .add_local_dir(REPO_ROOT / "cpp", remote_path=f"{REMOTE_REPO}/cpp", copy=True, ignore=CPP_IGNORE)
    .run_commands(
        f"cd {REMOTE_REPO}/cpp && mkdir -p build && cd build"
        " && cmake .. -DUSE_BACKEND=CUDA -DNO_GIT_REVISION=1 -DCMAKE_BUILD_TYPE=Release"
        " && make -j$(nproc)"
        f" && {KATAGO_BIN} version"
    )
    .env({"PYTHONUNBUFFERED": "1"})
    # Python training code is mounted at container start so edits don't rebuild the image.
    .add_local_dir(REPO_ROOT / "python", remote_path=PY_DIR, ignore=PY_IGNORE)
)

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# ----------------------------------------------------------------------------
# Helpers (run inside containers)
# ----------------------------------------------------------------------------


def run_cmd(cmd, cwd, log_path=None, env=None):
    """Run a subprocess, streaming output to stdout and optionally teeing to a log file."""
    print("+ " + " ".join(shlex.quote(str(c)) for c in cmd), flush=True)
    logf = open(log_path, "a") if log_path else None
    try:
        proc = subprocess.Popen(
            [str(c) for c in cmd], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            if logf:
                logf.write(line)
        proc.wait()
    finally:
        if logf:
            logf.close()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(map(str, cmd))}")


def read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def dataset_dir(dataset):
    return f"{DATA}/shuffled/{dataset}"


def run_root_for(run_name):
    return f"{DATA}/runs/{run_name}"


def eval_root_for(eval_name):
    return f"{DATA}/eval/{eval_name}"


def train_run(cfg, datadir, run_root, pos_len=19, board_size=19):
    """Run python/train.py for one model on a fixed dataset. Resumes if a checkpoint exists."""
    traindir = f"{run_root}/train"
    exportdir = f"{run_root}/torchmodels"
    os.makedirs(traindir, exist_ok=True)
    os.makedirs(exportdir, exist_ok=True)
    with open(f"{run_root}/run.json", "w") as f:
        json.dump({**cfg, "pos_len": pos_len, "board_size": board_size}, f, indent=2)

    cmd = [
        sys.executable, "train.py",
        "-traindir", traindir,
        "-datadir", datadir,
        "-exportdir", exportdir,
        "-exportprefix", cfg["run_name"],
        "-pos-len", pos_len,
        "-batch-size", cfg["batch_size"],
        "-model-kind", cfg["model_kind"],
        "-samples-per-epoch", cfg["samples_per_epoch"],
        "-epochs-per-export", cfg["epochs_per_export"],
        "-max-training-samples", cfg["max_samples"],
        "-lr-schedule", cfg["lr_schedule"],
        "-max-val-samples", cfg["max_val_samples"],
        "-seed", cfg["seed"],
        "-quit-if-no-data",
    ]
    cmd += OPTIMIZER_FLAGS[cfg["optimizer"]]
    cmd += shlex.split(cfg.get("extra_args", "") or "")

    t0 = time.time()
    run_cmd(cmd, cwd=PY_DIR, log_path=f"{run_root}/stdout.txt")
    elapsed = time.time() - t0

    vals = read_jsonl(f"{traindir}/metrics_val.json")
    last = vals[-1] if vals else {}
    summary = {
        "run_name": cfg["run_name"],
        "model_kind": cfg["model_kind"],
        "elapsed_sec": round(elapsed),
        "val_epochs_logged": len(vals),
        "last_val": {k: last.get(k) for k in VAL_METRIC_KEYS},
        "exports": sorted(d for d in os.listdir(exportdir) if not d.endswith(".tmp")),
    }
    with open(f"{run_root}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def export_run_dir(run_root):
    """Convert every torch checkpoint under <run_root>/torchmodels into a .bin.gz the C++ engine can load."""
    src = f"{run_root}/torchmodels"
    dst = f"{run_root}/exported"
    os.makedirs(dst, exist_ok=True)
    done = []
    for name in sorted(os.listdir(src)):
        if name.endswith(".tmp"):
            continue
        ckpt = f"{src}/{name}/model.ckpt"
        if not os.path.exists(ckpt):
            continue
        out = f"{dst}/{name}"
        if glob.glob(f"{out}/model.bin*"):
            done.append(name)
            continue
        tmp = out + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        run_cmd(
            [
                sys.executable, "export_model_pytorch.py",
                "-checkpoint", ckpt,
                "-export-dir", tmp,
                "-model-name", name,
                "-filename-prefix", "model",
                "-use-swa",
            ],
            cwd=PY_DIR,
            log_path=f"{run_root}/export_stdout.txt",
        )
        # Mirror export_model_for_selfplay.sh: gzip the .bin so the engine gets the usual .bin.gz.
        subprocess.run(["gzip", "-f", f"{tmp}/model.bin"], check=True)
        os.rename(tmp, out)
        done.append(name)
    return done


def list_exported(run_root):
    """Exported checkpoints of a run as [{name, samples, path}], sorted by sample count."""
    out = []
    exp = f"{run_root}/exported"
    if not os.path.isdir(exp):
        return out
    for name in os.listdir(exp):
        m = re.match(r"^(.*)-s(\d+)-d(\d+)$", name)
        files = glob.glob(f"{exp}/{name}/model.bin*")
        if not m or not files:
            continue
        out.append({"name": f"{m.group(1)}-s{m.group(2)}", "samples": int(m.group(2)), "path": files[0]})
    out.sort(key=lambda e: e["samples"])
    return out


def log_spaced_subset(entries, n):
    """Pick n entries with roughly log-spaced sample counts, always including the first and last."""
    import math

    if n <= 0 or n >= len(entries):
        return list(entries)
    lo = max(entries[0]["samples"], 1)
    hi = max(entries[-1]["samples"], 1)
    chosen, used = [], set()
    for i in range(n):
        target = lo * (hi / lo) ** (i / (n - 1)) if hi > lo else lo
        candidates = [e for e in entries if e["samples"] not in used]
        best = min(candidates, key=lambda e: abs(math.log(max(e["samples"], 1)) - math.log(target)))
        used.add(best["samples"])
        chosen.append(best)
    chosen.sort(key=lambda e: e["samples"])
    return chosen


def write_match_config(path, bots, secondary_idxs, visits, num_games, game_threads, board_size=19, komi=7.0, komi_auto=False):
    """Write a complete `katago match` config: fixed rules, one board size, fixed (or auto) komi, fixed visits, fp32."""
    lines = [
        "# Generated by experiments/modal_pair1/app.py",
        "logSearchInfo = false",
        "logMoves = false",
        "logGamesEvery = 25",
        "logToStdout = true",
        "",
        f"numBots = {len(bots)}",
    ]
    for i, b in enumerate(bots):
        lines.append(f"botName{i} = {b['name']}")
        lines.append(f"nnModelFile{i} = {b['path']}")
    if secondary_idxs:
        lines.append("secondaryBots = " + ",".join(str(i) for i in secondary_idxs))
    lines += [
        "",
        f"numGameThreads = {game_threads}",
        f"numGamesTotal = {num_games}",
        "maxMovesPerGame = 1200",
        "allowResignation = true",
        "resignThreshold = -0.95",
        "resignConsecTurns = 6",
        "",
        "koRules = POSITIONAL",
        "scoringRules = AREA",
        "taxRules = NONE",
        "multiStoneSuicideLegals = false",
        "hasButtons = false",
        f"bSizes = {int(board_size)}",
        "bSizeRelProbs = 1",
        f"komiAuto = {'true' if komi_auto else 'false'}",
        *([] if komi_auto else [f"komiMean = {float(komi)}"]),
        "handicapProb = 0.0",
        "handicapCompensateKomiProb = 1.0",
        "",
        f"maxVisits = {visits}",
        "numSearchThreads = 1",
        f"nnMaxBatchSize = {max(8, game_threads)}",
        "nnCacheSizePowerOfTwo = 20",
        "nnMutexPoolSizePowerOfTwo = 16",
        "nnRandomize = true",
        "numNNServerThreadsPerModel = 1",
        "useFP16 = false",
        "",
        "chosenMoveTemperatureEarly = 0.60",
        "chosenMoveTemperature = 0.20",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def compute_elos(sgf_dir, elo_prior_games=2.0):
    """Bayes Elo over every game under sgf_dir, via the repo's own summarize_sgfs machinery."""
    if PY_DIR not in sys.path:
        sys.path.insert(0, PY_DIR)
    from summarize_sgfs import GoGameResultSummary  # noqa: WPS433 (runtime import inside container)

    summ = GoGameResultSummary(elo_prior_games=elo_prior_games, estimate_first_player_advantage=False)
    summ.add_games_from_file_or_dir(sgf_dir, recursive=True)
    info = summ.get_elos()
    rows = [
        {
            "bot": p,
            "elo": float(info.elo[p]),
            "stderr": float(info.elo_stderr[p]),
            "games": float(info.effective_game_count[p]),
        }
        for p in info.players
    ]
    rows.sort(key=lambda r: -r["elo"])
    return rows, str(info)


def run_match(eval_root, bots, secondary_idxs, visits, games_per_bot, game_threads, board_size=19, komi=7.0, komi_auto=False):
    """Play a round-robin among bots (secondary bots only play primaries) and compute Elo."""
    os.makedirs(eval_root, exist_ok=True)
    primary = len(bots) - len(secondary_idxs)
    num_games = max(1, games_per_bot * primary // 2)
    cfg_path = f"{eval_root}/match.cfg"
    sgf_dir = f"{eval_root}/sgfs"
    os.makedirs(sgf_dir, exist_ok=True)
    write_match_config(cfg_path, bots, secondary_idxs, visits, num_games, game_threads, board_size, komi, komi_auto)
    with open(f"{eval_root}/bots.json", "w") as f:
        json.dump({"bots": bots, "secondary": secondary_idxs, "visits": visits, "num_games": num_games,
                   "board_size": board_size, "komi": komi, "komi_auto": komi_auto}, f, indent=2)

    t0 = time.time()
    run_cmd(
        [KATAGO_BIN, "match", "-config", cfg_path, "-log-file", f"{eval_root}/match.log", "-sgf-output-dir", sgf_dir],
        cwd=eval_root,
        log_path=f"{eval_root}/stdout.txt",
    )
    elapsed = time.time() - t0

    rows, text = compute_elos(sgf_dir)
    with open(f"{eval_root}/elo.txt", "w") as f:
        f.write(text + "\n")
    with open(f"{eval_root}/elo.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(text, flush=True)
    return {"num_games": num_games, "elapsed_sec": round(elapsed), "elos": rows}


def build_report(run_roots, eval_root):
    """Join per-epoch validation metrics with checkpoint Elos into report.md / report.json."""
    elo_path = f"{eval_root}/elo.json"
    elos = {}
    if os.path.exists(elo_path):
        with open(elo_path) as f:
            elos = {r["bot"]: r for r in json.load(f)}

    per_run = {}
    for run, root in run_roots.items():
        curve = []
        for v in read_jsonl(f"{root}/train/metrics_val.json"):
            curve.append(
                {
                    "samples": int(v.get("nsamp_train", 0) or 0),
                    "p0loss": v.get("p0loss"),
                    "vloss": v.get("vloss"),
                    "pacc1": v.get("pacc1"),
                    "loss": v.get("loss"),
                }
            )
        checkpoints = []
        for bot, row in elos.items():
            m = re.match(rf"^{re.escape(run)}-s(\d+)$", bot)
            if not m:
                continue
            s = int(m.group(1))
            nearest = min(curve, key=lambda c: abs(c["samples"] - s)) if curve else {}
            checkpoints.append({"bot": bot, "samples": s, "elo": row["elo"], "stderr": row["stderr"], "games": row["games"], **{k: nearest.get(k) for k in ("p0loss", "vloss", "pacc1")}})
        checkpoints.sort(key=lambda c: c["samples"])
        per_run[run] = {"curve": curve, "checkpoints": checkpoints}

    def fmt(x, nd=4):
        return "-" if x is None else f"{x:.{nd}f}"

    md = ["# Pair 1 report", ""]
    md.append(f"Runs: {', '.join(run_roots)}")
    md.append(f"Eval: {eval_root}")
    md.append("")
    for run, d in per_run.items():
        md.append(f"## {run}")
        md.append("")
        md.append("| samples | p0loss | vloss | pacc1 | Elo | +/- | games |")
        md.append("|---|---|---|---|---|---|---|")
        for c in d["checkpoints"]:
            md.append(
                f"| {c['samples']:,} | {fmt(c['p0loss'])} | {fmt(c['vloss'])} | {fmt(c['pacc1'])} | {c['elo']:.0f} | {c['stderr']:.0f} | {c['games']:.0f} |"
            )
        if not d["checkpoints"]:
            md.append("| (no evaluated checkpoints) | | | | | | |")
        md.append("")
        if d["curve"]:
            last = d["curve"][-1]
            md.append(f"Final validation ({last['samples']:,} samples): p0loss {fmt(last['p0loss'])}, vloss {fmt(last['vloss'])}, pacc1 {fmt(last['pacc1'])}")
            md.append("")

    runs = list(per_run)
    if len(runs) == 2:
        a, b = runs
        by_s_a = {c["samples"]: c for c in per_run[a]["checkpoints"]}
        by_s_b = {c["samples"]: c for c in per_run[b]["checkpoints"]}
        common = sorted(set(by_s_a) & set(by_s_b))
        if common:
            md.append(f"## Head to head: {b} minus {a}")
            md.append("")
            md.append("| samples | Elo diff | +/- (approx) | p0loss diff | vloss diff |")
            md.append("|---|---|---|---|---|")
            for s in common:
                ca, cb = by_s_a[s], by_s_b[s]
                se = (ca["stderr"] ** 2 + cb["stderr"] ** 2) ** 0.5
                dp = None if ca["p0loss"] is None or cb["p0loss"] is None else cb["p0loss"] - ca["p0loss"]
                dv = None if ca["vloss"] is None or cb["vloss"] is None else cb["vloss"] - ca["vloss"]
                md.append(f"| {s:,} | {cb['elo'] - ca['elo']:+.0f} | {se:.0f} | {fmt(dp)} | {fmt(dv)} |")
            md.append("")
            md.append("Positive Elo diff means the second run is stronger; negative loss diff means the second run fits better.")
            md.append("")

    os.makedirs(eval_root, exist_ok=True)
    with open(f"{eval_root}/report.md", "w") as f:
        f.write("\n".join(md))
    with open(f"{eval_root}/report.json", "w") as f:
        json.dump(per_run, f, indent=2)
    text = "\n".join(md)
    print(text, flush=True)
    return text


def make_train_cfg(p, run_name, model_kind, extra_args=""):
    return {
        "run_name": run_name,
        "model_kind": model_kind,
        "dataset": p["dataset"],
        "max_samples": p["max_samples"],
        "batch_size": p["batch_size"],
        "samples_per_epoch": p["samples_per_epoch"],
        "epochs_per_export": p["epochs_per_export"],
        "lr_schedule": p["lr_schedule"],
        "max_val_samples": p["max_val_samples"],
        "seed": p["seed"],
        "optimizer": p["optimizer"],
        "extra_args": (p["train_extra_args"] + " " + extra_args).strip(),
    }


def shuffle_pool(raw_dir, tmp_out, keep_rows, val_keep_rows, num_waves=4, val_frac=0.05):
    """Shuffle every npz under raw_dir into tmp_out/{train,val} with shuffle.py.
    5% of files by path hash become validation, the same rule as python/selfplay/shuffle.sh.
    keep_rows / val_keep_rows are row counts or "all". Each wave writes its own output files, so
    small pools should use num_waves=1: train.py drops partial batches, and a validation file with
    fewer rows than the batch size contributes nothing."""
    os.makedirs(tmp_out, exist_ok=True)
    nproc = os.cpu_count() or 16

    def shuffle(split, keep, lbound, ubound):
        shutil.rmtree(f"/tmp/shuf/{split}", ignore_errors=True)
        os.makedirs(f"/tmp/shuf/{split}", exist_ok=True)
        run_cmd(
            [
                sys.executable, "shuffle.py", raw_dir,
                # shuffle.py quits if the pool has fewer than -min-rows rows, and with the linear
                # defaults below the window is (rows - min_rows) + min_rows, i.e. the entire pool.
                "-min-rows", 1,
                "-expand-window-per-row", 1.0,
                "-taper-window-exponent", 1.0,
                "-keep-target-rows", keep,
                "-out-dir", f"{tmp_out}/{split}",
                "-out-tmp-dir", f"/tmp/shuf/{split}",
                "-approx-rows-per-out-file", 70000,
                "-num-processes", nproc,
                "-num-waves", num_waves,
                "-only-include-md5-path-prop-lbound", lbound,
                "-only-include-md5-path-prop-ubound", ubound,
            ],
            cwd=PY_DIR,
            log_path=f"{tmp_out}/shuffle_{split}.txt",
        )

    t1 = time.time()
    split = 1.0 - float(val_frac)
    shuffle("val", val_keep_rows, split, 1.00)
    shuffle("train", keep_rows, 0.00, split)
    print(f"Shuffle took {time.time()-t1:.0f}s", flush=True)


def count_npz_rows(root):
    """Total training rows and file count under root (reads one array per npz)."""
    import numpy as np

    rows, files = 0, 0
    for dirpath, _, names in os.walk(root):
        for n in names:
            if n.endswith(".npz"):
                with np.load(os.path.join(dirpath, n)) as z:
                    rows += int(z["binaryInputNCHWPacked"].shape[0])
                files += 1
    return rows, files


def split_counts(tmp_out):
    def count(split):
        files = glob.glob(f"{tmp_out}/{split}/*.npz")
        return {"files": len(files), "bytes": sum(os.path.getsize(f) for f in files)}

    with open(f"{tmp_out}/train.json") as f:
        train_range = json.load(f).get("range")
    return {"train": count("train"), "val": count("val"), "train_range": train_range}


# ----------------------------------------------------------------------------
# Modal functions
# ----------------------------------------------------------------------------


@app.function(image=image, cpu=16.0, memory=65536, timeout=8 * 3600, volumes={DATA: vol})
def prepare_data(dataset: str, dates: list, keep_rows: int, val_keep_rows: int, force: bool = False) -> dict:
    """Download kata1 daily archives, extract on local disk, shuffle into /data/shuffled/<dataset>/{train,val}."""
    from concurrent.futures import ThreadPoolExecutor

    vol.reload()
    out_root = dataset_dir(dataset)
    manifest_path = f"{out_root}/dataset.json"
    if os.path.exists(f"{out_root}/train.json") and not force:
        print(f"Dataset already prepared at {out_root}; skipping (use --force-data to rebuild)", flush=True)
        with open(manifest_path) as f:
            return json.load(f)
    shutil.rmtree(out_root + ".tmp", ignore_errors=True)
    if force:
        shutil.rmtree(out_root, ignore_errors=True)

    raw = "/tmp/raw"
    dl = "/tmp/dl"
    os.makedirs(raw, exist_ok=True)
    os.makedirs(dl, exist_ok=True)

    def fetch(date):
        url = ARCHIVE_URL.format(date=date)
        tgz = f"{dl}/{date}.tgz"
        dest = f"{raw}/{date}"
        r = subprocess.run(["wget", "-q", "--tries=3", "-O", tgz, url])
        if r.returncode != 0 or not os.path.exists(tgz) or os.path.getsize(tgz) == 0:
            if os.path.exists(tgz):
                os.remove(tgz)
            print(f"MISSING {url}", flush=True)
            return date, False, 0
        size = os.path.getsize(tgz)
        os.makedirs(dest, exist_ok=True)
        subprocess.run(["tar", "-I", "pigz", "-xf", tgz, "-C", dest], check=True)
        os.remove(tgz)
        n = sum(len([f for f in files if f.endswith(".npz")]) for _, _, files in os.walk(dest))
        print(f"OK {date}: {size/1e9:.2f} GB, {n} npz files", flush=True)
        return date, True, n

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(fetch, dates))
    got = [d for d, ok, _ in results if ok]
    missing = [d for d, ok, _ in results if not ok]
    total_files = sum(n for _, ok, n in results if ok)
    if not got:
        raise RuntimeError("No archives downloaded; check the date range against https://katagoarchive.org/kata1/trainingdata/")
    print(f"Downloaded {len(got)} days ({total_files} npz files) in {time.time()-t0:.0f}s; missing: {missing}", flush=True)

    tmp_out = out_root + ".tmp"
    shuffle_pool(raw, tmp_out, keep_rows, val_keep_rows)
    manifest = {
        "dataset": dataset,
        "source": "kata1-archive",
        "board_size": 19,
        "pos_len": 19,
        "dates": got,
        "missing_dates": missing,
        "raw_npz_files": total_files,
        "keep_rows": keep_rows,
        "val_keep_rows": val_keep_rows,
        **split_counts(tmp_out),
        "created": _dt.datetime.utcnow().isoformat() + "Z",
    }
    with open(f"{tmp_out}/dataset.json", "w") as f:
        json.dump(manifest, f, indent=2)
    os.rename(tmp_out, out_root)
    vol.commit()
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


# Modal list prices (September 2026), only used for the cost estimate written to the dataset manifest.
MODAL_USD_PER_SEC = {"L4": 0.000222, "cpu_core": 0.0000131, "gib": 0.00000222}
GEN_MEMORY_MB = 32768
GEN_SYNC_INTERVAL_SEC = 60


def _teacher_cache(teacher_url: str) -> tuple:
    """Download the teacher net into the volume cache if needed. Returns (cached path, net name)."""
    teacher_file = os.path.basename(urlparse(teacher_url).path)
    teacher_name = teacher_file[: -len(".bin.gz")] if teacher_file.endswith(".bin.gz") else teacher_file
    cached = f"{DATA}/teachers/{teacher_file}"
    if not os.path.exists(cached):
        os.makedirs(f"{DATA}/teachers", exist_ok=True)
        subprocess.run(["wget", "-q", "--tries=3", "-O", cached + ".tmp", teacher_url], check=True)
        os.rename(cached + ".tmp", cached)
        vol.commit()
    return cached, teacher_name


def _sync_shard_output(local_dir: str, shard_dir: str, final: bool = False) -> int:
    """Copy finished training files (and, when final, sgfs and logs) from the engine's local output
    directory to the shard's directory on the volume, skipping files already there. The engine renames
    each npz into place once complete, so anything ending in .npz is safe to copy while it runs."""
    copied = 0
    for dirpath, _, names in os.walk(local_dir):
        rel = os.path.relpath(dirpath, local_dir)
        for n in names:
            if not (n.endswith(".npz") or (final and not n.endswith(".tmp"))):
                continue
            dst = os.path.join(shard_dir, rel, n)
            if os.path.exists(dst):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(os.path.join(dirpath, n), dst + ".part")
            os.rename(dst + ".part", dst)
            copied += 1
    return copied


def _generate_teacher_shard(dataset: str, shard: int, num_shards: int, teacher_file: str, target_rows: int,
                            overrides: dict) -> dict:
    """One L4 container: play self-play games with the teacher until this shard holds target_rows rows.
    The engine writes to local disk; finished npz files are copied to
    /data/shuffled/<dataset>.tmp/raw/shard_<i>/ every GEN_SYNC_INTERVAL_SEC, so a killed container keeps
    all but its last minute of rows. Returns per-shard throughput statistics."""
    import resource
    import threading

    teacher_name = teacher_file[: -len(".bin.gz")] if teacher_file.endswith(".bin.gz") else teacher_file
    shard_dir = f"{dataset_dir(dataset)}.tmp/raw/shard_{shard:02d}"
    os.makedirs(shard_dir, exist_ok=True)
    models_dir = "/tmp/gen/models"
    os.makedirs(f"{models_dir}/{teacher_name}", exist_ok=True)
    shutil.copy(f"{DATA}/teachers/{teacher_file}", f"{models_dir}/{teacher_name}/model.bin.gz")
    cfg_path = "/tmp/gen/selfplay.cfg"
    shutil.copy(SELFPLAY_TEMPLATE_CFG, cfg_path)
    override_str = ",".join(f"{k}={v}" for k, v in overrides.items())
    out_dir = "/tmp/gen/selfplay"
    os.makedirs(out_dir, exist_ok=True)
    log_path = f"{out_dir}/selfplay_stdout.txt"
    board_size, cheap_prob, game_threads = overrides["dataBoardLen"], overrides["cheapSearchProb"], overrides["numGameThreads"]

    stop = threading.Event()

    def syncer():
        while not stop.wait(GEN_SYNC_INTERVAL_SEC):
            try:
                _sync_shard_output(out_dir, shard_dir)
            except Exception as e:  # the final sync below retries; keep generating
                print(f"shard {shard}: sync warning: {e}", flush=True)

    # Only moves with positive training weight are written (cheap searches have weight 0), so a game
    # yields about (1 - cheapSearchProb) of its moves as rows: 6.2 rows per 5x5 game measured at 0.75.
    # Generate in chunks and re-estimate rows per game.
    rows_per_game = max(3.0, (1.0 - float(cheap_prob)) * board_size * board_size)
    rows, files, total_games, chunk = 0, 0, 0, 0
    ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.time()
    sync_thread = threading.Thread(target=syncer, daemon=True)
    sync_thread.start()
    try:
        while rows < target_rows and chunk < 4:
            games = max(int(game_threads), int(math.ceil((target_rows - rows) / rows_per_game * 1.1)))
            run_cmd(
                [
                    KATAGO_BIN, "selfplay",
                    "-models-dir", models_dir,
                    "-output-dir", out_dir,
                    "-config", cfg_path,
                    "-override-config", override_str,
                    "-max-games-total", games,
                ],
                cwd="/tmp/gen",
                log_path=log_path,
            )
            total_games += games
            chunk += 1
            rows, files = count_npz_rows(out_dir)
            rows_per_game = max(1.0, rows / max(1, total_games))
            print(f"shard {shard}/{num_shards} chunk {chunk}: {total_games} games, {rows:,} rows in {files} files, "
                  f"{rows_per_game:.1f} rows/game, {rows/(time.time()-t0):.0f} rows/s", flush=True)
    finally:
        stop.set()
        sync_thread.join()
    gen_sec = time.time() - t0
    ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_sec = (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)
    with open(log_path) as f:
        log = f.read()
    nn_rows = sum(int(x) for x in re.findall(r"Final NN rows: (\d+)", log))
    nn_batches = sum(int(x) for x in re.findall(r"Final NN batches: (\d+)", log))
    synced = _sync_shard_output(out_dir, shard_dir, final=True)
    vol.commit()
    stats = {
        "shard": shard,
        "rows": rows,
        "files": files,
        "games": total_games,
        "chunks": chunk,
        "gen_sec": round(gen_sec, 1),
        "cpu_sec": round(cpu_sec, 1),
        "nn_evals": nn_rows,
        "nn_batches": nn_batches,
        "synced_files": synced,
    }
    print(f"shard {shard}/{num_shards} done: {json.dumps(stats)}", flush=True)
    return stats


# One shard function per CPU size (Modal fixes resources per function); --gen-cpu picks one. Each
# takes the keyword arguments of _generate_teacher_shard as a dict so it works with .map.
@app.function(image=image, gpu="L4", cpu=8.0, memory=GEN_MEMORY_MB, timeout=12 * 3600, volumes={DATA: vol})
def generate_teacher_shard_cpu8(args: dict) -> dict:
    return _generate_teacher_shard(**args)


@app.function(image=image, gpu="L4", cpu=16.0, memory=GEN_MEMORY_MB, timeout=12 * 3600, volumes={DATA: vol})
def generate_teacher_shard_cpu16(args: dict) -> dict:
    return _generate_teacher_shard(**args)


@app.function(image=image, gpu="L4", cpu=32.0, memory=GEN_MEMORY_MB, timeout=12 * 3600, volumes={DATA: vol})
def generate_teacher_shard_cpu32(args: dict) -> dict:
    return _generate_teacher_shard(**args)


GEN_SHARD_FUNCTIONS = {8: generate_teacher_shard_cpu8, 16: generate_teacher_shard_cpu16, 32: generate_teacher_shard_cpu32}
assert tuple(GEN_SHARD_FUNCTIONS) == GEN_CPU_SIZES


@app.function(image=image, cpu=8.0, memory=32768, timeout=4 * 3600, volumes={DATA: vol})
def shuffle_teacher_pool(dataset: str, manifest: dict) -> dict:
    """Shuffle the merged raw shard output of a dataset into train/val and finalize the dataset directory.
    The raw files stay under <dataset>/raw so the pool can be reshuffled (e.g. with another --val-frac)."""
    vol.reload()
    out_root = dataset_dir(dataset)
    tmp_out = out_root + ".tmp"
    raw_dir = f"{tmp_out}/raw"
    rows, files = count_npz_rows(raw_dir)
    reported = sum(s["rows"] for s in manifest["shards_stats"])
    if rows != reported:
        print(f"WARNING: shards reported {reported:,} rows but {rows:,} are on the volume", flush=True)
    shuffle_pool(raw_dir, tmp_out, "all", "all", num_waves=4 if rows >= 2_000_000 else 1, val_frac=manifest["val_frac"])
    manifest = {
        **manifest,
        "raw_rows": rows,
        "raw_npz_files": files,
        **split_counts(tmp_out),
        "created": _dt.datetime.utcnow().isoformat() + "Z",
    }
    with open(f"{tmp_out}/dataset.json", "w") as f:
        json.dump(manifest, f, indent=2)
    os.rename(tmp_out, out_root)
    vol.commit()
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def generate_teacher_data(dataset: str, board_size: int, teacher_url: str, target_rows: int, visits: int,
                          cheap_visits: int, game_threads: int, nn_threads: int = 2, cheap_prob: float = 0.75,
                          force: bool = False, val_frac: float = 0.05, cpu: int = 8, shards: int = 1) -> dict:
    """Generate a fixed training pool at one board size: `shards` L4 containers run `katago selfplay` with a
    released net as the teacher, each writing ceil(target_rows / shards) rows of raw files to the volume, then
    one CPU container shuffles the merged pool into /data/shuffled/<dataset>/{train,val} with
    pos_len = board_size. Runs inside run_pipeline's container, which has the volume mounted."""
    vol.reload()
    out_root = dataset_dir(dataset)
    if os.path.exists(f"{out_root}/train.json") and not force:
        print(f"Dataset already prepared at {out_root}; skipping (use --force-data to rebuild)", flush=True)
        with open(f"{out_root}/dataset.json") as f:
            return json.load(f)
    tmp_out = out_root + ".tmp"
    shutil.rmtree(tmp_out, ignore_errors=True)
    if force:
        shutil.rmtree(out_root, ignore_errors=True)
    os.makedirs(f"{tmp_out}/raw", exist_ok=True)
    cached, teacher_name = _teacher_cache(teacher_url)

    # Self-play config: the repo's small-board template with size, search and threading overridden.
    overrides = {
        "bSizes": int(board_size),
        "bSizeRelProbs": 1,
        "allowRectangleProb": 0.0,
        "dataBoardLen": int(board_size),
        "maxVisits": int(visits),
        "cheapSearchVisits": int(cheap_visits),
        # Cheap searches write no rows (cheapSearchTargetWeight = 0 in the template), so lowering this
        # raises rows per game at the price of more correlated rows from each game.
        "cheapSearchProb": float(cheap_prob),
        "numGameThreads": int(game_threads),
        "nnMaxBatchSize": max(32, int(game_threads)),
        # Small boards make the GPU launch-overhead bound: a second server thread overlaps batches.
        "numNNServerThreadsPerModel": int(nn_threads),
        "handicapProb": 0.0,
        # Enough raw files (>= ~200) that the 5% file-hash validation split is populated even on small pools.
        "maxRowsPerTrainFile": max(100, min(1000, int(target_rows) // 200)),
        "logGamesEvery": 100,
    }
    shutil.copy(SELFPLAY_TEMPLATE_CFG, f"{tmp_out}/selfplay.cfg")
    with open(f"{tmp_out}/selfplay_overrides.txt", "w") as f:
        f.write(",".join(f"{k}={v}" for k, v in overrides.items()) + "\n")
    vol.commit()

    per_shard = int(math.ceil(target_rows / shards))
    shard_args = [
        dict(dataset=dataset, shard=i, num_shards=shards, teacher_file=os.path.basename(cached),
             target_rows=per_shard, overrides=overrides)
        for i in range(shards)
    ]
    print(f"Generating {target_rows:,} rows of {board_size}x{board_size} with {teacher_name}: {shards} shard(s) of "
          f"{per_shard:,} rows on L4 containers with {cpu} cores, {game_threads} game threads, {nn_threads} NN threads",
          flush=True)
    t0 = time.time()
    shard_stats = list(GEN_SHARD_FUNCTIONS[cpu].map(shard_args))
    wall_sec = time.time() - t0
    rows = sum(s["rows"] for s in shard_stats)
    games = sum(s["games"] for s in shard_stats)
    container_sec = sum(s["gen_sec"] for s in shard_stats)
    cpu_sec = sum(s["cpu_sec"] for s in shard_stats)
    nn_evals = sum(s["nn_evals"] for s in shard_stats)
    nn_batches = sum(s["nn_batches"] for s in shard_stats)
    usd_per_container_sec = (MODAL_USD_PER_SEC["L4"] + cpu * MODAL_USD_PER_SEC["cpu_core"]
                             + GEN_MEMORY_MB / 1024 * MODAL_USD_PER_SEC["gib"])
    manifest = {
        "dataset": dataset,
        "source": "teacher-selfplay",
        "board_size": int(board_size),
        "pos_len": int(board_size),
        "teacher": teacher_name,
        "teacher_url": teacher_url,
        "target_rows": target_rows,
        "games": games,
        "rows_per_game": round(rows / max(1, games), 2),
        "visits": visits,
        "cheap_visits": cheap_visits,
        "cheap_prob": cheap_prob,
        "game_threads": game_threads,
        "nn_threads": nn_threads,
        "cpu": cpu,
        "shards": shards,
        "val_frac": val_frac,
        # Wall-clock of the generation step vs. summed generation time across shard containers
        # (container start-up, teacher copy and the final sync add roughly a minute per container).
        "generation_sec": round(wall_sec),
        "container_sec": round(container_sec),
        "rows_per_sec": round(rows / max(1.0, wall_sec), 1),
        "rows_per_container_sec": round(rows / max(1.0, container_sec), 1),
        "nn_evals": nn_evals,
        "nn_avg_batch": round(nn_evals / max(1, nn_batches), 1),
        "nn_evals_per_sec": round(nn_evals / max(1.0, container_sec)),
        "nn_evals_per_row": round(nn_evals / max(1, rows)),
        "cpu_cores_avg": round(cpu_sec / max(1.0, container_sec), 2),
        "est_gen_cost_usd": round(container_sec * usd_per_container_sec, 3),
        "est_usd_per_1m_rows": round(1e6 / max(1, rows) * container_sec * usd_per_container_sec, 2),
        "overrides": overrides,
        "shards_stats": shard_stats,
    }
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("overrides", "shards_stats")}, indent=2),
          flush=True)
    return shuffle_teacher_pool.remote(dataset, manifest)


@app.function(image=image, gpu="H100", cpu=8.0, memory=32768, timeout=24 * 3600, volumes={DATA: vol})
def train_model(cfg: dict) -> dict:
    """Train one model kind on the prepared dataset. Re-running resumes from the last checkpoint."""
    vol.reload()
    datadir = dataset_dir(cfg["dataset"])
    if not os.path.exists(f"{datadir}/train.json"):
        raise RuntimeError(f"Dataset not found at {datadir}; run --stage data first")
    manifest = {}
    if os.path.exists(f"{datadir}/dataset.json"):
        with open(f"{datadir}/dataset.json") as f:
            manifest = json.load(f)
    pos_len = int(manifest.get("pos_len", 19))
    board_size = int(manifest.get("board_size", pos_len))
    summary = train_run(cfg, datadir, run_root_for(cfg["run_name"]), pos_len=pos_len, board_size=board_size)
    vol.commit()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.function(image=image, cpu=4.0, memory=16384, timeout=2 * 3600, volumes={DATA: vol})
def export_run(run_name: str) -> list:
    vol.reload()
    done = export_run_dir(run_root_for(run_name))
    vol.commit()
    print(f"{run_name}: {len(done)} exported checkpoints", flush=True)
    return done


@app.function(image=image, gpu="L4", cpu=8.0, memory=16384, timeout=12 * 3600, volumes={DATA: vol})
def evaluate(run_names: list, eval_name: str, checkpoints_per_run: int, visits: int, games_per_bot: int,
             game_threads: int = 64, board_size: int = 0, komi: float = 0.0, komi_auto: bool = False) -> dict:
    """Round-robin match among log-spaced checkpoints of each run (+ any anchors in /data/anchors), then Bayes Elo.
    board_size 0 means "take it from the first run's run.json"; komi 0 means the size's default fair komi."""
    vol.reload()
    if not board_size:
        run_json = f"{run_root_for(run_names[0])}/run.json"
        with open(run_json) as f:
            board_size = int(json.load(f).get("board_size", 19))
    if not komi:
        komi = default_komi(board_size)
    print(f"Evaluating on {board_size}x{board_size}, komi {'auto' if komi_auto else komi}, {visits} visits", flush=True)
    bots = []
    for run in run_names:
        entries = list_exported(run_root_for(run))
        if not entries:
            raise RuntimeError(f"No exported checkpoints for {run}; run --stage export first")
        chosen = log_spaced_subset(entries, checkpoints_per_run)
        print(f"{run}: using {len(chosen)}/{len(entries)} checkpoints: {[e['samples'] for e in chosen]}", flush=True)
        bots += [{"name": e["name"], "path": e["path"], "run": run} for e in chosen]
    secondary = []
    for path in sorted(glob.glob(f"{DATA}/anchors/*.bin.gz")):
        secondary.append(len(bots))
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.basename(path)[: -len(".bin.gz")])
        bots.append({"name": "anchor-" + name, "path": path, "run": "anchor"})
    if secondary:
        print(f"Anchors (secondary bots): {[bots[i]['name'] for i in secondary]}", flush=True)

    eval_root = eval_root_for(eval_name)
    result = run_match(eval_root, bots, secondary, visits, games_per_bot, game_threads, board_size, komi, komi_auto)
    vol.commit()
    return result


@app.function(image=image, cpu=2.0, memory=4096, timeout=3600, volumes={DATA: vol})
def report(run_names: list, eval_name: str) -> str:
    vol.reload()
    text = build_report({r: run_root_for(r) for r in run_names}, eval_root_for(eval_name))
    vol.commit()
    return text


@app.function(image=image, gpu="L4", cpu=4.0, memory=16384, timeout=3600, volumes={DATA: vol})
def smoke(conv_kind: str, tf_kind: str, optimizer: str, smoke_samples: int = 8192, extra_args: str = "") -> dict:
    """End-to-end check on the 1024-row test file: benchmark, tiny train, export, 8-game match, Elo, report.
    smoke_samples > 8192 gives a longer training run (still on the 8K-row mini set) for sanity-checking learning."""
    root = "/tmp/smoke"
    shutil.rmtree(root, ignore_errors=True)
    mini = f"{root}/mini"
    src = f"{PY_DIR}/testdata/benchmark_data_1024.npz"
    for split, copies in (("train", 8), ("val", 1)):
        os.makedirs(f"{mini}/{split}", exist_ok=True)
        for i in range(copies):
            shutil.copy(src, f"{mini}/{split}/mini{i}.npz")
            # train.py reads the row count from a per-file sidecar that shuffle.py normally writes.
            with open(f"{mini}/{split}/mini{i}.json", "w") as f:
                json.dump({"num_rows": 1024}, f)
    with open(f"{mini}/train.json", "w") as f:
        json.dump({"range": [0, 8 * 1024]}, f)

    out = {"benchmark": {}, "train": {}, "export": {}}
    kinds = {"smoke-conv": conv_kind, "smoke-tf": tf_kind}
    bench_opt = {"sgd": "sgd", "adamw": "adam", "muon": "muon"}[optimizer]
    for run, kind in kinds.items():
        run_cmd(
            [
                sys.executable, "benchmark_fresh_model.py",
                "-model-kind", kind, "-optimizer", bench_opt, "-batch-size", 256,
                "-data", src, "-mode", "trainloop", "-num-iters", 10, "-warmup-iters", 3,
            ],
            cwd=PY_DIR,
            log_path=f"{root}/benchmark_{run}.txt",
        )
        out["benchmark"][run] = f"{root}/benchmark_{run}.txt"

    for run, kind in kinds.items():
        cfg = {
            "run_name": run, "model_kind": kind, "dataset": "mini",
            "max_samples": smoke_samples, "batch_size": 256, "samples_per_epoch": max(4096, smoke_samples // 2),
            "epochs_per_export": 1, "lr_schedule": "(0,8.0)", "max_val_samples": 1024, "seed": 1,
            "optimizer": optimizer, "extra_args": extra_args,
        }
        out["train"][run] = train_run(cfg, mini, f"{root}/runs/{run}")
        out["export"][run] = export_run_dir(f"{root}/runs/{run}")

    bots = []
    for run in kinds:
        bots += [{"name": e["name"], "path": e["path"], "run": run} for e in list_exported(f"{root}/runs/{run}")]
    out["match"] = run_match(f"{root}/eval", bots, [], visits=16, games_per_bot=4, game_threads=8)
    out["train_metrics_tail"] = {}
    for run in kinds:
        rows = read_jsonl(f"{root}/runs/{run}/train/metrics_train.json")
        keep = ("nsamp", "p0loss", "vloss", "loss", "gnorm_batch", "exgnorm_batch", "gnorm_cap_batch", "pslr_batch")
        out["train_metrics_tail"][run] = [{k: r.get(k) for k in keep if k in r} for r in rows[-3:]]
    out["report"] = build_report({r: f"{root}/runs/{r}" for r in kinds}, f"{root}/eval")

    stamp = _dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    dest = f"{DATA}/smoke/{stamp}"
    shutil.copytree(root, dest, ignore=shutil.ignore_patterns("mini"))
    vol.commit()
    out["saved_to"] = dest
    print(f"Smoke artifacts saved under {dest}", flush=True)
    return out


@app.function(image=image, cpu=1.0, memory=2048, timeout=24 * 3600, volumes={DATA: vol})
def run_pipeline(p: dict) -> dict:
    """Orchestrates stages remotely so `modal run --detach` survives the local client exiting."""
    stage = p["stage"]
    run_names = [p["conv_run"], p["tf_run"]]
    out = {"params": p}

    if stage == "gen" or (stage in ("data", "all") and p["data_source"] == "teacher"):
        out["data"] = generate_teacher_data(
            p["dataset"], p["board_size"], p["teacher_url"], p["gen_rows"], p["gen_visits"], p["gen_cheap_visits"],
            p["gen_threads"], nn_threads=p["gen_nn_threads"], cheap_prob=p["gen_cheap_prob"], force=p["force_data"],
            val_frac=p["val_frac"], cpu=p["gen_cpu"], shards=p["gen_shards"],
        )
    elif stage in ("data", "all"):
        out["data"] = prepare_data.remote(p["dataset"], p["dates"], p["keep_rows"], p["val_keep_rows"], p["force_data"])

    if stage in ("train", "all"):
        cfgs = [
            make_train_cfg(p, p["conv_run"], p["conv_kind"], p["conv_extra_args"]),
            make_train_cfg(p, p["tf_run"], p["tf_kind"], p["tf_extra_args"]),
        ]
        out["train"] = list(train_model.map(cfgs))  # both models train in parallel, one GPU each

    if stage in ("export", "all"):
        out["export"] = list(export_run.map(run_names))

    if stage in ("eval", "all"):
        out["eval"] = evaluate.remote(run_names, p["eval_name"], p["checkpoints_per_run"], p["visits"], p["games_per_bot"],
                                      64, 0, p["komi"], p["komi_auto"])

    if stage in ("report", "all"):
        out["report"] = report.remote(run_names, p["eval_name"])

    return out


# ----------------------------------------------------------------------------
# Diagnostic: benchmark variants (gradient-norm health + throughput) on L4 or H100
# ----------------------------------------------------------------------------

DIAG_VARIANTS = [
    # name, model kind, extra benchmark args, env overrides
    ("tf-muon-compile",         "b5c192h3nbttfrs-fson-silu-rsnh", ["-optimizer", "muon"], {}),
    ("tf-muon-nocompile",       "b5c192h3nbttfrs-fson-silu-rsnh", ["-optimizer", "muon", "-no-compile"], {}),
    ("tf-sgd-nocompile",        "b5c192h3nbttfrs-fson-silu-rsnh", ["-optimizer", "sgd", "-no-compile"], {}),
    ("tf-muon-bf16-nocompile",  "b5c192h3nbttfrs-fson-silu-rsnh", ["-optimizer", "muon", "-no-compile", "-use-bf16"], {}),
    ("tf-nomaskskip-nocompile", "b5c192h3nbttfrs-fson-silu-rsnh", ["-optimizer", "muon", "-no-compile"], {"KATAGO_TRANSFORMER_SKIP_REDUNDANT_MASKS": "0"}),
    ("tf-ropecast0-nocompile",  "b5c192h3nbttfrs-fson-silu-rsnh", ["-optimizer", "muon", "-no-compile"], {"KATAGO_LEARNED_ROPE_CAST_TO_INPUT_DTYPE": "0"}),
    ("tf-fson-silu-nocompile",  "b5c192h3nbttfrs-fson-silu",      ["-optimizer", "muon", "-no-compile"], {}),
    ("tfbase-muon-nocompile",   "b5c192h3nbttfrs",                ["-optimizer", "muon", "-no-compile"], {}),
    ("conv-muon-nocompile",     "b5c192nbt-fson-mish-rvglr-bnh",  ["-optimizer", "muon", "-no-compile"], {}),
    ("tf-muon-bf16-compile",    "b5c192h3nbttfrs-fson-silu-rsnh", ["-optimizer", "muon", "-use-bf16"], {}),
    ("tf-muon-tf32-compile",    "b5c192h3nbttfrs-fson-silu-rsnh", ["-optimizer", "muon", "-use-tf32-matmul"], {}),
    ("conv-muon-compile",       "b5c192nbt-fson-mish-rvglr-bnh",  ["-optimizer", "muon"], {}),
    ("conv-muon-bf16-compile",  "b5c192nbt-fson-mish-rvglr-bnh",  ["-optimizer", "muon", "-use-bf16"], {}),
]


def _run_diag(names: list) -> dict:
    out = {}
    for name, kind, extra, env_over in DIAG_VARIANTS:
        if names and name not in names:
            continue
        cmd = [sys.executable, "benchmark_fresh_model.py", "-model-kind", kind, "-batch-size", "256",
               "-data", f"{PY_DIR}/testdata/benchmark_data_1024.npz", "-mode", "trainloop",
               "-num-iters", "6", "-warmup-iters", "2"] + extra
        print("+ " + " ".join(cmd), env_over, flush=True)
        env = dict(os.environ, **env_over)
        p = subprocess.run(cmd, cwd=PY_DIR, capture_output=True, text=True, env=env)
        text = p.stdout + p.stderr
        bad = re.search(r"Bad-gnorm batches: .*", text)
        thr = re.search(r"Throughput: .*", text)
        tail = "\n".join(text.strip().splitlines()[-8:])
        out[name] = {"rc": p.returncode, "bad_gnorm": bad.group(0) if bad else None, "throughput": thr.group(0) if thr else None, "tail": tail}
        print(f"=== {name}: rc={p.returncode} | {out[name]['bad_gnorm']} | {out[name]['throughput']}", flush=True)
        if p.returncode != 0:
            print(tail, flush=True)
    return out


@app.function(image=image, gpu="L4", cpu=4.0, memory=16384, timeout=3600)
def diagnose(names: list) -> dict:
    return _run_diag(names)


@app.function(image=image, gpu="H100", cpu=8.0, memory=32768, timeout=3600)
def diagnose_h100(names: list) -> dict:
    return _run_diag(names)


# ----------------------------------------------------------------------------
# Local entrypoint
# ----------------------------------------------------------------------------


def date_range(end_date: str, days: int):
    end = _dt.date.fromisoformat(end_date)
    return [(end - _dt.timedelta(days=i)).isoformat() for i in range(days - 1, -1, -1)]


@app.local_entrypoint()
def main(
    stage: str = "all",
    # data: kata1 archive (19x19) or teacher self-play at --board-size
    data_source: str = "archive",
    dataset: str = "",
    board_size: int = 19,
    teacher_url: str = DEFAULT_TEACHER_URL,
    gen_rows: int = 1_000_000,
    gen_visits: int = 600,
    gen_cheap_visits: int = 100,
    gen_cheap_prob: float = 0.75,
    gen_threads: int = 512,
    gen_nn_threads: int = 2,
    gen_cpu: int = 8,
    gen_shards: int = 1,
    val_frac: float = 0.05,
    end_date: str = "2025-12-04",
    days: int = 30,
    keep_rows: int = 50_000_000,
    val_keep_rows: int = 2_000_000,
    force_data: bool = False,
    # models / training
    conv_kind: str = DEFAULT_CONV_KIND,
    tf_kind: str = DEFAULT_TF_KIND,
    run_tag: str = "pair1",
    seed: int = 1,
    max_samples: int = 200_000_000,
    batch_size: int = 256,
    samples_per_epoch: int = 2_000_000,
    epochs_per_export: int = 5,
    lr_schedule: str = DEFAULT_LR_SCHEDULE,
    optimizer: str = "muon",
    train_extra_args: str = "-no-compile -use-bf16",
    conv_extra_args: str = "",
    tf_extra_args: str = "",
    max_val_samples: int = 500_000,
    # smoke / diag
    smoke_samples: int = 8192,
    diag_names: str = "",
    diag_gpu: str = "L4",
    # evaluation
    checkpoints_per_run: int = 7,
    visits: int = 200,
    games_per_bot: int = 400,
    komi: float = 0.0,
    komi_auto: bool = False,
    eval_name: str = "",
):
    stages = ("smoke", "diag", "gen", "data", "train", "export", "eval", "report", "all")
    if stage not in stages:
        raise SystemExit(f"--stage must be one of {stages}")
    if optimizer not in OPTIMIZER_FLAGS:
        raise SystemExit(f"--optimizer must be one of {list(OPTIMIZER_FLAGS)}")
    if stage == "gen":
        data_source = "teacher"
    if data_source not in ("archive", "teacher"):
        raise SystemExit("--data-source must be 'archive' or 'teacher'")
    if not 2 <= board_size <= 19:
        raise SystemExit("--board-size must be between 2 and 19")
    if not 0.0 < val_frac < 1.0:
        raise SystemExit("--val-frac must be between 0 and 1")
    if not 0.0 <= gen_cheap_prob <= 1.0:
        raise SystemExit("--gen-cheap-prob must be between 0 and 1")
    if gen_threads < 1 or gen_nn_threads < 1:
        raise SystemExit("--gen-threads and --gen-nn-threads must be positive")
    if gen_cheap_visits > gen_visits:
        raise SystemExit("--gen-cheap-visits must not exceed --gen-visits")
    if gen_cpu not in GEN_CPU_SIZES:
        raise SystemExit(f"--gen-cpu must be one of {GEN_CPU_SIZES}")
    if gen_shards < 1:
        raise SystemExit("--gen-shards must be positive")
    if data_source == "archive" and board_size != 19:
        raise SystemExit("--board-size other than 19 requires --data-source teacher (the kata1 archive is 19x19 data)")
    if not dataset:
        dataset = f"teacher-b{board_size}-{gen_rows // 1000}k" if data_source == "teacher" else "kata1-30d-2025-12-04"
    size_tag = "" if board_size == 19 else f"-b{board_size}"

    if stage == "diag":
        fn = diagnose_h100 if diag_gpu.lower() == "h100" else diagnose
        res = fn.remote([n for n in diag_names.split(",") if n])
        print(f"{'variant':26s} {'rc':>2s}  gradient norms | throughput")
        for k, v in res.items():
            print(f"{k:26s} {v['rc']:>2d}  {v['bad_gnorm']} | {v['throughput']}")
        return

    if stage == "smoke":
        result = smoke.remote(conv_kind, tf_kind, optimizer, smoke_samples, train_extra_args)
        print(json.dumps({k: v for k, v in result.items() if k != "report"}, indent=2, default=str))
        print(f"\nSmoke OK. Artifacts: modal volume ls {VOLUME_NAME} {result['saved_to'].replace(DATA + '/', '')}")
        return

    p = {
        "stage": stage,
        "dataset": dataset,
        "dates": date_range(end_date, days),
        "keep_rows": keep_rows,
        "val_keep_rows": val_keep_rows,
        "force_data": force_data,
        "conv_kind": conv_kind,
        "tf_kind": tf_kind,
        "conv_run": f"{run_tag}{size_tag}-conv-s{seed}",
        "tf_run": f"{run_tag}{size_tag}-tf-s{seed}",
        "data_source": data_source,
        "board_size": board_size,
        "teacher_url": teacher_url,
        "gen_rows": gen_rows,
        "gen_visits": gen_visits,
        "gen_cheap_visits": gen_cheap_visits,
        "gen_cheap_prob": gen_cheap_prob,
        "gen_threads": gen_threads,
        "gen_nn_threads": gen_nn_threads,
        "gen_cpu": gen_cpu,
        "gen_shards": gen_shards,
        "val_frac": val_frac,
        "komi": komi,
        "komi_auto": komi_auto,
        "seed": seed,
        "max_samples": max_samples,
        "batch_size": batch_size,
        "samples_per_epoch": samples_per_epoch,
        "epochs_per_export": epochs_per_export,
        "lr_schedule": lr_schedule,
        "optimizer": optimizer,
        "train_extra_args": train_extra_args,
        "conv_extra_args": conv_extra_args,
        "tf_extra_args": tf_extra_args,
        "max_val_samples": max_val_samples,
        "checkpoints_per_run": checkpoints_per_run,
        "visits": visits,
        "games_per_bot": games_per_bot,
        "eval_name": eval_name or f"{run_tag}{size_tag}-s{seed}-v{visits}",
    }
    print("Pipeline parameters:\n" + json.dumps(p, indent=2), flush=True)
    result = run_pipeline.remote(p)
    if "report" in result:
        print(result["report"])
    else:
        print(json.dumps({k: v for k, v in result.items() if k != "params"}, indent=2, default=str))
    print(
        f"\nResults live on the '{VOLUME_NAME}' volume, e.g.:\n"
        f"  modal volume ls {VOLUME_NAME} runs\n"
        f"  modal volume get {VOLUME_NAME} eval/{p['eval_name']}/report.md ./report.md"
    )
