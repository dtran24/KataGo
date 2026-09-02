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

See README.md next to this file for details, costs and knobs.
"""

import datetime as _dt
import glob
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

APP_NAME = "katago-pair1"
VOLUME_NAME = "katago-pair1"
DATA = "/data"  # volume mount point inside containers

REPO_ROOT = Path(__file__).resolve().parents[2]
REMOTE_REPO = "/root/KataGo"
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


def train_run(cfg, datadir, run_root):
    """Run python/train.py for one model on a fixed dataset. Resumes if a checkpoint exists."""
    traindir = f"{run_root}/train"
    exportdir = f"{run_root}/torchmodels"
    os.makedirs(traindir, exist_ok=True)
    os.makedirs(exportdir, exist_ok=True)
    with open(f"{run_root}/run.json", "w") as f:
        json.dump(cfg, f, indent=2)

    cmd = [
        sys.executable, "train.py",
        "-traindir", traindir,
        "-datadir", datadir,
        "-exportdir", exportdir,
        "-exportprefix", cfg["run_name"],
        "-pos-len", 19,
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


def write_match_config(path, bots, secondary_idxs, visits, num_games, game_threads):
    """Write a complete `katago match` config: fixed rules, 19x19, fixed komi, fixed visits, fp32."""
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
        "bSizes = 19",
        "bSizeRelProbs = 1",
        "komiAuto = false",
        "komiMean = 7.0",
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


def run_match(eval_root, bots, secondary_idxs, visits, games_per_bot, game_threads):
    """Play a round-robin among bots (secondary bots only play primaries) and compute Elo."""
    os.makedirs(eval_root, exist_ok=True)
    primary = len(bots) - len(secondary_idxs)
    num_games = max(1, games_per_bot * primary // 2)
    cfg_path = f"{eval_root}/match.cfg"
    sgf_dir = f"{eval_root}/sgfs"
    os.makedirs(sgf_dir, exist_ok=True)
    write_match_config(cfg_path, bots, secondary_idxs, visits, num_games, game_threads)
    with open(f"{eval_root}/bots.json", "w") as f:
        json.dump({"bots": bots, "secondary": secondary_idxs, "visits": visits, "num_games": num_games}, f, indent=2)

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


def make_train_cfg(p, run_name, model_kind):
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
        "extra_args": p["train_extra_args"],
    }


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
    os.makedirs(tmp_out, exist_ok=True)
    nproc = os.cpu_count() or 16

    def shuffle(split, keep, lbound, ubound):
        shutil.rmtree(f"/tmp/shuf/{split}", ignore_errors=True)
        os.makedirs(f"/tmp/shuf/{split}", exist_ok=True)
        run_cmd(
            [
                sys.executable, "shuffle.py", raw,
                "-min-rows", 100_000_000_000,  # window = the entire pool
                "-keep-target-rows", keep,
                "-out-dir", f"{tmp_out}/{split}",
                "-out-tmp-dir", f"/tmp/shuf/{split}",
                "-approx-rows-per-out-file", 70000,
                "-num-processes", nproc,
                "-num-waves", 4,
                # Same file-hash split as python/selfplay/shuffle.sh: 5% of files become validation.
                "-only-include-md5-path-prop-lbound", lbound,
                "-only-include-md5-path-prop-ubound", ubound,
            ],
            cwd=PY_DIR,
            log_path=f"{tmp_out}/shuffle_{split}.txt",
        )

    t1 = time.time()
    shuffle("val", val_keep_rows, 0.95, 1.00)
    shuffle("train", keep_rows, 0.00, 0.95)
    print(f"Shuffle took {time.time()-t1:.0f}s", flush=True)

    def count(split):
        files = glob.glob(f"{tmp_out}/{split}/*.npz")
        return {"files": len(files), "bytes": sum(os.path.getsize(f) for f in files)}

    with open(f"{tmp_out}/train.json") as f:
        train_range = json.load(f).get("range")
    manifest = {
        "dataset": dataset,
        "dates": got,
        "missing_dates": missing,
        "raw_npz_files": total_files,
        "keep_rows": keep_rows,
        "val_keep_rows": val_keep_rows,
        "train_range": train_range,
        "train": count("train"),
        "val": count("val"),
        "created": _dt.datetime.utcnow().isoformat() + "Z",
    }
    with open(f"{tmp_out}/dataset.json", "w") as f:
        json.dump(manifest, f, indent=2)
    os.rename(tmp_out, out_root)
    vol.commit()
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


@app.function(image=image, gpu="H100", cpu=8.0, memory=32768, timeout=24 * 3600, volumes={DATA: vol})
def train_model(cfg: dict) -> dict:
    """Train one model kind on the prepared dataset. Re-running resumes from the last checkpoint."""
    vol.reload()
    datadir = dataset_dir(cfg["dataset"])
    if not os.path.exists(f"{datadir}/train.json"):
        raise RuntimeError(f"Dataset not found at {datadir}; run --stage data first")
    summary = train_run(cfg, datadir, run_root_for(cfg["run_name"]))
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
def evaluate(run_names: list, eval_name: str, checkpoints_per_run: int, visits: int, games_per_bot: int, game_threads: int = 64) -> dict:
    """Round-robin match among log-spaced checkpoints of each run (+ any anchors in /data/anchors), then Bayes Elo."""
    vol.reload()
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
    result = run_match(eval_root, bots, secondary, visits, games_per_bot, game_threads)
    vol.commit()
    return result


@app.function(image=image, cpu=2.0, memory=4096, timeout=3600, volumes={DATA: vol})
def report(run_names: list, eval_name: str) -> str:
    vol.reload()
    text = build_report({r: run_root_for(r) for r in run_names}, eval_root_for(eval_name))
    vol.commit()
    return text


@app.function(image=image, gpu="L4", cpu=4.0, memory=16384, timeout=3600, volumes={DATA: vol})
def smoke(conv_kind: str, tf_kind: str, optimizer: str) -> dict:
    """End-to-end check on the 1024-row test file: benchmark, tiny train, export, 8-game match, Elo, report."""
    root = "/tmp/smoke"
    shutil.rmtree(root, ignore_errors=True)
    mini = f"{root}/mini"
    src = f"{PY_DIR}/testdata/benchmark_data_1024.npz"
    for split, copies in (("train", 8), ("val", 1)):
        os.makedirs(f"{mini}/{split}", exist_ok=True)
        for i in range(copies):
            shutil.copy(src, f"{mini}/{split}/mini{i}.npz")
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
            "max_samples": 8192, "batch_size": 256, "samples_per_epoch": 4096, "epochs_per_export": 1,
            "lr_schedule": "(0,8.0)", "max_val_samples": 1024, "seed": 1, "optimizer": optimizer, "extra_args": "",
        }
        out["train"][run] = train_run(cfg, mini, f"{root}/runs/{run}")
        out["export"][run] = export_run_dir(f"{root}/runs/{run}")

    bots = []
    for run in kinds:
        bots += [{"name": e["name"], "path": e["path"], "run": run} for e in list_exported(f"{root}/runs/{run}")]
    out["match"] = run_match(f"{root}/eval", bots, [], visits=16, games_per_bot=4, game_threads=8)
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

    if stage in ("data", "all"):
        out["data"] = prepare_data.remote(p["dataset"], p["dates"], p["keep_rows"], p["val_keep_rows"], p["force_data"])

    if stage in ("train", "all"):
        cfgs = [make_train_cfg(p, p["conv_run"], p["conv_kind"]), make_train_cfg(p, p["tf_run"], p["tf_kind"])]
        out["train"] = list(train_model.map(cfgs))  # both models train in parallel, one GPU each

    if stage in ("export", "all"):
        out["export"] = list(export_run.map(run_names))

    if stage in ("eval", "all"):
        out["eval"] = evaluate.remote(run_names, p["eval_name"], p["checkpoints_per_run"], p["visits"], p["games_per_bot"])

    if stage in ("report", "all"):
        out["report"] = report.remote(run_names, p["eval_name"])

    return out


# ----------------------------------------------------------------------------
# Local entrypoint
# ----------------------------------------------------------------------------


def date_range(end_date: str, days: int):
    end = _dt.date.fromisoformat(end_date)
    return [(end - _dt.timedelta(days=i)).isoformat() for i in range(days - 1, -1, -1)]


@app.local_entrypoint()
def main(
    stage: str = "all",
    # data
    dataset: str = "kata1-30d-2025-12-04",
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
    optimizer: str = "sgd",
    train_extra_args: str = "",
    max_val_samples: int = 500_000,
    # evaluation
    checkpoints_per_run: int = 7,
    visits: int = 200,
    games_per_bot: int = 400,
    eval_name: str = "",
):
    stages = ("smoke", "data", "train", "export", "eval", "report", "all")
    if stage not in stages:
        raise SystemExit(f"--stage must be one of {stages}")
    if optimizer not in OPTIMIZER_FLAGS:
        raise SystemExit(f"--optimizer must be one of {list(OPTIMIZER_FLAGS)}")

    if stage == "smoke":
        result = smoke.remote(conv_kind, tf_kind, optimizer)
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
        "conv_run": f"{run_tag}-conv-s{seed}",
        "tf_run": f"{run_tag}-tf-s{seed}",
        "seed": seed,
        "max_samples": max_samples,
        "batch_size": batch_size,
        "samples_per_epoch": samples_per_epoch,
        "epochs_per_export": epochs_per_export,
        "lr_schedule": lr_schedule,
        "optimizer": optimizer,
        "train_extra_args": train_extra_args,
        "max_val_samples": max_val_samples,
        "checkpoints_per_run": checkpoints_per_run,
        "visits": visits,
        "games_per_bot": games_per_bot,
        "eval_name": eval_name or f"{run_tag}-s{seed}-v{visits}",
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
