"""One-seed, supervised Un-0 pilot. Reuses pair-1 training, never C++ export.

modal run experiments/un0_5x5/app.py --stage diag
modal run --detach experiments/un0_5x5/app.py --stage train
modal run experiments/un0_5x5/app.py --stage evaluate
"""

import hashlib
import json
from pathlib import Path
import sys
import time

import modal

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "python/train.py").exists()),
            Path("/root/KataGo"))
sys.path.insert(0, str(ROOT))
from experiments.modal_pair1 import app as pair

app = modal.App("katago-un0-5x5")
image = pair.image.add_local_dir(ROOT / "experiments", remote_path="/root/KataGo/experiments",
                                ignore=["**/__pycache__/**", "**/.pytest_cache/**"])
vol = modal.Volume.from_name("katago-pair1", create_if_missing=False)
MODEL = "un0-b5c192-n1250-e10"
DATASET = "teacher-b5-1000k"
PREFIX = "un0-pilot-v1"
SCHEDULE = "(0,8.0),(10M,4.0),(14M,2.0),(17M,1.0),(19M,0.5)"


def dataset_manifest():
    root = Path(pair.dataset_dir(DATASET))
    manifest = json.loads((root / "dataset.json").read_text())
    if manifest["pos_len"] != 5 or manifest["board_size"] != 5:
        raise ValueError("Expected the existing full 5x5 teacher pool")
    return root, manifest


def cfg_for(seed, max_samples=20_000_000):
    if seed != 1:
        raise ValueError("This pilot is limited to seed 1")
    return dict(run_name=f"{PREFIX}-s{seed}", dataset=DATASET, model_kind=MODEL,
                seed=seed, batch_size=2048, max_samples=max_samples,
                samples_per_epoch=500_000, epochs_per_export=1,
                max_val_samples=50_000, lr_schedule=SCHEDULE,
                optimizer="muon", extra_args="-no-compile -use-bf16")


def provenance():
    paths = list((ROOT / "python").rglob("*.py")) + list((ROOT / "experiments/un0_5x5").glob("*.py"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


@app.function(image=image, gpu="H100", cpu=8, memory=32768, timeout=900,
              volumes={"/data": vol}, retries=0)
def diagnose():
    vol.reload()
    datadir, manifest = dataset_manifest()
    out = Path("/data/studies") / PREFIX / "diag"
    out.mkdir(parents=True, exist_ok=True)
    datafile = sorted((datadir / "train").glob("*.npz"))[0]
    for name, kind in [("un0", MODEL), ("conv", pair.DEFAULT_CONV_KIND), ("tf", pair.DEFAULT_TF_KIND)]:
        pair.run_cmd([sys.executable, "benchmark_fresh_model.py", "-model-kind", kind,
                      "-optimizer", "muon", "-batch-size", 2048, "-pos-len", 5,
                      "-data", datafile, "-mode", "trainloop", "-num-iters", 12,
                      "-warmup-iters", 4, "-no-compile", "-use-bf16"],
                     cwd=pair.PY_DIR, log_path=str(out / f"{name}.txt"))
    (out / "provenance.json").write_text(json.dumps(provenance(), indent=2))
    (out / "dataset.json").write_text(json.dumps(manifest, indent=2))
    vol.commit()
    return {"directory": str(out)}


@app.function(image=image, gpu="H100", cpu=8, memory=32768, timeout=7200,
              volumes={"/data": vol}, retries=0)
def train_seed(seed, max_samples):
    vol.reload()
    datadir, manifest = dataset_manifest()
    cfg = cfg_for(seed, max_samples)
    root = Path(pair.run_root_for(cfg["run_name"]))
    root.mkdir(parents=True, exist_ok=True)
    spec = {"config": cfg, "source_sha256": provenance(), "dataset": manifest}
    specpath = root / "study.json"
    if specpath.exists() and json.loads(specpath.read_text()) != spec:
        raise ValueError("Run name already exists with different code/config/data; choose a new PREFIX")
    specpath.write_text(json.dumps(spec, indent=2))
    vol.commit()
    started = time.time()
    try:
        summary = pair.train_run(cfg, str(datadir), str(root), pos_len=5, board_size=5)
        return {**summary, "wall_seconds": time.time() - started}
    finally:
        vol.commit()


@app.function(image=image, gpu="H100", cpu=4, memory=32768, timeout=3600,
              volumes={"/data": vol}, retries=0)
def evaluate():
    vol.reload()
    datadir, _ = dataset_manifest()
    out = Path("/data/studies") / PREFIX / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for seed in (1,):
        run_root = Path(pair.run_root_for(f"{PREFIX}-s{seed}"))
        if not (run_root / "summary.json").exists():
            raise RuntimeError("Wait for seed 1 to finish before final evaluation")
        paths.append(run_root / "train/checkpoint.ckpt")
        for family in ("conv", "tf"):
            paths.append(Path(pair.run_root_for(f"pair1-b5-{family}-s{seed}")) / "train/checkpoint.ckpt")
    if not all(p.exists() for p in paths):
        raise FileNotFoundError([str(p) for p in paths if not p.exists()])
    pair.run_cmd([sys.executable, str(ROOT / "experiments/un0_5x5/evaluate.py"),
                  "--data", str(datadir / "val"), "--output", str(out / "report.json"),
                  "--audit", str(Path("/data/studies") / PREFIX / "data-audit.json"),
                  *[str(p) for p in paths]], cwd=pair.PY_DIR, log_path=str(out / "stdout.txt"))
    vol.commit()
    return json.loads((out / "report.json").read_text())


@app.function(image=image, cpu=4, memory=16384, timeout=600,
              volumes={"/data": vol}, retries=0)
def audit_data():
    vol.reload()
    datadir, _ = dataset_manifest()
    output = Path("/data/studies") / PREFIX / "data-audit.json"
    pair.run_cmd([sys.executable, str(ROOT / "experiments/un0_5x5/audit_data.py"),
                  str(datadir), str(output)], cwd=pair.PY_DIR)
    vol.commit()
    return json.loads(output.read_text())


@app.function(image=image, cpu=0.25, memory=512, timeout=7500, retries=0)
def run_training(max_samples):
    calls = [train_seed.spawn(1, max_samples)]
    return [call.get() for call in calls]


@app.local_entrypoint()
def main(stage: str = "diag", max_samples: int = 20_000_000):
    if max_samples <= 0 or max_samples > 20_000_000:
        raise ValueError("Use 1..20M samples for this pilot")
    if stage == "diag":
        print(json.dumps(diagnose.remote(), indent=2))
    elif stage == "train":
        call = run_training.spawn(max_samples)
        print(f"Spawned seed-1 pilot: {call.object_id}", flush=True)
        print(json.dumps(call.get(), indent=2))
    elif stage == "evaluate":
        print(json.dumps(evaluate.remote(), indent=2))
    elif stage == "audit":
        print(json.dumps(audit_data.remote(), indent=2))
    else:
        raise ValueError("stage must be diag, train, evaluate, or audit")
