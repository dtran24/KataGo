"""One-seed Fourier neural operator pilot using the frozen pair-1 recipe."""
import hashlib
import json
from pathlib import Path
import sys
import time

import modal

ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'python/train.py').exists()), Path('/root/KataGo'))
sys.path.insert(0, str(ROOT))
from experiments.modal_pair1 import app as pair
from experiments.un0_5x5.app import dataset_manifest, cfg_for as prior_cfg

app = modal.App('katago-fno-5x5')
image = pair.image.add_local_dir(ROOT / 'experiments', remote_path='/root/KataGo/experiments',
                                ignore=['**/__pycache__/**', '**/.pytest_cache/**'])
vol = modal.Volume.from_name('katago-pair1', create_if_missing=False)
MODEL = 'fno-b5c192-w112-m2-p1'
PREFIX = 'fno-pilot-v1'


def cfg_for(seed=1, max_samples=20_000_000):
    cfg = prior_cfg(seed, max_samples)
    return {**cfg, 'run_name': f'{PREFIX}-s{seed}', 'model_kind': MODEL}


def provenance():
    paths = list((ROOT / 'python').rglob('*.py'))
    for study in ('modal_pair1', 'un0_5x5', 'fno_5x5'):
        paths += list((ROOT / 'experiments' / study).glob('*.py'))
    paths += [ROOT / 'experiments/fno_5x5/README.md']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


@app.function(image=image, gpu='H100', cpu=8, memory=32768, timeout=900,
              volumes={'/data': vol}, retries=0)
def diagnose():
    vol.reload()
    datadir, manifest = dataset_manifest()
    out = Path('/data/studies') / PREFIX / 'diag'
    out.mkdir(parents=True, exist_ok=True)
    datafile = sorted((datadir / 'train').glob('*.npz'))[0]
    pair.run_cmd([sys.executable, 'benchmark_fresh_model.py', '-model-kind', MODEL,
                  '-optimizer', 'muon', '-batch-size', 2048, '-pos-len', 5,
                  '-data', datafile, '-mode', 'trainloop', '-num-iters', 24,
                  '-warmup-iters', 8, '-no-compile', '-use-bf16'],
                 cwd=pair.PY_DIR, log_path=str(out / 'fno.txt'))
    (out / 'provenance.json').write_text(json.dumps(provenance(), indent=2))
    (out / 'dataset.json').write_text(json.dumps(manifest, indent=2))
    vol.commit()
    return {'directory': str(out)}


@app.function(image=image, gpu='H100', cpu=8, memory=32768, timeout=7200,
              volumes={'/data': vol}, retries=0)
def train_seed(max_samples):
    vol.reload()
    datadir, manifest = dataset_manifest()
    cfg = cfg_for(1, max_samples)
    root = Path(pair.run_root_for(cfg['run_name']))
    root.mkdir(parents=True, exist_ok=True)
    spec = {'config': cfg, 'source_sha256': provenance(), 'dataset': manifest}
    path = root / 'study.json'
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError('Existing run has different code/config/data; choose a new PREFIX')
    path.write_text(json.dumps(spec, indent=2))
    vol.commit()
    started = time.time()
    try:
        return {**pair.train_run(cfg, str(datadir), str(root), pos_len=5, board_size=5),
                'wall_seconds': time.time() - started}
    finally:
        vol.commit()


@app.function(image=image, gpu='H100', cpu=4, memory=32768, timeout=3600,
              volumes={'/data': vol}, retries=0)
def evaluate():
    vol.reload()
    datadir, _ = dataset_manifest()
    out = Path('/data/studies') / PREFIX / 'evaluation'
    out.mkdir(parents=True, exist_ok=True)
    runs = [f'{PREFIX}-s1', 'pair1-b5-conv-s1', 'pair1-b5-tf-s1', 'un0-pilot-v1-s1']
    roots = [Path(pair.run_root_for(name)) for name in runs]
    if not all((root / 'summary.json').exists() for root in roots):
        raise RuntimeError('All training runs must finish before final evaluation')
    paths = [root / 'train/checkpoint.ckpt' for root in roots]
    pair.run_cmd([sys.executable, str(ROOT / 'experiments/un0_5x5/evaluate.py'),
                  '--data', str(datadir / 'val'), '--output', str(out / 'report.json'),
                  '--audit', '/data/studies/un0-pilot-v1/data-audit.json',
                  *[str(p) for p in paths]], cwd=pair.PY_DIR, log_path=str(out / 'stdout.txt'))
    pair.run_cmd([sys.executable, str(ROOT / 'experiments/fno_5x5/diagnostics.py'),
                  str(paths[0]), str(datadir / 'val'), str(out / 'diagnostics.json')], cwd=pair.PY_DIR)
    (out / 'provenance.json').write_text(json.dumps(provenance(), indent=2))
    vol.commit()
    return {'directory': str(out), 'complete': True}


@app.function(image=image, cpu=0.25, memory=512, timeout=7500, retries=0)
def run_training(max_samples):
    call = train_seed.spawn(max_samples)
    print(f'Seed 1 call: {call.object_id}', flush=True)
    return call.get()


@app.local_entrypoint()
def main(stage: str = 'diag', max_samples: int = 20_000_000):
    if not 0 < max_samples <= 20_000_000:
        raise ValueError('Use 1..20M samples for this pilot')
    if stage == 'diag':
        result = diagnose.remote()
    elif stage == 'train':
        call = run_training.spawn(max_samples)
        print(f'Spawned one-seed FNO pilot: {call.object_id}', flush=True)
        result = call.get()
    elif stage == 'evaluate':
        result = evaluate.remote()
    else:
        raise ValueError('stage must be diag, train, or evaluate')
    print(json.dumps(result, indent=2))
