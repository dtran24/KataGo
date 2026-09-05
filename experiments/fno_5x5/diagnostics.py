"""Inference-only intervention; not a separately trained spectral ablation."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'python')]
from experiments.un0_5x5.evaluate import score
from katago.train.load_model import load_model


def main():
    checkpoint, data, output = sys.argv[1:]
    model, _, _ = load_model(checkpoint, use_swa=False, device='cuda', pos_len=5)
    report = {'checkpoint_sha256': hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
              'description': 'First 2048 validation rows, fp32, all eight D4 transforms; inference intervention only',
              'base': score(model, data, max_rows=2048)}
    for block in model.blocks:
        block.spectral.enabled = False
    report['spectral_off'] = score(model, data, max_rows=2048)
    Path(output).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
