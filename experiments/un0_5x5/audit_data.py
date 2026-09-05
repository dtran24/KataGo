"""Audit game overlap and model-input overlap in the existing 5x5 pool.

Game IDs are the six exact integer-valued channels 41:47 documented in
cpp/dataio/trainingwrite.cpp. Input hashes include all 22 spatial and 19 global
features, with packed padding and signed zero normalized. SHA-256 collisions
are negligible. D4 canonicalization takes the minimum hash over all 8 transforms.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def fingerprints(path):
    with np.load(path) as data:
        if data["binaryInputNCHWPacked"].shape[1:] != (22, 4) or data["globalInputNC"].shape[1:] != (19,):
            raise ValueError(f"Not a 5x5 V7 feature file: {path}")
        targets = data["globalTargetsNC"]
        game_parts = targets[:, 41:47]
        if game_parts.shape[1] != 6 or not np.all(game_parts == game_parts.astype(np.uint32)):
            raise ValueError(f"Invalid game IDs in {path}")
        games = np.ascontiguousarray(game_parts, dtype="<u4").view("V24").ravel().tolist()
        spatial = np.unpackbits(data["binaryInputNCHWPacked"], axis=2)[:, :, :25].reshape(-1, 22, 5, 5)
        glob = data["globalInputNC"].astype("<f4")
    glob[glob == 0] = 0  # canonical +0.0
    global_bytes = [row.tobytes() for row in glob]
    exact, canonical = None, None
    for symmetry in range(8):
        x = spatial
        if symmetry & 1:
            x = x[:, :, ::-1, :]
        if symmetry & 2:
            x = x[:, :, :, ::-1]
        if symmetry & 4:
            x = x.transpose(0, 1, 3, 2)
        packed = np.packbits(x.reshape(len(x), 22, 25), axis=2)
        hashes = [hashlib.sha256(row.tobytes() + g).digest() for row, g in zip(packed, global_bytes)]
        if symmetry == 0:
            exact, canonical = hashes, hashes
        else:
            canonical = [min(a, b) for a, b in zip(canonical, hashes)]
    return games, exact, canonical


def audit(root, mask_output=None):
    train_files, val_files = sorted((root / "train").glob("*.npz")), sorted((root / "val").glob("*.npz"))
    if not train_files or not val_files:
        raise ValueError("Need train and val NPZ files")
    val = [Counter(), Counter(), Counter()]
    val_keys = {}
    for path in val_files:
        val_keys[path.name] = fingerprints(path)
        for count, keys in zip(val, val_keys[path.name]):
            count.update(keys)
    seen = [set(), set(), set()]
    train_games = set()
    train_rows = 0
    for path in train_files:
        keys = fingerprints(path)
        train_rows += len(keys[0])
        train_games.update(keys[0])
        for found, counts, items in zip(seen, val, keys):
            found.update(k for k in items if k in counts)
        print(f"Audited {path.name}: {train_rows:,} training rows", flush=True)
    masks = {name: np.array([g not in seen[0] and p not in seen[2]
                            for g, p in zip(keys[0], keys[2])], dtype=np.bool_)
             for name, keys in val_keys.items()}
    if mask_output is not None:
        mask_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(mask_output, **masks)
    return {
        "train_rows": train_rows, "validation_rows": sum(val[0].values()),
        "train_unique_games": len(train_games), "validation_unique_games": len(val[0]),
        "shared_games": len(seen[0]),
        "validation_rows_from_shared_games": sum(val[0][k] for k in seen[0]),
        "validation_rows_with_exact_train_input": sum(val[1][k] for k in seen[1]),
        "validation_rows_with_D4_equivalent_train_input": sum(val[2][k] for k in seen[2]),
        "validation_unique_exact_inputs": len(val[1]),
        "validation_unique_D4_inputs": len(val[2]),
        "validation_rows_with_unseen_game_and_D4_input": sum(int(m.sum()) for m in masks.values()),
        "input_scope": "full-history 22 spatial + 19 global features; excludes targets",
        "files_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in train_files + val_files},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = audit(args.dataset, args.output.with_suffix(".masks.npz"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "files_sha256"}, indent=2))


if __name__ == "__main__":
    main()
