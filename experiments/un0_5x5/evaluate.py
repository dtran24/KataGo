"""Fixed full-history, all-row validation, identical for old and new checkpoints.

Report mean per-symmetry losses (not the loss of an ensemble). Include tails;
preserve KataGo's example/target weights and metric denominator. No RNG use.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from katago.train.data_processing_pytorch import apply_symmetry, apply_symmetry_policy
from katago.train.load_model import load_model


def batches(directory, batch_size, device, masks=None):
    files = sorted(Path(directory).glob("*.npz"))
    if not files:
        raise ValueError(f"No NPZ files in {directory}")
    for path in files:
        with np.load(path) as data:
            packed = data["binaryInputNCHWPacked"]
            glob = data["globalInputNC"].astype(np.float32)
            targets = data["globalTargetsNC"].astype(np.float32)
            policy = data["policyTargetsNCMove"][:, :1].astype(np.float32)
        if packed.shape[1:] != (22, 4) or glob.shape[1:] != (19,) or policy.shape[-1] != 26:
            raise ValueError(f"Not a 5x5 V7 feature file: {path}")
        keep = np.ones(len(glob), dtype=np.bool_) if masks is None else masks[path.name]
        if keep.shape != (len(glob),) or keep.dtype != np.bool_:
            raise ValueError(f"Invalid unseen-input mask for {path}")
        for start in range(0, len(glob), batch_size):
            sl = slice(start, start + batch_size)
            spatial = np.unpackbits(packed[sl], axis=2)[:, :, :25].reshape(-1, 22, 5, 5)
            values = [spatial, glob[sl], policy[sl], targets[sl]]
            yield [torch.as_tensor(a, device=device, dtype=torch.float32) for a in values] + [
                torch.as_tensor(keep[sl], device=device)]


def metric_sums(policy_logits, value_logits, policy, targets):
    p = policy[:, 0]
    mass = p.sum(-1, keepdim=True)
    if not bool(torch.all(mass > 0)):
        raise ValueError("Policy target with zero mass")
    p = p / mass
    w = targets[:, 25]
    pw = w * targets[:, 26]
    vw = w * (1 - targets[:, 35])
    entropy = -(p * p.clamp_min(1e-30).log()).sum(-1)
    ce = -(p * F.log_softmax(policy_logits, -1)).sum(-1)
    vce = -(targets[:, :3] * F.log_softmax(value_logits, -1)).sum(-1)
    return torch.stack([w.sum(), (pw * ce).sum(), (1.2 * vw * vce).sum(),
                        (pw * (policy_logits.argmax(-1) == p.argmax(-1))).sum(),
                        (pw * entropy).sum(), (pw * (ce - entropy)).sum()]).double()


@torch.inference_mode()
def score(model, directory, batch_size=512, symmetries=8, max_rows=None, masks=None):
    model.eval()
    totals = torch.zeros(6, dtype=torch.float64, device=model.device)
    unseen_totals = torch.zeros_like(totals)
    unseen_rows = 0
    rows = 0
    for x, g, p, targets, keep in batches(directory, batch_size, model.device, masks):
        if max_rows is not None:
            n = min(x.shape[0], max_rows - rows)
            if n <= 0:
                break
            x, g, p, targets, keep = [a[:n] for a in (x, g, p, targets, keep)]
        if not bool(torch.all(x[:, 0] == 1)):
            raise ValueError("Expected full 5x5 masks")
        # Raw NPZ features already contain full available history. No train-time
        # 2% history truncation or random symmetry is applied here.
        for symmetry in range(symmetries):
            out = model(apply_symmetry(x, symmetry).contiguous(), g)[0]
            policy = apply_symmetry_policy(p, symmetry, 5)
            totals += metric_sums(out[0][:, 0], out[1], policy, targets)
            if masks is not None and bool(keep.any()):
                unseen_totals += metric_sums(out[0][keep, 0], out[1][keep], policy[keep], targets[keep])
        unseen_rows += int(keep.sum())
        rows += x.shape[0]
    if totals[0] <= 0 or not bool(torch.isfinite(totals).all()):
        raise ValueError("Empty or non-finite evaluation")
    values = (totals[1:] / totals[0]).tolist()
    names = ("p0loss", "vloss", "pacc1", "ptentr", "policy_kl")
    result = dict(rows=rows, symmetries=symmetries, **dict(zip(names, values)))
    if masks is not None:
        if unseen_totals[0] <= 0 or not bool(torch.isfinite(unseen_totals).all()):
            raise ValueError("Empty or non-finite unseen-game/input subset")
        result["unseen_game_and_D4_input"] = dict(rows=unseen_rows,
            **dict(zip(names, (unseen_totals[1:] / unseen_totals[0]).tolist())))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--symmetries", type=int, choices=(1, 8), default=8)
    parser.add_argument("--audit", type=Path, help="Data audit JSON and adjacent .masks.npz for secondary unseen-input metrics")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"complete": False,
              "protocol": "raw weights; fp32; full history; all rows; mean per-D4-transform loss",
              "dataset_files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(Path(args.data).glob("*.npz"))}, "models": []}
    masks = None
    if args.audit is not None:
        audit = json.loads(args.audit.read_text())
        for name, digest in report["dataset_files"].items():
            if audit["files_sha256"].get("val/" + name) != digest:
                raise ValueError("Audit masks refer to different validation data")
        with np.load(args.audit.with_suffix(".masks.npz")) as data:
            masks = {key: data[key] for key in data.files}
        report["audit"] = audit
    for checkpoint in args.checkpoints:
        model, _, state = load_model(checkpoint, use_swa=False, device=args.device, pos_len=5)
        start = time.perf_counter()
        metrics = score(model, args.data, args.batch_size, args.symmetries, masks=masks)
        entry = {"checkpoint": checkpoint, "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                 "parameters": sum(p.numel() for p in model.parameters()),
                 "samples": state.get("train_state", {}).get("global_step_samples"),
                 "seconds": time.perf_counter() - start, **metrics}
        oscillator = next((b for b in model.blocks if hasattr(b, "evolve")), None)
        if oscillator is not None:
            # Diagnostic interventions are NOT separately trained controls.
            entry["diagnostics_first_2048"] = {}
            original_steps = oscillator.steps
            for label, steps, enabled in [("base", original_steps, True),
                                           ("double_steps", original_steps * 2, True),
                                           ("coupling_off", original_steps, False)]:
                oscillator.steps, oscillator.coupling_enabled = steps, enabled
                entry["diagnostics_first_2048"][label] = score(
                    model, args.data, args.batch_size, args.symmetries, max_rows=2048)
            oscillator.steps, oscillator.coupling_enabled = original_steps, True
        report["models"].append(entry)
        print(json.dumps(entry), flush=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        del model
    report["complete"] = True
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
