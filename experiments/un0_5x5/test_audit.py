import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("un0_audit", Path(__file__).with_name("audit_data.py"))
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def write(path, boards, game_ids, glob):
    targets = np.zeros((len(boards), 80), dtype=np.float32)
    targets[:, 41] = game_ids
    np.savez(path, binaryInputNCHWPacked=np.packbits(boards.reshape(-1, 22, 25), axis=2),
             globalInputNC=glob, globalTargetsNC=targets)


def test_game_exact_and_symmetry_overlap_are_distinct(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    train = np.zeros((3, 22, 5, 5), dtype=np.uint8)
    train[:, 0] = 1
    train[0, 1, 0, 1] = train[0, 1, 2, 4] = 1
    train[1, 1, 2, 2] = 1
    train[2, 2, 3, 3] = 1
    val = train.copy()
    val[0] = np.rot90(train[0], axes=(-2, -1))
    global_train = np.zeros((3, 19), dtype=np.float32)
    global_val = global_train.copy()
    global_val[2, 5] = 1  # a different global feature means a different input
    write(tmp_path / "train/a.npz", train, [10, 20, 30], global_train)
    write(tmp_path / "val/b.npz", val, [10, 99, 98], global_val)
    result = audit_module.audit(tmp_path, tmp_path / "masks.npz")
    assert result["train_rows"] == result["validation_rows"] == 3
    assert result["shared_games"] == result["validation_rows_from_shared_games"] == 1
    assert result["validation_rows_with_exact_train_input"] == 1
    assert result["validation_rows_with_D4_equivalent_train_input"] == 2
    assert result["validation_rows_with_unseen_game_and_D4_input"] == 1
    with np.load(tmp_path / "masks.npz") as masks:
        np.testing.assert_array_equal(masks["b.npz"], [False, False, True])
