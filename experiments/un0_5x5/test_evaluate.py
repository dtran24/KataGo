import importlib.util
import math
from pathlib import Path

import numpy as np
import torch

from katago.train import modelconfigs
from katago.train.metrics_pytorch import Metrics
from katago.train.model_pytorch import Model

spec = importlib.util.spec_from_file_location("un0_evaluate", Path(__file__).with_name("evaluate.py"))
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_weighted_metrics_match_katago():
    model = Model(modelconfigs.config_of_name["b5c192nbt-fson-mish-rvglr-bnh"], 5)
    metrics = Metrics(1, model)
    torch.manual_seed(3)
    policy_logits, value_logits = torch.randn(3, 26), torch.randn(3, 3)
    policy = torch.rand(3, 1, 26)
    targets = torch.zeros(3, 80)
    targets[:, :3] = torch.softmax(torch.randn(3, 3), -1)
    targets[:, 25] = torch.tensor([1., 2., 0.])
    targets[:, 26] = torch.tensor([0.5, 1., 0.])
    targets[:, 35] = torch.tensor([0., 0.2, 1.])
    p = policy[:, 0] / policy[:, 0].sum(-1, keepdim=True)
    actual = evaluation.metric_sums(policy_logits, value_logits, policy, targets)
    expected = torch.stack([
        targets[:, 25].sum(),
        metrics.loss_policy_player_samplewise(policy_logits, p, targets[:, 26], targets[:, 25]).sum(),
        metrics.loss_value_samplewise(value_logits, targets[:, :3], 1 - targets[:, 35], targets[:, 25]).sum(),
        metrics.accuracy1(policy_logits, p, targets[:, 26], targets[:, 25]),
    ]).double()
    torch.testing.assert_close(actual[:4], expected)


class UniformModel(torch.nn.Module):
    device = torch.device("cpu")

    def forward(self, x, g):
        return ((torch.zeros(len(x), 1, 26), torch.zeros(len(x), 3)),)


def test_all_rows_and_tail_are_scored_without_rng(tmp_path):
    spatial = np.zeros((5, 22, 32), dtype=np.uint8)
    spatial[:, 0, :25] = 1
    targets = np.zeros((5, 80), dtype=np.float32)
    targets[:, 0] = targets[:, 25] = targets[:, 26] = 1
    np.savez(tmp_path / "data.npz", binaryInputNCHWPacked=np.packbits(spatial, axis=2),
             globalInputNC=np.zeros((5, 19), dtype=np.float32),
             policyTargetsNCMove=np.ones((5, 1, 26)), globalTargetsNC=targets)
    rng_before = torch.random.get_rng_state()
    result = evaluation.score(UniformModel(), tmp_path, batch_size=3, symmetries=8)
    assert result["rows"] == 5
    assert math.isclose(result["p0loss"], math.log(26), abs_tol=1e-6)
    assert math.isclose(result["vloss"], 1.2 * math.log(3), abs_tol=1e-6)
    assert abs(result["policy_kl"]) < 1e-6
    assert torch.equal(rng_before, torch.random.get_rng_state())
    other = evaluation.score(UniformModel(), tmp_path, batch_size=2, symmetries=8)
    for key in result:
        assert math.isclose(result[key], other[key], abs_tol=1e-6)
    filtered = evaluation.score(UniformModel(), tmp_path, batch_size=3, symmetries=8,
        masks={"data.npz": np.array([True, False, False, False, True])})
    assert filtered["rows"] == 5
    assert filtered["unseen_game_and_D4_input"]["rows"] == 2
    assert math.isclose(filtered["unseen_game_and_D4_input"]["p0loss"], math.log(26), abs_tol=1e-6)
