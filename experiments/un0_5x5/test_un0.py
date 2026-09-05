import copy
import io

import pytest
import torch

from katago.train import modelconfigs
from katago.train.model_pytorch import Model
from katago.train.un0 import Un0Block, kuramoto_velocity


def block(steps=10, solver="euler"):
    torch.manual_seed(7)
    obj = Un0Block("test", 8, {"un0_channels": 1, "un0_steps": steps,
                               "un0_solver": solver}, 5)
    obj.initialize()
    return obj


def test_velocity_and_gradients_against_pairwise_equation():
    torch.manual_seed(1)
    theta = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    coupling = torch.randn(5, 5, dtype=torch.float64, requires_grad=True)
    omega = torch.randn(5, dtype=torch.float64, requires_grad=True)
    direct = omega + (coupling * (theta[:, None, :] - theta[:, :, None]).sin()).sum(-1)
    fast = kuramoto_velocity(theta, coupling, omega)
    torch.testing.assert_close(fast, direct)
    for a, b in zip(torch.autograd.grad(fast.square().sum(), (theta, coupling, omega)),
                    torch.autograd.grad(direct.square().sum(), (theta, coupling, omega))):
        torch.testing.assert_close(a, b)


def test_zero_coupling_and_drive_have_analytic_solution():
    obj = block()
    with torch.no_grad():
        obj.coupling.zero_()
    actual = obj.evolve(torch.zeros(2, 25))
    expected = (obj.initial_phase + obj.integration_time * obj.omega).expand(2, -1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_step_refinement_converges():
    obj = block()
    drive = torch.linspace(-1, 1, 25)[None, :]
    obj.steps, obj.solver = 256, "rk4"
    reference = obj.evolve(drive).detach()
    errors = []
    for steps in (10, 20, 40):
        obj.steps, obj.solver = steps, "euler"
        errors.append((obj.evolve(drive).detach() - reference).square().mean())
    assert errors[2] < errors[1] < errors[0]


def test_full_model_gradients_registration_and_checkpoint():
    config = copy.deepcopy(modelconfigs.config_of_name["un0-b5c192-n1250-e10"])
    config["un0_channels"] = 1
    model = Model(config, 5)
    model.initialize()
    x = torch.randn(2, 22, 5, 5)
    x[:, 0] = 1
    g = torch.randn(2, 19)
    model.eval()
    outputs = model(x, g)
    assert outputs[0][0].shape[-1] == 26
    sum(o.square().mean() for head in outputs for o in head).backward()
    obj = model.blocks[0]
    for p in obj.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert obj.coupling.grad.abs().sum() > 0
    assert obj.coupling.grad.diagonal().abs().max() == 0
    groups = {}
    model.add_reg_dict(groups)
    registered = [id(p) for group in groups.values() for p in group]
    assert len(registered) == len(set(registered))
    assert set(registered) == {id(p) for p in model.parameters()}
    payload = io.BytesIO()
    torch.save(model.state_dict(), payload)
    payload.seek(0)
    restored = Model(config, 5)
    restored.initialize()  # KataGo initialization also sets fixed norm scales.
    restored.eval()
    restored.load_state_dict(torch.load(payload, weights_only=True))
    for expected, actual in zip(outputs[0], restored(x, g)[0]):
        torch.testing.assert_close(expected, actual)
    for expected, actual in zip(outputs[0], model(x[:1], g[:1])[0]):
        torch.testing.assert_close(expected[:1], actual, atol=2e-5, rtol=2e-5)


def test_rejects_wrong_board():
    with pytest.raises(ValueError, match="5x5"):
        Un0Block("test", 8, {"un0_channels": 1, "un0_steps": 10}, 9)
    obj = block()
    mask = torch.ones(1, 1, 5, 5)
    mask[..., 0, 0] = 0
    with pytest.raises(ValueError, match="full 5x5"):
        obj(torch.zeros(1, 8, 5, 5), mask, None, None)
