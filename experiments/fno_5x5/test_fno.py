import copy
import io

import pytest
import torch

from katago.train import modelconfigs
from katago.train.fno import SpectralConv2d
from katago.train.model_pytorch import Model
from muon.muon import SingleDeviceMuonWithAuxAdam


def test_spectral_values_and_gradients_against_explicit_dft():
    torch.manual_seed(3)
    layer = SpectralConv2d(2, modes=2, padding=1).double()
    layer.initialize()
    x = torch.randn(2, 2, 5, 5, dtype=torch.float64, requires_grad=True)
    padded = torch.nn.functional.pad(x, (1, 1, 1, 1))
    coords = torch.arange(7, dtype=torch.float64)
    y, z = torch.meshgrid(coords, coords, indexing='ij')
    reference = torch.einsum('oi,bi->bo', layer.dc, padded.mean((-2, -1)))[:, :, None, None].expand(-1, -1, 7, 7)
    for (kx, ky), real, imag in zip(layer.frequencies, layer.real, layer.imag):
        basis = torch.exp(2j * torch.pi * (kx * y + ky * z) / 7)
        coefficient = (padded * basis.conj()).sum((-2, -1)) / 49
        transformed = torch.einsum('oi,bi->bo', torch.complex(real, imag), coefficient)
        reference = reference + 2 * (transformed[:, :, None, None] * basis).real
    reference = reference[:, :, 1:6, 1:6]
    actual = layer(x)
    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
    params = (x, *layer.parameters())
    for a, b in zip(torch.autograd.grad(actual.square().sum(), params),
                    torch.autograd.grad(reference.square().sum(), params)):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)


def test_all_modes_identity_and_discarded_high_frequency():
    layer = SpectralConv2d(2, modes=2, padding=0)
    with torch.no_grad():
        layer.dc.copy_(torch.eye(2))
        for real, imag in zip(layer.real, layer.imag):
            real.copy_(torch.eye(2))
            imag.zero_()
    x = torch.randn(2, 2, 5, 5)
    torch.testing.assert_close(layer(x), x)
    # On a larger grid a frequency outside the retained square must be removed.
    z = torch.arange(7)
    high = torch.cos(2 * torch.pi * 3 * z / 7)[None, None, None, :].expand(1, 2, 7, 7)
    assert layer(high).abs().max() < 1e-6
    with pytest.raises(ValueError, match='Nyquist'):
        layer(torch.zeros(1, 2, 4, 4))


def test_full_model_gradients_muon_and_checkpoint():
    torch.manual_seed(1)
    config = copy.deepcopy(modelconfigs.config_of_name['fno-b5c192-w112-m2-p1'])
    config['fno_channels'] = 8
    model = Model(config, 5)
    model.initialize()
    model.eval()
    x, g = torch.randn(2, 22, 5, 5), torch.randn(2, 19)
    x[:, 0] = 1
    outputs = model(x, g)
    assert outputs[0][0].shape[-1] == 26
    sum(o.square().mean() for head in outputs for o in head).backward()
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    for block in model.blocks:
        assert all(p.grad.abs().sum() > 0 for p in block.spectral.parameters())
    groups = {}
    model.add_reg_dict(groups)
    registered = [id(p) for group in groups.values() for p in group]
    assert len(registered) == len(set(registered))
    assert set(registered) == {id(p) for p in model.parameters()}
    spectral_params = [p for b in model.blocks for p in b.spectral.parameters()]
    optimizer = SingleDeviceMuonWithAuxAdam([dict(params=spectral_params, use_muon=True, lr=1e-3, weight_decay=0.0)])
    before = [p.clone() for p in spectral_params]
    optimizer.step()
    assert all(torch.isfinite(p).all() and not torch.equal(p, q) for p, q in zip(spectral_params, before))
    outputs = model(x, g)
    payload = io.BytesIO()
    torch.save(model.state_dict(), payload)
    payload.seek(0)
    restored = Model(config, 5)
    restored.initialize()
    restored.eval()
    restored.load_state_dict(torch.load(payload, weights_only=True))
    for expected, actual in zip(outputs[0], restored(x, g)[0]):
        torch.testing.assert_close(expected, actual)
    for expected, actual in zip(outputs[0], model(x[:1], g[:1])[0]):
        torch.testing.assert_close(expected[:1], actual, atol=2e-5, rtol=2e-5)
    with pytest.raises(ValueError, match='5x5'):
        Model(config, 9)
    x[:, 0, 0, 0] = 0
    with pytest.raises(ValueError, match='full 5x5'):
        model(x, g)
