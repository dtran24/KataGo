"""Fourier neural operator blocks for the fixed 5x5 teacher-learning pilot.

Implements the spectral integral operator + pointwise path from Li et al.,
ICLR 2021 (https://arxiv.org/abs/2010.08895), inside a KataGo residual bottleneck.
Real parameter matrices enforce conjugate symmetry without unused parameters.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, channels, modes=2, padding=1):
        super().__init__()
        if channels < 1 or modes < 1 or padding < 0:
            raise ValueError("Invalid spectral channels, modes, or padding")
        self.channels, self.modes, self.padding = channels, modes, padding
        # Representatives of conjugate pairs in the square [-m,m]^2.
        self.frequencies = [(kx, ky) for ky in range(1, modes + 1)
                            for kx in range(-modes, modes + 1)]
        self.frequencies += [(kx, 0) for kx in range(1, modes + 1)]
        # Each is a real 2D channel matrix, compatible with existing batched Muon.
        self.dc = nn.Parameter(torch.empty(channels, channels))
        self.real = nn.ParameterList([nn.Parameter(torch.empty(channels, channels))
                                      for _ in self.frequencies])
        self.imag = nn.ParameterList([nn.Parameter(torch.empty(channels, channels))
                                      for _ in self.frequencies])
        self.enabled = True  # inference intervention only

    def initialize(self):
        nn.init.normal_(self.dc, std=self.channels ** -0.5)
        for p in list(self.real) + list(self.imag):
            nn.init.normal_(p, std=(2 * self.channels) ** -0.5)

    def forward(self, x):
        h, w = x.shape[-2:]
        p = self.padding
        hp, wp = h + 2 * p, w + 2 * p
        if 2 * self.modes >= min(hp, wp):
            raise ValueError("Spectral modes must exclude Nyquist and fit the padded grid")
        # FFT and complex products stay fp32 during bf16 training. Preserve double
        # precision for numerical reference tests; no complex trainable parameters.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x if x.dtype == torch.float64 else x.float()
            ft = torch.fft.rfft2(F.pad(x, (p, p, p, p)), norm="ortho")
            out = torch.zeros_like(ft)
            out[:, :, 0, 0] = F.linear(ft[:, :, 0, 0].real, self.dc)
            ix = [kx % hp for kx, ky in self.frequencies]
            iy = [ky for kx, ky in self.frequencies]
            weights = torch.complex(torch.stack(tuple(self.real)), torch.stack(tuple(self.imag)))
            values = torch.einsum("bim,moi->bom", ft[:, :, ix, iy], weights)
            out[:, :, ix, iy] = values
            # ky=0 has both positive/negative kx stored by rfft2. Tie them exactly.
            for j in range(len(self.frequencies) - self.modes, len(self.frequencies)):
                kx, _ = self.frequencies[j]
                out[:, :, -kx, 0] = values[:, :, j].conj()
            result = torch.fft.irfft2(out, s=(hp, wp), norm="ortho")
            result = result[:, :, p:p+h, p:p+w]
            return result if self.enabled else result * 0


class FNOBlock(nn.Module):
    def __init__(self, name, c_main, config, pos_len):
        super().__init__()
        from .model_pytorch import NormMask, act
        if pos_len != 5:
            raise ValueError("The FNO pilot supports exactly 5x5 inputs")
        self.name = name
        width = int(config["fno_channels"])
        self.norm = NormMask(c_main, config, fixup_use_gamma=False)
        self.activation = act(config["activation"])
        self.lift = nn.Conv2d(c_main, width, 1, bias=False)
        self.spectral = SpectralConv2d(width, config["fno_modes"], config["fno_padding"])
        self.local = nn.Conv2d(width, width, 1)
        self.project = nn.Conv2d(width, c_main, 1, bias=False)

    def initialize(self, fixup_scale=1.0):
        self.norm.set_scale(fixup_scale)
        nn.init.xavier_normal_(self.lift.weight)
        self.spectral.initialize()
        nn.init.xavier_normal_(self.local.weight)
        nn.init.zeros_(self.local.bias)
        nn.init.xavier_normal_(self.project.weight)

    def forward(self, x, mask, mask_sum_hw, mask_sum, extra_outputs=None, block_shared_data=None):
        if x.shape[-2:] != (5, 5) or not bool(torch.all(mask == 1)):
            raise ValueError("FNO requires a full 5x5 board without padded points")
        z = self.lift(self.activation(self.norm(x, mask, mask_sum_hw, mask_sum)))
        z = (self.spectral(z) + self.local(z)) / math.sqrt(2)
        residual = self.project(self.activation(z))
        if extra_outputs is not None:
            extra_outputs.report(self.name + ".out", residual)
        return residual

    def add_reg_dict(self, reg_dict):
        self.norm.add_reg_dict(reg_dict)
        reg_dict["normal"].extend([self.lift.weight, self.local.weight, self.project.weight,
                                   *self.spectral.parameters()])
        reg_dict["noreg"].append(self.local.bias)

    def set_brenorm_params(self, renorm_avg_momentum, rmax, dmax):
        self.norm.set_brenorm_params(renorm_avg_momentum, rmax, dmax)

    def add_brenorm_clippage(self, upper_rclippage, lower_rclippage, dclippage):
        self.norm.add_brenorm_clippage(upper_rclippage, lower_rclippage, dclippage)
