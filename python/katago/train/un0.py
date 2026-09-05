"""Experimental board-conditioned Kuramoto trunk; Python inference only.

Dynamics follow Un-0's ConditionalKuramotoDynamics (MIT), reference revision
75243cab846c092527734925c2433b556f5e5ee7, https://github.com/unconv-ai/Un-0.
This Go adaptation uses a spatial encoder for drive strengths, one driver,
fixed initial phases, and a spatial decoder into KataGo's existing heads.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def kuramoto_velocity(theta, coupling, omega):
    """K[i,j] couples source j into destination i. No normalization here."""
    sin, cos = theta.sin(), theta.cos()
    return omega + cos * F.linear(sin, coupling) - sin * F.linear(cos, coupling)


class Un0Block(nn.Module):
    def __init__(self, name, c_main, config, pos_len):
        super().__init__()
        if pos_len != 5:
            raise ValueError("The experimental Un-0 trunk supports exactly 5x5 inputs")
        self.name = name
        self.channels = int(config["un0_channels"])
        self.steps = int(config["un0_steps"])
        self.integration_time = float(config.get("un0_time", 1.0))
        self.solver = config.get("un0_solver", "euler")
        if self.channels < 1 or self.steps < 1 or self.integration_time <= 0:
            raise ValueError("Un-0 channels, steps, and integration time must be positive")
        if self.solver not in ("euler", "rk4"):
            raise ValueError("Un-0 solver must be euler or rk4")
        self.n = self.channels * 25
        self.drive = nn.Conv2d(c_main, self.channels, 1)
        self.readout = nn.Conv2d(2 * self.channels, c_main, 1, bias=False)
        self.coupling = nn.Parameter(torch.empty(self.n, self.n))
        self.omega = nn.Parameter(torch.empty(self.n))
        self.driver_omega = nn.Parameter(torch.empty(()))
        # Fixed across examples and train/eval. A checkpoint contains these phases.
        self.register_buffer("initial_phase", torch.empty(self.n))
        self.register_buffer("driver_phase", torch.empty(()))
        self.coupling_enabled = True  # inference diagnostic only, not a trained ablation

    def initialize(self, fixup_scale=1.0):
        with torch.no_grad():
            nn.init.normal_(self.coupling, std=self.n ** -0.5)
            self.coupling.fill_diagonal_(0)
            nn.init.normal_(self.omega)
            nn.init.normal_(self.driver_omega)
            nn.init.uniform_(self.initial_phase, -math.pi, math.pi)
            nn.init.uniform_(self.driver_phase, -math.pi, math.pi)
            nn.init.xavier_normal_(self.drive.weight)
            nn.init.zeros_(self.drive.bias)
            nn.init.xavier_normal_(self.readout.weight, gain=fixup_scale)

    def evolve(self, drive):
        # Keep phase accumulation and trigonometry fp32 under bf16 autocast.
        # Dense products follow the caller's autocast, then accumulate in fp32.
        theta = self.initial_phase.float().expand(drive.shape[0], -1)
        coupling = self.coupling - torch.diag_embed(self.coupling.diagonal())
        if not self.coupling_enabled:
            coupling = coupling * 0
        dt = self.integration_time / self.steps

        def velocity(state, time):
            driver = self.driver_phase.float() + time * self.driver_omega.float()
            return (kuramoto_velocity(state, coupling, self.omega.float())
                    + drive.float() * torch.sin(driver - state))

        for step in range(self.steps):
            time = step * dt
            if self.solver == "euler":
                theta = theta + dt * velocity(theta, time)
            else:
                k1 = velocity(theta, time)
                k2 = velocity(theta + dt * k1 / 2, time + dt / 2)
                k3 = velocity(theta + dt * k2 / 2, time + dt / 2)
                k4 = velocity(theta + dt * k3, time + dt)
                theta = theta + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return theta

    def forward(self, x, mask, mask_sum_hw, mask_sum, extra_outputs=None, block_shared_data=None):
        if x.shape[-2:] != (5, 5):
            raise ValueError("Un-0 requires a 5x5 tensor")
        # Dense coupling is meaningful only for the fixed full-board pilot.
        if not bool(torch.all(mask == 1)):
            raise ValueError("Un-0 requires a full 5x5 board, without padded points")
        drive = self.drive(x).flatten(1)
        theta = self.evolve(drive)
        relative = theta - theta[:, :1]
        features = torch.cat((relative.sin().reshape(-1, self.channels, 5, 5),
                              relative.cos().reshape(-1, self.channels, 5, 5)), dim=1)
        residual = self.readout(features)
        if extra_outputs is not None:
            extra_outputs.report(self.name + ".phases", theta)
        return residual

    def add_reg_dict(self, reg_dict):
        reg_dict["normal"].extend([self.coupling, self.drive.weight, self.readout.weight])
        reg_dict["noreg"].extend([self.omega, self.driver_omega, self.drive.bias])

    def set_brenorm_params(self, renorm_avg_momentum, rmax, dmax):
        pass

    def add_brenorm_clippage(self, upper_rclippage, lower_rclippage, dclippage):
        pass
