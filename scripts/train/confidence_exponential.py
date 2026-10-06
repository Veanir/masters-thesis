"""Numerical amendment: zero adjoints of overflowed finite-input confidence exp.

Forward is the original torch.exp. Undefined nonzero adjoints, NaN inputs and
infinite inputs remain errors. This is NOT a general nonfinite-gradient filter.
"""

import importlib

import torch

from scripts.common.paths import script_help

script_help(__doc__, __name__)


EVENTS = []
CONTEXT = {}


class ConfidenceExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = x.exp()
        ctx.save_for_backward(x, y)
        return y

    @staticmethod
    def backward(ctx, adjoint):
        x, y = ctx.saved_tensors
        irrelevant_overflow = torch.isfinite(x) & torch.isposinf(y) & (adjoint == 0)
        if bool(irrelevant_overflow.any()):
            EVENTS.append(
                {
                    **CONTEXT,
                    "count": int(irrelevant_overflow.sum()),
                    "dtype": str(x.dtype),
                    "input_min": float(x[irrelevant_overflow].min()),
                    "input_max": float(x[irrelevant_overflow].max()),
                    "upstream_adjoint_exactly_zero": True,
                }
            )
        return adjoint * torch.where(irrelevant_overflow, torch.zeros_like(y), y)


def install():
    module = importlib.import_module("models.heads.postprocess")
    original = module.reg_dense_conf

    def reg_dense_conf(x, mode):
        name, vmin, vmax = mode
        if name == "exp":
            return vmin + ConfidenceExp.apply(x).clip(max=vmax - vmin)
        return original(x, mode)

    module.reg_dense_conf = reg_dense_conf
    return original
