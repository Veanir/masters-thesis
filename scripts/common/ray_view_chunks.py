"""Bound the decoder batch in inference; keep the original camera pipeline."""

from contextlib import contextmanager

import torch

from scripts.common.paths import script_help

script_help(__doc__, __name__)


def slice_batch(value, start, stop, count):
    if isinstance(value, dict):
        return {k: slice_batch(v, start, stop, count) for k, v in value.items()}
    if not isinstance(value, torch.Tensor) or value.ndim == 0 or value.shape[0] != count:
        raise ValueError("Unexpected non-batched value in prepared RaySt3R batch")
    return value[start:stop]


@contextmanager
def decoder_view_chunks(model, size):
    """Patch only the independent-view forward pass, for unscored inference.

    Caller must use eval/inference mode. Diagnostic loss aggregation is omitted;
    the enclosing official eval_model adds its normal diagnostic eval_pred.
    This wrapper is never for training or validation-loss selection.
    """
    if size < 1 or model.training:
        raise ValueError("Positive chunk size and model.eval() required")
    original = model.forward

    def forward(batch, mode="loss"):
        if mode != "viz" or torch.is_grad_enabled():
            raise RuntimeError("View chunks are exclusively for inference")
        count = batch["new_cams"]["depths"].shape[0]
        if count <= size:
            return original(batch, mode=mode)
        predictions = []
        for start in range(0, count, size):
            sliced = slice_batch(batch, start, min(start + size, count), count)
            pred, _, _ = original(sliced, mode=mode)
            predictions.append(pred)
        keys = set(predictions[0])
        assert all(set(p) == keys for p in predictions)
        merged = {k: torch.cat([p[k] for p in predictions], dim=0) for k in keys}
        return merged, batch["new_cams"], {}

    model.forward = forward
    try:
        yield
    finally:
        model.forward = original
