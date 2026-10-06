"""Choose DINO RGB context without changing depth masks or model parameters."""

from scripts.common.paths import script_help

script_help(__doc__, __name__)


def install():
    import torch
    from utils import batch_prep

    assert not hasattr(batch_prep, "_evolution_original_dino")
    original = batch_prep.compute_dino_and_store_features
    batch_prep._evolution_original_dino = original
    batch_prep._evolution_rgb_context = "masked"

    def features(dino_model, rgb, mask, dino_layers=None):
        context = batch_prep._evolution_rgb_context
        assert context in ["masked", "full"]
        # Only the local RGB multiplier changes. The caller's depth validity,
        # target masks and pointmap construction remain untouched.
        effective = mask if context == "masked" else torch.ones_like(mask)
        return original(dino_model, rgb, effective, dino_layers)

    batch_prep.compute_dino_and_store_features = features


def set_context(value):
    from utils import batch_prep

    assert value in ["masked", "full"] and hasattr(batch_prep, "_evolution_original_dino")
    batch_prep._evolution_rgb_context = value
