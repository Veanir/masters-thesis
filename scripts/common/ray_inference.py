"""Explicit inference-only portability profile; never apply during training."""

from functools import wraps

import numpy as np

from scripts.common.paths import script_help

script_help(__doc__, __name__)


PROFILE = "fp32-canonical-poses-v1"


def canonical_poses(poses):
    """Order the existing camera set by the float32 values consumed by the loader."""
    poses = np.asarray(poses)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError("Expected finite N x 4 x 4 camera matrices")
    keys = poses.astype(np.float32).reshape(len(poses), -1)
    order = np.lexsort(keys[:, ::-1].T)
    return poses[order]


def install():
    import torch
    from eval_wrapper import eval as evaluation
    from eval_wrapper import sample_poses

    assert not getattr(evaluation.EvalWrapper, "_portable_profile", None), (
        "Profile already installed"
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    original_sample = sample_poses.sample_camera_poses

    @wraps(original_sample)
    def sample(*args, **kwargs):
        return canonical_poses(original_sample(*args, **kwargs))

    sample_poses.sample_camera_poses = sample
    original_init = evaluation.EvalWrapper.__init__

    @wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        original_forward = self.model.forward

        @wraps(original_forward)
        def forward(*args, **kwargs):
            if self.model.training or torch.is_grad_enabled():
                raise RuntimeError("Portable profile is exclusively for eval/inference mode")
            with torch.autocast(device_type="cuda", enabled=False):
                return original_forward(*args, **kwargs)

        self.model.forward = forward

    evaluation.EvalWrapper.__init__ = init
    evaluation.EvalWrapper._portable_profile = PROFILE
