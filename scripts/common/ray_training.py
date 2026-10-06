"""Explicit trainable subset and official RaySt3R batch contract."""

import os
import sys
from pathlib import Path

import numpy as np
import torch

from scripts.common.paths import ROOT

BASE = Path(os.environ.get("RGBD_MODEL_WORK", str(ROOT)))
VENDOR = BASE / "vendor/rayst3r"


def load_models():
    sys.path.insert(0, str(VENDOR))
    os.chdir(VENDOR)
    from eval_wrapper.eval import EvalWrapper

    model = EvalWrapper(str(BASE / "weights/rayst3r.pth")).model
    dino = torch.hub.load(
        str(BASE / "vendor/dinov2"), "dinov2_vitl14_reg", source="local", pretrained=False
    )
    dino.load_state_dict(
        torch.load(
            BASE / "weights/dinov2_vitl14_reg4_pretrain.pth", map_location="cpu", weights_only=True
        ),
        strict=True,
    )
    dino.eval().cuda().requires_grad_(False)
    model.eval().requires_grad_(False)
    for block in list(model.decoder_blocks)[-2:]:
        block.train().requires_grad_(True)
    model.pts_head.train().requires_grad_(True)
    model.classifier_head.train().requires_grad_(True)
    trainable = {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}
    assert trainable and sum(trainable.values()) < sum(p.numel() for p in model.parameters())
    return model, dino, trainable


def make_batch(data, views, rgb=None):
    rgb = data["rgb"] if rgb is None else rgb
    K = torch.from_numpy(data["K"].copy()).float()
    mask = torch.from_numpy(data["input_mask"].copy()).bool()
    return {
        "input_cams": {
            "c2ws": torch.eye(4)[None, None],
            "Ks": K[None, None],
            "depths": torch.from_numpy(data["input_depth_uint16"].astype(np.float32))[None, None],
            "valid_masks": mask[None, None],
            "original_valid_masks": mask[None, None].clone(),
            "imgs": torch.from_numpy(rgb.copy())[None, None],
        },
        "new_cams": {
            "c2ws": torch.from_numpy(data["novel_c2ws"][views].copy())[None],
            "Ks": K[None, None].repeat(1, len(views), 1, 1),
            "depths": torch.from_numpy(data["novel_depth_uint16"][views].astype(np.float32))[None],
            "valid_masks": torch.from_numpy(data["novel_masks"][views].copy())[None].bool(),
        },
    }


def loss_for(model, dino, batch):
    from engine import eval_model

    return eval_model(model, batch, mode="loss", dino_model=dino)["loss"]
