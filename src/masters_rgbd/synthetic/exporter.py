"""Dataset output writing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from masters_rgbd.synthetic.io_utils import write_json
from masters_rgbd.synthetic.renderer import RenderedScene
from masters_rgbd.synthetic.schemas import CameraMetadata, SceneMetadata


def write_rendered_scene(
    scene_dir: Path,
    rendered: RenderedScene,
    camera: CameraMetadata,
    scene: SceneMetadata,
) -> None:
    scene_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rendered.rgb).save(scene_dir / "rgb.png")
    np.save(scene_dir / "depth.npy", rendered.depth.astype(np.float32, copy=False))
    Image.fromarray(rendered.instance_mask.astype(np.uint16, copy=False)).save(
        scene_dir / "instance_mask.png"
    )
    write_json(scene_dir / "camera.json", camera.to_json())
    write_json(scene_dir / "scene.json", scene.to_json())
