"""Independent serialized-input and evaluator-coordinate boundary check."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.common.paths import script_help
from scripts.data.select_homebrewed_observations import OUT, ROOT, sha

script_help(__doc__, __name__)


def main():
    inputs = ROOT / "runs/research-evolution-hb-inputs-v2"
    obs = ROOT / "runs/research-evolution-hb-observations-v2"
    im = json.loads((inputs / "complete.json").read_text())
    om = json.loads((obs / "complete.json").read_text())
    rows = []
    assert im["contract"] == om["contract"] and not im["gt_exported"]
    assert len(im["rows"]) == len(om["rows"]) == 198
    for left, right in zip(im["rows"], om["rows"], strict=True):
        sid = left["sample_id"]
        assert sid == right["sample_id"]
        folder = inputs / sid
        assert {f.name for f in folder.iterdir()} == {
            "rgb.png",
            "depth.png",
            "mask.png",
            "camera.json",
        }
        assert all(sha(folder / n) == h for n, h in left["export_sha256"].items())
        encoded = np.asarray(Image.open(folder / "depth.png"))
        mask = np.asarray(Image.open(folder / "mask.png")) > 0
        rgb = np.asarray(Image.open(folder / "rgb.png"))
        camera = json.loads((folder / "camera.json").read_text())
        k = np.asarray(camera["K"])
        assert set(camera) == {"K", "cam2world"} and np.array_equal(camera["cam2world"], np.eye(4))
        assert (
            encoded.dtype == np.uint16
            and rgb.shape == (480, 640, 3)
            and mask.shape == encoded.shape == (480, 640)
        )
        assert sha(obs / (sid + ".npz")) == right["output_sha256"]
        with np.load(obs / (sid + ".npz")) as data:
            decoded = encoded.astype(float) * 10 / 65535
            assert (
                np.max(abs(decoded - data["depth_m"])) < 2e-15
                and np.array_equal(data["mask"], mask)
                and np.array_equal(data["rgb"], rgb)
            )
            assert np.array_equal(data["intrinsics"], k[[0, 1, 0, 1], [0, 1, 2, 2]])
            v, u = np.nonzero(mask)
            z = decoded[v, u]
            points = np.column_stack(
                ((u - k[0, 2]) * z / k[0, 0], -(v - k[1, 2]) * z / k[1, 1], -z)
            )
            n = min(512, len(points))
            idx = (
                np.floor((np.arange(n) + 0.5) * len(points) / n).astype(int)
                if n
                else np.empty(0, int)
            )
            assert data["partial_points_camera_m"].shape == (n, 3)
            error = float(np.max(abs(points[idx] - data["partial_points_camera_m"]))) if n else 0.0
            assert (
                error < 2e-15
                and left["input_pixels"] == len(points)
                and left["input_eligible"] == (len(points) >= 512)
            )
            assert (
                np.isfinite(data["mesh_vertices_camera_m"]).all()
                and data["mesh_vertices_camera_m"][:, 2].max() < 0
            )
        rows.append(
            {
                "sample_id": sid,
                "input_pixels": len(points),
                "input_eligible": left["input_eligible"],
                "backprojection_error_m": error,
            }
        )
    path = OUT / "export-v2-verification.json"
    assert not path.exists()
    result = {
        "status": "serialized_HB_export_verified",
        "rows": rows,
        "all198_preserved": True,
        "eligible": sum(r["input_eligible"] for r in rows),
        "zero_sensor_points": sum(r["input_pixels"] == 0 for r in rows),
        "gt_files_in_inference_inputs": False,
        "model_predictions_computed": False,
        "input_manifest_sha256": sha(inputs / "complete.json"),
        "observation_manifest_sha256": sha(obs / "complete.json"),
        "script_sha256": sha(Path(__file__)),
    }
    path.write_text(json.dumps(result, indent=2))
    print(
        "VERIFIED198, eligible",
        result["eligible"],
        "zero_depth",
        result["zero_sensor_points"],
        flush=True,
    )


if __name__ == "__main__":
    main()
