"Two fixed orientation physics admissions; no RGB or model result selection."


def main():

    import argparse
    import hashlib
    import json
    import time
    import traceback
    from pathlib import Path

    import numpy as np

    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a, _ = p.parse_known_args()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    import carb
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.simulation_manager import SimulationManager
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    SimulationManager.switch_physics_engine("physx", verbose=True)
    assert SimulationManager.get_active_physics_engine() == "physx"
    carb.settings.get_settings().set_bool("/physics/updateToUsd", True)
    from scripts.data.isaac_scene_objects import digest, make_object, world_vertices

    a.output.mkdir(exist_ok=False, parents=True)
    manifest = json.loads((a.assets / "candidate-split.json").read_text())
    rows = []
    try:
        for index, row in enumerate(sorted(manifest["rows"], key=lambda r: r["key"])):
            asset = a.assets / row["asset_id"]
            assert digest(asset / "asset.json") == row["asset_json_sha256"]
            meta = json.loads((asset / "asset.json").read_text())
            outcomes = []
            started = time.monotonic()
            assert (
                meta["welded_watertight"]
                and meta["welded_winding_consistent"]
                and meta["welded_volume_m3"] > 0
            )
            assert digest(asset / "physical-mesh.npz") == meta["physical_mesh_sha256"]
            for orientation in range(2):
                omni.usd.get_context().new_stage()
                world = World(stage_units_in_meters=1.0, physics_dt=1 / 120)
                stage = omni.usd.get_context().get_stage()
                table = UsdGeom.Cube.Define(stage, "/World/Table")
                table.CreateSizeAttr(1.0)
                table.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.1))
                table.AddScaleOp().Set(Gf.Vec3f(10, 10, 0.2))
                UsdPhysics.CollisionAPI.Apply(table.GetPrim())
                root, data, _ = make_object(stage, asset, "/World/Target", (0.0, 0.0, 0.0))
                # Upright then quarter turn around object x; yaw fixed by family hash.
                yaw = (
                    int(hashlib.sha256(row["family_id"].encode()).hexdigest()[:8], 16) / 2**32 * 360
                )
                rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw) * Gf.Rotation(
                    Gf.Vec3d(1, 0, 0), 90 * orientation
                )
                q = rotation.GetQuat()
                root.GetPrim().GetAttribute("xformOp:orient").Set(Gf.Quatf(q))
                verts, _ = world_vertices(root, data)
                height = float(-verts[:, 2].min() + 0.06)
                root.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(0, 0, height))
                for prim in stage.Traverse():
                    if prim.IsA(UsdPhysics.Scene):
                        PhysxSchema.PhysxSceneAPI.Apply(prim).CreateEnableCCDAttr(True)
                world.reset()
                for _ in range(600):
                    world.step(render=False)
                before, _ = world_vertices(root, data)
                for _ in range(60):
                    world.step(render=False)
                after, transform = world_vertices(root, data)
                movement = float(np.linalg.norm(after - before, axis=1).max())
                penetration = float(max(0, -after[:, 2].min()))
                passed = bool(
                    np.isfinite(after).all() and movement <= 0.001 and penetration <= 0.002
                )
                outcomes.append(
                    {
                        "orientation": orientation,
                        "yaw_degrees": yaw,
                        "x_degrees": 90 * orientation,
                        "initial_height_m": height,
                        "max_vertex_movement_m": movement,
                        "floor_penetration_m": penetration,
                        "world_T_object": transform.tolist(),
                        "passed": passed,
                    }
                )
                world.stop()
                world.clear_instance()
            result = {
                "key": row["key"],
                "asset_id": row["asset_id"],
                "family_id": row["family_id"],
                "split_pool": row["split_pool"],
                "admission_priority": row["admission_priority"],
                "orientations": outcomes,
                "passed": all(r["passed"] for r in outcomes),
                "seconds": time.monotonic() - started,
            }
            rows.append(result)
            (a.output / "progress.json").write_text(json.dumps(rows, indent=2))
            print(
                "ADMISSION",
                index + 1,
                len(manifest["rows"]),
                row["asset_id"],
                result["passed"],
                result["seconds"],
                flush=True,
            )
        (a.output / "complete.json").write_text(
            json.dumps(
                {
                    "status": "candidate_physics_admission_complete",
                    "rows": rows,
                    "manifest_sha256": digest(a.assets / "candidate-split.json"),
                    "script_sha256": digest(Path(__file__)),
                    "collider_source_sha256": digest(
                        Path(__file__)
                        .parents[2]
                        .joinpath("scripts/data/data/isaac_scene_objects.py")
                    ),
                    "physics_dt": 1 / 120,
                    "steps": 660,
                    "scope": (
                        "Asset admission only; scene pair penetration and frame RGB-D "
                        "checks remain required."
                    ),
                },
                indent=2,
            )
        )
    except BaseException:
        (a.output / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
