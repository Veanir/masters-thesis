"""Bounded physical arrangements with paired RGB, clean axial depth and GT poses.

This generator requires an admitted family manifest. A small first batch must
pass external camera/RGB-D QA before the full dataset is admitted to training."""


def main():

    import argparse
    import hashlib
    import itertools
    import json
    import time
    import traceback
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from scripts.data.audit_rgb_frames import rgb_frame_qa

    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--admission", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--arrangement", type=int, required=True)
    a, _ = p.parse_known_args()
    assert a.count == 1 and 0 <= a.arrangement < 8
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "renderer": "PathTracing",
            "samples_per_pixel_per_frame": 64,
            "anti_aliasing": 0,
            "denoiser": False,
        }
    )
    import carb
    import omni.replicator.core as rep
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.simulation_manager import SimulationManager
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade

    from scripts.data.audit_triangle_meshes import check_pair, load_mesh

    SimulationManager.switch_physics_engine("physx", verbose=True)
    assert SimulationManager.get_active_physics_engine() == "physx"
    carb.settings.get_settings().set_bool("/physics/updateToUsd", True)
    rep.orchestrator.set_capture_on_play(False)
    from scripts.data.isaac_scene_objects import digest, make_object, world_vertices

    def rng_for(name):
        return np.random.default_rng(
            int(hashlib.sha256(("evolution-scenes-v1/" + name).encode()).hexdigest()[:16], 16)
        )

    def table_surface(stage, folder, style, rng):
        """Change the material of an existing flat surface, with no new geometry."""
        size = 512
        y, x = np.mgrid[:size, :size] / size
        if style == 0:
            wave = np.sin((x * 45 + np.sin(y * 8) * 0.5) * 2 * np.pi) + 0.35 * np.sin(
                x * 321 + y * 3
            )
            colors = np.array([0.60, 0.40, 0.22])[None, None, :] + wave[:, :, None] * np.array(
                [0.055, 0.045, 0.025]
            )
        elif style == 1:
            field = rng.normal(0, 0.025, (size, size, 1))
            colors = np.array([0.25, 0.28, 0.31])[None, None, :] + field
        else:
            colors = np.array([0.72, 0.70, 0.65])[None, None, :] + rng.normal(
                0, 0.045, (size, size, 1)
            )
        texture = folder / "table-texture.png"
        Image.fromarray(np.uint8(np.clip(colors, 0, 1) * 255)).save(texture)
        # Visual plane on the top of the physical table. It introduces no depth
        # offset: the top of the collision cube and this plane are both exactly z0.
        surface = UsdGeom.Mesh.Define(stage, "/World/TableSurface")
        surface.CreatePointsAttr(
            [Gf.Vec3f(-5, -5, 0), Gf.Vec3f(5, -5, 0), Gf.Vec3f(5, 5, 0), Gf.Vec3f(-5, 5, 0)]
        )
        surface.CreateFaceVertexCountsAttr([4])
        surface.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        surface.CreateSubdivisionSchemeAttr("none")
        uv = UsdGeom.PrimvarsAPI(surface).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        )
        uv.Set([Gf.Vec2f(0, 0), Gf.Vec2f(10, 0), Gf.Vec2f(10, 10), Gf.Vec2f(0, 10)])
        material = UsdShade.Material.Define(stage, "/World/TableMaterial")
        shader = UsdShade.Shader.Define(stage, "/World/TableMaterial/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set([0.55, 0.35, 0.7][style])
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        tex = UsdShade.Shader.Define(stage, "/World/TableMaterial/Texture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture.resolve()))
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        # st spans0..10. PNG has no wrap metadata; USD's default useMetadata
        # falls back to black outside0..1. Repetition must be authored explicitly.
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        reader = UsdShade.Shader.Define(stage, "/World/TableMaterial/UV")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.ConnectableAPI(), "rgb"
        )
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(surface.GetPrim()).Bind(material)
        return {
            "profile": ["light_wood", "dark_laminate", "pale_stone"][style],
            "texture_sha256": digest(texture),
            "geometry": "same plane z0",
            "wrapS": "repeat",
            "wrapT": "repeat",
        }

    def capture(annotators):
        def read():
            return (
                annotators["rgb"].get_data()[:, :, :3].copy(),
                annotators["distance_to_image_plane"].get_data().copy(),
                annotators["semantic_segmentation"].get_data(),
                annotators["camera_params"].get_data(),
            )

        rgb, depth, labels, params = read()
        attempts = []
        for retry in range(4):
            qa = rgb_frame_qa(rgb)
            attempts.append({"attempt": retry, **qa})
            if qa["valid"] or retry == 3:
                break
            old_depth = depth.copy()
            old_labels = labels["data"].copy()
            old_camera = {
                k: np.asarray(params[k]).copy() for k in ["cameraViewTransform", "cameraProjection"]
            }
            rep.orchestrator.step(rt_subframes=16, delta_time=0.0)
            rgb, depth, labels, params = read()
            stable = [
                bool(np.array_equal(old_depth, depth, equal_nan=True)),
                bool(np.array_equal(old_labels, labels["data"])),
                all(
                    np.array_equal(v, np.asarray(params[k]), equal_nan=True)
                    for k, v in old_camera.items()
                ),
            ]
            attempts[-1]["static_depth_labels_camera"] = stable
            assert all(stable), "RGB recapture changed a frozen geometry channel"
        return rgb, depth, labels, params, qa, attempts

    def main():
        admission = json.loads(a.admission.read_text())
        assert admission["status"] == "geometry_admission_frozen_before_full_scene_generation"
        all_rows = sorted(
            admission["rows"], key=lambda r: (r["split_pool"] != "train", r["admission_priority"])
        )
        selected = all_rows[a.start : a.start + a.count]
        assert len(selected) == a.count and a.count > 0
        a.output.mkdir(parents=True, exist_ok=False)
        results = []
        for ordinal, row in enumerate(selected, a.start):
            started = time.monotonic()
            target_folder = a.output / row["asset_id"]
            target_folder.mkdir()
            frames = []
            arrangements = []
            pool = [
                r
                for r in all_rows
                if r["split_pool"] == row["split_pool"] and r["family_id"] != row["family_id"]
            ]
            for arrangement in [a.arrangement]:
                if len(frames) >= 20:
                    break
                name = row["key"] + f"/arrangement{arrangement}"
                rng = rng_for(name)
                folder = target_folder / f"arrangement-{arrangement:02d}"
                folder.mkdir()
                occluders = [pool[i] for i in rng.choice(len(pool), 2, replace=False)]
                scene_rows = [row, *occluders]
                assert len({r["family_id"] for r in scene_rows}) == 3
                omni.usd.get_context().new_stage()
                world = World(stage_units_in_meters=1.0, physics_dt=1 / 120)
                stage = omni.usd.get_context().get_stage()
                table = UsdGeom.Cube.Define(stage, "/World/Table")
                table.CreateSizeAttr(1.0)
                table.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.1))
                table.AddScaleOp().Set(Gf.Vec3f(10, 10, 0.2))
                # Physics-only cube; visibility is supplied by the coplanar material surface.
                table.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
                UsdPhysics.CollisionAPI.Apply(table.GetPrim())
                material = table_surface(stage, folder, arrangement % 3, rng)
                objects = []
                radii = []
                for j, asset_row in enumerate(scene_rows):
                    asset = a.assets / asset_row["asset_id"]
                    assert digest(asset / "asset.json") == asset_row["asset_json_sha256"]
                    root, data, meta = make_object(stage, asset, f"/World/obj_{j}", (0.0, 0.0, 0.0))
                    # Reuse a certified settled orientation, yaw it, lift15mm and
                    # settle again jointly. The actual final meshes are audited below.
                    pose = np.asarray(
                        asset_row["physics_orientations"][(arrangement + j) % 2]["world_T_object"]
                    )
                    yaw = float(rng.uniform(0, 360))
                    turn = np.asarray(Gf.Matrix3d(Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw))).T
                    rotation = turn @ pose[:3, :3]
                    root.GetPrim().GetAttribute("xformOp:orient").Set(
                        Gf.Quatf(
                            Gf.Matrix3d(*rotation.T.ravel().tolist()).ExtractRotation().GetQuat()
                        )
                    )
                    verts, _ = world_vertices(root, data)
                    radius = float(np.linalg.norm(np.ptp(verts[:, :2], axis=0)) / 2)
                    radii.append(radius)
                    if j == 0:
                        xy = np.zeros(2)
                    else:
                        angle = [0.3, 2.8][j - 1] + float(rng.uniform(-0.25, 0.25))
                        distance = radii[0] + radius + float(rng.uniform(0.02, 0.08))
                        xy = distance * np.array([np.cos(angle), np.sin(angle)])
                    translation = [float(xy[0]), float(xy[1]), float(-verts[:, 2].min() + 0.015)]
                    root.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(*translation))
                    objects.append((root, data, meta))
                for prim in stage.Traverse():
                    if prim.IsA(UsdPhysics.Scene):
                        PhysxSchema.PhysxSceneAPI.Apply(prim).CreateEnableCCDAttr(True)
                world.reset()
                for _ in range(600):
                    world.step(render=False)
                before = [world_vertices(root, data)[0] for root, data, _ in objects]
                for _ in range(60):
                    world.step(render=False)
                after = [world_vertices(root, data)[0] for root, data, _ in objects]
                movement = [
                    float(np.linalg.norm(x - y, axis=1).max())
                    for x, y in zip(before, after, strict=False)
                ]
                floor = [float(max(0, -x[:, 2].min())) for x in after]
                meshes = []
                object_records = []
                for j, (root, data, meta) in enumerate(objects):
                    verts, pose = world_vertices(root, data)
                    path = folder / f"object-{j}.npz"
                    np.savez_compressed(
                        path, vertices_world_m=verts, faces=data["faces"], world_T_object=pose
                    )
                    meshes.append(load_mesh(path))
                    object_records.append(
                        {
                            "key": scene_rows[j]["key"],
                            "family_id": scene_rows[j]["family_id"],
                            "split_pool": scene_rows[j]["split_pool"],
                            "file": path.name,
                            "sha256": digest(path),
                            "world_T_object": pose.tolist(),
                            "source_mesh_sha256": meta["mesh_npz_sha256"],
                        }
                    )
                    UsdPhysics.RigidBodyAPI(root.GetPrim()).GetRigidBodyEnabledAttr().Set(False)
                pairs = [
                    {"left": i, "right": j, **check_pair(meshes[i], meshes[j])}
                    for i, j in itertools.combinations(range(3), 2)
                ]
                geometry_pass = (
                    max(movement) <= 0.001
                    and max(floor) <= 0.002
                    and all(m.is_volume for m in meshes)
                    and all(r["passed"] for r in pairs)
                )
                scene = {
                    "sample_group": name,
                    "objects": object_records,
                    "max_vertex_movement_m": movement,
                    "floor_penetration_m": floor,
                    "pairs": pairs,
                    "geometry_passed": geometry_pass,
                    "table_material": material,
                    "captures": [],
                }
                if geometry_pass:
                    center = (after[0].max(0) + after[0].min(0)) / 2
                    extent = float(np.ptp(after[0], axis=0).max())
                    dome = float(rng.uniform(300, 650))
                    intensity = float(rng.uniform(700, 1700))
                    light_angle = float(rng.uniform(0, 2 * np.pi))
                    color = [(1.0, 0.84, 0.68), (1.0, 1.0, 1.0), (0.78, 0.88, 1.0)][arrangement % 3]
                    rep.create.light(light_type="Dome", intensity=dome, color=color)
                    light_position = (
                        float(np.cos(light_angle)),
                        float(np.sin(light_angle)),
                        float(rng.uniform(1.4, 2.2)),
                    )
                    rep.create.light(
                        light_type="Sphere",
                        position=light_position,
                        intensity=intensity,
                        scale=float(rng.uniform(0.2, 0.5)),
                        color=color,
                    )
                    scene["lighting"] = {
                        "dome": dome,
                        "sphere": intensity,
                        "position": light_position,
                        "color": color,
                    }
                    focal = [24.0, 36.0, 48.0, 36.0][arrangement % 4]
                    # Vary physical focal length and camera distance together.
                    # After640px export this spans fx480/720/960, without digitally
                    # magnifying an old image or changing object occupancy alone.
                    scene["camera_recipe"] = {
                        "focal_length_mm": focal,
                        "horizontal_aperture_mm": 32.0,
                        "expected_source_fx": 960 * focal / 32,
                        "distance_multiplier_for_focal": focal / 24,
                        "resolution": [960, 720],
                    }
                    camera = rep.create.camera(
                        focal_length=focal, horizontal_aperture=32, clipping_range=(0.05, 100.0)
                    )
                    render = rep.create.render_product(camera, (960, 720))
                    annotators = {
                        n: rep.AnnotatorRegistry.get_annotator(n)
                        for n in [
                            "rgb",
                            "distance_to_image_plane",
                            "semantic_segmentation",
                            "camera_params",
                        ]
                    }
                    for annotator in annotators.values():
                        annotator.attach(render)
                    azimuth = float(rng.uniform(0, 2 * np.pi))
                    elevation = [0.35, 0.57, 0.8, 0.57][arrangement % 4]
                    for view in range(5):
                        angle = azimuth + 2 * np.pi * view / 5
                        direction = np.array([np.cos(angle), np.sin(angle), elevation])
                        direction /= np.linalg.norm(direction)
                        position = (
                            center
                            + max(0.16, extent * float(rng.uniform(2.0, 2.5)))
                            * (focal / 24)
                            * direction
                        )
                        with camera:
                            rep.modify.pose(position=tuple(position), look_at=tuple(center))
                        for _ in range(2):
                            rep.orchestrator.step(rt_subframes=4, delta_time=0.0)
                        rgb, depth, labels, params, qa, attempts = capture(annotators)
                        ids = []
                        for key, value in labels["info"]["idToLabels"].items():
                            names = value.get("class", [])
                            names = [names] if isinstance(names, str) else names
                            if row["asset_id"].casefold() in [str(x).casefold() for x in names]:
                                ids.append(int(key))
                        mask = np.isin(labels["data"], ids)
                        finite = mask & np.isfinite(depth) & (depth > 0)
                        valid = (
                            qa["valid"] and int(mask.sum()) >= 1000 and int(finite.sum()) >= 1000
                        )
                        record = {
                            "view": view,
                            "frame_id": f"{row['asset_id']}-a{arrangement:02d}-v{view}",
                            "rgb_qa": qa,
                            "capture_attempts": attempts,
                            "mask_pixels": int(mask.sum()),
                            "valid_depth_pixels": int(finite.sum()),
                            "valid": bool(valid),
                            "camera_position_world": position.tolist(),
                            "target_semantic_ids": ids,
                            "labels": labels["info"]["idToLabels"],
                        }
                        # Keep every attempted RGB for auditing; depth/camera also remain
                        # available for failures, so exclusion can be reproduced.
                        Image.fromarray(rgb).save(folder / f"rgb-{view}.png")
                        np.savez_compressed(
                            folder / f"rgbd-{view}.npz", depth_clean_m=depth, target_mask=mask
                        )
                        np.savez_compressed(
                            folder / f"camera-{view}.npz",
                            **{k: np.asarray(v) for k, v in params.items()},
                        )
                        record["files"] = {
                            f"{kind}-{view}.{suffix}": digest(folder / f"{kind}-{view}.{suffix}")
                            for kind, suffix in [("rgb", "png"), ("rgbd", "npz"), ("camera", "npz")]
                        }
                        scene["captures"].append(record)
                        if valid:
                            frames.append(
                                {"relative_folder": str(folder.relative_to(a.output)), **record}
                            )
                        if len(frames) >= 20:
                            break
                    for annotator in annotators.values():
                        annotator.detach(render)
                    render.destroy()
                (folder / "scene.json").write_text(json.dumps(scene, indent=2))
                arrangements.append(
                    {
                        "folder": folder.name,
                        "geometry_passed": geometry_pass,
                        "valid_frames": sum(r["valid"] for r in scene["captures"]),
                    }
                )
                world.stop()
                world.clear_instance()
            result = {
                "key": row["key"],
                "asset_id": row["asset_id"],
                "family_id": row["family_id"],
                "split_pool": row["split_pool"],
                "frames": frames,
                "arrangements": arrangements,
                "required_frames": 20,
                "complete_20_frames": len(frames) == 20,
                "seconds": time.monotonic() - started,
            }
            (target_folder / "complete.json").write_text(json.dumps(result, indent=2))
            results.append(result)
            (a.output / "progress.json").write_text(
                json.dumps(
                    [
                        {
                            "asset_id": r["asset_id"],
                            "valid_frames": len(r["frames"]),
                            "seconds": r["seconds"],
                        }
                        for r in results
                    ],
                    indent=2,
                )
            )
            print(
                "SCENES", ordinal + 1, row["asset_id"], len(frames), result["seconds"], flush=True
            )
        (a.output / "complete.json").write_text(
            json.dumps(
                {
                    "status": "one_isolated_arrangement_complete_requires_external_QA",
                    "rows": results,
                    "admission_sha256": digest(a.admission),
                    "script_sha256": digest(Path(__file__)),
                    "collider_source_sha256": digest(
                        Path(__file__)
                        .parents[2]
                        .joinpath("scripts/data/data/isaac_scene_objects.py")
                    ),
                    "source_resolution": [960, 720],
                    "physics_dt": 1 / 120,
                    "physics_steps": 660,
                    "target_frames": 20,
                    "max_arrangements": 8,
                    "training_ready": False,
                    "sensor_depth": (
                        "Derived later once per observation from clean depth using frozen "
                        "development calibration; shared across RGB arms."
                    ),
                },
                indent=2,
            )
        )

    try:
        main()
    except BaseException:
        a.output.mkdir(parents=True, exist_ok=True)
        (a.output / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
