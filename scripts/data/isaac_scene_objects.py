"""Isaac object construction and metric world-space export."""

import hashlib
import json

import numpy as np

from scripts.common.paths import script_help

script_help(__doc__, __name__)
if __name__ != "__main__":
    from isaacsim.core.utils.semantics import add_labels
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_object(stage, asset, path, position):
    meta = json.loads((asset / "asset.json").read_text())
    assert digest(asset / "mesh.npz") == meta["mesh_npz_sha256"]
    assert digest(asset / "texture.png") == meta["texture_sha256"]
    data = np.load(asset / "mesh.npz")
    root = UsdGeom.Xform.Define(stage, path)
    root.AddTranslateOp().Set(Gf.Vec3d(*position))
    root.AddOrientOp().Set(Gf.Quatf(1.0))
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(0.25)
    rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim())
    rigid.CreateSolverPositionIterationCountAttr(16)
    rigid.CreateSolverVelocityIterationCountAttr(4)
    rigid.CreateEnableCCDAttr(True)
    mesh = UsdGeom.Mesh.Define(stage, path + "/Mesh")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(data["vertices"]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(data["faces"]), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(data["faces"].ravel()))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    uv.Set(Vt.Vec2fArray.FromNumpy(data["uv"]))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexDecomposition")
    decomposition = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(mesh.GetPrim())
    decomposition.CreateVoxelResolutionAttr(1000000)
    decomposition.CreateErrorPercentageAttr(0.5)
    decomposition.CreateMaxConvexHullsAttr(64)
    decomposition.CreateShrinkWrapAttr(True)
    decomposition.CreateMinThicknessAttr(0.00025)
    collision = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
    collision.CreateContactOffsetAttr(0.002)
    collision.CreateRestOffsetAttr(0.0)
    material = UsdShade.Material.Define(stage, path + "/Material")
    shader = UsdShade.Shader.Define(stage, path + "/Material/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    texture = UsdShade.Shader.Define(stage, path + "/Material/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        str((asset / "texture.png").resolve())
    )
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    reader = UsdShade.Shader.Define(stage, path + "/Material/UV")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result"
    )
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.ConnectableAPI(), "rgb"
    )
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    add_labels(root.GetPrim(), [asset.name])
    return root, data, meta


def world_vertices(root, data):
    # USD matrices act on row vectors. Transpose before column-vector export.
    transform = np.array(
        UsdGeom.XformCache().GetLocalToWorldTransform(root.GetPrim()), dtype=np.float64
    ).T
    vertices = data["vertices"].astype(np.float64) @ transform[:3, :3].T + transform[:3, 3]
    return vertices, transform
