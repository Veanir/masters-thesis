"""Compact B1 RGB-D implicit-field model.

The model consumes only observations available at inference time: the masked
RGB-D image, the corresponding visible partial point cloud, camera intrinsics,
and camera-frame query coordinates.  Ground-truth meshes, object identifiers,
and mesh-to-camera transforms are deliberately absent from the interface.

``encode_observation`` and ``decode_queries`` are separate so a single image
and point-cloud encoding can be reused across bounded query chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class B1Prediction:
    """Metric unsigned distance and occupancy logit for each query point."""

    udf_m: Tensor
    occupancy_logits: Tensor


@dataclass(frozen=True)
class ObservationEncoding:
    """Reusable inference-time features for one RGB-D observation batch."""

    image_features: Tensor
    global_image_features: Tensor
    depth_m: Tensor
    mask: Tensor
    partial_points_camera_m: Tensor
    point_features: Tensor
    global_point_features: Tensor
    point_statistics: Tensor
    roi_center_camera_m: Tensor
    roi_half_extent_m: Tensor
    image_height: int
    image_width: int


RGBMode = Literal["rgbd", "depth_only", "luminance"]


def _rgb_for_mode(rgb: Tensor, mask: Tensor, mode: RGBMode) -> Tensor:
    """Apply the deterministic RGB input ablation after target masking."""

    masked_rgb = rgb * mask
    if mode == "depth_only":
        return torch.zeros_like(masked_rgb)
    if mode == "luminance":
        weights = masked_rgb.new_tensor((0.2126, 0.7152, 0.0722)).view(1, 3, 1, 1)
        luminance = torch.sum(masked_rgb * weights, dim=1, keepdim=True)
        return luminance.expand(-1, 3, -1, -1)
    return masked_rgb


class _ConvBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int, *, stride: int) -> None:
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups=min(8, output_channels), num_channels=output_channels),
            nn.SiLU(inplace=True),
        )


class _ImageEncoder(nn.Module):
    feature_channels = 48
    global_channels = 128
    spatial_moment_count = 9

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _ConvBlock(5, 16, stride=2),
            _ConvBlock(16, 32, stride=2),
            _ConvBlock(32, self.feature_channels, stride=1),
        )
        self.global_projection = nn.Sequential(
            nn.Linear(
                self.feature_channels * self.spatial_moment_count,
                self.global_channels,
            ),
            nn.SiLU(inplace=True),
        )

    def forward(self, rgbd_mask: Tensor) -> tuple[Tensor, Tensor]:
        spatial = self.features(rgbd_mask)
        y = torch.linspace(-1.0, 1.0, spatial.shape[2], device=spatial.device, dtype=spatial.dtype)
        x = torch.linspace(-1.0, 1.0, spatial.shape[3], device=spatial.device, dtype=spatial.dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        basis = torch.stack(
            (
                torch.ones_like(grid_x),
                grid_x,
                grid_y,
                grid_x.square(),
                grid_y.square(),
                grid_x * grid_y,
                torch.sin(torch.pi * grid_x),
                torch.sin(torch.pi * grid_y),
                torch.sin(torch.pi * (grid_x + grid_y)),
            )
        )
        moments = torch.einsum("bchw,khw->bck", spatial, basis) / float(
            spatial.shape[2] * spatial.shape[3]
        )
        pooled = moments.flatten(start_dim=1)
        return spatial, self.global_projection(pooled)


class _PointEncoder(nn.Module):
    point_channels = 48
    global_channels = 64

    def __init__(self) -> None:
        super().__init__()
        self.points = nn.Sequential(
            nn.Linear(3, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, self.point_channels),
            nn.SiLU(inplace=True),
        )
        self.global_projection = nn.Sequential(
            nn.Linear(self.point_channels, self.global_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, points_camera_m: Tensor) -> tuple[Tensor, Tensor]:
        point_features = self.points(points_camera_m)
        global_features = point_features.amax(dim=1)
        return point_features, self.global_projection(global_features)


class B1FieldModel(nn.Module):
    """Small RGB-D-conditioned dual UDF/occupancy field.

    ``intrinsics`` uses ``[fx, fy, cx, cy]`` ordering and shape ``[B, 4]``.
    Camera coordinates follow the project convention: visible points have
    negative Z and positive depth is ``-Z``.
    """

    def __init__(
        self,
        *,
        conditioning: Literal["pixel_knn", "global"] = "pixel_knn",
        rgb_mode: RGBMode = "rgbd",
        knn_neighbors: int = 8,
        decoder_channels: int = 128,
        minimum_roi_half_extent_m: float = 0.15,
        roi_padding_fraction: float = 0.55,
        ray_descriptor: bool = False,
    ) -> None:
        super().__init__()
        if conditioning not in {"pixel_knn", "global"}:
            raise ValueError("conditioning must be 'pixel_knn' or 'global'")
        if rgb_mode not in {"rgbd", "depth_only", "luminance"}:
            raise ValueError("rgb_mode must be 'rgbd', 'depth_only', or 'luminance'")
        if knn_neighbors <= 0:
            raise ValueError("knn_neighbors must be positive")
        if decoder_channels <= 0:
            raise ValueError("decoder_channels must be positive")
        if minimum_roi_half_extent_m <= 0.0:
            raise ValueError("minimum_roi_half_extent_m must be positive")
        if roi_padding_fraction < 0.0:
            raise ValueError("roi_padding_fraction cannot be negative")

        self.conditioning = conditioning
        self.rgb_mode = rgb_mode
        self.knn_neighbors = knn_neighbors
        self.minimum_roi_half_extent_m = minimum_roi_half_extent_m
        self.roi_padding_fraction = roi_padding_fraction
        self.ray_descriptor = ray_descriptor
        self.image_encoder = _ImageEncoder()
        self.point_encoder = _PointEncoder()

        local_decoder_channels = 0
        if conditioning == "pixel_knn":
            local_input_channels = _PointEncoder.point_channels + 3 + 1
            self.local_geometry = nn.Sequential(
                nn.Linear(local_input_channels, 64),
                nn.SiLU(inplace=True),
                nn.Linear(64, 64),
                nn.SiLU(inplace=True),
            )
            local_decoder_channels = _ImageEncoder.feature_channels + 1 + 64
            if ray_descriptor:
                local_decoder_channels += 6

        decoder_input_channels = (
            3
            + 3 * 4 * 2
            + _ImageEncoder.global_channels
            + _PointEncoder.global_channels
            + 16
            + local_decoder_channels
        )
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_channels, decoder_channels),
            nn.SiLU(inplace=True),
            nn.Linear(decoder_channels, decoder_channels),
            nn.SiLU(inplace=True),
            nn.Linear(decoder_channels, 64),
            nn.SiLU(inplace=True),
        )
        self.udf_head = nn.Linear(64, 1)
        self.occupancy_head = nn.Linear(64, 1)

    @staticmethod
    def _validate_observation(
        rgb: Tensor,
        depth_m: Tensor,
        mask: Tensor,
        partial_points_camera_m: Tensor,
    ) -> tuple[int, int, int]:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"rgb must have shape [B, 3, H, W], got {tuple(rgb.shape)}")
        batch_size, _, height, width = rgb.shape
        expected_raster_shape = (batch_size, 1, height, width)
        if tuple(depth_m.shape) != expected_raster_shape:
            raise ValueError(
                f"depth_m must have shape {expected_raster_shape}, got {tuple(depth_m.shape)}"
            )
        if tuple(mask.shape) != expected_raster_shape:
            raise ValueError(
                f"mask must have shape {expected_raster_shape}, got {tuple(mask.shape)}"
            )
        if (
            partial_points_camera_m.ndim != 3
            or partial_points_camera_m.shape[0] != batch_size
            or partial_points_camera_m.shape[2] != 3
        ):
            raise ValueError(
                "partial_points_camera_m must have shape [B, P, 3], "
                f"got {tuple(partial_points_camera_m.shape)}"
            )
        if partial_points_camera_m.shape[1] == 0:
            raise ValueError("partial_points_camera_m cannot be empty")
        if rgb.device != depth_m.device or rgb.device != mask.device:
            raise ValueError("rgb, depth_m, and mask must be on the same device")
        if rgb.device != partial_points_camera_m.device:
            raise ValueError("raster inputs and partial points must be on the same device")
        return batch_size, height, width

    def encode_observation(
        self,
        rgb: Tensor,
        depth_m: Tensor,
        mask: Tensor,
        partial_points_camera_m: Tensor,
    ) -> ObservationEncoding:
        """Encode RGB-D and visible geometry once for later query chunks."""

        _, height, width = self._validate_observation(rgb, depth_m, mask, partial_points_camera_m)
        model_dtype = next(self.parameters()).dtype
        if not rgb.is_floating_point() or rgb.dtype != model_dtype:
            rgb = rgb.to(dtype=model_dtype)
        if not depth_m.is_floating_point() or depth_m.dtype != model_dtype:
            depth_m = depth_m.to(dtype=model_dtype)
        if mask.dtype != model_dtype:
            mask = mask.to(dtype=model_dtype)
        if (
            not partial_points_camera_m.is_floating_point()
            or partial_points_camera_m.dtype != model_dtype
        ):
            partial_points_camera_m = partial_points_camera_m.to(dtype=model_dtype)

        mask = mask.clamp(0.0, 1.0)
        masked_rgb = _rgb_for_mode(rgb, mask, self.rgb_mode)
        raster_input = torch.cat((masked_rgb, depth_m * mask, mask), dim=1)
        image_features, global_image_features = self.image_encoder(raster_input)
        point_features, global_point_features = self.point_encoder(partial_points_camera_m)
        point_minimum = partial_points_camera_m.amin(dim=1)
        point_maximum = partial_points_camera_m.amax(dim=1)
        roi_center = 0.5 * (point_minimum + point_maximum)
        roi_half = (
            0.5
            * (point_maximum - point_minimum).amax(dim=1, keepdim=True)
            * (1.0 + self.roi_padding_fraction)
        ).clamp_min(self.minimum_roi_half_extent_m)
        point_statistics = torch.cat(
            (
                partial_points_camera_m.mean(dim=1),
                partial_points_camera_m.std(dim=1, unbiased=False),
                point_minimum,
                point_maximum,
                roi_center,
                roi_half,
            ),
            dim=-1,
        )
        return ObservationEncoding(
            image_features=image_features,
            global_image_features=global_image_features,
            depth_m=depth_m,
            mask=mask,
            partial_points_camera_m=partial_points_camera_m,
            point_features=point_features,
            global_point_features=global_point_features,
            point_statistics=point_statistics,
            roi_center_camera_m=roi_center,
            roi_half_extent_m=roi_half,
            image_height=height,
            image_width=width,
        )

    @staticmethod
    def _query_projection(
        encoding: ObservationEncoding,
        query_points_camera_m: Tensor,
        intrinsics: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size, query_count, _ = query_points_camera_m.shape
        if intrinsics.shape != (batch_size, 4):
            raise ValueError(
                f"intrinsics must have shape [{batch_size}, 4], got {tuple(intrinsics.shape)}"
            )
        intrinsics = intrinsics.to(
            device=query_points_camera_m.device,
            dtype=query_points_camera_m.dtype,
        )
        fx, fy, cx, cy = intrinsics.unbind(dim=-1)
        depth_m = -query_points_camera_m[..., 2]
        safe_depth_m = depth_m.clamp_min(torch.finfo(query_points_camera_m.dtype).eps)
        u = cx[:, None] + fx[:, None] * query_points_camera_m[..., 0] / safe_depth_m
        v = cy[:, None] - fy[:, None] * query_points_camera_m[..., 1] / safe_depth_m

        width_denominator = max(encoding.image_width - 1, 1)
        height_denominator = max(encoding.image_height - 1, 1)
        u_normalized = 2.0 * u / width_denominator - 1.0
        v_normalized = 2.0 * v / height_denominator - 1.0
        projection_valid = (
            (depth_m > 0.0)
            & (u_normalized >= -1.0)
            & (u_normalized <= 1.0)
            & (v_normalized >= -1.0)
            & (v_normalized <= 1.0)
        )

        sampling_grid = torch.stack((u_normalized, v_normalized), dim=-1).view(
            batch_size, query_count, 1, 2
        )
        return sampling_grid, projection_valid, u_normalized, v_normalized, depth_m

    @classmethod
    def _sample_pixel_features(
        cls,
        encoding: ObservationEncoding,
        query_points_camera_m: Tensor,
        intrinsics: Tensor,
    ) -> tuple[Tensor, Tensor]:
        sampling_grid, projection_valid, _, _, _ = cls._query_projection(
            encoding, query_points_camera_m, intrinsics
        )
        sampled = F.grid_sample(
            encoding.image_features,
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled = sampled.squeeze(-1).transpose(1, 2)
        sampled = sampled * projection_valid.unsqueeze(-1).to(dtype=sampled.dtype)
        return sampled, projection_valid.unsqueeze(-1).to(dtype=sampled.dtype)

    @classmethod
    def _sample_ray_descriptor(
        cls,
        encoding: ObservationEncoding,
        query_points_camera_m: Tensor,
        intrinsics: Tensor,
    ) -> Tensor:
        sampling_grid, projection_valid, u_normalized, v_normalized, query_depth_m = (
            cls._query_projection(encoding, query_points_camera_m, intrinsics)
        )
        sampled = (
            F.grid_sample(
                torch.cat((encoding.depth_m, encoding.mask), dim=1),
                sampling_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            .squeeze(-1)
            .transpose(1, 2)
        )
        observed_depth_m = sampled[..., 0]
        sampled_mask = sampled[..., 1]
        scale = encoding.roi_half_extent_m
        descriptor = torch.stack(
            (
                u_normalized,
                v_normalized,
                query_depth_m / scale,
                observed_depth_m / scale,
                (query_depth_m - observed_depth_m) / scale,
                sampled_mask,
            ),
            dim=-1,
        ).clamp(-4.0, 4.0)
        return descriptor * projection_valid.unsqueeze(-1).to(dtype=descriptor.dtype)

    def _local_geometry_features(
        self,
        encoding: ObservationEncoding,
        query_points_camera_m: Tensor,
    ) -> Tensor:
        distances = torch.cdist(query_points_camera_m, encoding.partial_points_camera_m)
        neighbor_count = min(self.knn_neighbors, encoding.partial_points_camera_m.shape[1])
        nearest_distances, nearest_indices = distances.topk(
            k=neighbor_count, dim=-1, largest=False, sorted=False
        )
        batch_indices = torch.arange(
            query_points_camera_m.shape[0], device=query_points_camera_m.device
        )[:, None, None]
        neighbor_points = encoding.partial_points_camera_m[batch_indices, nearest_indices]
        neighbor_features = encoding.point_features[batch_indices, nearest_indices]
        scale = encoding.roi_half_extent_m[:, None, None, :]
        offsets = (neighbor_points - query_points_camera_m.unsqueeze(2)) / scale
        nearest_distances = nearest_distances / scale.squeeze(-1)
        local_input = torch.cat(
            (neighbor_features, offsets, nearest_distances.unsqueeze(-1)), dim=-1
        )
        return self.local_geometry(local_input).amax(dim=2)

    def decode_queries(
        self,
        encoding: ObservationEncoding,
        query_points_camera_m: Tensor,
        intrinsics: Tensor,
    ) -> B1Prediction:
        """Decode one bounded camera-frame query chunk from cached features."""

        if query_points_camera_m.ndim != 3 or query_points_camera_m.shape[2] != 3:
            raise ValueError(
                "query_points_camera_m must have shape [B, Q, 3], "
                f"got {tuple(query_points_camera_m.shape)}"
            )
        batch_size, query_count, _ = query_points_camera_m.shape
        if batch_size != encoding.image_features.shape[0]:
            raise ValueError("query batch does not match observation batch")
        if query_count == 0:
            raise ValueError("query_points_camera_m cannot be empty")
        if query_points_camera_m.device != encoding.image_features.device:
            raise ValueError("queries and observation encoding must be on the same device")
        if query_points_camera_m.dtype != encoding.image_features.dtype:
            query_points_camera_m = query_points_camera_m.to(dtype=encoding.image_features.dtype)

        query_model_roi = (
            query_points_camera_m - encoding.roi_center_camera_m[:, None, :]
        ) / encoding.roi_half_extent_m[:, None, :]
        frequencies = query_points_camera_m.new_tensor((1.0, 2.0, 4.0, 8.0))
        phases = torch.pi * query_model_roi.unsqueeze(-1) * frequencies
        query_fourier = torch.cat((phases.sin(), phases.cos()), dim=-1).flatten(start_dim=2)
        global_image = encoding.global_image_features[:, None, :].expand(-1, query_count, -1)
        global_points = encoding.global_point_features[:, None, :].expand(-1, query_count, -1)
        point_statistics = encoding.point_statistics[:, None, :].expand(-1, query_count, -1)
        if self.conditioning == "pixel_knn":
            pixel_features, projection_valid = self._sample_pixel_features(
                encoding, query_points_camera_m, intrinsics
            )
            local_features = self._local_geometry_features(encoding, query_points_camera_m)
            local_decoder_features: tuple[Tensor, ...] = (
                query_model_roi,
                query_fourier,
                pixel_features,
                projection_valid,
            )
            if self.ray_descriptor:
                local_decoder_features += (
                    self._sample_ray_descriptor(encoding, query_points_camera_m, intrinsics),
                )
            decoder_features = local_decoder_features + (
                global_image,
                local_features,
                global_points,
                point_statistics,
            )
        else:
            decoder_features = (
                query_model_roi,
                query_fourier,
                global_image,
                global_points,
                point_statistics,
            )
        decoder_input = torch.cat(decoder_features, dim=-1)
        shared_features = self.decoder(decoder_input)
        udf_m = F.softplus(self.udf_head(shared_features).squeeze(-1)) * encoding.roi_half_extent_m
        occupancy_logits = self.occupancy_head(shared_features).squeeze(-1)
        return B1Prediction(udf_m=udf_m, occupancy_logits=occupancy_logits)

    def forward(
        self,
        rgb: Tensor,
        depth_m: Tensor,
        mask: Tensor,
        partial_points_camera_m: Tensor,
        query_points_camera_m: Tensor,
        intrinsics: Tensor,
    ) -> B1Prediction:
        encoding = self.encode_observation(rgb, depth_m, mask, partial_points_camera_m)
        return self.decode_queries(encoding, query_points_camera_m, intrinsics)
