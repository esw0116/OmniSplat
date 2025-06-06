from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ...dataset import DatasetCfg
from ..types import Gaussians
# from .cuda_splatting import DepthRenderingMode, render_cuda, render_depth_cuda
from .cuda_yin_splatting import DepthRenderingMode as YinDepthRenderingMode, render_cuda as yin_render_cuda, render_depth_cuda as yin_render_depth_cuda
from .cuda_yang_splatting import DepthRenderingMode as YangDepthRenderingMode, render_cuda as yang_render_cuda, render_depth_cuda as yang_render_depth_cuda

from .decoder import Decoder, DecoderOutput, DecoderOutputCUDACfg


@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]


class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderSplattingCUDACfg,
        dataset_cfg: DatasetCfg,
    ) -> None:
        super().__init__(cfg, dataset_cfg)
        self.register_buffer(
            "background_color",
            torch.tensor(dataset_cfg.background_color, dtype=torch.float32),
            persistent=False,
        )


    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        omnidirectional: Literal["omni", "yin", "yang"] | None = None,
        depth_mode: YinDepthRenderingMode | YangDepthRenderingMode | None = None,
    ) -> DecoderOutput:
        b, v, _, _ = extrinsics.shape
        
        if omnidirectional=="yin":
            color = yin_render_cuda(
                rearrange(extrinsics, "b v i j -> (b v) i j"),
                rearrange(intrinsics, "b v i j -> (b v) i j"),
                rearrange(near, "b v -> (b v)"),
                rearrange(far, "b v -> (b v)"),
                image_shape,
                repeat(self.background_color, "c -> (b v) c", b=b, v=v),
                repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
                repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
                repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
                repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            )
        elif omnidirectional=="yang":
            color = yang_render_cuda(
                rearrange(extrinsics, "b v i j -> (b v) i j"),
                rearrange(intrinsics, "b v i j -> (b v) i j"),
                rearrange(near, "b v -> (b v)"),
                rearrange(far, "b v -> (b v)"),
                image_shape,
                repeat(self.background_color, "c -> (b v) c", b=b, v=v),
                repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
                repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
                repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
                repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            )
        else:
            raise NotImplementedError()

        if isinstance(color, DecoderOutputCUDACfg):
            alpha = color.alpha
            depth = color.depth
            color = color.color
            color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)
            depth = rearrange(depth, "(b v) h w -> b v h w", b=b, v=v)
            alpha = rearrange(alpha, "(b v) h w -> b v h w", b=b, v=v)
        
        return DecoderOutput(
            color,
            depth
            if depth_mode is None
            else self.render_depth(
                gaussians, extrinsics, intrinsics, near, far, image_shape, omnidirectional, depth_mode
            ),
            alpha
        )

    def render_depth(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        omnidirectional: Literal["omni", "yin", "yang"] | None = None,
        mode: YinDepthRenderingMode | YangDepthRenderingMode = "depth",
    ) -> Float[Tensor, "batch view height width"]:
        b, v, _, _ = extrinsics.shape
        if omnidirectional=="yin":
            result = yin_render_depth_cuda(
                rearrange(extrinsics, "b v i j -> (b v) i j"),
                rearrange(intrinsics, "b v i j -> (b v) i j"),
                rearrange(near, "b v -> (b v)"),
                rearrange(far, "b v -> (b v)"),
                image_shape,
                repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
                repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
                repeat(gaussians.opacities, "b g -> (b v) g", v=v),
                mode=mode,
            )
        elif omnidirectional=="yang":
            result = yang_render_depth_cuda(
                rearrange(extrinsics, "b v i j -> (b v) i j"),
                rearrange(intrinsics, "b v i j -> (b v) i j"),
                rearrange(near, "b v -> (b v)"),
                rearrange(far, "b v -> (b v)"),
                image_shape,
                repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
                repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
                repeat(gaussians.opacities, "b g -> (b v) g", v=v),
                mode=mode,
            )
        else:
            raise NotImplementedError()
        
        return rearrange(result, "(b v) h w -> b v h w", b=b, v=v)
