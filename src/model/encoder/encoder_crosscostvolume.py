from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from collections import OrderedDict

from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import (
    BackboneMultiview,
)
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, Gaussians as rawGaussians
from .encoder import Encoder
from .costvolume.depth_predictor_crossmultiview import DepthPredictorCrossMultiView

from ...global_cfg import get_cfg

from .epipolar.epipolar_sampler import EpipolarSampler
from ..encodings.positional_encoding import PositionalEncoding
from .encoder_costvolume import EncoderCostVolumeCfg


def combine_Gaussian(gs_lst:list) -> rawGaussians:
    mean, cov, scale, rot, harmonic, opacity = [], [], [], [], [], []
    for gaussian in gs_lst:
        mean.append(gaussian.means)
        scale.append(gaussian.scales)
        rot.append(gaussian.rotations)
        cov.append(gaussian.covariances)
        harmonic.append(gaussian.harmonics)
        opacity.append(gaussian.opacities)
        
    mean = torch.cat(mean, dim=1)
    scales = torch.cat(scale, dim=1)
    rotation = torch.cat(rot, dim=1)
    cov = torch.cat(cov, dim=1)
    harmonic = torch.cat(harmonic, dim=1)
    opacity = torch.cat(opacity, dim=1)
    
    return rawGaussians(mean, cov, scales, rotation, harmonic, opacity)


class EncoderCrossCostVolume(Encoder[EncoderCostVolumeCfg]):
    backbone: BackboneMultiview
    depth_predictor:  DepthPredictorCrossMultiView
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderCostVolumeCfg) -> None:
        super().__init__(cfg)

        # multi-view Transformer backbone
        if cfg.use_epipolar_trans:
            self.epipolar_sampler = EpipolarSampler(
                num_views=get_cfg().dataset.view_sampler.num_context_views,
                num_samples=32,
            )
            self.depth_encoding = nn.Sequential(
                (pe := PositionalEncoding(10)),
                nn.Linear(pe.d_out(1), cfg.d_feature),
            )
        self.backbone = BackboneMultiview(
            feature_channels=cfg.d_feature,
            downscale_factor=cfg.downscale_factor,
            no_cross_attn=cfg.wo_backbone_cross_attn,
            use_epipolar_trans=cfg.use_epipolar_trans,
        )
        ckpt_path = cfg.unimatch_weights_path
        if get_cfg().mode == 'train':
            if cfg.unimatch_weights_path is None:
                print("==> Init multi-view transformer backbone from scratch")
            else:
                print("==> Load multi-view transformer backbone checkpoint: %s" % ckpt_path)
                unimatch_pretrained_model = torch.load(ckpt_path)["model"]
                updated_state_dict = OrderedDict(
                    {
                        k: v
                        for k, v in unimatch_pretrained_model.items()
                        if k in self.backbone.state_dict()
                    }
                )
                # NOTE: when wo cross attn, we added ffns into self-attn, but they have no pretrained weight
                is_strict_loading = not cfg.wo_backbone_cross_attn
                self.backbone.load_state_dict(updated_state_dict, strict=is_strict_loading)

        # gaussians convertor
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        # cost volume based depth predictor
        self.depth_predictor = DepthPredictorCrossMultiView(
            feature_channels=cfg.d_feature,
            upscale_factor=cfg.downscale_factor,
            num_depth_candidates=cfg.num_depth_candidates,
            costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
            costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
            costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
            gaussian_raw_channels=cfg.num_surfaces * (self.gaussian_adapter.d_in + 2),
            gaussians_per_pixel=cfg.gaussians_per_pixel,
            num_views=get_cfg().dataset.view_sampler.num_context_views,
            depth_unet_feat_dim=cfg.depth_unet_feat_dim,
            depth_unet_attn_res=cfg.depth_unet_attn_res,
            depth_unet_channel_mult=cfg.depth_unet_channel_mult,
            wo_depth_refine=cfg.wo_depth_refine,
            wo_cost_volume=cfg.wo_cost_volume,
            wo_cost_volume_refine=cfg.wo_cost_volume_refine,
        )

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,  # use deterministic to visualize the feature matching results
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ): # -> Gaussians:
        
        image_key_yin, intrinsics_key_yin = "image_yin", "intrinsics_yin"
        image_key_yang, intrinsics_key_yang = "image_yang", "intrinsics_yang"

        device = context[image_key_yin].device
        b, v, _, h, w = context[image_key_yin].shape

        # Encode the context images.
        if self.cfg.use_epipolar_trans:
            breakpoint()
            epipolar_kwargs = {
                "epipolar_sampler": self.epipolar_sampler,
                "depth_encoding": self.depth_encoding,
                "extrinsics": context["extrinsics"],
                "intrinsics": context[intrinsics_key_yin],
                "near": context["near"],
                "far": context["far"],
            }
        else:
            epipolar_kwargs = None
        
        # Add: rotated yang images
        rotate_yang = torch.rot90(context[image_key_yang], k=1, dims=(3, 4))
        assert h, w == rotate_yang.shape[-2:] 
        total_context = torch.cat([context[image_key_yin], rotate_yang], dim=1) # b, 2*v, c, h, w
        
        
        
        # trans_features_yin, cnn_features_yin = self.backbone(
        #     context[image_key_yin],
        #     attn_splits=self.cfg.multiview_trans_attn_split,
        #     return_cnn_features=True,
        #     epipolar_kwargs=epipolar_kwargs,
        # )
        
        # trans_features_yang, cnn_features_yang = self.backbone(
        #     context[image_key_yang],
        #     attn_splits=self.cfg.multiview_trans_attn_split,
        #     return_cnn_features=True,
        #     epipolar_kwargs=epipolar_kwargs,
        # )
        # trans_features_yang = torch.rot90(trans_features_yang, k=1, dims=(-2, -1))
        # cnn_features_yang = torch.rot90(cnn_features_yang, k=1, dims=(-2, -1))
        
        # trans_features = torch.cat([trans_features_yin, trans_features_yang], dim=1)
        # cnn_features = torch.cat([cnn_features_yin, cnn_features_yang], dim=1)
        
        
        
        trans_features, cnn_features = self.backbone(
            total_context,
            attn_splits=self.cfg.multiview_trans_attn_split,
            return_cnn_features=True,
            epipolar_kwargs=epipolar_kwargs,
        )  # b, 2*v, c, h, w
        
        
        # Sample depths from the resulting features.
        in_feats = trans_features
        extra_info = {}
        # extra_info['images'] = rearrange(context[image_key], "b v c h w -> (v b) c h w")
        extra_info['images'] = rearrange(total_context,  "b v c h w -> (v b) c h w")
        extra_info["scene_names"] = scene_names
        gpp = self.cfg.gaussians_per_pixel
        
        if deterministic:
            return self.depth_predictor(
                in_feats,
                context[intrinsics_key_yin],
                context["extrinsics"],
                context["near"],
                context["far"],
                gaussians_per_pixel=gpp,
                deterministic=deterministic,
                extra_info=extra_info,
                cnn_features=cnn_features,
            )
        
        depths, densities, raw_gaussians = self.depth_predictor(
            in_feats,
            context[intrinsics_key_yin],
            context["extrinsics"],
            context["near"],
            context["far"],
            gaussians_per_pixel=gpp,
            deterministic=deterministic,
            extra_info=extra_info,
            cnn_features=cnn_features,
        )
        # shape of depths:  b v (h w) srf dpt y
        assert raw_gaussians.size(-1) == 2
        
        # Convert the features and depths into Gaussians.
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        gpp = self.cfg.gaussians_per_pixel

        depths_yin, densities_yin, raw_gaussians_yin = depths[..., 0], densities[..., 0], raw_gaussians[..., 0]
        depths_yang, densities_yang, raw_gaussians_yang = depths[..., 1], densities[..., 1], raw_gaussians[..., 1]
        
        gaussians_yin = rearrange(
            raw_gaussians_yin,
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        )
        offset_xy_yin = gaussians_yin[..., :2].sigmoid()
        xy_ray_yin = xy_ray + (offset_xy_yin - 0.5) * pixel_size
        gaussians_yin = self.gaussian_adapter.forward(
            rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
            rearrange(context[intrinsics_key_yin], "b v i j -> b v () () () i j"),
            rearrange(xy_ray_yin, "b v r srf xy -> b v r srf () xy"),
            depths_yin,
            self.map_pdf_to_opacity(densities_yin, global_step) / gpp,
            rearrange(
                gaussians_yin[..., 2:],
                "b v r srf c -> b v r srf () c",
            ),
            (h, w),
            omnidirectional='yin',
        )
        
        gaussians_yang = rearrange(
            raw_gaussians_yang,
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        )
        offset_xy_yang = gaussians_yang[..., :2].sigmoid()
        xy_ray_yang = xy_ray + (offset_xy_yang - 0.5) * pixel_size
        gaussians_yang = self.gaussian_adapter.forward(
            rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
            rearrange(context[intrinsics_key_yin], "b v i j -> b v () () () i j"), # rotated 90 degree so yin intrinsics is right
            rearrange(xy_ray_yang, "b v r srf xy -> b v r srf () xy"),
            depths_yang,
            self.map_pdf_to_opacity(densities_yang, global_step) / gpp,
            rearrange(
                gaussians_yang[..., 2:],
                "b v r srf c -> b v r srf () c",
            ),
            (h, w),
            omnidirectional='yang90',
        )
        
        gaussians = combine_Gaussian([gaussians_yin, gaussians_yang])
        depths = torch.cat([depths_yin, depths_yang], dim=1)
        
        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )

        # Optionally apply a per-pixel opacity.
        opacity_multiplier = 1

        return Gaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                opacity_multiplier * gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
        )

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
                * self.cfg.downscale_factor,
            )

            # if self.cfg.apply_bounds_shim:
            #     _, _, _, h, w = batch["context"][image_key].shape
            #     near_disparity = self.cfg.near_disparity * min(h, w)
            #     batch = apply_bounds_shim(batch, near_disparity, self.cfg.far_disparity)

            return batch

        return data_shim

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None
