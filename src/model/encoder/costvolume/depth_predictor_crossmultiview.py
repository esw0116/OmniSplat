import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ..backbone.unimatch.geometry import coords_grid, yin_to_3d, yang90_to_3d, yin_from_3d, yang90_from_3d
from .ldm_unet.unet import UNetModel


def cross_warp_with_pose_depth_candidates(
    feature1,
    intrinsics,
    pose,
    depth,
    clamp_min_depth=1e-3,
    warp_padding_mode="zeros",
):
    """
    feature1: [B, C, H, W, 2] # yin, yang90 order
    intrinsics: [B, 3, 3]
    pose: [B, 4, 4]
    depth: [B, D, H, W]
    """

    assert intrinsics.size(1) == intrinsics.size(2) == 3
    assert pose.size(1) == pose.size(2) == 4
    assert depth.dim() == 4
    assert feature1.size(-1) == 2

    b, d, h, w = depth.size()
    c = feature1.size(1)
    
    feature_yin = feature1[..., 0] # b c h w
    feature_yang = feature1[..., 1]
    
    feature_one = torch.ones_like(feature_yin)[:, 0:1] # b 1 h w
    feature_yin = torch.cat([feature_yin, feature_one], dim=1) # b (c+1) h w
    feature_yang = torch.cat([feature_yang, feature_one], dim=1) # b (c+1) h w
    
    with torch.no_grad():
        # pixel coordinates
        grid = coords_grid(
            b, h, w, homogeneous=True, device=depth.device
        )  # [B, 3, H, W]
        # back project to 3D and transform viewpoint

        points_from_yin = yin_to_3d(grid.view(b, 3, -1), h, w)   # [B, 3, H*W]
        points_from_yin = torch.bmm(pose[:, :3, :3], points_from_yin).unsqueeze(2).repeat(
            1, 1, d, 1
        ) * depth.view(
            b, 1, d, h * w
        )  # [B, 3, D, H*W]
        points_from_yin = points_from_yin + pose[:, :3, -1:].unsqueeze(-1)  # [B, 3, D, H*W]
        
        points_from_yang = yang90_to_3d(grid.view(b, 3, -1), h, w)   # [B, 3, H*W]
        points_from_yang = torch.bmm(pose[:, :3, :3], points_from_yang).unsqueeze(2).repeat(
            1, 1, d, 1
        ) * depth.view(
            b, 1, d, h * w
        )  # [B, 3, D, H*W]
        points_from_yang = points_from_yang + pose[:, :3, -1:].unsqueeze(-1)  # [B, 3, D, H*W]
        
        points_to_yin_from_yin = yin_from_3d(points_from_yin, h, w)
        points_to_yang_from_yin = yang90_from_3d(points_from_yin, h, w)
        points_to_yin_from_yang = yin_from_3d(points_from_yang, h, w)
        points_to_yang_from_yang = yang90_from_3d(points_from_yang, h, w)

        pixel_coords_to_yin_from_yin = points_to_yin_from_yin[:, :2] / points_to_yin_from_yin[:, -1:].clamp(min=clamp_min_depth)  # [B, 2, D, H*W]
        pixel_coords_to_yang_from_yin = points_to_yang_from_yin[:, :2] / points_to_yang_from_yin[:, -1:].clamp(min=clamp_min_depth)  # [B, 2, D, H*W]
        pixel_coords_to_yin_from_yang = points_to_yin_from_yang[:, :2] / points_to_yin_from_yang[:, -1:].clamp(min=clamp_min_depth)  # [B, 2, D, H*W]
        pixel_coords_to_yang_from_yang = points_to_yang_from_yang[:, :2] / points_to_yang_from_yang[:, -1:].clamp(min=clamp_min_depth)  # [B, 2, D, H*W]

        # normalize to [-1, 1]
        x_grid_tifi = 2 * pixel_coords_to_yin_from_yin[:, 0] / (w - 1) - 1
        y_grid_tifi = 2 * pixel_coords_to_yin_from_yin[:, 1] / (h - 1) - 1
        grid_tifi = torch.stack([x_grid_tifi, y_grid_tifi], dim=-1)  # [B, D, H*W, 2]

        x_grid_tafi = 2 * pixel_coords_to_yang_from_yin[:, 0] / (w - 1) - 1
        y_grid_tafi = 2 * pixel_coords_to_yang_from_yin[:, 1] / (h - 1) - 1
        grid_tafi = torch.stack([x_grid_tafi, y_grid_tafi], dim=-1)  # [B, D, H*W, 2]

        x_grid_tifa = 2 * pixel_coords_to_yin_from_yang[:, 0] / (w - 1) - 1
        y_grid_tifa = 2 * pixel_coords_to_yin_from_yang[:, 1] / (h - 1) - 1
        grid_tifa = torch.stack([x_grid_tifa, y_grid_tifa], dim=-1)  # [B, D, H*W, 2]

        x_grid_tafa = 2 * pixel_coords_to_yang_from_yang[:, 0] / (w - 1) - 1
        y_grid_tafa = 2 * pixel_coords_to_yang_from_yang[:, 1] / (h - 1) - 1
        grid_tafa = torch.stack([x_grid_tafa, y_grid_tafa], dim=-1)  # [B, D, H*W, 2]

    # sample features
    warped_features_tifi = F.grid_sample(
        feature_yin,
        grid_tifi.view(b, d * h, w, 2),
        mode="bilinear",
        padding_mode=warp_padding_mode,
        align_corners=True,
    ).view(
        b, c+1, d, h, w
    )  # [B, C+1, D, H, W]
    warped_feature_tifi, warped_weight_tifi = warped_features_tifi[:, :c], warped_features_tifi[:, c:]

    warped_features_tifa = F.grid_sample(
        # feature_yang,
        feature_yin,
        grid_tifa.view(b, d * h, w, 2),
        mode="bilinear",
        padding_mode=warp_padding_mode,
        align_corners=True,
    ).view(
        b, c+1, d, h, w
    )  # [B, C+1, D, H, W]
    warped_feature_tifa, warped_weight_tifa = warped_features_tifa[:, :c], warped_features_tifa[:, c:]

    # weight_yin = warped_weight_tifi / (warped_weight_tifi + warped_weight_tifa + 1e-8)
    # warped_feature_yin = weight_yin * warped_feature_tifi + (1-weight_yin) * warped_feature_tifa

    warped_features_tafi = F.grid_sample(
        # feature_yin,
        feature_yang,
        grid_tafi.view(b, d * h, w, 2),
        mode="bilinear",
        padding_mode=warp_padding_mode,
        align_corners=True,
    ).view(
        b, c+1, d, h, w
    )  # [B, C+1, D, H, W]
    warped_feature_tafi, warped_weight_tafi = warped_features_tafi[:, :c], warped_features_tafi[:, c:]

    warped_features_tafa = F.grid_sample(
        feature_yang,
        grid_tafa.view(b, d * h, w, 2),
        mode="bilinear",
        padding_mode=warp_padding_mode,
        align_corners=True,
    ).view(
        b, c+1, d, h, w
    )  # [B, C+1, D, H, W]
    warped_feature_tafa, warped_weight_tafa = warped_features_tafa[:, :c], warped_features_tafa[:, c:]

    # weight_yang = warped_weight_tafa / (warped_weight_tafi + warped_weight_tafa + 1e-8)
    # warped_feature_yang = (1-weight_yang) * warped_feature_tafi + weight_yang * warped_feature_tafa
    
    # warped_feature = torch.stack([warped_feature_yin, warped_feature_yang], dim=-1) # b c d h w 2

    warped_feature_tafa, warped_weight_tafa = warped_features_tafa[:, :c], warped_features_tafa[:, c:]

    weight_yin = warped_weight_tifi / (warped_weight_tifi + warped_weight_tafi + 1e-8)
    warped_feature_yin = weight_yin * warped_feature_tifi + (1-weight_yin) * warped_feature_tafi

    weight_yang = warped_weight_tafa / (warped_weight_tifa + warped_weight_tafa + 1e-8)
    warped_feature_yang = (1-weight_yang) * warped_feature_tifa + weight_yang * warped_feature_tafa
    
    warped_feature = torch.stack([warped_feature_yin, warped_feature_yang], dim=-1) # b c d h w 2

    return warped_feature


def prepare_feat_proj_data_lists(
    features, intrinsics, extrinsics, near, far, num_samples
):
    # prepare features
    b, v, _, _, h, w = features.shape

    feat_lists = []
    pose_curr_lists = []
    init_view_order = list(range(v))
    feat_lists.append(rearrange(features, "b v ... -> (v b) ..."))  # (vxb c h w)
    for idx in range(1, v):
        cur_view_order = init_view_order[idx:] + init_view_order[:idx]
        cur_feat = features[:, cur_view_order]
        feat_lists.append(rearrange(cur_feat, "b v ... -> (v b) ..."))  # (vxb c h w)

        # calculate reference pose
        # NOTE: not efficient, but clearer for now
        if v > 2:
            cur_ref_pose_to_v0_list = []
            for v0, v1 in zip(init_view_order, cur_view_order):
                cur_ref_pose_to_v0_list.append(
                    extrinsics[:, v1].clone().detach().inverse()
                    @ extrinsics[:, v0].clone().detach()
                )
            cur_ref_pose_to_v0s = torch.cat(cur_ref_pose_to_v0_list, dim=0)  # (vxb c h w)
            pose_curr_lists.append(cur_ref_pose_to_v0s)
    
    # get 2 views reference pose
    # NOTE: do it in such a way to reproduce the exact same value as reported in paper
    if v == 2:
        pose_ref = extrinsics[:, 0].clone().detach()
        pose_tgt = extrinsics[:, 1].clone().detach()
        pose = pose_tgt.inverse() @ pose_ref
        pose_curr_lists = [torch.cat((pose, pose.inverse()), dim=0),]

    # unnormalized camera intrinsic
    intr_curr = intrinsics[:, :, :3, :3].clone().detach()  # [b, v, 3, 3]
    intr_curr[:, :, 0, :] *= float(w)
    intr_curr[:, :, 1, :] *= float(h)
    intr_curr = rearrange(intr_curr, "b v ... -> (v b) ...", b=b, v=v)  # [vxb 3 3]

    # prepare depth bound (inverse depth) [v*b, d]
    min_depth = rearrange(1.0 / far.clone().detach(), "b v -> (v b) 1")
    max_depth = rearrange(1.0 / near.clone().detach(), "b v -> (v b) 1")
    depth_candi_curr = (
        min_depth
        + torch.linspace(0.0, 1.0, num_samples).unsqueeze(0).to(min_depth.device)
        * (max_depth - min_depth)
    ).type_as(features)
    depth_candi_curr = repeat(depth_candi_curr, "vb d -> vb d () ()")  # [vxb, d, 1, 1]
    return feat_lists, intr_curr, pose_curr_lists, depth_candi_curr


class DepthPredictorCrossMultiView(nn.Module):
    """IMPORTANT: this model is in (v b), NOT (b v), due to some historical issues.
    keep this in mind when performing any operation related to the view dim"""

    def __init__(
        self,
        feature_channels=128,
        upscale_factor=4,
        num_depth_candidates=32,
        costvolume_unet_feat_dim=128,
        costvolume_unet_channel_mult=(1, 1, 1),
        costvolume_unet_attn_res=(),
        gaussian_raw_channels=-1,
        gaussians_per_pixel=1,
        num_views=2,
        depth_unet_feat_dim=64,
        depth_unet_attn_res=(),
        depth_unet_channel_mult=(1, 1, 1),
        wo_depth_refine=False,
        wo_cost_volume=False,
        wo_cost_volume_refine=False,
        **kwargs,
    ):
        super(DepthPredictorCrossMultiView, self).__init__()
        self.num_depth_candidates = num_depth_candidates
        self.regressor_feat_dim = costvolume_unet_feat_dim
        self.upscale_factor = upscale_factor
        # ablation settings
        # Table 3: base
        self.wo_depth_refine = wo_depth_refine
        # Table 3: w/o cost volume
        self.wo_cost_volume = wo_cost_volume
        # Table 3: w/o U-Net
        self.wo_cost_volume_refine = wo_cost_volume_refine

        # Cost volume refinement: 2D U-Net
        input_channels = feature_channels if wo_cost_volume else (num_depth_candidates + feature_channels)
        channels = self.regressor_feat_dim
        if wo_cost_volume_refine:
            self.corr_project = nn.Conv2d(input_channels, channels, 3, 1, 1)
        else:
            modules = [
                nn.Conv2d(input_channels, channels, 3, 1, 1),
                nn.GroupNorm(8, channels),
                nn.GELU(),
                UNetModel(
                    image_size=None,
                    in_channels=channels,
                    model_channels=channels,
                    out_channels=channels,
                    num_res_blocks=1,
                    attention_resolutions=costvolume_unet_attn_res,
                    channel_mult=costvolume_unet_channel_mult,
                    num_head_channels=32,
                    dims=2,
                    postnorm=True,
                    num_frames=num_views,
                    use_cross_view_self_attn=True,
                ),
                nn.Conv2d(channels, num_depth_candidates, 3, 1, 1)
            ]
            self.corr_refine_net = nn.Sequential(*modules)
            # cost volume u-net skip connection
            self.regressor_residual = nn.Conv2d(
                input_channels, num_depth_candidates, 1, 1, 0
            )

        # Depth estimation: project features to get softmax based coarse depth
        self.depth_head_lowres = nn.Sequential(
            nn.Conv2d(num_depth_candidates, num_depth_candidates * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_depth_candidates * 2, num_depth_candidates, 3, 1, 1),
        )

        # CNN-based feature upsampler
        proj_in_channels = feature_channels + feature_channels
        upsample_out_channels = feature_channels
        self.upsampler = nn.Sequential(
            nn.Conv2d(proj_in_channels, upsample_out_channels, 3, 1, 1),
            nn.Upsample(
                scale_factor=upscale_factor,
                mode="bilinear",
                align_corners=True,
            ),
            nn.GELU(),
        )
        self.proj_feature = nn.Conv2d(
            upsample_out_channels, depth_unet_feat_dim, 3, 1, 1
        )

        # Depth refinement: 2D U-Net
        input_channels = 3 + depth_unet_feat_dim + 1 + 1
        channels = depth_unet_feat_dim
        if wo_depth_refine:  # for ablations
            self.refine_unet = nn.Conv2d(input_channels, channels, 3, 1, 1)
        else:
            self.refine_unet = nn.Sequential(
                nn.Conv2d(input_channels, channels, 3, 1, 1),
                nn.GroupNorm(4, channels),
                nn.GELU(),
                UNetModel(
                    image_size=None,
                    in_channels=channels,
                    model_channels=channels,
                    out_channels=channels,
                    num_res_blocks=1, 
                    attention_resolutions=depth_unet_attn_res,
                    channel_mult=depth_unet_channel_mult,
                    num_head_channels=32,
                    dims=2,
                    postnorm=True,
                    num_frames=num_views,
                    use_cross_view_self_attn=True,
                ),
            )

        # Gaussians prediction: covariance, color
        gau_in = depth_unet_feat_dim + 3 + feature_channels
        self.to_gaussians = nn.Sequential(
            nn.Conv2d(gau_in, gaussian_raw_channels * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(
                gaussian_raw_channels * 2, gaussian_raw_channels, 3, 1, 1
            ),
        )

        # Gaussians prediction: centers, opacity
        if not wo_depth_refine:
            channels = depth_unet_feat_dim
            disps_models = [
                nn.Conv2d(channels, channels * 2, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(channels * 2, gaussians_per_pixel * 2, 3, 1, 1),
            ]
            self.to_disparity = nn.Sequential(*disps_models)

    def forward(
        self,
        features,
        intrinsics,
        extrinsics,
        near,
        far,
        gaussians_per_pixel=1,
        deterministic=True,
        extra_info=None,
        cnn_features=None,
    ):
        """IMPORTANT: this model is in (v b), NOT (b v), due to some historical issues.
        keep this in mind when performing any operation related to the view dim"""

        # format the input
        b, V, c, h, w = features.shape  # here v is multiplied by 2 because we concatenate yin, yang features
        assert V % 2 == 0
        v = V // 2
        
        features_yin = features[:, :v]
        features_yang = features[:, v:]
        features = torch.stack([features_yin, features_yang], dim=-1) # b, v, c, h, w, 2
        
        feat_comb_lists, intr_curr, pose_curr_lists, disp_candi_curr = (
            prepare_feat_proj_data_lists(
                features,
                intrinsics,
                extrinsics,
                near,
                far,
                num_samples=self.num_depth_candidates,
            )
        )
        
        
        if cnn_features is not None:
            cnn_features_yin = rearrange(cnn_features[:, :v], "b v ... -> (v b) ...")
            cnn_features_yang = rearrange(cnn_features[:, v:], "b v ... -> (v b) ...")
            cnn_features = torch.cat([cnn_features_yin, cnn_features_yang], dim=0) # ybv c h w
            # cnn_features = rearrange(cnn_features, "b v ... -> (v b) ...")

        # cost volume constructions
        feat01 = feat_comb_lists[0] # (v b) c h w 2
        if self.wo_cost_volume:
            raw_correlation_in_yin = feat01[..., 0]
            raw_correlation_in_yang = feat01[..., 1]
        else:
            raw_correlation_in_yin_lists, raw_correlation_in_yang_lists = [], []
            for feat10, pose_curr in zip(feat_comb_lists[1:], pose_curr_lists):
                # sample feat01 from feat10 via camera projection
                feat01_warped = cross_warp_with_pose_depth_candidates(
                    feat10,
                    intr_curr,
                    pose_curr,
                    1.0 / disp_candi_curr.repeat([1, 1, *feat10.shape[-3:-1]]),
                    warp_padding_mode="zeros",
                )  # [B=v*b, C, D, H, W, 2]
                
                # calculate similarity
                raw_correlation_in_yin = (feat01[..., 0].unsqueeze(2) * feat01_warped[..., 0]).sum(
                    1
                ) / (
                    c**0.5
                )  # [B, D, H, W]

                raw_correlation_in_yin_lists.append(raw_correlation_in_yin)
                
                raw_correlation_in_yang = (feat01[..., 1].unsqueeze(2) * feat01_warped[..., 1]).sum(
                    1
                ) / (
                    c**0.5
                )  # [B, D, H, W]
                raw_correlation_in_yang_lists.append(raw_correlation_in_yang)


            # #Without any refinement
            # raw_correlation_yin = torch.mean(
            #     torch.stack(raw_correlation_in_yin_lists, dim=0), dim=0, keepdim=False
            # )  # [vxb, d, h, w]
            # raw_correlation_yang = torch.mean(
            #     torch.stack(raw_correlation_in_yang_lists, dim=0), dim=0, keepdim=False
            # )  # [vxb, d, h, w]
            # raw_correlation = torch.cat([raw_correlation_yin, raw_correlation_yang], dim=0) # [yB, D, H, W] y=2
            # pdf = raw_correlation
            
            
            # average all cost volumes
            raw_correlation_in_yin = torch.mean(
                torch.stack(raw_correlation_in_yin_lists, dim=0), dim=0, keepdim=False
            )  # [vxb, d, h, w]
            raw_correlation_in_yin = torch.cat((raw_correlation_in_yin, feat01[..., 0]), dim=1) # [vxb, d+c, h, w]

            raw_correlation_in_yang = torch.mean(
                torch.stack(raw_correlation_in_yang_lists, dim=0), dim=0, keepdim=False
            )  # [vxb, d, h, w]
            raw_correlation_in_yang = torch.cat((raw_correlation_in_yang, feat01[..., 0]), dim=1) # [vxb, d+c, h, w]
            


        
        # refine cost volume via 2D u-net
        if self.wo_cost_volume_refine:
            raw_correlation_yin = self.corr_project(raw_correlation_in_yin)
            raw_correlation_yang = self.corr_project(raw_correlation_in_yang)
        else:
            raw_correlation_yin = self.corr_refine_net(raw_correlation_in_yin)  # (vb d h w)
            raw_correlation_yang = self.corr_refine_net(raw_correlation_in_yang)  # (vb d h w)
            # apply skip connection
            raw_correlation_yin = raw_correlation_yin + self.regressor_residual(
                raw_correlation_in_yin
            )
            raw_correlation_yang = raw_correlation_yang + self.regressor_residual(
                raw_correlation_in_yang
            )
        
        # concatenate yin yang since there is no cross attention between them
        raw_correlation = torch.cat([raw_correlation_yin, raw_correlation_yang], dim=0) # [yB, D, H, W] y=2
        
        # softmax to get coarse depth and density
        pdf = F.softmax(
            self.depth_head_lowres(raw_correlation), dim=1
        )  # [yB, D, H, W]
        

        disp_candi_curr = repeat(disp_candi_curr, "b d h w -> (y b) d h w", y=2)
        coarse_disps = (disp_candi_curr * pdf).sum(
            dim=1, keepdim=True
        )  # (yvb, 1, h, w)
        pdf_max = torch.max(pdf, dim=1, keepdim=True)[0]  # argmax
        
        # # use mode depth
        # pdf_argmax = torch.max(pdf, dim=1, keepdim=True)[1]
        # disp_expand = repeat(disp_candi_curr, "b d 1 1 -> b d h w", h=pdf_argmax.shape[-2], w=pdf_argmax.shape[-1])
        # coarse_disps = torch.gather(disp_expand, dim=1, index=pdf_argmax)

        if deterministic: # prepare for visualization
            pdf_argmax = torch.max(pdf, dim=1, keepdim=True)[1]
            disp_expand = repeat(disp_candi_curr, "b d 1 1 -> b d h w", h=pdf_argmax.shape[-2], w=pdf_argmax.shape[-1])
            depth_mode = F.interpolate(torch.reciprocal(torch.gather(disp_expand, dim=1, index=pdf_argmax)), scale_factor=self.upscale_factor)
            
            pdf_b, _, pdf_h, pdf_w = pdf_max.shape
            pdf_max_flatten = rearrange(pdf_max, "b 1 h w -> b 1 (h w)")
            # top_confidence_idx = torch.topk(pdf_max_flatten, k=10, dim=2)[1]
            top_confidence_idx = repeat(torch.randint(0, pdf_h*pdf_w, size=(50,)), "k -> b 1 k", b=pdf_b)
            top_confidence_hw = torch.stack([top_confidence_idx // pdf_w, top_confidence_idx % pdf_w], dim=-1) # b 1 20 2
            top_confidence_hw = top_confidence_hw * self.upscale_factor + self.upscale_factor // 2
            # return fine_disps, depth_mode, top_confidence_hw, disp_candi_curr
            
            
            # 각 target view 에서 pdf top k 계산하기
            # pdf_topk_value, pdf_topk_idx = torch.topk(pdf, 5, dim=1, keepdim=True) # B k h w
            
            
            
        pdf_max = F.interpolate(pdf_max, scale_factor=self.upscale_factor)

        fullres_disps = F.interpolate(
            coarse_disps,
            scale_factor=self.upscale_factor,
            mode="bilinear",
            align_corners=True,
        )

        # depth refinement
        feat01_cat = rearrange(feat01, "B c h w y -> (y B) c h w") # y=2
        proj_feat_in_fullres = self.upsampler(torch.cat((feat01_cat, cnn_features), dim=1)) # (y B) 2c sh sw
        proj_feature = self.proj_feature(proj_feat_in_fullres)  # (y B) u sh sw
        refine_out = self.refine_unet(torch.cat(
            (extra_info["images"], proj_feature, fullres_disps, pdf_max), dim=1
        ))

        # gaussians head
        raw_gaussians_in = [refine_out,
                            extra_info["images"], proj_feat_in_fullres]
        raw_gaussians_in = torch.cat(raw_gaussians_in, dim=1)
        raw_gaussians = self.to_gaussians(raw_gaussians_in)
        raw_gaussians = rearrange(
            raw_gaussians, "(y v b) c h w -> b v (h w) c y", v=v, b=b, y=2
        )

        if self.wo_depth_refine:
            densities = repeat(
                pdf_max,
                "(y v b) dpt h w -> b v (h w) srf dpt y",
                b=b,
                v=v,
                srf=1,
                y=2,
            )
            depths = 1.0 / fullres_disps
            depths = repeat(
                depths,
                "(y v b) dpt h w -> b v (h w) srf dpt y",
                b=b,
                v=v,
                srf=1,
                y=2,
            )
        else:
            # delta fine depth and density
            delta_disps_density = self.to_disparity(refine_out)
            delta_disps, raw_densities = delta_disps_density.split(
                gaussians_per_pixel, dim=1
            )

            # combine coarse and fine info and match shape
            densities = repeat(
                F.sigmoid(raw_densities),
                "(y v b) dpt h w -> b v (h w) srf dpt y",
                b=b,
                v=v,
                srf=1,
                y = 2
            )

            fine_disps = (fullres_disps + delta_disps).clamp(
                1.0 / repeat(far, "b v -> (2 v b) () () ()"),
                1.0 / repeat(near, "b v -> (2 v b) () () ()"),
            )
            depths = 1.0 / fine_disps
            depths = repeat(
                depths,
                "(y v b) dpt h w -> b v (h w) srf dpt y",
                b=b,
                v=v,
                srf=1,
                y=2,
            )
            
        if deterministic:
            return fine_disps, depth_mode, top_confidence_hw, disp_candi_curr


        return depths, densities, raw_gaussians
