import torch
from torch.nn import functional as F
from einops import rearrange

from .unimatch.backbone import CNNEncoder
from .multiview_transformer import MultiViewFeatureTransformer
from .unimatch.utils import split_feature, merge_splits
from .unimatch.position import PositionEmbeddingSine

from ..costvolume.conversions import depth_to_relative_disparity
from ....geometry.epipolar_lines import get_depth


def erp_to_yinyang(image):
    """
    Convert erp to sphere
    image: (B, N_Views, C, H, W), range [0, 1]
    """
    b, v, c, h, w = image.shape
    image = image.reshape(b*v, c, h, w)
    # theta_values = torch.linspace(-torch.pi / 2, torch.pi / 2, h)  # Latitude (from -pi/2 to pi/2)
    # phi_values = torch.linspace(-torch.pi, torch.pi, w)             # Longitude (from -pi to pi) 
    # theta, phi = torch.meshgrid(theta_values, phi_values, indexing='ij') # [h, w]

    # Convert spherical coordinates (theta, phi) to Cartesian coordinates (x, y, z)
    # x = torch.cos(theta) * torch.sin(phi) 
    # y = -torch.sin(theta)
    # z = torch.cos(theta) * torch.cos(phi)
    
    # yin_filter = torch.logical_and(
    #         torch.logical_and(-torch.pi / 4 <= theta, theta <= torch.pi / 4),
    #         torch.logical_and(-3 * torch.pi / 4 <= phi, phi <= 3 * torch.pi / 4)
    #     )
    
    def theta_to_nheight(theta):
        # return (h-1) / torch.pi * theta - (h-1) / 2
        return -2 / torch.pi * theta
    
    def phi_to_nwidth(phi):
        # return (w-1) / (2*torch.pi) * phi + (w-1) / 2
        return 1 / torch.pi * phi
    
    yin_h, yin_w = h, 3*h
    yin_y = theta_to_nheight(torch.linspace(torch.pi / 4, -torch.pi / 4, yin_h))
    yin_x = phi_to_nwidth(torch.linspace(-3*torch.pi / 4, 3*torch.pi / 4, yin_w))
    yin_yx = torch.meshgrid(yin_y, yin_x, indexing='ij') # [h, 3h]
    yin_grid = torch.stack(yin_yx, dim=-1).unsqueeze(0).expand(b,-1,-1,-1)[..., [1,0]]
    
    yang_h, yang_w = 3*h, h
    yang_h_14 = yang_h // 4
    yang_h_23 = yang_h - 2*yang_h_14
    yang_w_23 = yang_w // 2
    
    yang1_y = theta_to_nheight(torch.linspace(torch.pi / 4, torch.pi / 2, yang_h_14))
    yang1_x = phi_to_nwidth(torch.linspace(-torch.pi / 4, torch.pi / 4, yang_w))
    yang1_yx = torch.meshgrid(yang1_y, yang1_x, indexing='ij') # [3h/4, h]
    
    yang2_y = theta_to_nheight(torch.linspace(torch.pi / 2, -torch.pi / 2, yang_h_23))
    yang2_x = phi_to_nwidth(torch.linspace(3*torch.pi / 4, torch.pi, yang_w_23))
    yang2_yx = torch.meshgrid(yang2_y, yang2_x, indexing='ij') # [3h/2, h/2]
    
    yang3_y = theta_to_nheight(torch.linspace(torch.pi / 2, -torch.pi / 2, yang_h_23))
    yang3_x = phi_to_nwidth(torch.linspace(-torch.pi, -3*torch.pi / 4, yang_w_23))
    yang3_yx = torch.meshgrid(yang3_y, yang3_x, indexing='ij') # [3h/2, h/2]
    
    yang4_y = theta_to_nheight(torch.linspace(-torch.pi / 2, -torch.pi / 4, yang_h_14))
    yang4_x = phi_to_nwidth(torch.linspace(-torch.pi / 4, torch.pi / 4, yang_w))
    yang4_yx = torch.meshgrid(yang4_y, yang4_x, indexing='ij') # [3h/4, h]
    
    yang23_yx = [torch.cat([yang2_yx[0], yang3_yx[0]], dim=1), torch.cat([yang2_yx[1], yang3_yx[1]], dim=1)] # [3h/2, h]
    yang_yx = [torch.cat([yang1_yx[0], yang23_yx[0], yang4_yx[0]], dim=0), torch.cat([yang1_yx[1], yang23_yx[1], yang4_yx[1]], dim=0)] # [3h, h]
    yang_grid = torch.stack(yang_yx, dim=-1).unsqueeze(0).expand(b,-1,-1,-1)[..., [1,0]]

    yin_image = F.grid_sample(image, yin_grid, align_corners=True)
    yang_image = F.grid_sample(image, yang_grid, align_corners=True)

    yin_image = rearrange(yin_image, "(b v) c h w -> b v c h w", b=b, v=v)
    yang_image = rearrange(yang_image, "(b v) c h w -> b v c h w", b=b, v=v)
    
    return yin_image, yang_image
    

def feature_add_position_list(features_list, attn_splits, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)

    if attn_splits > 1:  # add position in splited window
        features_splits = [
            split_feature(x, num_splits=attn_splits) for x in features_list
        ]

        position = pos_enc(features_splits[0])
        features_splits = [x + position for x in features_splits]

        out_features_list = [
            merge_splits(x, num_splits=attn_splits) for x in features_splits
        ]

    else:
        position = pos_enc(features_list[0])

        out_features_list = [x + position for x in features_list]

    return out_features_list


class BackboneMultiview(torch.nn.Module):
    """docstring for BackboneMultiview."""

    def __init__(
        self,
        feature_channels=128,
        num_transformer_layers=6,
        ffn_dim_expansion=4,
        no_self_attn=False,
        no_cross_attn=False,
        num_head=1,
        no_split_still_shift=False,
        no_ffn=False,
        global_attn_fast=True,
        downscale_factor=8,
        use_epipolar_trans=False,
    ):
        super(BackboneMultiview, self).__init__()
        self.feature_channels = feature_channels
        # Table 3: w/o cross-view attention
        self.no_cross_attn = no_cross_attn
        # Table B: w/ Epipolar Transformer
        self.use_epipolar_trans = use_epipolar_trans

        # NOTE: '0' here hack to get 1/4 features
        self.backbone = CNNEncoder(
            output_dim=feature_channels,
            num_output_scales=1 if downscale_factor == 8 else 0,
        )

        self.transformer = MultiViewFeatureTransformer(
            num_layers=num_transformer_layers,
            d_model=feature_channels,
            nhead=num_head,
            ffn_dim_expansion=ffn_dim_expansion,
            no_cross_attn=no_cross_attn,
        )

    def normalize_images(self, images):
        '''Normalize image to match the pretrained GMFlow backbone.
            images: (B, N_Views, C, H, W)
        '''
        shape = [*[1]*(images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(
            *shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(
            *shape).to(images.device)

        return (images - mean) / std

    def extract_feature(self, images):
        b, v = images.shape[:2]
        concat = rearrange(images, "b v c h w -> (b v) c h w")

        # list of [nB, C, H, W], resolution from high to low
        features = self.backbone(concat)
        if not isinstance(features, list):
            features = [features]
        # reverse: resolution from low to high
        features = features[::-1]

        features_list = [[] for _ in range(v)]
        for feature in features:
            feature = rearrange(feature, "(b v) c h w -> b v c h w", b=b, v=v)
            for idx in range(v):
                features_list[idx].append(feature[:, idx])

        return features_list

    def forward(
        self,
        images,
        attn_splits=2,
        return_cnn_features=False,
        epipolar_kwargs=None,
    ):
        ''' images: (B, N_Views, C, H, W), range [0, 1] '''
        # resolution low to high
        features_list = self.extract_feature(
            self.normalize_images(images))  # list of features

        cur_features_list = [x[0] for x in features_list]

        if return_cnn_features:
            cnn_features = torch.stack(cur_features_list, dim=1)  # [B, V, C, H, W]

        if self.use_epipolar_trans:
            # NOTE: Epipolar Transformer, only for ablation used
            # we only abalate Epipolar Transformer under 2 views setting
            assert (
                epipolar_kwargs is not None
            ), "must provide camera params to apply epipolar transformer"
            assert len(cur_features_list) == 2, "only use 2 views input for Epipolar Transformer ablation"
            feature0, feature1 = cur_features_list
            epipolar_sampler = epipolar_kwargs["epipolar_sampler"]
            depth_encoding = epipolar_kwargs["depth_encoding"]

            features = torch.stack((feature0, feature1), dim=1)  # [B, V, C, H, W]
            extrinsics = epipolar_kwargs["extrinsics"]
            intrinsics = epipolar_kwargs["intrinsics"]
            near = epipolar_kwargs["near"]
            far = epipolar_kwargs["far"]
            # Get the samples used for epipolar attention.
            sampling = epipolar_sampler.forward(
                features, extrinsics, intrinsics, near, far
            )
            # similar to pixelsplat, use camera distance as position encoding
            # Compute positionally encoded depths for the features.
            collect = epipolar_sampler.collect
            depths = get_depth(
                rearrange(sampling.origins, "b v r xyz -> b v () r () xyz"),
                rearrange(sampling.directions, "b v r xyz -> b v () r () xyz"),
                sampling.xy_sample,
                rearrange(collect(extrinsics), "b v ov i j -> b v ov () () i j"),
                rearrange(collect(intrinsics), "b v ov i j -> b v ov () () i j"),
            )

            # Clip the depths. This is necessary for edge cases where the context views
            # are extremely close together (or possibly oriented the same way).
            depths = depths.maximum(near[..., None, None, None])
            depths = depths.minimum(far[..., None, None, None])
            depths = depth_to_relative_disparity(
                depths,
                rearrange(near, "b v -> b v () () ()"),
                rearrange(far, "b v -> b v () () ()"),
            )
            depths = depth_encoding(depths[..., None])
            target = sampling.features + depths
            source = features

            features = self.transformer((source, target), attn_type="epipolar")
        else:
            # add position to features
            cur_features_list = feature_add_position_list(
                cur_features_list, attn_splits, self.feature_channels)

            # Transformer
            cur_features_list = self.transformer(
                cur_features_list, attn_num_splits=attn_splits)

            features = torch.stack(cur_features_list, dim=1)  # [B, V, C, H, W]

        if return_cnn_features:
            out_lists = [features, cnn_features]
        else:
            out_lists = [features, None]

        return out_lists
