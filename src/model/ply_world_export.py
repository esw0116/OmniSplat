from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from torch import Tensor


def inverse_sigmoid(x):
    return torch.log(x/(1-x))


def construct_list_of_attributes(num_rest: int) -> list[str]:
    attributes = ["x", "y", "z", "nx", "ny", "nz", "red", "green", "blue"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    for i in range(num_rest):
        attributes.append(f"f_rest_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")
    
    return attributes


def export_ply(
    # extrinsics: Float[Tensor, "4 4"],
    means: Float[Tensor, "gaussian 3"],
    scales: Float[Tensor, "gaussian 3"],
    rotations: Float[Tensor, "gaussian 4"],
    harmonics: Float[Tensor, "gaussian 3 d_sh"],
    opacities: Float[Tensor, " gaussian"],
    path: Path,
):

    # Apply the rotation to the Gaussian rotations.
    rotations = R.from_quat(rotations.detach().cpu().numpy()).as_matrix()
    rotations = R.from_matrix(rotations).as_quat()
    x, y, z, w = rearrange(rotations, "g xyzw -> xyzw g")
    rotations = np.stack((w, x, y, z), axis=-1)

    # Since our axes are swizzled for the spherical harmonics, we only export the DC band.
    num_rest = 3 * (int(np.sqrt(harmonics.shape[-1]))**2 - 1)
    harmonics_view_invariant = harmonics[..., 0]
    harmonics_view_rest = harmonics[..., 1:].flatten(start_dim=1)
    
    opacities = inverse_sigmoid(opacities)

    dtype_full = [(attribute, "f4") if attribute not in ["red", "green", "blue"]  else (attribute, "u1") for attribute in construct_list_of_attributes(num_rest=num_rest)]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = (
        means.detach().cpu().numpy(),
        torch.zeros_like(means).detach().cpu().numpy(),
        np.round(np.clip(harmonics_view_invariant.detach().cpu().contiguous().numpy()* 0.28209479177387814, -1, 1)*127.5+127.5).astype(np.uint8),
        harmonics_view_invariant.detach().cpu().contiguous().numpy(),
        harmonics_view_rest.detach().cpu().contiguous().numpy(),
        opacities[..., None].detach().cpu().numpy(),
        scales.log().detach().cpu().numpy(),
        rotations,
    )
    attributes = np.concatenate(attributes, axis=1)

    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)
