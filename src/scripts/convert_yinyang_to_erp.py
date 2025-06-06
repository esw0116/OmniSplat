import os, glob
import torch
from torch.nn.functional import interpolate, pad
from PIL import Image
import torchvision
from torchvision.utils import save_image
import equilib


def combine_yinyang(yin_image, yang_image, yin_alpha, yang_alpha):
    b, c, h, w = yin_image.shape
    assert w == 2*h
    
    yin_alpha, yang_alpha = yin_alpha[:, 0:1], yang_alpha[:, 0:1]
    sum_alpha = yin_alpha + yang_alpha + 1e-8
    yin_portion = yin_alpha / sum_alpha
    yang_portion = yang_alpha / sum_alpha
    yin_portion[sum_alpha < 0.03] = 0
    yang_portion[sum_alpha < 0.03] = 0
    
    erp_image = yin_image * yin_portion + yang_image * yang_portion
    erp_alpha = yin_portion + yang_portion
    
    return erp_image, erp_alpha


def convert_yinyang(yin_image, yang_image, yin_alpha, yang_alpha, target_shape=[512, 1024]):
    b, c, h, w = yin_image.shape
    alpha_c = yin_alpha.shape[1]
    assert w == 3*h
    assert yang_image.shape[-2] == w
    assert yang_image.shape[-1] == h
    
    erp_h, erp_w = target_shape
    h_half = erp_h // 4
    assert erp_h == h_half * 4
    rot_image = equilib.Equi2Equi(height=erp_h, width=erp_w, mode='bicubic')
    
    yang_erp_image, yin_erp_image = torch.zeros((b, c, erp_h, erp_w)),  torch.zeros((b, c, erp_h, erp_w))
    yang_erp_alpha, yin_erp_alpha = torch.zeros((b, alpha_c, erp_h, erp_w)),  torch.zeros((b, alpha_c, erp_h, erp_w))
    
    yang_erp_image[..., h_half:3*h_half, h_half:7*h_half] = interpolate(torch.rot90(yang_image, 1, [2,3]), size=(2*h_half, 6*h_half), mode='bicubic', align_corners=True)
    yang_erp_image = rot_image(src=yang_erp_image, rots=[{'yaw': 0, 'roll':3*torch.pi/2, 'pitch': 0}]*b)
    yang_erp_image = rot_image(src=yang_erp_image, rots=[{'yaw':torch.pi, 'roll': 0, 'pitch': 0}]*b)
    
    yang_erp_alpha[..., h_half:3*h_half, h_half:7*h_half] = interpolate(torch.rot90(yang_alpha, 1, [2,3]), size=(2*h_half, 6*h_half), mode='bicubic', align_corners=True)
    yang_erp_alpha = rot_image(src=yang_erp_alpha, rots=[{'yaw': 0, 'roll':3*torch.pi/2, 'pitch': 0}]*b)
    yang_erp_alpha = rot_image(src=yang_erp_alpha, rots=[{'yaw':torch.pi, 'roll': 0, 'pitch': 0}]*b)
    
    yin_erp_image[..., h_half:3*h_half, h_half:7*h_half] = interpolate(yin_image, size=(2*h_half, 6*h_half), mode='bicubic', align_corners=True)
    yin_erp_alpha[..., h_half:3*h_half, h_half:7*h_half] = interpolate(yin_alpha, size=(2*h_half, 6*h_half), mode='bicubic', align_corners=True)
    
    erp_image, erp_alpha = combine_yinyang(yin_erp_image, yang_erp_image, yin_erp_alpha, yang_erp_alpha)
    
    return erp_image, erp_alpha


def convert_yinyang_original(yin_image, yang_image, target_shape=[512, 1024]):
    b, c, h, w = yin_image.shape
    assert w == 3*h
    assert yang_image.shape[-2] == w
    assert yang_image.shape[-1] == h
    
    erp_h, erp_w = target_shape
    h_half = erp_h // 4
    assert erp_h == h_half * 4
    rot_image = equilib.Equi2Equi(height=erp_h, width=erp_w, mode='bilinear')
    
    erp_image = torch.zeros((b, c, erp_h, erp_w))
    yang_pad = pad(interpolate(torch.rot90(yang_image, 1, [2,3]), size=(2*h_half, 6*h_half), mode='bicubic', align_corners=True), pad=(1,1,1,1), mode='replicate')
    erp_image[..., h_half-1:3*h_half+1, h_half-1:7*h_half+1] = yang_pad
    
    erp_image = rot_image(src=erp_image, rots=[{'roll':-torch.pi/2, 'yaw': 0, 'pitch': torch.pi}]*b)
    erp_image[..., h_half:3*h_half, h_half:7*h_half] = interpolate(yin_image, size=(2*h_half, 6*h_half), mode='bicubic', align_corners=True)
    
    return erp_image

if __name__ == '__main__':

    yin_dir = 'dataset/OmniDatasets_1024x512/Ricoh360/bricks/yin256'
    yang_dir = 'dataset/OmniDatasets_1024x512/Ricoh360/bricks/yang256'
    output_dir = 'outputs/temp'
    
    yin_img_paths = sorted(glob.glob(os.path.join(yin_dir, '*.png')))
    yang_img_paths = sorted(glob.glob(os.path.join(yang_dir, '*.png')))
    yin_alpha_paths, yang_alpha_paths = yin_img_paths, yang_img_paths
    
    
    pil_to_tensor = torchvision.transforms.ToTensor()

    for yin_img_path, yang_img_path, yin_alpha_path, yang_alpha_path in zip(yin_img_paths, yang_img_paths, yin_alpha_paths, yang_alpha_paths):
        folder_name = os.path.basename(os.path.dirname(os.path.dirname(yin_img_path)))
        img_name = os.path.basename(yin_img_path)
        if not img_name.endswith('.png'):
            continue
        assert folder_name == os.path.basename(os.path.dirname(os.path.dirname(yang_img_path)))
        assert img_name == os.path.basename(yang_img_path)
        
        os.makedirs(os.path.join(output_dir, folder_name, 'color'), exist_ok=True)
        
        yin_img = pil_to_tensor(Image.open(yin_img_path).convert('RGB')).to('cuda').unsqueeze(0)
        yang_img = pil_to_tensor(Image.open(yang_img_path).convert('RGB')).to('cuda').unsqueeze(0)
        
        erp_img = convert_yinyang_original(yin_img, yang_img).cpu()
        
        save_image(erp_img, os.path.join(output_dir, folder_name, img_name))
    