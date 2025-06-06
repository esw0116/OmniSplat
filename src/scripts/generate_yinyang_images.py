import torch
from torch.nn import functional as F
from einops import rearrange

def erp_to_yin(image):
    """
    Convert erp to sphere
    image: (B, C, H, W), range [0, 1]
    """
    b, c, h, w = image.shape
    
    def theta_to_nheight(theta):
        # return (h-1) / torch.pi * theta - (h-1) / 2
        return -2 / torch.pi * theta
    
    def phi_to_nwidth(phi):
        # return (w-1) / (2*torch.pi) * phi + (w-1) / 2
        return 1 / torch.pi * phi
    
    yin_h, yin_w = h //2, 3*(h//2)
    yin_y = theta_to_nheight(torch.linspace(torch.pi / 4, -torch.pi / 4, yin_h))
    yin_x = phi_to_nwidth(torch.linspace(-3*torch.pi / 4, 3*torch.pi / 4, yin_w))
    yin_yx = torch.meshgrid(yin_y, yin_x, indexing='ij') # [h, 3h]
    yin_grid = torch.stack(yin_yx, dim=-1).unsqueeze(0).expand(b,-1,-1,-1)[..., [1,0]]

    yin_image = F.grid_sample(image, yin_grid, align_corners=True)    
    return yin_image


if __name__ == '__main__':
    from PIL import Image
    import argparse
    from torch.nn import functional as F
    import torchvision
    from torchvision.utils import save_image
    import os, glob, tqdm
    import equilib
    
    
    # dataset_root = 'dataset/OmniDatasets_1024x512/Ricoh360'
    # short_length = 256
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', '-d', type=str, required=True, help='input directory')
    parser.add_argument('--short_length', '-sl', type=int, default=256, help='short length')
    args = parser.parse_args()
    
    img_paths = sorted(glob.glob(os.path.join(args.dataset_root, '*', 'images', '*.png')) + glob.glob(os.path.join(args.dataset_root, '*', 'images', '*.jpg')))
    pbar = tqdm.tqdm(img_paths, ncols=100)
    
    for img_path in pbar:
        img_dir = os.path.dirname(os.path.dirname(img_path))
        img_name = os.path.basename(img_path)
        yin_dir = os.path.join(img_dir, f'yin{args.short_length}')
        yang_dir = os.path.join(img_dir, f'yang{args.short_length}')
        
        os.makedirs(yin_dir, exist_ok=True)
        os.makedirs(yang_dir, exist_ok=True)
        
        img = Image.open(img_path).convert('RGB')
        img = torchvision.transforms.ToTensor()(img).unsqueeze(0)
        h, w = img.shape[-2:]
        
        if args.short_length == h / 2:
            pass
        else:
            img = F.interpolate(img, size=(2*args.short_length, 4*args.short_length), mode='bicubic', align_corners=True)
            h, w = img.shape[-2:]
        
        rot_image = equilib.Equi2Equi(height=h, width=w, mode='bilinear')
        
        # new_image = rot_image(src=img, rots=[{'yaw':torch.pi, 'roll': 0, 'pitch': 0}])
        # new_image = rot_image(src=new_image, rots=[{'yaw': 0, 'roll':torch.pi/2, 'pitch': 0}])
        new_image = rot_image(src=img, rots=[{'roll':-torch.pi/2, 'yaw': 0, 'pitch': torch.pi}])
    
        yin_img = erp_to_yin(img)
        yang_img = erp_to_yin(new_image)
        yang_img = torch.rot90(yang_img, 3, [2,3])
        
        save_image(yin_img, os.path.join(yin_dir, img_name))
        save_image(yang_img, os.path.join(yang_dir, img_name))
