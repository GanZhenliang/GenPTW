import os
import torch
import numpy as np
from torch import nn
import random
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional
from augly.image import functional as augly_functional
from torchvision import transforms


class TransformNet(nn.Module):
    """"
    This class is adapted from RoSteALS paper
    """
    def __init__(self, rnd_bri=0.3, rnd_hue=0.1, rnd_noise=0.03, rnd_sat=1.0, rnd_trans=0.1,contrast=[0.5, 2.0], ramp=1000,
                 apply_many_crops = False, apply_required_attacks = False, required_attack_list = ['resize','random_crop']) -> None:
        super().__init__()
        self.rnd_bri = rnd_bri
        self.rnd_hue = rnd_hue
        self.rnd_noise = rnd_noise
        self.rnd_sat = rnd_sat
        self.rnd_trans = rnd_trans
        self.contrast_low, self.contrast_high = contrast
        self.ramp = ramp
        self.apply_many_crops = apply_many_crops
        self.apply_required_attacks = apply_required_attacks
        self.register_buffer('step0', torch.tensor(0))  # large number
        self.required_attack_list = required_attack_list
        # self.imagenet_mean = nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1), requires_grad=False)
        # self.imagenet_std = nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1), requires_grad = False)
        self.imagenet_mean = nn.Parameter(torch.Tensor([0.5, 0.5, 0.5]).view(-1, 1, 1), requires_grad=False)
        self.imagenet_std = nn.Parameter(torch.Tensor([0.5, 0.5, 0.5]).view(-1, 1, 1), requires_grad=False)

    def imgnet_unnormalize(self,x):
        return (x * self.imagenet_std) + self.imagenet_mean
    
    def imgnet_normalize(self,x):
        return (x - self.imagenet_mean) / self.imagenet_std
    
    def jpeg_compress(self, x, quality_factor):
        """ jpeg code from Stable Signature"""
        """ Apply jpeg compression to image
        Args:
            x: normalized Tensor image between [-1, 1]
            quality_factor: quality factor
        """
        with torch.no_grad():
            to_pil = transforms.ToPILImage()
            to_tensor = transforms.ToTensor()
            img_jpeg = torch.zeros_like(x, device=x.device)
            ## clampt the values:
            x_clip = torch.round(255 * self.imgnet_unnormalize(x)).clamp(0, 255) / 255.0
            for ii,img in enumerate(x_clip):
                pil_img = to_pil(img)
                img_jpeg[ii] = to_tensor(augly_functional.encoding_quality(pil_img, quality=quality_factor))
            img_gap = self.imgnet_normalize(img_jpeg) - x
            img_gap = img_gap.detach()
        img_jpeg_compressed_differentiable = x + img_gap   
        return img_jpeg_compressed_differentiable
    
 
    def forward(self, x, global_step, active_step = 0, p=0.9,dtype=torch.float32):
        
        # if torch.rand(1)[0] >= p:
        #     return x

        batch_size, sh, device = x.shape[0], x.size(), x.device
        if active_step == 0:
            ramp_fn = lambda ramp: np.min([(global_step-self.step0.cpu().item()) / ramp, 1.])
        else:
            ramp_fn = lambda ramp: np.min([(global_step-active_step) / ramp, 1.])
            
        rnd_bri = ramp_fn(self.ramp) * self.rnd_bri
        rnd_hue = ramp_fn(self.ramp) * self.rnd_hue
        rnd_brightness = get_rnd_brightness_torch(rnd_bri, rnd_hue, batch_size).to(device)  # [batch_size, 3, 1, 1]
        rnd_noise = torch.rand(1)[0] * ramp_fn(self.ramp) * self.rnd_noise

        contrast_low = 1. - (1. - self.contrast_low) * ramp_fn(self.ramp)
        contrast_high = 1. + (self.contrast_high - 1.) * ramp_fn(self.ramp)

        contrast_params = [contrast_low, contrast_high]

        rnd_sat = torch.rand(1)[0] * ramp_fn(self.ramp) * self.rnd_sat
        
        if self.apply_many_crops:
            selected_attack = 'crop'
            print(selected_attack + ' from crop list')
        
        elif self.apply_required_attacks:
            selected_attack = random.choice(self.required_attack_list)
            # print(selected_attack + ' from required attack list')
        
        else:
            selected_attack = random.choice(['blur', 'noise','contrast','brightness','saturation','jpeg', 'rotation','rotation', 'crop', 'crop', 'resize', 'resize'])
            print(selected_attack + ' from all attack list')
        
        if selected_attack == 'blur':
            # blur
            x = (x + 1.) / 2.
            N_blur = 7
            # N_blur = int(np.random.choice(np.arange(3, 12, 2)))  # 选奇数
            f = random_blur_kernel(probs=[.25, .25], N_blur=N_blur, sigrange_gauss=[1., 3.], sigrange_line=[.25, 1.],
                                        wmin_line=3).to(device)
            x = F.conv2d(x, f, bias=None, padding=int((N_blur - 1) / 2))
            x = torch.clamp(x, 0, 1)
            x = (x * 2.) - 1.

        elif selected_attack == 'noise':
            # noise
            x = (x + 1.) / 2.
            noise = torch.normal(mean=0, std=rnd_noise, size=x.size(), dtype=dtype).to(device)
            x = x + noise
            x = torch.clamp(x, 0, 1)
            x = (x * 2.) - 1.

        elif selected_attack == 'contrast':
            # contrast & brightness
            x = (x + 1.) / 2.
            contrast_scale = torch.Tensor(x.size()[0]).uniform_(contrast_params[0], contrast_params[1])
            contrast_scale = contrast_scale.reshape(x.size()[0], 1, 1, 1).to(device)
            x = x * contrast_scale
            x = torch.clamp(x, 0, 1)
            
            x = (x * 2.) - 1.

        elif selected_attack == 'brightness':
            
            x = (x + 1.) / 2.            
            x = x + rnd_brightness
            x = torch.clamp(x, 0, 1) 
            x = (x * 2.) - 1.
        
        elif selected_attack == 'saturation':
            # saturation
            x = (x + 1.) / 2.    
            sat_weight = torch.FloatTensor([.3, .6, .1]).reshape(1, 3, 1, 1).to(device)
            encoded_image_lum = torch.mean(x * sat_weight, dim=1).unsqueeze_(1)
            x = (1 - rnd_sat) * x + rnd_sat * encoded_image_lum
            x = torch.clamp(x, 0, 1)  
            x = (x * 2.) - 1.

        elif selected_attack == 'rotation':
            # rotation
            rotation_angle_range=(2, 30)
            angle = int(np.random.uniform(*rotation_angle_range))
            x= functional.rotate(x, angle= angle, interpolation=functional.InterpolationMode('bilinear'))
        
        elif selected_attack == 'jpeg':
            # augly implementation:
            x = (x + 1.) / 2.
            x = self.imgnet_normalize(x)
            # jpeg_quality = int(np.random.uniform(40. , 100.))
            jpeg_quality = int(np.random.uniform(60. , 100.))
            x = self.jpeg_compress(x, jpeg_quality)
            x = self.imgnet_unnormalize(x)
            x = (x * 2.) - 1.
        
        elif selected_attack == 'center_crop':
            ### center crop
            # # crop using area:
            x = (x + 1.) / 2.
            crop_scale_range=(0.08, 0.94)
            crop_scale = int(np.sqrt(np.random.uniform(*crop_scale_range)) * x.size()[2])
            x = functional.center_crop(x, crop_scale)
            x = (x * 2.) - 1.

        elif selected_attack == 'random_crop':
            ### random crop
            # # crop using area
            x = (x + 1.) / 2.
            crop_scale_range=(0.08, 0.94)
            crop_size = int(np.sqrt(np.random.uniform(*crop_scale_range)) * x.size()[2])
            crop_transform = transforms.RandomCrop(crop_size)
            x = crop_transform(x)
            x = (x * 2.) - 1.
    
        elif selected_attack == 'resize':

            ## resize 2:  
            old_h = x.size()[3]
            old_w = x.size()[2]
            x = (x + 1.) / 2. 
            resize_scale_range = (0.5, 1.5)
            new_w = int(np.random.uniform(*resize_scale_range) * x.size()[3])
            new_h = int(np.random.uniform(*resize_scale_range) * x.size()[2])
            x = functional.resize(x, (new_h, new_w), interpolation=functional.InterpolationMode('bilinear'))
            x = functional.resize(x, (old_h, old_w), interpolation=functional.InterpolationMode('bilinear'))
            x = (x * 2.) - 1.
        return x
    
def random_blur_kernel(probs, N_blur, sigrange_gauss, sigrange_line, wmin_line):
    N = N_blur
    coords = torch.from_numpy(np.stack(np.meshgrid(range(N_blur), range(N_blur), indexing='ij'), axis=-1)) - (0.5 * (N-1)) # （7,7,2)
    manhat = torch.sum(torch.abs(coords), dim=-1)   # (7, 7)

    # nothing, default
    vals_nothing = (manhat < 0.5).float()           # (7, 7)

    # gauss
    sig_gauss = torch.rand(1)[0] * (sigrange_gauss[1] - sigrange_gauss[0]) + sigrange_gauss[0]
    vals_gauss = torch.exp(-torch.sum(coords ** 2, dim=-1) /2. / sig_gauss ** 2)

    # line
    theta = torch.rand(1)[0] * 2.* np.pi
    v = torch.FloatTensor([torch.cos(theta), torch.sin(theta)]) # (2)
    dists = torch.sum(coords * v, dim=-1)                       # (7, 7)

    sig_line = torch.rand(1)[0] * (sigrange_line[1] - sigrange_line[0]) + sigrange_line[0]
    w_line = torch.rand(1)[0] * (0.5 * (N-1) + 0.1 - wmin_line) + wmin_line

    vals_line = torch.exp(-dists ** 2 / 2. / sig_line ** 2) * (manhat < w_line) # (7, 7)

    t = torch.rand(1)[0]
    vals = vals_nothing
    if t < (probs[0] + probs[1]):
        vals = vals_line
    else:
        vals = vals
    if t < probs[0]:
        vals = vals_gauss
    else:
        vals = vals

    v = vals / torch.sum(vals)   
    z = torch.zeros_like(v)     
    f = torch.stack([v,z,z, z,v,z, z,z,v], dim=0).reshape([3, 3, N, N])
    return f

def get_rnd_brightness_torch(rnd_bri, rnd_hue, batch_size):
    rnd_hue = torch.FloatTensor(batch_size, 3, 1, 1).uniform_(-rnd_hue, rnd_hue)
    rnd_brightness = torch.FloatTensor(batch_size, 1, 1, 1).uniform_(-rnd_bri, rnd_bri)
    return rnd_hue + rnd_brightness