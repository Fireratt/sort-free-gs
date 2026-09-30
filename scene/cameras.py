#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import load_image_tensor_from_path

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, resolution_scale, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", resolution=None,
                 image_path=None, white_background=False, lazy_load_image=False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.resolution_scale = resolution_scale
        self.gt_alpha_mask = gt_alpha_mask
        self.image_path = image_path
        self.white_background = white_background
        self.lazy_load_image = lazy_load_image

        if data_device != 'disk':
            try:
                self.data_device = torch.device(data_device)
            except Exception as e:
                print(e)
                print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
                self.data_device = torch.device("cuda")
        else:
            self.data_device = "disk"
            
        self.image_width = resolution[0]
        self.image_height = resolution[1]
        
        if not self.lazy_load_image:
            assert image is not None, "Eager image loading expects a preloaded tensor."
            self.gt_image = image[:3, ...].clamp(0.0, 1.0).contiguous()
            if self.data_device != "disk":
                self.gt_image = self.gt_image.to(self.data_device)
        else:
            self.gt_image = None
            
            
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    @property
    def original_image(self):
        if self.gt_image is not None:
            return self.gt_image

        image = load_image_tensor_from_path(
            self.image_path,
            (self.image_width, self.image_height),
            self.white_background,
        )[:3, ...]
        if self.data_device != "disk":
            image = image.to(self.data_device)
        return image

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
