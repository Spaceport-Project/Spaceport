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
import math
# from pytorch3d.renderer.cameras import FoVPerspectiveCameras
import numpy as np
from utils.graphics_utils import getWorld2View, getWorld2View3, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", time = 0
                 , render_img_size = None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx #FoVx*1.6/FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.time = time
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        
        
       
        if render_img_size:
            self.image_width = render_img_size[0]
            self.image_height = render_img_size[1]
        else:
            self.original_image = image.clamp(0.0, 1.0)
            # .to(self.data_device)
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]


            if gt_alpha_mask is not None:
            
                self.original_image *= gt_alpha_mask
                # .to(self.data_device)
            else:
                self.original_image *= torch.ones((1, self.image_height, self.image_width)).to(device='cuda')
                                                    #   , device=self.data_device)

            
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale



        # R = -R
        # R[:,0] =  -R[:,0]
        # R[:,1] =  -R[:,1]
        # T = -T.dot(R)


        R = torch.Tensor(R).unsqueeze(0)
        T = torch.Tensor(T).unsqueeze(0)

        # R = torch.Tensor(R).unsqueeze(0)
        # T = torch.Tensor(T).unsqueeze(0)
        
        # R = -R
        # R[:,0] =  -R[:,0]
        # R[:,1] =  -R[:,1]
       
        # persp_cam = FoVPerspectiveCameras(device="cuda", R = R, T = T, zfar = self.zfar, znear = self.znear, fov = self.FoVy, degrees=False, aspect_ratio=self.FoVx/self.FoVy)
        # self.world_view_transform = persp_cam.get_world_to_view_transform().get_matrix()
     
        # self.full_proj_transform = persp_cam.get_full_projection_transform().get_matrix()
      
        # self.camera_center = persp_cam.get_camera_center()

        self.world_view_transform = getWorld2View3(R, T, translate=torch.Tensor([0,0,-1])).transpose(0,1)
        self.projection_matrix = projection_matrix(self.znear, self.zfar, self.FoVx, self.FoVy, device="cpu").transpose(0,1)
        self.full_proj_transform = self.world_view_transform.mm(self.projection_matrix)

        self.camera_center = self.world_view_transform.inverse()[3, :3]
        pass
        

        
def projection_matrix(znear, zfar, fovx, fovy, device = "cpu"):
    """
    Constructs an OpenGL-style perspective projection matrix.
    """
    t = znear * math.tan(0.5 * fovy)
    b = -t
    r = znear * math.tan(0.5 * fovx)
    l = -r
    n = znear
    f = zfar
    return torch.tensor(
        [
            [2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, (f + n) / (f - n), -1.0 * f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device=device,
    )

# class Camera(nn.Module):
#     def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
#                  image_name, uid,
#                  trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", time = 0
#                  ):
#         super(Camera, self).__init__()

#         self.uid = uid
#         self.colmap_id = colmap_id
#         self.R = R
#         self.T = T
#         self.FoVx = FoVx
#         self.FoVy = FoVy
#         self.image_name = image_name
#         self.time = time
#         try:
#             self.data_device = torch.device(data_device)
#         except Exception as e:
#             print(e)
#             print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
#             self.data_device = torch.device("cuda")
#         self.original_image = image.clamp(0.0, 1.0)
#         # .to(self.data_device)
#         self.image_width = self.original_image.shape[2]
#         self.image_height = self.original_image.shape[1]

#         if gt_alpha_mask is not None:
#             self.original_image *= gt_alpha_mask
#             # .to(self.data_device)
#         else:
#             self.original_image *= torch.ones((1, self.image_height, self.image_width))
#                                                 #   , device=self.data_device)

            
#         self.zfar = 100.0
#         self.znear = 0.01

#         self.trans = trans
#         self.scale = scale

#         self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
#         # .cuda()
#         self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
#         # .cuda()
#         self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
#         self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, time):
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
        self.time = time

