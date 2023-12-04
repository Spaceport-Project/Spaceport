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
from pytorch3d.renderer.cameras import FoVPerspectiveCameras
import numpy as np
from utils.graphics_utils import getWorld2View2, getWorld2View3, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", time = 0
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx*1.8/FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.time = time
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        self.original_image = image.clamp(0.0, 1.0)
        # .to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask
            # .to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width))
                                                #   , device=self.data_device)

            
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale





        R = torch.Tensor(R).unsqueeze(0)
        T = torch.Tensor(T).unsqueeze(0)
        
        R = -R
       
        persp_cam = FoVPerspectiveCameras(device="cuda", R = R, T = T, zfar = self.zfar, znear = self.znear, fov = self.FoVy, degrees=False, aspect_ratio=self.FoVx/self.FoVy)
        # self.world_view_transform = persp_cam.get_full_projection_transform()._transforms[0]._matrix.squeeze(0)
        self.world_view_transform = persp_cam.get_world_to_view_transform().get_matrix()
        # self.world_view_transform[:,1]*= -1
        # self.world_view_transform[:,0]*= -1
        # print(persp_cam.get_full_projection_transform().get_matrix())
        self.full_proj_transform = persp_cam.get_full_projection_transform().get_matrix()
        # persp_proj_matrix = persp_cam.get_full_projection_transform()._transforms[1]._matrix.squeeze(0)
        # self.full_proj_transform = persp_proj_matrix.matmul(self.world_view_transform)
        # self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.camera_center = persp_cam.get_camera_center()
        

        # self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        # world_view_trans = getWorld2View3(R, -T)
        # world_view_trans[:,2] *= -1
        # self.world_view_transform =  torch.tensor(world_view_trans).transpose(0,1)
        # # self.world_view_transform = torch.tensor(getWorld2View3(R, T)).transpose(0,1)
        
        # # self.world_view_transform[:,1] *= -1
        # # self.world_view_transform[:,2] *= -1
        # # .cuda()
        # self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
        # # .cuda()
        # # world_view_trans_unsq = self.world_view_transform.unsqueeze(0)
        # # proj_matrix_unsq = self.projection_matrix.unsqueeze(0)
        # # self.full_proj_transform = (world_view_trans_unsq.bmm(proj_matrix_unsq)).squeeze(0)
        # # self.full_proj_transform = (proj_matrix_unsq.bmm(world_view_trans_unsq)).squeeze(0)
        # self.full_proj_transform =  self.projection_matrix.matmul(self.world_view_transform)
        # # self.full_proj_transform[0,2] = 10*self.full_proj_transform[0,2]
        # # self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # self.camera_center = self.world_view_transform.inverse()[3, :3]




        pass

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

