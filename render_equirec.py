
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import gc
import imageio
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
import threading
import concurrent.futures
def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def render_set_all(model_path, name, iteration, views, gaussians, pipeline, background, num_frames, render_img_size):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
 
    cnt=0
    for id, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_images = []
        gt_list = []
        gt_list2 = []
        render_list = []
        for idx in tqdm(range(num_frames)):
            if idx == 0:
                time1 = time()
            # if idx > 0:
            #     break
            view.time = idx/num_frames
            # print(f"views[{idx}]: {view.image_name}, {view.R}")
            rendering = render(view, gaussians, pipeline, background, render_img_size=render_img_size, equirec_flag=True)["render"]
            # transform = torchvision.transforms.CenterCrop((int(render_img_size[1]*3/4), int(render_img_size[0])))
            # cropped_rendering = transform(rendering)

            # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0}_{1:02d}'.format(id, idx) + ".png"))
            render_images.append(to8b(rendering).transpose(1,2,0))
            render_list.append(rendering)
            if name in ["train", "test"]:
                gt_list2.append(to8b(view.original_image).transpose(1,2,0))
                gt = view.original_image[0:3, :, :]
                # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            
                gt_list.append(gt)
        time2=time()
        print("FPS:",(len(views)-1)/(time2-time1))
        count = 0
        print("writing training images.")
        if len(gt_list) != 0:
            for image in tqdm(gt_list):
                torchvision.utils.save_image(image, os.path.join(gts_path, '{0:05d}'.format(count) + ".png"))
                count+=1
        count = 0
       
        
        imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), f'video_rgb_{cnt}.mp4'), render_images, fps=30, quality=8)
        cnt+=1
     
        torch.cuda.empty_cache()


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_images = []
    gt_list = []
    gt_list2 = []
    render_list = []
   
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:time1 = time()
        # print(f"views[{idx}]: {view.image_name}, {view.R}")
        rendering = render(view, gaussians, pipeline, background)["render"]
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        render_images.append(to8b(rendering).transpose(1,2,0))
        # print(to8b(rendering).shape)
        render_list.append(rendering)
        if name in ["train", "test"]:
            gt_list2.append(to8b(view.original_image).transpose(1,2,0))
            gt = view.original_image[0:3, :, :]
            # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
           
            gt_list.append(gt)
    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))
    # count = 0
    # print("writing training images.")
    # if len(gt_list) != 0:
    #     for image in tqdm(gt_list):
    #         torchvision.utils.save_image(image, os.path.join(gts_path, '{0:05d}'.format(count) + ".png"))
    #         count+=1
    # count = 0
    # print("writing rendering images.")
    # if len(render_list) != 0:
    #     for image in tqdm(render_list):
    #         torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(count) + ".png"))
    #         count +=1
    # print("writing training images.")

    multithread_write(gt_list, gts_path)
    # print("writing rendering images.")

    multithread_write(render_list, render_path)
    
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4'), render_images, fps=30, quality=8)
    # imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb_gt.mp4'), gt_list2, fps=30, quality=8)

def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool, skip_grid_render:bool, render_img_size, test_or_train):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, skip_grid_render=skip_grid_render, render_img_size=render_img_size, test_or_train =test_or_train)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
        if not skip_video:
            # render_set(dataset.model_path,"video",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background)
            if not skip_grid_render:
                render_set_all(dataset.model_path,"video",scene.loaded_iter, scene.getVideoCameras(),gaussians, pipeline, background, num_frames= scene.maxtime, render_img_size=render_img_size)
            else:
                render_set(dataset.model_path,"video",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")

    model = ModelParams(parser, sentinel=True)
   
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str, default="arguments/dynerf/default.py")
    parser.add_argument("--skip_grid_render", action="store_true")
    parser.add_argument("--override_render_img_size", nargs='+', default = (None, None), type=int)

    args = get_combined_args(parser)
 
    if args.configs:
        print("Loading configs from ", args.configs)
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    if args.override_render_img_size:
        render_img_size = tuple(args.override_render_img_size)
    else :
        render_img_size = None
        
    if args.skip_train and args.skip_test:
        test_or_train = "test"
    else:
        test_or_train = "train"

    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video, args.skip_grid_render, render_img_size, test_or_train)