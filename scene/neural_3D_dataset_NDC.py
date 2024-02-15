import concurrent.futures
import gc
import glob
import os
import json
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm
from scipy.spatial.transform import Rotation as Rot
from utils.general_utils import PILtoTorch


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    # x = normalize(np.cross(z, y_))  # (3)
    x = normalize(np.cross(y_, z))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    #y = np.cross(x, z)  # (3)
    y = np.cross(z, x)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    poses = poses @ blender2opencv
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[
        :3
    ] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = np.concatenate(
        [poses, last_row], 1
    )  # (N_images, 4, 4) homogeneous coordinate
    pose_avg_homo_inv = np.linalg.inv(pose_avg_homo)
    poses_centered = pose_avg_homo_inv @ poses_homo  # (N_images, 4, 4)

    #     poses_centered = poses_centered  @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
   
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([vec0, vec1, vec2, pos], 1)
    return m



def viewmatrix2(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
   
    vec1 = normalize(np.cross(vec2, vec0))
    cw = np.stack([vec0, vec1, vec2])
    # r = Rot.from_euler('x', 180,  degrees=True)
    # cw = np.matmul(r.as_matrix(), cw)
    # pos = np.matmul(-pos, cw)


    m = np.eye(4)
    m[:3,:3] = cw 
    m[:3,3] = pos
    return m

  
def get_grid2(poses, near_fars):
  
    # center pose
    _, poses_avg = center_poses(poses, np.eye(4))

    # poses_avg[:3,1] *= -1
    # poses_avg[:3,2] *= -1
    # r = Rot.from_euler('yz', (-30, -5),  degrees=True)
    # poses_avg[:3,:3] = np.matmul(r.as_matrix(), poses_avg[:3,:3])

    GRID_SIZE_X = 15
    GRID_SIZE_Z = 15
    # min_x, max_x = -1.0, 0.5
    # min_z, max_z = -1.2, 0
    min_x, max_x = -1.0, 0.5
    min_z, max_z =  -2, -1.2

    # min_x, max_x = -5, 7
    # min_z, max_z = -10, 2
    render_poses = []

    step_x = (max_x - min_x)/GRID_SIZE_X
    step_z = (max_z - min_z)/GRID_SIZE_Z
    for j, z in enumerate(np.arange(max_z, min_z , -step_z)):
        for i, x in enumerate(np.arange(min_x, max_x, step_x)):
            poses_avg_clone = poses_avg.copy()

            cam_center = poses_avg_clone[:3, 3]
            cx = poses_avg_clone[:3, 0]*x
            cam_center = cam_center + cx
            cz = poses_avg_clone[:3, 2]*z
            cam_center = cam_center + cz
           
            poses_avg_clone[:3, 3] = cam_center
            
            
            render_poses.append(np.linalg.inv(poses_avg_clone))
   
   
    return np.stack(render_poses)

def get_grid(poses, near_fars):
  
   
    c2w = np.eye(4)
  
    
    r = Rot.from_euler('yz', (30, 5),  degrees=True)
    # #print(r.as_matrix())
    c2w_r = np.matmul(r.as_matrix(), c2w[:3,:3])
    up = c2w_r[:3,1]
    z = c2w_r[:3,2]

    
   
    render_poses = render_grid(
        z, up
    )
    return np.stack(render_poses)

def render_grid(z1, up):
    render_poses = []
    GRID_SIZE_X = 1
    GRID_SIZE_Z = 1
    # min_x, max_x = -2 , 2
    # min_z, max_z = 0, 2
   
    min_x, max_x = -5, 5
    min_z, max_z = 0, 10

    step_x = (max_x - min_x)/GRID_SIZE_X
    step_z = (max_z - min_z)/GRID_SIZE_Z
    for j, z in enumerate(np.arange(min_z, max_z , step_z)):
        for i, x in enumerate(np.arange(max_x, min_x, -step_x)):
            c = np.array([x, 0.2, z])
            
            # z1 = normalize(c - z_tmp)
            # z1 = np.array([0.0, 0.0, 1.0])
            # z1 = c2w[:3,2]

            render_poses.append(viewmatrix(z1, up, c))

     
    return render_poses


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.0])

    for theta in np.linspace(0.0, 2.0 * np.pi * N_rots, N + 1)[:-1]:
        c2w_tmp = c2w[:3, :4]
        tmp_array = np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0]) * rads,
        c = np.dot(
            c2w_tmp,
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0]) * rads,
           
        )
        z_tmp = np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0]))
        z = normalize(c - z_tmp)
        render_poses.append(viewmatrix(z, up, c))
    return render_poses



def process_video(video_data_save, video_path, img_wh, downsample, transform):
    """
    Load video_path data to video_data_save tensor.
    """
    video_frames = cv2.VideoCapture(video_path)
    count = 0
    video_images_path = video_path.split('.')[0]
    image_path = os.path.join(video_images_path,"images")

    if not os.path.exists(image_path):
        os.makedirs(image_path)
        while video_frames.isOpened():
            ret, video_frame = video_frames.read()
            if ret:
                video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
                video_frame = Image.fromarray(video_frame)
                if downsample != 1.0:
                    
                    img = video_frame.resize(img_wh, Image.LANCZOS)
                img.save(os.path.join(image_path,"%04d.png"%count))

                img = transform(img)
                video_data_save[count] = img.permute(1,2,0)
                count += 1
            else:
                break
      
    else:
        images_path = os.listdir(image_path)
        images_path.sort()
        
        for path in images_path:
            img = Image.open(os.path.join(image_path,path))
            if downsample != 1.0:  
                img = img.resize(img_wh, Image.LANCZOS)
                img = transform(img)
                video_data_save[count] = img.permute(1,2,0)
                count += 1
        
    video_frames.release()
    print(f"Video {video_path} processed.")
    return None


# define a function to process all videos
def process_videos(videos, skip_index, img_wh, downsample, transform, num_workers=1):
    """
    A multi-threaded function to load all videos fastly and memory-efficiently.
    To save memory, we pre-allocate a tensor to store all the images and spawn multi-threads to load the images into this tensor.
    """
    #all_imgs = torch.zeros(len(videos) - 1, 300, img_wh[-1] , img_wh[-2], 3)
    all_imgs = torch.zeros(len(videos) - 1, 201, img_wh[-1] , img_wh[-2], 3)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # start a thread for each video
        current_index = 0
        futures = []
        for index, video_path in enumerate(videos):
            # skip the video with skip_index (eval video)
            if index == skip_index:
                continue
            else:
                future = executor.submit(
                    process_video,
                    all_imgs[current_index],
                    video_path,
                    img_wh,
                    downsample,
                    transform,
                )
                futures.append(future)
                current_index += 1
    return all_imgs




def get_spiral(c2ws_all, near_fars, rads_scale=1.0, N_views=120):
    """
    Generate a set of poses using NeRF's spiral camera trajectory as validation poses.
    """
    # center pose
    c2w = average_poses(c2ws_all)

    # Get average pose
    c2ws_all_tmp = c2ws_all[:, :3, 1]
    up = normalize(c2ws_all_tmp.sum(0))

    # Find a reasonable "focus depth" for this dataset
    dt = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / ((1.0 - dt) / close_depth + dt / inf_depth)

    # Get radii for spiral path
    zdelta = near_fars.min() * 0.2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    render_poses = render_path_spiral(
        c2w, up, rads, focal, zdelta, zrate=0.5, N=N_views
    )
    return np.stack(render_poses)

def load_cam_poses(path):
    with open(path, 'r', encoding="utf-8") as infile:
        data_in = json.load(infile)
    
    return np.array(data_in["cam_poses"])

class Neural3D_NDC_Dataset(Dataset):
    def __init__(
        self,
        datadir,
        split="train",
        downsample=1.0,
        is_stack=True,
        cal_fine_bbox=False,
        N_vis=-1,
        time_scale=1.0,
        scene_bbox_min=[-1.0, -1.0, -1.0],
        scene_bbox_max=[1.0, 1.0, 1.0],
        N_random_pose=1000,
        bd_factor=0.75,
        eval_step=1,
        eval_index=0,
        sphere_scale=1.0,
        skip_grid_render=False,
        image_paths = None,
        images = None
    ):
        cam01_images_fold = os.path.join(datadir,"cam01","images")
        cam01_images = [file for file in os.listdir(cam01_images_fold) if file.endswith(".png") or file.endswith(".jpg") ]
                
        if image_paths is None:
            self.images_paths = []
        if images is None:
            self.images = []
        
        self.split = split

        if self.split == "train":
            for folder in os.listdir(datadir):
                if not folder == "cam00":
                    d = os.path.join(datadir,folder,"images")
                    if os.path.isdir(d):
                        for file in os.listdir(d):
                            if file.endswith(".png") or file.endswith(".jpg"):
                                self.images_paths.append(os.path.join(d,file))
        else:
            test_images = os.listdir(os.path.join(datadir,"cam00","images"))
            for image_path in test_images:
                if image_path.endswith(".png") or image_path.endswith(".jpg"):
                    self.images_paths.append(os.path.join(datadir,"cam00","images",image_path))
                    
        for image_path in self.images_paths:
            img = Image.open(image_path)
            im_data = np.array(img.convert("RGBA"))
            bg = np.array([1,1,1])
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            img = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            img = PILtoTorch(img,(img.size[0], img.size[1]))
            self.images.append(img)

        self.len_images = len(cam01_images)
        img = Image.open(os.path.join(cam01_images_fold,cam01_images[0]))


        self.img_wh = (
           img.size[0], # int(1352 / downsample),
           img.size[1], # int(1014 / downsample),
        ) 
      
        self.root_dir = datadir
        self.split = split
        self.downsample = downsample 
        self.is_stack = is_stack
        self.N_vis = N_vis
        self.time_scale = time_scale
        self.scene_bbox = torch.tensor([scene_bbox_min, scene_bbox_max])
        self.skip_grid_render = skip_grid_render

        self.world_bound_scale = 1.1
        self.bd_factor = bd_factor
        self.eval_step = eval_step
        self.eval_index = eval_index
        self.blender2opencv = np.eye(4)
        self.transform = T.ToTensor()

        self.near = 0.0
        self.far = 1.0
        self.near_far = [self.near, self.far]  # NDC near far is [0, 1.0]
        self.white_bg = False
        self.ndc_ray = True
        self.depth_data = False

        self.load_meta()
        print(f"meta data loaded, total image:{len(self)}")

    def load_meta(self):
        """
        Load meta data from the dataset.
        """
        # Read poses and video file paths.
        poses_arr = np.load(os.path.join(self.root_dir, "poses_bounds.npy"))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
        self.near_fars = poses_arr[:, -2:]
        videos = glob.glob(os.path.join(self.root_dir, "cam*"))
        videos = sorted(videos)
        #assert len(videos) == poses_arr.shape[0]

        H, W, focal = poses[0, :, -1]
        focal = focal / self.downsample
        self.focal = [focal, focal]
        # self.focal = [self.img_wh[0]/2, self.img_wh[1]]
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # poses = np.concatenate([poses[..., :1], poses[..., 1:2], poses[..., 2:4]], -1)

        # poses, _ = center_poses(
        #     poses, self.blender2opencv
        # )  # Re-center poses so that the average is near the center.

        # near_original = self.near_fars.min()
        # scale_factor = near_original * 0.75
        # self.near_fars /= (
        #     scale_factor  # rescale nearest plane so that it is at z = 4/3.
        # )
        # poses[..., 3] /= scale_factor

        # Sample N_views poses for validation - NeRF-like camera trajectory.
        N_views = 120
        if self.skip_grid_render:
            self.val_poses = get_spiral(poses, self.near_fars, N_views=N_views)
        else:
            self.val_poses= get_grid2(poses, self.near_fars)
      
        W, H = self.img_wh
        poses_i_train = []

        for i in range(len(poses)):
            if i != self.eval_index:
                poses_i_train.append(i)
        self.poses = poses[poses_i_train]
        self.poses_all = poses
        self.image_paths, self.image_poses, self.image_times, N_cam, N_time = self.load_images_path(videos, self.split)
        self.cam_number = N_cam
        self.time_number = N_time
    def get_val_pose(self):
        render_poses = self.val_poses
        render_times = torch.linspace(0.0, 1.0, render_poses.shape[0]) * 2.0 - 1.0
        return render_poses, self.time_scale * render_times
    def load_images_path(self,videos,split):
        image_paths = []
        image_poses = []
        image_times = []
        N_cams = 0
        N_time = 0
        countss = self.len_images
        for index, video_path in enumerate(videos):
            
            if index == self.eval_index:
                if split =="train":
                    continue
            else:
                if split == "test":
                    continue
            N_cams +=1
            count = 0
            video_images_path = video_path.split('.')[0]
            image_path = os.path.join(video_images_path,"images")
            video_frames = cv2.VideoCapture(video_path)
            if not os.path.exists(image_path):
                print(f"no images saved in {image_path}, extract images from video.")
                os.makedirs(image_path)
                this_count = 0
                while video_frames.isOpened():
                    ret, video_frame = video_frames.read()
                    if this_count >= countss:break
                    if ret:
                        video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
                        video_frame = Image.fromarray(video_frame)
                        if self.downsample != 1.0:

                            img = video_frame.resize(self.img_wh, Image.LANCZOS)
                        img.save(os.path.join(image_path,"%04d.png"%count))

                        # img = transform(img)
                        count += 1
                        this_count+=1
                    else:
                        break
                    
            images_path = os.listdir(image_path)
            images_path.sort()
            if index == 0: 
                self.num_frames = len(images_path)
            this_count = 0
            for idx, path in enumerate(images_path):
                if this_count >=countss:break
                image_paths.append(os.path.join(image_path,path))
                pose = np.array(self.poses_all[index])
                R = pose[:3,:3]
                T = pose[:3,3]
                R = -R
                R[:,0] = -R[:,0]
                T = -pose[:3,3].dot(R)
                image_times.append(idx/countss)
                image_poses.append((R,T))
                # if self.downsample != 1.0:
                #     img = video_frame.resize(self.img_wh, Image.LANCZOS)
                # img.save(os.path.join(image_path,"%04d.png"%count))
                this_count+=1  
            N_time = len(images_path)

                #     video_data_save[count] = img.permute(1,2,0)
                #     count += 1
        return image_paths, image_poses, image_times, N_cams, N_time
    def __len__(self):
        return len(self.image_paths)
    # def __getitem__(self,index):
    #     img = Image.open(self.image_paths[index])
    #     # img = img.resize(self.img_wh, Image.LANCZOS)
       
    #     im_data = np.array(img.convert("RGBA"))

    #     bg = np.array([1,1,1]) 

    #     norm_data = im_data / 255.0
    #     arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
    #     img = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
    
    #     img = PILtoTorch(img,(img.size[0], img.size[1]))

    #     # img = self.transform(img)
    #     ##rint("Returning image from getitem")
    #     return img, self.image_poses[index], self.image_times[index]

    def __getitem__(self,index):
        return self.images[index], self.image_poses[index], self.image_times[index]
    
    def load_pose(self,index):
        return self.image_poses[index]

