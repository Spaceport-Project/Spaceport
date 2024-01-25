
# Masking 4DGS Data for Preprocess/Postprocess

This readme file explains the necessary neural network setups and how to run masking script for different use cases.

## Necessary Neural Network Setups

In order to run masking on images/videos, we are employing two different neural networks. The first model is for detecting the bounding boxes of 
object of interest, the second one for pixel-wise segmentation of object of interest. For the current use case the object of interest is persons
in the scene. 

For object detection YOLO-X model is employed. You can clone the the repository from [here](https://github.com/Megvii-BaseDetection/YOLOX) 
and the pretrained model weights can be downloaded from [here](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth).
There are different versions of the arcticture, the suggested architecture is YOLOX_x which is the best performing model among alternatives.

Also for the object segmentation part SegmentAnyting model is employed. This repository can be cloned from [here](https://github.com/facebookresearch/segment-anything), also pretrained model weights can be downloaded from [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth). The suggested model architecture is 
sam_vit_h for this use case.

Once the both repositories cloned on the same level hierarchy in the file system, 'generate_masked_data.py' should also be placed on the same level with cloned repositories in order to avoid any import errors. For the environment setup nerfstudio environment should be capable of running this task since it is python>=3.7 
and torch>=1.9

## Generating Masked Training Data (Preprocess)

```bash
python generate_masked_training_data.py image -n yolox-x -c /home/hamit/Softwares/repos/cloned_repos/YOLOX/model_weights/yolox_x.pth --sam_checkpoint /home/hamit/Softwares/repos/cloned_repos/segment-anything/model_weights/sam_vit_h_4b8939.pth --sam_model_type vit_h --path /media/hamit/Elements1/03-01-2024_DATA/processed_datas/data_lum90_green_png_30fps/new/31-45/undistorted_images --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu
```

## Generating Masked Grid Videos (Postprocess)

```bash
 python generate_masked_training_data.py video -n yolox-x -c /home/hamit/Softwares/repos/cloned_repos/YOLOX/model_weights/yolox_x.pth --sam_checkpoint /home/hamit/Softwares/repos/cloned_repos/segment-anything/model_weights/sam_vit_h_4b8939.pth --sam_model_type vit_h --path /media/hamit/Elements/video_grid_data/video_grid_1-17/ --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu 
```
