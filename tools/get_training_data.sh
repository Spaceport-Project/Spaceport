#!/bin/bash

# Usage: ./data_preparation.sh <conda_env_nerf> <conda_env_llff> <input_dir> <output_dir> <llff_path>

# Check if the correct number of arguments are provided
if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <conda_env_nerf> <conda_env_llff> <input_dir> <output_dir> <llff_path> <image_extension>" 
    exit 1
fi

eval "$(conda shell.bash hook)"

# Activate the 'nerfspaceport' conda environment
conda init bash
conda activate "$1"

# Set input and output directories
input_dir="$3"
output_dir="$4"
llff_path="$5"
image_extension="$6"

# Store the current working directory
previous_dir=$(pwd)

# Run organize_image_folders.py
python organize_image_folders.py "$input_dir" "$output_dir" "$image_extension"
echo "Enter starting frame index (leave blank for no limit): "
read start_frame
echo "Enter ending frame index (leave blank for no limit): "
read end_frame

# Run Colmap to get transforms.json file
colmap_input_dir="$output_dir/$start_frame-$end_frame/colmap_input"
colmap_output_dir="$output_dir/$start_frame-$end_frame/colmap_output"

# Replace 'your_matching_method' with your desired method
matching_method="exhaustive"
ns-process-data images --data "$colmap_input_dir" --matching_method "$matching_method" --output_dir "$colmap_output_dir"

# Undistort images
undistorted_input_dir="$colmap_output_dir"
undistorted_output_dir="$output_dir/$start_frame-$end_frame/undistorted_LLFF_input"
python undistort_images.py "$undistorted_input_dir" "$undistorted_output_dir" --img_ext "$image_extension"

nested_undistorted_input_dir="$output_dir/$start_frame-$end_frame/original_images"
nested_undistorted_output_dir="$output_dir/$start_frame-$end_frame/undistorted_images"
json_file="$colmap_output_dir/transforms.json"
python undistort_images_nested.py "$nested_undistorted_input_dir" "$nested_undistorted_output_dir" "$json_file" --img_ext "$image_extension"

# Deactivate the 'nerfspaceport' conda environment
conda deactivate

# Activate the 'LLFF' conda environment
conda activate "$2"

# Run imgs2poses.py
cd "$llff_path"
imgs2poses_input_dir="$undistorted_output_dir"
python imgs2poses.py "$imgs2poses_input_dir"

# Rename poses_bounds.npy
#mv "$undistorted_output_dir/poses_bounds.npy" "$nested_undistorted_output_dir/poses_bounds_spaceport.npy"
cp "$undistorted_output_dir/poses_bounds.npy" "$nested_undistorted_output_dir/poses_bounds_spaceport.npy"

# Return to the previous working directory
cd "$previous_dir"

# Copy the images folder to colmap_pcd directory
colmap_pcd_dir="$output_dir/$start_frame-$end_frame/colmap_pcd"
mkdir "$colmap_pcd_dir"
cp -r "$imgs2poses_input_dir/images" "$colmap_pcd_dir/images"

# Run Colmap automatic_reconstructor
colmap automatic_reconstructor --workspace_path "$colmap_pcd_dir" --image_path "$colmap_pcd_dir/images"

echo "Saved ply file to: $colmap_pcd_dir/dense/0/fused.ply"
# Deactivate the 'LLFF' conda environment
conda deactivate

conda activate "$1"
# Python script to downsample point cloud
python downsample_point.py "$colmap_pcd_dir/dense/0/fused.ply" "$colmap_pcd_dir/dense/0/fused_downsample.ply"

conda deactivate

# Return the path to the undistorted training data
echo "Undistorted training data is located at: $undistorted_output_dir"
