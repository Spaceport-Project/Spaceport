import cv2
import os
import json
import numpy as np
import argparse
import multiprocessing
from pathlib import Path

def load_camera_params(json_file_path):
    """Load camera parameters from a JSON file."""
    with open(json_file_path, 'r') as f:
        camera_params = json.load(f)
    camera_matrix = np.array([[camera_params['fl_x'], 0, camera_params['cx']],
                              [0, camera_params['fl_y'], camera_params['cy']],
                              [0, 0, 1]])
    distortion_coefficients = np.array([camera_params['k1'], camera_params['k2'],
                                        camera_params['p1'], camera_params['p2']])
    return camera_matrix, distortion_coefficients

def undistort_images(image_directory, output_directory, camera_matrix, distortion_coefficients, img_ext):
    """Undistort images in the specified directory."""
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    print("Saving undistorted images to " + output_directory)
    for file in os.listdir(image_directory):
        if not file.endswith(img_ext):
            continue
        image_path = os.path.join(image_directory, file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Unable to read image {image_path}. Skipping.")
            continue
        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (w, h), 1, (w, h))
        undistorted_image = cv2.undistort(image, camera_matrix, distortion_coefficients, None, new_camera_matrix)
        x, y, w, h = roi
        undistorted_image = undistorted_image[y:y+h, x:x+w]
        basename = os.path.basename(image_path)
        cv2.imwrite(os.path.join(output_directory, basename), undistorted_image)

def worker_task(args):
    image_directory, output_directory, camera_matrix, distortion_coefficients, img_ext = args
    undistort_images(image_directory, output_directory, camera_matrix, distortion_coefficients, img_ext)

def main():
    parser = argparse.ArgumentParser(description="Undistort images using camera parameters")
    parser.add_argument("base_input_directory", help="Base directory containing images to undistort")
    parser.add_argument("base_output_directory", help="Base directory to save undistorted images")
    parser.add_argument("json_file_path", help="Path to the JSON file containing camera parameters")
    parser.add_argument("--img_ext", default=".png", help="Image file extension to process (default: .png)")
    args = parser.parse_args()

    # Load camera parameters
    camera_matrix, distortion_coefficients = load_camera_params(args.json_file_path)

    # Prepare directory pairs for multiprocessing
    directory_pairs = []
    for cam_dir in os.listdir(args.base_input_directory):
        if os.path.isdir(os.path.join(args.base_input_directory, cam_dir)):
            image_directory = os.path.join(args.base_input_directory, cam_dir, 'images')
            relative_path = os.path.relpath(image_directory, args.base_input_directory)
            output_directory = os.path.join(args.base_output_directory, relative_path)
            directory_pairs.append((image_directory, output_directory, camera_matrix, distortion_coefficients, args.img_ext))

    # Multiprocessing
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    pool.map(worker_task, directory_pairs)
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
