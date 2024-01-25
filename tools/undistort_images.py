import cv2
import os
import json
import numpy as np
import argparse

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
    for file in os.listdir(image_directory):
        if not file.endswith(img_ext):
            continue
        image_path = os.path.join(image_directory, file)
        print('Processing image:', image_path)
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

def main():
    parser = argparse.ArgumentParser(description="Undistort images using camera parameters")
    parser.add_argument("image_directory", help="Directory containing images to undistort")
    parser.add_argument("output_directory", help="Directory to save undistorted images")
    parser.add_argument("--img_ext", default=".png", help="Image file extension to process (default: .png)")
    
    args = parser.parse_args()
    image_directory = os.path.join(args.image_directory, 'images')
    output_directory = os.path.join(args.output_directory, 'images')
    json_file_path = os.path.join(args.image_directory, 'transforms.json')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    print("Loading images from", image_directory)
    print("Saving undistorted images to", output_directory)
    print("Loading camera parameters from", json_file_path)

    camera_matrix, distortion_coefficients = load_camera_params(json_file_path)
    undistort_images(image_directory, output_directory, camera_matrix, distortion_coefficients, args.img_ext)

if __name__ == "__main__":
    main()
