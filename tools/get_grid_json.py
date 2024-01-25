import json

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Given keyframes
keyframes = [
    {
      "matrix": "[-0.05050521304467104,0.9987237973810936,1.6653345369377348e-16,0,-0.016501846752466187,-0.0008344942696361102,0.9998634870186388,0,0.9985874586179571,0.05049831842746444,0.016522933363296688,0,-0.0653132042308587,-0.0033028724287530297,-0.0010806922437654192,1]",
      "fov": 45,
      "aspect": 1.621621622,
      "properties": "[[\"FOV\",35],[\"NAME\",\"Camera 0\"],[\"TIME\",0]]"
    },
    {
      "matrix": "[-0.05050521304467104,0.9987237973810936,1.6653345369377348e-16,0,-0.016501846752466187,-0.0008344942696361102,0.9998634870186388,0,0.9985874586179571,0.05049831842746444,0.016522933363296688,0,-0.07137382979621941,0.1426573214195612,-0.0010806922437654207,1]",
      "fov": 45,
      "aspect": 1.621621622,
      "properties": "[[\"FOV\",35],[\"NAME\",\"Camera 1\"],[\"TIME\",0.3333333333333333]]"
    },
    {
      "matrix": "[-0.050505213044671926,0.9987237973810938,5.551115123125783e-17,0,-0.016501846752466076,-0.0008344942696361102,0.9998634870186389,0,0.9985874586179571,0.050498318427464994,0.016522933363296355,0,0.05844253982411485,0.14508339124208902,0.0010672890934631022,1]",
      "fov": 45,
      "aspect": 1.621621622,
      "properties": "[[\"FOV\",35],[\"NAME\",\"Camera 2\"],[\"TIME\",0.6666666666666666]]"
    },
    {
      "matrix": "[-0.05050521304467237,0.998723797381094,5.551115123125783e-17,0,-0.01650184675246591,-0.0008344942696363322,0.9998634870186391,0,0.9985874586179573,0.050498318427465494,0.016522933363296133,0,0.06450316538947501,0.003261908966817442,0.0010672890934630749,1]",
      "fov": 45,
      "aspect": 1.621621622,
      "properties": "[[\"FOV\",35],[\"NAME\",\"Camera 3\"],[\"TIME\",1]]"
    }
]

# Extract the last three 3D points from each keyframe
points = []
for keyframe in keyframes:
    matrix = np.array(eval(keyframe["matrix"]))
    # Get the first 3 elements of the last 4 elements of the matrix
    points.append(matrix[12:15])

# Define your four 3D coordinates (replace these with your own coordinates)
coordinates = np.array(points)

# Find the minimum and maximum values for each dimension
min_coords = np.min(coordinates, axis=0)
max_coords = np.max(coordinates, axis=0)

# Number of grid centers
num_centers = 10

# Calculate the step size for each dimension
step_x = (max_coords[0] - min_coords[0]) / (num_centers - 1)
step_y = (max_coords[1] - min_coords[1]) / (num_centers - 1)

# Limit the spread of grid centers along the z-axis (adjust as needed)
min_z = min_coords[2]
max_z = min_coords[2] + 0.1  # Adjust this value to control the spread along the z-axis


# Generate the grid centers on the rectangular plane
grid_centers = []
for i in range(num_centers):
    for j in range(num_centers):
        x = min_coords[0] + i * step_x
        y = min_coords[1] + j * step_y
        z = (max_z + min_z) / 2
        grid_centers.append([x, y, z])

# Create a 3D scatter plot for the extracted points
points = np.array(points)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o', label='Keyframe Points')

# Assuming you already have the grid_centers from the previous code
grid_centers = np.array(grid_centers)
ax.scatter(grid_centers[:, 0], grid_centers[:, 1], grid_centers[:, 2], c='b', marker='x', label='Grid Centers')

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.legend()
plt.show()

given_matrix_str = [0.0017185946975362132,0.9999985232150423,1.1102230246251565e-16,0,0.005474302260813246,-0.000009408120732024017,0.9999850158508599,0,0.9999835390880305,-0.0017185689458568132,-0.00547431034519219,0]

# Parse the given matrix into a NumPy array
new_matrices = []
camera_path = []

print(grid_centers)
# Iterate through each grid center
for i in range(len(grid_centers)):
    rotation_matrix = np.reshape(given_matrix_str, (3,4))  # Reshape the matrix into a 4x3 matrix
    # Get transpose of the rotation matrix
    rotation_matrix = np.transpose(rotation_matrix)
    # Create a new transformation matrix by appending the translation components and a 1
    camera_to_world = [rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2], grid_centers[i][0],
                    rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2], grid_centers[i][1],
                    rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2], grid_centers[i][2],
                    0.0, 0.0, 0.0, 1.0]
    frame = {
        "camera_to_world": camera_to_world,
        "fov": 45,
        "aspect": 1.621621622
    }
    camera_path.append(frame)

# Create the JSON structure
json_data = {
    "keyframes": keyframes,
    "camera_type": "perspective",
    "render_height": 1184,
    "render_width": 1920,
    "camera_path": camera_path,
    "fps": 25,
    "seconds": 4,
    "smoothness_value": 0,

}

# Save the JSON data to a file
with open("output_test_close_fov45.json", "w") as json_file:
    json.dump(json_data, json_file, indent=4)