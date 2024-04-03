import numpy as np
import pyvista as pv
from PIL import Image
import cv2 as cv
import math
from scipy.spatial.transform import Rotation as R

def quaternion_to_euler(w, x, y, z):
    # Calculate roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Calculate pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp) # use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # Calculate yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

# Define the positions
leader_pos = np.array([10000.0, 0.0, 0.0])  # Leader position
chaser_pos = np.array([0.0, 0.0, 0.0])  # Chaser position

# Quaternions representing orientation
leader_quat = [0.707, 0.707, 0, 0]
#leader_quat = [1, 0, 0, 0]  # w, x, y, z for the leader
angles_l = np.array(quaternion_to_euler(*leader_quat))


#chaser_quat = [0.707, 0, 0.707, 0]  # w, x, y, z for the chaser
chaser_quat = [1, 0, 0, 0]
angles_c = np.array(quaternion_to_euler(*chaser_quat))
angles = angles_l- angles_c
angles_rel = [-angles[1],angles[2],angles[0]]
# Step 1: Compute relative position
relative_pos = leader_pos - chaser_pos

# Step 2: Apply the inverse rotation of the chaser to the relative position
# Create a rotation object from the chaser's quaternion (note scipy uses scalar-last convention by default)
chaser_rotation = R.from_quat([chaser_quat[1], chaser_quat[2], chaser_quat[3], chaser_quat[0]])

# Inverse of chaser's rotation
chaser_rotation_inv = chaser_rotation.inv()

# Rotate the relative position
rotated_pos = chaser_rotation_inv.apply(relative_pos)


camera_frame_pos = [-rotated_pos[1], rotated_pos[2], rotated_pos[0]]

R_rad = np.array(angles_rel) * math.pi / 180
rotation_matrix = cv.Rodrigues(R_rad)[0]
# Your translation vector
translation_vector = np.array(camera_frame_pos)  # Replace with your translation vector

# Create a 4x4 transformation matrix
transformation_matrix = np.eye(4)  # Initialize as identity matrix
transformation_matrix[:3, :3] = rotation_matrix  # Set the top-left 3x3 to your rotation matrix
transformation_matrix[:3, 3] = translation_vector  # Set the top-right 3x1 to your translation vector

# Load the image
image = Image.open('aruco2.png')  # Change 'example.jpg' to the path of your image

# Convert the image to a numpy array
image_array = np.array(image)

# Create a PyVista texture
texture = pv.numpy_to_texture(image_array)

# Create a plane surface mesh
plane = pv.Plane(i_size=image.width, j_size=image.height)

# Apply the texture to the surface mesh
plane.texture_map_to_plane(inplace=True)

# Apply the transformation
plane.transform(transformation_matrix)

# Create a PyVista plotter
plotter = pv.Plotter()

# Add the transformed plane with texture to the plotter
plotter.add_mesh(plane, texture=texture)


# origin = np.array([[0, 00, 10000]])  # Coordinates of the origin point
# origin_poly = pv.PolyData(origin)
# plotter.add_mesh(origin_poly, color="red", point_size=10, render_points_as_spheres=True)


# Adjust the camera settings as needed
plotter.view_xy()
plotter.set_position([0000, 0, 000])
plotter.camera.view_angle = 60.0

# Display the plot
plotter.show()

# Save the screenshot
plotter.screenshot('screenshot.png')
