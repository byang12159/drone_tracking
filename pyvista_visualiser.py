import numpy as np
import pyvista as pv
from PIL import Image
import cv2 as cv
import math


R_rad = np.array((45.0, 0.0, 0.0)) * math.pi / 180
rotation_matrix = cv.Rodrigues(R_rad)[0]
# Your translation vector
translation_vector = np.array([1000, 1000, 10000])  # Replace with your translation vector

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


origin = np.array([[0, 00, 10000]])  # Coordinates of the origin point
origin_poly = pv.PolyData(origin)
plotter.add_mesh(origin_poly, color="red", point_size=10, render_points_as_spheres=True)


# Adjust the camera settings as needed
plotter.view_xy()
plotter.set_position([0000, 0, 000])
plotter.camera.view_angle = 60.0

# Display the plot
plotter.show()

# Save the screenshot
plotter.screenshot('screenshot.png')
