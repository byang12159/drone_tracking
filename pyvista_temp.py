import pyvista as pv
from pyvista import demos
plotter = demos.orientation_plotter()
# Create a sphere centered at (0, 0, 0)
# sphere = pv.Sphere()

# # Create a plotter
# plotter = pv.Plotter()

# # Add the sphere to the plotter
# plotter.add_mesh(sphere)

# Set the camera position and orientation
# Move the camera to (x, y, z)
# Apply yaw, pitch, and roll rotations
x = 0
y = 0
z = 20
yaw = 0 
pitch = 90
roll = 0
plotter.camera_position = [(x, y, z), (2, 1, 0), (yaw, pitch, roll)]

# Take a screenshot
plotter.show()
screenshot = plotter.screenshot()

# Save the screenshot to a file
screenshot.save('screenshot.png')
