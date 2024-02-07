# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import UnivariateSpline

# # Sample data for demonstration
# x = np.linspace(0, 10, 100)
# y = np.sin(x)

# # Fit a spline curve to the data
# spline = UnivariateSpline(x, y)

# # Generate points along the spline curve
# x_points = np.linspace(0, 10, 1000)
# y_points = spline(x_points)

# # Calculate tangent vectors
# t_values = np.linspace(0, 10, 20)  # Choose parameter values where you want to evaluate the tangent vectors
# tangent_vectors = spline.derivative(n=1)(t_values)
# print(tangent_vectors)
# # Plot the spline curve
# plt.plot(x_points, y_points, label='Spline Curve')

# # Plot tangent vectors
# for i in range(len(t_values)):
#     tangent = tangent_vectors[i]
#     print(tangent)
#     tangent /= np.linalg.norm(tangent)  # Normalize the tangent vector for plotting
#     plt.arrow(x_points[i], y_points[i], tangent[0], tangent[1], color='red', head_width=0.1, length_includes_head=True)

# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Spline Curve and Tangent Vectors')
# plt.legend()
# plt.grid(True)
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create a figure and axis
fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro')  # Create an empty plot to be updated later

# Set axis limits
ax.set_xlim(0, 10)
ax.set_ylim(-1, 1)

# Initialization function: plot the background of each frame
def init():
    ln.set_data([], [])
    return ln,

# Update function: called for each frame with the next set of data points
def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,

# Generate frames for animation
frames = np.linspace(0, 10, 100)

# Create the animation
ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)

# Show the animation
plt.show()
