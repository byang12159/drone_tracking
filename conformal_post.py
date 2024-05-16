import csv
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# Path to your CSV file
csv_file_path = 'simulation_data_stationary_randomuniform.csv'

position_x = []
position_y = []
position_z = []

accel_x = []
accel_y = []
accel_z = []

# Open the file
with open(csv_file_path, mode='r', newline='') as file:
    # Create a CSV reader
    csv_reader = csv.reader(file)
    
    # Iterate over the rows in the file
    first = True
    position_x_i = []
    position_y_i = []
    position_z_i = []
    accel_x_i = []
    accel_y_i = []
    accel_z_i = []

    for row in csv_reader:
        if first:
            first=False
        elif row[0].startswith("Iteration"):
            position_x.append(position_x_i)
            position_y.append(position_y_i)
            position_z.append(position_z_i)
            accel_x.append(accel_x_i)
            accel_y.append(accel_y_i)
            accel_z.append(accel_z_i)

            position_x_i = []
            position_y_i = []
            position_z_i = []
            accel_x_i = []
            accel_y_i = []
            accel_z_i = []
        else:
            float_row = [float(value) for value in row]

            position_x_i.append(float_row[0])
            position_y_i.append(float_row[1])
            position_z_i.append(float_row[2])

            accel_x_i.append(float_row[10])
            accel_y_i.append(float_row[11])
            accel_z_i.append(float_row[12])
    position_x.append(position_x_i)
    position_y.append(position_y_i)
    position_z.append(position_z_i)
    accel_x.append(accel_x_i)
    accel_y.append(accel_y_i)
    accel_z.append(accel_z_i)



# Create subplots with a single column and three rows
fig, axs = plt.subplots(3, 1, figsize=(8, 10))

# Plot data on the first subplot
for j in range(len(position_x)):
    times = np.arange(0, 0.01*(len(position_x[j])-1), 0.01)
    # print(j, len(times), len(position_x[j]), len(position_y[j]), len(position_z[j]))
    axs[0].plot(times, position_x[j][0:len(times)], color='blue')
    axs[1].plot(times, position_y[j][0:len(times)], color='red')
    axs[2].plot(times, position_z[j][0:len(times)], color='green')

axs[0].set_title('Plot 1: Position x')
axs[1].set_title('Plot 2: Position y')
axs[2].set_title('Plot 3: Position z')
plt.tight_layout()


# fig, axs = plt.subplots(3, 1, figsize=(8, 10))

# # Plot data on the first subplot
# axs[0].plot(times, accel_x, color='blue')
# axs[0].set_title('Plot 1: Accel x')
# # Plot data on the second subplot
# axs[1].plot(times, accel_y, color='red')
# axs[1].set_title('Plot 2: Accel y')
# # Plot data on the third subplot
# axs[2].plot(times, accel_z, color='green')
# axs[2].set_title('Plot 3: Accel z')

# # Add some space between subplots
# plt.tight_layout()

plt.show()


# Create a figure
fig = plt.figure()

# Add a 3D subplot
ax = fig.add_subplot(111, projection='3d')

# Create a 3D scatter plot
for j in range(len(position_x)):
    ax.plot(position_x[j],position_y[j],position_z[j],'-',linewidth=0.5)
   
# Set labels for the axes
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# Set title
ax.set_title('3D Scatter Plot')

# Show the plot
plt.show()