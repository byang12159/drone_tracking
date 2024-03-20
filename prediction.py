import numpy as np
import matplotlib.pyplot as plt

class Prediction():
    def __init__(self):
        pass

    def compute_cam_bound(self, depth):
        # Given FOV of pinhole camera and distance from camera, computes the rectangle range of observable image
        fov_h = 100 #degrees
        fov_d = 138 #degrees

        rec_width = 2 * (np.tan(np.deg2rad(fov_h/2)) * depth )
        b = 2 * (np.tan(np.deg2rad(fov_d/2)) * depth )
        rec_height = np.sqrt(b**2 - rec_width**2)

        return rec_width,rec_height


    def plot_rec(self, ax, min_x,max_x,min_y,max_y,min_z,max_z):
        x = [min_x, max_x, max_x, min_x, min_x, max_x, max_x, min_x]
        y = [min_y, min_y, max_y, max_y, min_y, min_y, max_y, max_y]
        z = [min_z, min_z, min_z, min_z, max_z, max_z, max_z, max_z]

        # Define connections between the corner points
        connections = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]

        # Plot wireframe
        for connection in connections:
            ax.plot([x[connection[0]], x[connection[1]]],
                    [y[connection[0]], y[connection[1]]],
                    [z[connection[0]], z[connection[1]]], 'k-', color='red')


    def prediction(self, initial_state, timestep, steps):
        num_trajectory = 100
        total_trajectories=[]

        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')

        # Generate Trajectories
        for i in range(num_trajectory):
            trajectory = [initial_state]
            for s in range(steps):
                a = np.random.uniform(-3.0, 3.0, size=3)
                v = trajectory[-1][3:6] + a * timestep
                p = trajectory[-1][:3] + v * timestep
                trajectory.append([p[0], p[1], p[2], v[0], v[1], v[2], a[0],a[1],a[2]])

            total_trajectories.append(trajectory)
            trajectory=np.array(trajectory)
            ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], color='b')
        

        # Find Hyper-rectangles of Trajectories
        total_trajectories=np.array(total_trajectories)
        rectangle = []
        for s in range(steps+1):
            min_x = np.min(total_trajectories[:,s,0])
            max_x = np.max(total_trajectories[:,s,0])
            min_y = np.min(total_trajectories[:,s,1])
            max_y = np.max(total_trajectories[:,s,1])
            min_z = np.min(total_trajectories[:,s,2])
            max_z = np.max(total_trajectories[:,s,2])
            rectangle.append([min_x,max_x,min_y,max_y,min_z,max_z])

            plot_rec(ax, min_x,max_x,min_y,max_y,min_z,max_z)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show
