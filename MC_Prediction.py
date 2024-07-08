import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import math

class Prediction():
    def __init__(self, fov_h= 70, fov_d = None ):
        # Given FOV of pinhole camera and distance from camera, computes the rectangle range of observable image
        # Arducam B0385 Global Shutter Camera
        self.fov_h = fov_h
        self.fov_d = fov_d
        self.aspect_ratio =  4 / 3

    # def compute_cam_bound(self, depth):

    #     rec_width = 2 * (np.tan(np.deg2rad(self.fov_h/2)) * depth )
    #     b = 2 * (np.tan(np.deg2rad(self.fov_d/2)) * depth )
    #     rec_height = np.sqrt(b**2 - rec_width**2)

    #     return rec_width,rec_height
    
    def calculate_viewable_area(self, distance):
        # Convert horizontal FOV from degrees to radians
        fov_h_radians = math.radians(self.fov_h)
        
        # Calculate the vertical FOV in radians
        fov_v_radians = 2 * math.atan(math.tan(fov_h_radians / 2) / self.aspect_ratio)
        
        # Calculate the viewable width
        viewable_width = 2 * distance * math.tan(fov_h_radians / 2)
        
        # Calculate the viewable height
        viewable_height = 2 * distance * math.tan(fov_v_radians / 2)
        
        return viewable_width, viewable_height


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

    def get_min_sphere(self, centerpoint, min_x,max_x,min_y,max_y,min_z,max_z):
        min_radius = 10000

        x = [min_x, max_x, max_x, min_x, min_x, max_x, max_x, min_x]
        y = [min_y, min_y, max_y, max_y, min_y, min_y, max_y, max_y]
        z = [min_z, min_z, min_z, min_z, max_z, max_z, max_z, max_z]

        for i in range(8):
            distance = la.norm(centerpoint - np.array([x[i],y[i],z[i]]))
            if distance < min_radius:
                min_radius = distance

        return min_radius


        

    def find_prediction(self, initial_state, ego_state, timestep, accel_range, steps = 9, num_trajectory = 100, visualize = False):
            
        total_trajectories=[]

        # Generate Trajectories
        for i in range(num_trajectory):
            trajectory = [initial_state]
            for s in range(steps):
                a = np.random.uniform(-accel_range, accel_range, size=3)
                v = trajectory[-1][3:6] + a * timestep
                p = trajectory[-1][:3] + v * timestep
                trajectory.append([p[0], p[1], p[2], v[0], v[1], v[2], a[0],a[1],a[2]])

            total_trajectories.append(trajectory)
            trajectory=np.array(trajectory)
        

        # Find Hyper-rectangles of Trajectories
        total_trajectories=np.array(total_trajectories)
        rectangle = []
        bound_y,bound_z = self.calculate_viewable_area(ego_state[0])

        for s in range(steps+1):
            min_x = np.min(total_trajectories[:,s,0])
            max_x = np.max(total_trajectories[:,s,0])
            min_y = np.min(total_trajectories[:,s,1])
            max_y = np.max(total_trajectories[:,s,1])
            min_z = np.min(total_trajectories[:,s,2])
            max_z = np.max(total_trajectories[:,s,2])

            mid_y = (min_y+max_y)/2
            mid_z = (min_z+max_z)/2

            if np.abs(max_y-min_y)  > bound_y:
                break
                min_y = rectangle[-1][2]
                max_y = rectangle[-1][3]
            if np.abs(max_z-min_z)  > bound_z:
                break
                min_z = rectangle[-1][4]
                max_z = rectangle[-1][5]
            rectangle.append([min_x,max_x,min_y,max_y,min_z,max_z])

        # Find predict trajectory by average
        predict_trajectory = []
        for s in range(steps+1):
            x_val = np.mean(total_trajectories[:,s,0])
            y_val = np.mean(total_trajectories[:,s,1])
            z_val = np.mean(total_trajectories[:,s,2])
            predict_trajectory.append([x_val, y_val, z_val])

        # Compute Variance
        variance = []
        for r in range(steps):
            variance.append(self.get_min_sphere(predict_trajectory[r],rectangle[r][0], rectangle[r][1], rectangle[r][2], rectangle[r][3], rectangle[r][4], rectangle[r][5]))

        if visualize:
            # Trajectory Plot
            # fig = plt.figure(1)
            # ax = fig.add_subplot(111, projection='3d')

            # predict_trajectory = np.array(predict_trajectory)
            # ax.plot(predict_trajectory[:,0], predict_trajectory[:,1], predict_trajectory[:,2], '-x',color='r', label="Avg Traj")

            # for t in range(num_trajectory):
            #     ax.plot(total_trajectories[t][:,0], total_trajectories[t][:,1], total_trajectories[t][:,2], '-x',color='b')

            # for r in range(len(rectangle)):
            #     self.plot_rec(ax, rectangle[r][0], rectangle[r][1], rectangle[r][2], rectangle[r][3], rectangle[r][4], rectangle[r][5])

            # # plot_rec(ax, 0,1,-bound_y/2,bound_y/2,-bound_z/2,bound_y/2)
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.set_zlabel("Z")
            # plt.show()

            # Variance Plot
            s = np.arange(0,steps,1)
            
            print(variance)
            plt.figure(figsize=(10, 5))
            plt.plot(s,variance, marker='o')
            plt.xlabel('Steps')
            plt.ylabel('Variance')
            plt.title('Prediction Variance')
            plt.grid(True)
            plt.show()


        return rectangle, total_trajectories, np.array(predict_trajectory), variance