from simple_excitation import excitation
import airsim
import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

# data = {"name": "John", "age": 30}




def compute_cam_bound(depth):
    # Given FOV of pinhole camera and distance from camera, computes the rectangle range of observable image
    fov_h = 100 #degrees
    fov_d = 138 #degrees

    rec_width = 2 * (np.tan(np.deg2rad(fov_h/2)) * depth )
    b = 2 * (np.tan(np.deg2rad(fov_d/2)) * depth )
    rec_height = np.sqrt(b**2 - rec_width**2)

    return rec_height, rec_width


def plot_rec(ax, min_x,max_x,min_y,max_y,min_z,max_z):
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
                [z[connection[0]], z[connection[1]]], '-', color='red')


def prediction(initial_state, timestep, steps, camera_depth,accel_range):
    num_trajectory = 200
    total_trajectories=[]

    # fig = plt.figure(1)
    # ax = fig.add_subplot(111, projection='3d')

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
        # ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], color='b')
    

    # Find Hyper-rectangles of Trajectories
    total_trajectories=np.array(total_trajectories)
    rectangle = []
    bound_y,bound_z = compute_cam_bound(depth=camera_depth)

    # print("bounds: ",bound_y,bound_z)
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

        # plot_rec(ax, min_x,max_x,min_y,max_y,min_z,max_z)

    # plot_rec(ax, 0,1,-bound_y/2,bound_y/2,-bound_z/2,bound_y/2)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show

    return len(rectangle)

if __name__ == "__main__":
    #MOVE IN NON X DIRECTION FOR V?????????????????
    effort = excitation(2,2)


    variable_V = np.linspace(0,20,5)
    variable_A = np.linspace(0,44,50)
    total_steps_predict = []
    total_d = []

    for v in variable_V:
        print("current v:",v)
        
        initial_V = v
        timestep = 0.1

        initalstate = np.zeros(9)
        initalstate[3]=initial_V

        d = np.linspace(0,3,50)
        steps_predict = []
        for i in d:
            num_steps = prediction(initial_state=initalstate, timestep=timestep,steps=30, camera_depth=i, accel_range=5)
            steps_predict.append(num_steps)
            
        total_steps_predict.append(steps_predict)
        total_d.append(d)
        # plt.plot(d,steps_predict)
        # plt.xlabel("Target  Distance from Cameqra (m)")
        # plt.ylabel("# Future Steps Within Cam Bounds")
        # plt.show()    
        
        # plt.plot(d,0.1*np.array(steps_predict))
        # plt.xlabel("Target  Distance from Camera (m)")
        # plt.ylabel("Future Time Within Cam Bounds (s)")
        # plt.title("System timstep:0.1s, Velocity:2m/s, max Accel:+-5m/s^2")
        # plt.show()

    # data=[variable_A,total_d,total_steps_predict]
    # with open("data_variableV_44a.pkl", "wb") as file:
    #     pickle.dump(data, file) 

    for i in range(len(variable_V)):
        plt.plot(total_d[i],0.1*np.array(total_steps_predict[i]),'-', label=f"V: {variable_V[i]}")
        
    plt.xlabel("Target  Distance from Camera (m)")
    plt.ylabel("Future Time Within Cam Bounds (s)")
    plt.title("System timstep:0.1s, Varying Velocity, max Accel:+-5m/s^2")
    plt.legend()
    plt.show()

    # Varying Acceleration
    # for v in variable_A:
    #     print("current a:",v)
        
    #     initial_V = 2
    #     timestep = 0.1

    #     initalstate = np.zeros(9)
    #     initalstate[5]=initial_V

    #     d = np.linspace(0,3,50)
    #     steps_predict = []
    #     for i in d:
    #         num_steps = prediction(initial_state=initalstate, timestep=timestep,steps=30, camera_depth=i, accel_range=v)
    #         steps_predict.append(num_steps)
            
    #     total_steps_predict.append(steps_predict)
    #     total_d.append(d)
    #     # plt.plot(d,steps_predict)
    #     # plt.xlabel("Target  Distance from Cameqra (m)")
    #     # plt.ylabel("# Future Steps Within Cam Bounds")
    #     # plt.show()    
        
    #     # plt.plot(d,0.1*np.array(steps_predict))
    #     # plt.xlabel("Target  Distance from Camera (m)")
    #     # plt.ylabel("Future Time Within Cam Bounds (s)")
    #     # plt.title("System timstep:0.1s, Velocity:2m/s, max Accel:+-5m/s^2")
    #     # plt.show()

    # data=[variable_A,total_d,total_steps_predict]
    # with open("data_variableA_44a.pkl", "wb") as file:
    #     pickle.dump(data, file) 

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for i in range(len(variable_A)):
    #     vselect= np.ones_like(total_d[i])*variable_A[i]
    #     ax.plot(total_d[i], 0.1*np.array(total_steps_predict[i]), vselect, 'x')

    # # Add labels and title
    # ax.set_xlabel("Target  Distance from Camera (m)")
    # ax.set_ylabel("Future Time Within Cam Bounds (s)")
    # ax.set_zlabel('Acceleration Bound (m/s^2)')
    # ax.set_title('System timstep:0.1s, Initial Velocity:2m/s, Varying Accel')

    # # Show plot
    # plt.show()