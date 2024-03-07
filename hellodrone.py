# ready to run example: PythonClient/multirotor/hello_drone.py
# note: async methods take a long time to execute, add join() to wait for it finish 
# NED coordinates: +x:forward +y:right +z:down
# todo:
# 1) set global coordinate, zero origin
import airsim
import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from particle_main import RunParticle
import traceback
import random
from controller_m.gen_traj import Generate
from perception.perception import Perception
from simple_excitation import excitation
lead = "Drone_L"
chase = "Drone_C"

def compute_cam_bound(depth):
    # Given FOV of pinhole camera and distance from camera, computes the rectangle range of observable image
    fov_h = 100 #degrees
    fov_d = 138 #degrees

    rec_width = 2 * (np.tan(np.deg2rad(fov_h/2)) * depth )
    b = 2 * (np.tan(np.deg2rad(fov_d/2)) * depth )
    rec_height = np.sqrt(b**2 - rec_width**2)

    return rec_width,rec_height


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
                [z[connection[0]], z[connection[1]]], 'k-', color='red')


def prediction(initial_state, timestep, steps):
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







# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
# client.reset()

curr_state = client.simGetVehiclePose(lead)
print("lead state", curr_state)
# curr_state.position.z_val = 0
# client.simSetVehiclePose(curr_state, True, lead)
# time.sleep(2)
# curr_state = client.simGetVehiclePose(lead)
# print("lead state", curr_state)

# curr_state2 = client.getMultirotorState(chase)
# print("chaser state", curr_state2)

client.enableApiControl(True,lead)
client.armDisarm(True, lead)
client.takeoffAsync(10.0, lead).join()

client.enableApiControl(True,chase)
client.armDisarm(True, chase)
client.takeoffAsync(10.0, chase).join()

# curr_state = client.getMultirotorState(lead)
# print("lead state", curr_state)
# curr_state2 = client.getMultirotorState(chase)
# print("chaser state", curr_state2)

# pose = client.simGetVehiclePose(lead)
# print("lead state", pose.position)

# Take picture ###################################################################################################################
# vision = Perception(client)
# img_rgb = vision.capture_RGB(client)
# cv2.imshow("pic",img_rgb)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img_segment = vision.capture_segment(client)
# cv2.imshow("pic",img_segment)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Chase Drone Movement ###################################################################################################################

count = 0
# lead_pose1 = [client.simGetVehiclePose(lead).position.x_val, client.simGetVehiclePose(lead).position.y_val, client.simGetVehiclePose(lead).position.z_val]
lead_pose1 = [client.simGetVehiclePose(lead).position.x_val, client.simGetVehiclePose(lead).position.y_val,client.simGetVehiclePose(lead).position.z_val]
print("Lead position",lead_pose1)

mcl = RunParticle(starting_state=lead_pose1)    

 
# Initialize mcl Position
est_states = np.zeros((len(mcl.ref_traj) ,6)) # x y z vx vy vz
gt_states  = np.zeros((len(mcl.ref_traj) ,16))
iteration_count = np.arange(0,len(mcl.ref_traj) , 1, dtype=int)

start_time = time.time()

pose_est_history_x = []
pose_est_history_y = []
pose_est_history_z = []
velocity_est_history_x = []
velocity_est_history_y =[]
velocity_est_history_z = []
PF_history_x = []
PF_history_y = []
PF_history_z = []
GT_state_history_x=[]
GT_state_history_y=[]
GT_state_history_z=[]
total_vest=[]

GT_state_history=[]
particle_state_est=[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]

PF_history_x.append(np.array(mcl.filter.particles['position'][:,0]).flatten())
PF_history_y.append(np.array(mcl.filter.particles['position'][:,1]).flatten())
PF_history_z.append(np.array(mcl.filter.particles['position'][:,2]).flatten())

# Assume constant time step between trajectory stepping
timestep = 0.1
oldpositions= lead_pose1
oldvelocity = [0,0,0]
totalcount = 100
start_time = time.time()

def random_traj(i,total_count):
    x= 3* np.sin(i* 2*np.pi/total_count)
    y= np.cos(i* 2*np.pi/total_count)
    z= 0.5*np.sin(i* 2*np.pi/total_count)
    return x,y,z

def circle_traj(i,total_count):
    radius = 10
    start=lead_pose1
    t = np.linspace(0,2*np.pi,totalcount)
    x = lead_pose1[0]-radius - radius * np.cos(i* 2*np.pi/total_count)
    y = lead_pose1[1]-radius * np.sin(i* 2*np.pi/total_count)
    z= lead_pose1[2]
    return x,y,z

effort = excitation(2,2)
try:

    while True:
        dt_move = 2
        # Lead Drone Movement ###################################################################################################################
        # client.moveByVelocityAsync(1,0,0,dt_move,vehicle_name=lead)
        # effortx,efforty,effortz = random_traj(count,totalcount)
        # if count >= 50:
        #     client.moveByVelocityBodyFrameAsync(6, 0, 0, timestep, vehicle_name = lead)
        # else:
        #     client.moveByVelocityBodyFrameAsync(3, 0, 0, timestep, vehicle_name = lead)
        client.moveByVelocityBodyFrameAsync(effort[count], 0, 0, timestep, vehicle_name = lead)

        # identify location of lead
        # lead_pose = [client.simGetVehiclePose(lead).position.x_val, client.simGetVehiclePose(lead).position.y_val, client.simGetVehiclePose(lead).position.z_val]
        lead_pose = [client.simGetVehiclePose(lead).position.x_val, client.simGetVehiclePose(lead).position.y_val,client.simGetVehiclePose(lead).position.z_val]
        # print("Lead position",lead_pose)

        state_est = mcl.rgb_run(current_pose=lead_pose, past_states = particle_state_est, last_vel= oldvelocity , time_step=0.1)   
        oldpositions = state_est[:3]
        oldvelocity = state_est[3:6]
        
        # if count == 50:
        #     prediction(state_est, timestep=0.1, steps=10    )

       
        GT_state_history.append(lead_pose)
        particle_state_est.append(state_est)

        PF_history_x.append(np.array(mcl.filter.particles['position'][:,0]).flatten())
        PF_history_y.append(np.array(mcl.filter.particles['position'][:,1]).flatten())
        PF_history_z.append(np.array(mcl.filter.particles['position'][:,2]).flatten())

        count += 1
        curr_time = time.time()
        print(f"Total simulation time: {round(curr_time-start_time,4)} sec")
        time.sleep(timestep)

        if count == totalcount:
            break

    GT_state_history = np.array(GT_state_history)
    particle_state_est = np.array(particle_state_est)
    PF_history_x = np.array(PF_history_x)
    PF_history_y = np.array(PF_history_y)
    PF_history_z = np.array(PF_history_z)
    # print(GT_state_history.shape)
    # print(pose_est_history_x)

    times = np.arange(0,particle_state_est.shape[0]-2)*timestep
    velocity_GT_x = (GT_state_history[1:,0]-GT_state_history[:-1,0])/timestep
    velocity_GT_y = (GT_state_history[1:,1]-GT_state_history[:-1,1])/timestep
    velocity_GT_z = (GT_state_history[1:,2]-GT_state_history[:-1,2])/timestep

    accel_GT_x = (velocity_GT_x[1:]-velocity_GT_x[:-1])/timestep
    accel_GT_y = (velocity_GT_y[1:]-velocity_GT_y[:-1])/timestep
    accel_GT_z = (velocity_GT_z[1:]-velocity_GT_z[:-1])/timestep

    # fig, (posx,posy,posz) = plt.subplots(3, 1, figsize=(14, 10))
    # posx.plot(times, particle_state_est[:,0], label = "Filter Pos x")
    # posx.plot(times, GT_state_history[:,0], label = "GT Pos x")
    # posx.legend()
    # posy.plot(times, particle_state_est[:,1], label = "Filter Pos y")    
    # posy.plot(times, GT_state_history[:,1], label = "GT Pos y")
    # posy.legend()
    # posz.plot(times, particle_state_est[:,2], label = "Filter Pos z")
    # posz.plot(times, GT_state_history[:,2], label = "GT Pos z")
    # posz.legend()

    # fig, (velx,vely,velz) = plt.subplots(3, 1, figsize=(14, 10))
    # velx.plot(times, particle_state_est[:,3], label = "Filter Vel x")
    # velx.plot(times[1:], velocity_GT_x, label = "GT Vel x")
    # # velx.set_ylim(-1,2)
    # velx.legend()
    # vely.plot(times, particle_state_est[:,4], label = "Filter Vel y")    
    # vely.plot(times[1:], velocity_GT_y, label = "GT Vel y")
    # vely.legend()
    # velz.plot(times, particle_state_est[:,5], label = "Filter Vel z")
    # velz.plot(times[1:], velocity_GT_z, label = "GT Vel z")
    # velz.legend()

    fig, (posx,velx,accelx) = plt.subplots(3, 1, figsize=(14, 10))
    posx.plot(times, particle_state_est[2:,0], label = "Filter Pos x")
    posx.plot(times, GT_state_history[:,0], label = "GT Pos x")
    posx.legend()
    velx.plot(times, particle_state_est[2:,3], label = "Filter Vel x")
    velx.plot(times[1:], velocity_GT_x, label = "GT Vel x")
    velx.legend()
    accelx.plot(times, particle_state_est[2:,6], label = "Filter acel x")    
    accelx.plot(times[2:], accel_GT_x, label = "GT accel x")
    accelx.legend()
    

    plt.show()
    
    # fig, (velx,vely,velz) = plt.subplots(3, 1, figsize=(14, 10))
    # velx.plot(times, total_vest[:,0], label = "Filter Vel x")
    # vely.plot(times, total_vest[:,1], label = "Filter Vel y")    
    # velz.plot(times, total_vest[:,2], label = "Filter Vel z")
    # velz.legend()
    # plt.show()

    # fig = plt.figure(1)
    # ax = fig.add_subplot(111, projection='3d')
    # # ax.plot(x, y, z, color='b')
    # # t = np.linspace(0, 32, 1000)
    # # x = mcl.ref_traj[:,0]
    # # y = mcl.ref_traj[:,1]
    # # z = mcl.ref_traj[:,2]
    # plt.figure(1)
    # # ax.plot(x,y,z, color = 'b')
    # ax.plot(pose_est_history_x,pose_est_history_y,pose_est_history_z, '*',color = 'g',label='PF Estimate state')
    # ax.plot(GT_state_history_x,GT_state_history_y,GT_state_history_z, color = 'r',label='GT state')
    # ax.plot(lead_pose1[0],lead_pose1[1],lead_pose1[2],'*', color = 'b')
    # ax.plot(PF_history_x[0],PF_history_y[0],PF_history_z[0],'*', color = 'b')
    # ax.plot(PF_history_x[1],PF_history_y[1],PF_history_z[1],'*', color = 'purple')
    # plt.legend()
    # plt.show()

    print("Finished")
    client.reset()
    client.armDisarm(False)

    # that's enough fun for now. let's quit cleanly
    client.enableApiControl(False)

except Exception as e:
    print("Error Occured, Canceling: ",e)
    traceback.print_exc()

    client.reset()
    client.armDisarm(False)

    # that's enough fun for now. let's quit cleanly
    client.enableApiControl(False)


# # client.moveToZAsync(10,2, vehicle_name=chase).join()
# curr_state = client.simGetVehiclePose(chase)
# print("chase state", curr_state)




# # Async methods returns Future. Call join() to wait for task to complete.
# client.takeoffAsync().join()
# client.moveToPositionAsync(-10, 10, -10, 5).join()

# # take images
# responses = client.simGetImages([
#     airsim.ImageRequest("0", airsim.ImageType.DepthVis),
#     airsim.ImageRequest("1", airsim.ImageType.DepthPlanar, True)])
# print('Retrieved images: %d', len(responses))

# # do something with the images
# for response in responses:
#     if response.pixels_as_float:
#         print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
#         airsim.write_pfm(os.path.normpath('py1.pfm'), airsim.get_pfm_array(response))
#     else:
#         print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
#         airsim.write_file(os.path.normpath('py1.png'), response.image_data_uint8)


