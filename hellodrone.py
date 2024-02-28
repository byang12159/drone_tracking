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

from controller_m.gen_traj import Generate
from perception.perception import Perception
lead = "Drone_L"
chase = "Drone_C"

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
lead_pose1 = [client.simGetVehiclePose(lead).position.x_val, 0,0]
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


PF_history_x.append(np.array(mcl.filter.particles['position'][:,0]).flatten())
PF_history_y.append(np.array(mcl.filter.particles['position'][:,1]).flatten())
PF_history_z.append(np.array(mcl.filter.particles['position'][:,2]).flatten())

# Assume constant time step between trajectory stepping
timestep = 0.1
oldpositions= lead_pose1
try:

    while True:
        dt_move = 2
        # Lead Drone Movement ###################################################################################################################
        # client.moveByVelocityAsync(1,0,0,dt_move,vehicle_name=lead)
        client.moveByVelocityBodyFrameAsync(1, 0, 0, 30, vehicle_name = lead)

        # identify location of lead
        # lead_pose = [client.simGetVehiclePose(lead).position.x_val, client.simGetVehiclePose(lead).position.y_val, client.simGetVehiclePose(lead).position.z_val]
        lead_pose = [client.simGetVehiclePose(lead).position.x_val, 0,0]
        # print("Lead position",lead_pose)

        state_est = mcl.rgb_run(current_pose=lead_pose, timestep=0.1, lastpose=oldpositions)   
        oldpositions = state_est[:3]
        
        GT_state_history_x.append(lead_pose[0])
        GT_state_history_y.append(lead_pose[1])
        GT_state_history_z.append(lead_pose[2])
        
        pose_est_history_x.append(state_est[0])
        pose_est_history_y.append(state_est[1])
        pose_est_history_z.append(state_est[2])
        velocity_est_history_x.append(state_est[3])
        velocity_est_history_y.append(state_est[4])
        velocity_est_history_z.append(state_est[5])

        PF_history_x.append(np.array(mcl.filter.particles['position'][:,0]).flatten())
        PF_history_y.append(np.array(mcl.filter.particles['position'][:,1]).flatten())
        PF_history_z.append(np.array(mcl.filter.particles['position'][:,2]).flatten())

        count += 1
        time.sleep(0.1)

        if count == 200:
            break

    pose_est_history_x=np.array(pose_est_history_x)
    pose_est_history_y=np.array(pose_est_history_y)
    pose_est_history_z=np.array(pose_est_history_z)
    PF_history_x = np.array(PF_history_x)
    PF_history_y = np.array(PF_history_y)
    PF_history_z = np.array(PF_history_z)
    GT_state_history_x = np.array(GT_state_history_x)
    GT_state_history_y = np.array(GT_state_history_y)
    GT_state_history_z = np.array(GT_state_history_z)
    # print(GT_state_history.shape)
    # print(pose_est_history_x)

    times = np.arange(0,len(pose_est_history_x))*timestep
    velocity_GT_x = (GT_state_history_x[1:]-GT_state_history_x[:-1])/timestep
    velocity_GT_y = (GT_state_history_y[1:]-GT_state_history_y[:-1])/timestep
    velocity_GT_z = (GT_state_history_z[1:]-GT_state_history_z[:-1])/timestep

    fig, (posx,posy,posz) = plt.subplots(3, 1, figsize=(14, 10))
    posx.plot(times, pose_est_history_x, label = "Filter Pos x")
    posx.plot(times, GT_state_history_x, label = "GT Pos x")
    # posx.set_ylim(-1,1)
    posx.legend()
    posy.plot(times, pose_est_history_y, label = "Filter Pos y")    
    posy.plot(times, GT_state_history_y, label = "GT Pos y")
    posy.legend()
    posz.plot(times, pose_est_history_z, label = "Filter Pos z")
    posz.plot(times, GT_state_history_z, label = "GT Pos z")
    posz.legend()

    fig, (velx,vely,velz) = plt.subplots(3, 1, figsize=(14, 10))
    velx.plot(times, velocity_est_history_x, label = "Filter Vel x")
    velx.plot(times[1:], velocity_GT_x, label = "GT Vel x")
    velx.set_ylim(-1,2)
    velx.legend()
    vely.plot(times, velocity_est_history_y, label = "Filter Vel y")    
    vely.plot(times[1:], velocity_GT_y, label = "GT Vel y")
    vely.legend()
    velz.plot(times, velocity_est_history_z, label = "Filter Vel z")
    velz.plot(times[1:], velocity_GT_z, label = "GT Vel z")
    velz.legend()
    
    plt.show()

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    # ax.plot(x, y, z, color='b')
    # t = np.linspace(0, 32, 1000)
    # x = mcl.ref_traj[:,0]
    # y = mcl.ref_traj[:,1]
    # z = mcl.ref_traj[:,2]
    plt.figure(1)
    # ax.plot(x,y,z, color = 'b')
    ax.plot(pose_est_history_x,pose_est_history_y,pose_est_history_z, '*',color = 'g',label='PF Estimate state')
    ax.plot(GT_state_history_x,GT_state_history_y,GT_state_history_z, color = 'r',label='GT state')
    ax.plot(lead_pose1[0],lead_pose1[1],lead_pose1[2],'*', color = 'b')
    ax.plot(PF_history_x[0],PF_history_y[0],PF_history_z[0],'*', color = 'b')
    ax.plot(PF_history_x[1],PF_history_y[1],PF_history_z[1],'*', color = 'purple')
    plt.legend()
    plt.show()

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


