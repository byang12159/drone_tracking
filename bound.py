from simple_excitation import excitation
import airsim
import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from samplecircle import samplecircle
from hellodrone import simulation
import traceback
# data = {"name": "John", "age": 30}

import pickle
# File path to save the data
file_path = 'data.pickle'


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
                [z[connection[0]], z[connection[1]]], '-',color='red')
def prediction(initial_state, timestep, steps, camera_depth,accel_range, num_trajectory = 100):
    
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
        # ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], '-x',color='b')
    

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
    # ax.set_zlabel("Z")
    # plt.show()

    # Find predict trajectory by average
    predict_trajectory = []
    for s in range(steps+1):
        x_val = np.mean(total_trajectories[:,s,0])
        y_val = np.mean(total_trajectories[:,s,1])
        z_val = np.mean(total_trajectories[:,s,2])
        predict_trajectory.append([x_val, y_val, z_val])
    



    return len(rectangle), total_trajectories, np.array(predict_trajectory)

if __name__ == "__main__":
    variable_a = np.arange(2,20.5,0.5)
    data = []
    for set_a in variable_a:
        lookahead_steps=[]
        raw_norms=[]
        for simulation_iteration in range(100):
            
            sim = simulation(totalcount=35)
            # np.random.seed(42)
            count =0
            # lead = "Drone_L"
            # client = airsim.MultirotorClient()
            # client.confirmConnection()
            # client.enableApiControl(True,lead)
            # client.armDisarm(True, lead)
            # client.takeoffAsync(30.0, lead).join()
            try:
                client = sim.client2
                while True:
                    if count <=20:
                        efforty=2
                        client.moveByVelocityBodyFrameAsync(0,efforty,0, sim.timestep, vehicle_name = sim.lead)
                    else:
                        effort = np.random.uniform(-set_a*0.1, set_a*0.1, 3)
                        print("random effort ",effort)
                        client.moveByVelocityBodyFrameAsync(effort[0],effort[1],effort[2], sim.timestep, vehicle_name = sim.lead)

                    lead_pose = [client.simGetObjectPose(sim.lead).position.x_val, client.simGetObjectPose(sim.lead).position.y_val,client.simGetObjectPose(sim.lead).position.z_val]
                    state_est = sim.mcl.rgb_run(current_pose=lead_pose, past_states = sim.particle_state_est, time_step=sim.timestep)  

                    curr_time = time.time()
                    print(f"{count}, Total simulation time: {round(curr_time-sim.start_time,4)} sec")
                    
                    if count == 20:
                        print("Compute prediction")
                        print(f"PF state est: {state_est}")
                        lead_pose = client.simGetObjectPose(sim.lead).position
                        print("actual position" ,lead_pose.x_val, lead_pose.y_val,lead_pose.z_val)
                        camera_depth = 2
                        steps =15
                        num_steps, trajectories, predict_traj = prediction(initial_state=state_est, timestep=sim.timestep,steps=steps, camera_depth=camera_depth, accel_range=set_a, num_trajectory=50)
                        
                        

                    lead_pose = [client.simGetObjectPose(sim.lead).position.x_val,client.simGetObjectPose(sim.lead).position.y_val,client.simGetObjectPose(sim.lead).position.z_val]
                    chase_pose = [client.simGetObjectPose(sim.chase).position.x_val,client.simGetObjectPose(sim.chase).position.y_val,client.simGetObjectPose(sim.chase).position.z_val]
                
                    sim.global_state_history_L.append(lead_pose)
                    sim.global_state_history_C.append(chase_pose)
                    sim.particle_state_est.append(state_est)

                    count += 1
                    if count >= sim.totalcount:
                        break

                    time.sleep(sim.timestep)

                print("Finished")       

                sim.client1.reset()
                sim.client1.armDisarm(False)
                sim.client1.enableApiControl(False)
                sim.client2.reset()
                sim.client2.armDisarm(False)
                sim.client2.enableApiControl(False)

                sim.global_state_history_L = np.array(sim.global_state_history_L)
                sim.global_state_history_C = np.array(sim.global_state_history_C)
                sim.particle_state_est = np.array(sim.particle_state_est)
                
                # fig = plt.figure(1)
                # ax = fig.add_subplot(111, projection='3d')
                # ax.plot(sim.global_state_history_L[19:,0], sim.global_state_history_L[19:,1], sim.global_state_history_L[19:,2], '-x',color='purple')
                # ax.plot(predict_traj[:,0], predict_traj[:,1], predict_traj[:,2], '-o',color='red')
                # # for trajectory in trajectories:
                # #     ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], '-',color='b', alpha=0.35)
                # ax.axis('equal')
                # ax.set_xlabel('X')
                # ax.set_ylabel('Y')
                # ax.set_zlabel("Z")
                # plt.show()

                print("LENGS",len(sim.global_state_history_L[20:,0]),len(predict_traj[:,0]))
                print(sim.global_state_history_L[19:,0])
                print(predict_traj[:,0])

                difference = sim.global_state_history_L[19:,0:3]-predict_traj[:,0:3]
                row_norm = np.linalg.norm(difference, axis=1)
                print("deiff shap",difference.shape)
                print("rownorm",row_norm.shape)
                print(row_norm)

                raw_norms.append(row_norm)
                for i,r in enumerate(row_norm):
                    if r>0.5:
                        print("first instance exceed",i)
                        lookahead_steps.append(i)
                        break
                print("lookahead",lookahead_steps)
                
                
                
                # fig = plt.figure(1)
                # ax = fig.add_subplot(111, projection='3d')
                # ax.plot(sim.global_state_history_L[20:,0], sim.global_state_history_L[20:,1], sim.global_state_history_L[20:,2], '-o',color='green')
                # ax.plot(predict_traj[:,0], predict_traj[:,1], predict_traj[:,2], '-o',color='red')

                # ax.axis('equal')
                # ax.set_xlabel('X')
                # ax.set_ylabel('Y')
                # ax.set_zlabel("Z")
                # plt.show()

                # times = np.arange(0,sim.particle_state_est.shape[0]-2)*sim.timestep
                # fig, (posx,posy,posz) = plt.subplots(3, 1, figsize=(14, 10))
                # posx.plot(times, sim.particle_state_est[2:,0], label = "Filter Pos x")
                # posx.plot(times, sim.global_state_history_L[:,0], label = "GT Pos x")
                # posx.legend()
                # posy.plot(times, sim.particle_state_est[2:,1], label = "Filter Pos y")    
                # posy.plot(times, sim.global_state_history_L[:,1], label = "GT Pos y")
                # posy.legend()
                # posz.plot(times, sim.particle_state_est[2:,2], label = "Filter Pos z")
                # posz.plot(times, sim.global_state_history_L[:,2], label = "GT Pos z")
                # posz.legend()
                # plt.show()


            except Exception as e:
                print("Error Occured, Canceling: ",e)
                traceback.print_exc()
                
                sim.client1.reset()
                sim.client1.armDisarm(False)
                sim.client1.enableApiControl(False)
                sim.client2.reset()
                sim.client2.armDisarm(False)
                sim.client2.enableApiControl(False)

            # #MOVE IN NON X DIRECTION FOR V?????????????????
            # effort = excitation(2,2)
            # variable_V = np.linspace(0,20,5)
            # variable_A = np.linspace(0,44,50)
            # total_steps_predict = []
            # total_d = []

            # variable_Vx,variable_Vy,variable_Vz = samplecircle()
            # ######################################
            # for v in range(len(variable_Vx)):
            #     print("current v:",v)
                
            #     # initial_V = v
            #     timestep = 0.1

            #     initalstate = np.zeros(9)
            #     initalstate[3]=2
            #     # initalstate[4]=variable_Vy[v]
            #     # initalstate[5]=variable_Vz[v]

            #     d = np.linspace(0,3,50)
            #     steps_predict = []
            #     for i in d:
            #         num_steps = prediction(initial_state=initalstate, timestep=timestep,steps=30, camera_depth=i, accel_range=5)
            #         steps_predict.append(num_steps)
            #         plt.show()
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

            # # data=[variable_A,total_d,total_steps_predict]
            # # with open("data_variableV_44a.pkl", "wb") as file:
            # #     pickle.dump(data, file) 

            # for i in range(len(variable_Vx)):
            #     plt.plot(total_d[i],0.1*np.array(total_steps_predict[i]),'-', label=f"V: {variable_Vx[i],variable_Vy[i],variable_Vz[i]}")
                
            # plt.xlabel("Target  Distance from Camera (m)")
            # plt.ylabel("Future Time Within Cam Bounds (s)")
            # plt.title("System timstep:0.1s, Velocity 5m/s, max Accel:+-5m/s^2")
            # # plt.legend()q
            # plt.show()

            # # Varying Acceleration ######################################
            # # for v in variable_A:
            # #     print("current a:",v)
                
            # #     initial_V = 2
            # #     timestep = 0.1

            # #     initalstate = np.zeros(9)
            # #     initalstate[5]=initial_V

            # #     d = np.linspace(0,3,50)
            # #     steps_predict = []
            # #     for i in d:
            # #         num_steps = prediction(initial_state=initalstate, timestep=timestep,steps=30, camera_depth=i, accel_range=v)
            # #         steps_predict.append(num_steps)
                    
            # #     total_steps_predict.append(steps_predict)
            # #     total_d.append(d)
            # #     # plt.plot(d,steps_predict)
            # #     # plt.xlabel("Target  Distance from Cameqra (m)")
            # #     # plt.ylabel("# Future Steps Within Cam Bounds")
            # #     # plt.show()    
                
            # #     # plt.plot(d,0.1*np.array(steps_predict))
            # #     # plt.xlabel("Target  Distance from Camera (m)")
            # #     # plt.ylabel("Future Time Within Cam Bounds (s)")
            # #     # plt.title("System timstep:0.1s, Velocity:2m/s, max Accel:+-5m/s^2")
            # #     # plt.show()

            # # data=[variable_A,total_d,total_steps_predict]
            # # with open("data_variableA_44a.pkl", "wb") as file:
            # #     pickle.dump(data, file) 

            # # fig = plt.figure()
            # # ax = fig.add_subplot(111, projection='3d')
            # # for i in range(len(variable_A)):
            # #     vselect= np.ones_like(total_d[i])*variable_A[i]
            # #     ax.plot(total_d[i], 0.1*np.array(total_steps_predict[i]), vselect, 'x')

            # # # Add labels and title
            # # ax.set_xlabel("Target  Distance from Camera (m)")
            # # ax.set_ylabel("Future Time Within Cam Bounds (s)")
            # # ax.set_zlabel('Acceleration Bound (m/s^2)')
            # # ax.set_title('System timstep:0.1s, Initial Velocity:2m/s, Varying Accel')

            # # # Show plot
            # # plt.show()

    
        print("lookahaed",lookahead_steps)
        data.append(set_a)
        data.append(lookahead_steps)
        data.append(raw_norms)

        # Open the file in binary write mode
        with open(file_path, 'wb') as file:
            # Serialize and save the data to the file
            pickle.dump(data, file)

        print("Data has been saved to", file_path)
