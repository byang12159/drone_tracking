import airsim
import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import traceback
import random
import matplotlib.animation as animation
from pyvista_visualiser import Perception_simulation
lead = "Drone_L"
chase = "Drone_C"
if __name__ == "__main__":
    try: 
        vis = Perception_simulation()


        # connect to the AirSim simulator
        client = airsim.MultirotorClient()
        client.confirmConnection()

        client.enableApiControl(True,lead)
        client.armDisarm(True, lead)
        client.takeoffAsync(20.0, lead).join()

        client.enableApiControl(True,chase)
        client.armDisarm(True, chase)
        client.takeoffAsync(20.0, chase).join()

        # total = client.getMultirotorState(lead).kinematics_estimated
        # current_pos = [client.getMultirotorState(lead).kinematics_estimated.position.x_val, client.getMultirotorState(lead).kinematics_estimated.position.y_val, client.getMultirotorState(lead).kinematics_estimated.position.z_val]
        # current_velocity = client.getMultirotorState(lead).kinematics_estimated.linear_velocity
        # acel = client.getMultirotorState(lead).kinematics_estimated.linear_acceleration
        # print("kinematics",total)
        print("sim",client.simGetVehiclePose(chase))
        print("start lead pose: ",client.simGetObjectPose(lead))
        print(" start chase pose: ",client.simGetObjectPose(chase))
        # waypoints = [
        #         airsim.Vector3r(0, 0 , 2),
        #         airsim.Vector3r(0, 2, 2),
        #         airsim.Vector3r(2, 2, 2),
        #         airsim.Vector3r(0,  2, 2)]

        # client.moveOnPathAsync(waypoints, 1, 4 ,airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,0), 1, 1, vehicle_name=lead).join()


        # client.moveByRollPitchYawThrottleAsync(roll=0, pitch=30, yaw=0, throttle=0.5, duration=2, vehicle_name = lead)
        # time.sleep(0.2)
        # print("object pose: ",client.simGetObjectPose(lead))
        # # print(f"1: {pos}, 2:{vel}, 3:{acel}")
        # time.sleep(4)
        # Define the positions
        start = time.time()
        lead_pose = client.simGetObjectPose(lead)
        chase_pose = client.simGetObjectPose(chase)

        leader_pos = np.array([lead_pose.position.x_val*1000, lead_pose.position.y_val*1000, lead_pose.position.z_val*1000 ])  # Leader position
        chaser_pos = np.array([chase_pose.position.x_val*1000, chase_pose.position.y_val*1000, chase_pose.position.z_val*1000 ])  # Chaser position
        leader_quat = np.array([lead_pose.orientation.w_val, lead_pose.orientation.x_val, lead_pose.orientation.y_val, -1 * lead_pose.orientation.z_val ])  # w, x, y, z for the leader
        chaser_quat = np.array([chase_pose.orientation.w_val, chase_pose.orientation.x_val, chase_pose.orientation.y_val, -1 * chase_pose.orientation.z_val ])

        transformation_matrix = vis.get_transform(leader_pos, leader_quat, chaser_pos, chaser_quat)
        print("Trans",transformation_matrix)
        difference = vis.get_image(transformation_matrix)
        difference = np.array(difference)/1000

        end_time = time.time()
        print("Time interval ",end_time-start)
        chase_state = client.simGetVehiclePose(chase).position
        client.moveToPositionAsync(difference[0]-chase_state.x_val,difference[1]-chase_state.y_val, chase_state.z_val-difference[2],2,vehicle_name=chase).join()
        print(client.simGetObjectPose(chase))
        print(client.simGetVehiclePose(chase))
        time.sleep(4)
        # client.moveToPositionAsync(2,2,0,2,vehicle_name=chase).join()
        # print("object" ,client.simGetObjectPose(chase))
        # print("sim",client.simGetVehiclePose(chase))

        client.reset()
    


    except Exception as e:
        print("Error Occured, Canceling: ",e)
        traceback.print_exc()

        client.reset()
        client.armDisarm(False)

        # that's enough fun for now. let's quit cleanly
        client.enableApiControl(False)