import setup_path 
import airsim
import tempfile
import os
import numpy as np
import cv2
import pprint
import time
import threading
import random
import transformations

# drone_name should match the name in ~/Document/AirSim/settings.json
class BaselineRacer(object):
    def __init__(self, drone_name="Drone1", viz_traj=True, viz_traj_color_rgba=[1.0, 0.0, 0.0, 1.0], viz_image_cv2=True):
        self.drone_name = drone_name
        self.gate_poses_ground_truth = None
        self.viz_image_cv2 = viz_image_cv2
        self.viz_traj = viz_traj
        self.viz_traj_color_rgba = viz_traj_color_rgba

        self.airsim_client = airsim.MultirotorClient()
        self.airsim_client.confirmConnection()
        self.airsim_client.race_tier = 1
        # we need two airsim MultirotorClient objects because the comm lib we use (rpclib) is not thread safe
        # so we poll images in a thread using one airsim MultirotorClient object
        # and use another airsim MultirotorClient for querying state commands
        self.airsim_client_images = airsim.MultirotorClient()
        self.airsim_client_images.confirmConnection()
        self.airsim_client_odom = airsim.MultirotorClient()
        self.airsim_client_odom.confirmConnection()
        self.level_name = None

        # self.image_callback_thread = threading.Thread(
        #     target=self.repeat_timer_image_callback, args=(self.image_callback, 0.03)
        # )
        # self.odometry_callback_thread = threading.Thread(
        #     target=self.repeat_timer_odometry_callback,
        #     args=(self.odometry_callback, 0.02),
        # )
        # self.is_image_thread_active = False
        # self.is_odometry_thread_active = False
        
    def run(self, num_sample = 1000):

        client.takeoffAsync().join()
        client.moveToPositionAsync(-10, 10, -10, 5).join()

        # # Sample drone position
        # x = 1
        # y = 0
        # z = -10
        # # Set drone position
        # pose = self.airsim_client.simGetVehiclePose(
        #     vehicle_name=self.drone_name
        # )
        # pose.position.x_val = x 
        # pose.position.y_val = y 
        # pose.position.z_val = z 
        # q = transformations.quaternion_from_euler(0, 0, -1.371073)
        # pose.orientation.w_val = q[0]
        # pose.orientation.x_val = q[1]
        # pose.orientation.y_val = q[2]
        # pose.orientation.z_val = q[3]
        # self.airsim_client.simSetVehiclePose(pose, True, self.drone_name)
        # # self.airsim_client.simSetVehiclePose(pose=pose, ignore_collison=True, vehicle_name = self.drone_name)




def main():
    baseline_racer = BaselineRacer(drone_name="Drone1", viz_traj_color_rgba=[1.0, 1.0, 0.0, 1.0])
    baseline_racer.run()


if __name__ == '__main__':
    droneL = 'Drone1'
    droneC = 'Drone2'

    # connect to the AirSim simulator
    client = airsim.MultirotorClient()
    client.confirmConnection()

    client.enableApiControl(True, droneL)
    client.armDisarm(True, droneL)
    client.enableApiControl(True, droneC)
    client.armDisarm(True, droneC)

    time.sleep(3)

    main()

    # Async methods returns Future. Call join() to wait for task to complete.
    # client.takeoffAsync().join()
    # client.moveToPositionAsync(-10, 10, -10, 5), droneL.join()

    print("finished")