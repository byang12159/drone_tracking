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
import threading
lead = "Drone_L"
chase = "Drone_C"


client = airsim.MultirotorClient()
client.confirmConnection()

client.enableApiControl(True,lead)
client.armDisarm(True, lead)
client.takeoffAsync(30.0, lead).join()

client.enableApiControl(True,chase)
client.armDisarm(True, chase)
client.takeoffAsync(30.0, chase).join()

count = 0
# lead_pose1 = [client.simGetVehiclePose(lead).position.x_val, client.simGetVehiclePose(lead).position.y_val, client.simGetVehiclePose(lead).position.z_val]
lead_pose1 = [client.simGetVehiclePose(lead).position.x_val, client.simGetVehiclePose(lead).position.y_val,client.simGetVehiclePose(lead).position.z_val]
lead_pose1_chase = [client.simGetVehiclePose(chase).position.x_val, client.simGetVehiclePose(chase).position.y_val,client.simGetVehiclePose(chase).position.z_val]
print("Lead position",lead_pose1)