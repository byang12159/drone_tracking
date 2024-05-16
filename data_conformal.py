import airsim
import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import csv

import threading

import traceback

from hellodrone import Simulation

class Excitation(Simulation):
    def __init__(self, Vx_effort, Vy_effort, Vz_effort):
        super().__init__()

        starttime = time.time()

        initalization_pose = airsim.Vector3r(0,0,10)
        self.client1.simSetVehiclePose(initalization_pose, ignore_collision=True, vehicle_name=self.lead)
        time.sleep(1)

        self.min_bounds = {'Vx':-9.0,'Vy':-9.0,'Vz':-9.0}
        self.max_bounds = {'Vx':9.0, 'Vy':9.0, 'Vz':9.0}

        self.data_lead = []

        sample_Vx = []
        sample_Yy = []
        sample_Vz = []

        self.Vx_effort = Vx_effort
        self.Vy_effort = Vy_effort
        self.Vz_effort = Vz_effort

    
    def move_lead(self, dt=0.1):
        try:
            
            # x=0-5
            # y=0 # YK
            # z=34.27  
            # center = airsim.Vector3r(0, y, z) 
            # radius = 10 
            # num_waypoints = 30
            # waypoints = []

            # lead_pose = self.client1.simGetVehiclePose(self.lead).position
            # lead_NED = [lead_pose.x_val, lead_pose.y_val,lead_pose.z_val]
            # print(lead_NED)
            # # for i in range(num_waypoints):
            # #     angle = 2 * np.pi * (i / (num_waypoints - 1))
            # #     x = center.x_val + radius * np.cos(angle)
            # #     y = center.y_val + radius * np.sin(angle)
            # #     z = center.z_val  # Maintain same altitude
            # #     waypoint = airsim.Vector3r(x, y, z)
            # #     waypoints.append(waypoint)

            # waypoint = airsim.Vector3r(lead_NED[0]+10,lead_NED[1]+10,lead_NED[2])
            # waypoints.append(waypoint)
            # waypoint = airsim.Vector3r(lead_NED[0]-10,lead_NED[1]+10,lead_NED[2])
            # waypoints.append(waypoint)

            # vel = 0.5
            # self.client1.moveOnPathAsync(waypoints, vel, airsim.YawMode(False,0), lookahead=1, adaptive_lookahead=1, vehicle_name = self.lead)

            for i in range(500):
                # print("CALLING LEAD")
                self.client1.moveByVelocityBodyFrameAsync(self.Vx_effort,self.Vy_effort,self.Vz_effort, self.timestep, vehicle_name = self.lead)
                time.sleep(0.01)
        
        except Exception as e:
            # Print the error message
            print(f"An error occurred: {e}")
            stop_flag.set()

        finally:
            stop_flag.set()  # Signal the other thread to stop
        
    def get_NED(self, vehicle_name, client, get_quaternion=False):
        # Return options: angular_acceleration, angular_velocity, linear_acceleration, linear_velocity, orientation, position

        GT_states = client.getMultirotorState(vehicle_name).kinematics_estimated
        
        position =              [GT_states.position.x_val,             GT_states.position.y_val,            GT_states.position.z_val]
        orientation =           [GT_states.orientation.w_val,          GT_states.orientation.x_val,         GT_states.orientation.y_val,          GT_states.orientation.z_val]
        linear_velocity =       [GT_states.linear_velocity.x_val,      GT_states.linear_velocity.y_val,     GT_states.linear_velocity.z_val]
        linear_acceleration =   [GT_states.linear_acceleration.x_val,  GT_states.linear_acceleration.y_val, GT_states.linear_acceleration.z_val]
        angular_velocity =      [GT_states.angular_velocity.x_val,     GT_states.angular_velocity.y_val,    GT_states.angular_velocity.z_val]
        angular_acceleration =  [GT_states.angular_acceleration.x_val, GT_states.angular_acceleration.y_val,GT_states.angular_acceleration.z_val]

        return position + orientation+ linear_velocity+ linear_acceleration+ angular_velocity+ angular_acceleration
        
    def capture_data(self, dt=0.01):
        while not stop_flag.is_set():
            # print("CALLING  DATA")
            current_state = self.get_NED(vehicle_name=self.lead, client=self.client2)

            self.data_lead.append(current_state)


            time.sleep(dt)

    
if __name__=="__main__":

    stop_flag = threading.Event()
    data_storage = []
    filename = 'simulation_data_stationary_randomuniform.csv'

    # start = -5
    # stop = 5
    start = -9
    stop = 9
    increment = 1.0

    # Initialize an empty list to store the combinations
    combinations = []

    # Iterate through x, y, and z dimensions
    for x in [i * increment for i in range(int((stop - start) / increment) + 1)]:
        for y in [i * increment for i in range(int((stop - start) / increment) + 1)]:
            for z in [i * increment for i in range(int((stop - start) / increment) + 1)]:
                # Calculate the actual value of x, y, and z
                x_value = start + x * increment
                y_value = start + y * increment
                z_value = start + z * increment
                # Append the combination to the list
                combinations.append((x_value, y_value, z_value))



    # Number of samples
    num_samples = 1000

    # Define the range
    low = -9
    high = 9

    # Randomly sample x, y, z values
    x = np.random.uniform(low, high, num_samples)
    y = np.random.uniform(low, high, num_samples)
    z = np.random.uniform(low, high, num_samples)
    # Combine x, y, z into a single array of shape (num_samples, 3)
    coordinates = np.column_stack((x, y, z))
    # Print the generated coordinates
    print(coordinates)



    for i in range(num_samples):
        sim = Excitation(Vx_effort=coordinates[i][0], Vy_effort=coordinates[i][1], Vz_effort=coordinates[i][2])

        error_event = threading.Event()
        threadL = threading.Thread(target=sim.move_lead, name='Thread lead')
        threadC = threading.Thread(target=sim.capture_data, name='Thread Data')

        try:

            print(f"####### STARTING Simulation {i} #######")
            
            # Start the threads
            threadL.start()
            threadC.start()

            # If one of the threads terminates, this loop will exit
            # and we can join the other thread to ensure both are finished
            threadL.join()
            threadC.join()

            # Terminate Simulation

            # Add title for the iteration
            title = [f"Iteration {i}","Vx:",coordinates[i][0],"Vy:",coordinates[i][1],"Vz:", coordinates[i][2]]
            # Append the title and iteration data to the CSV file
            with open(filename, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(title)  # Write title row
                csvwriter.writerows(sim.data_lead)  # Write data rows

            data_storage.append(sim.data_lead)
            # write_csv(data_storage, output_file="lead_output.csv")

            sim.client1.reset()
            sim.client1.armDisarm(False)
            sim.client1.enableApiControl(False)
            # sim.client2.reset()
            # sim.client2.armDisarm(False)
            # sim.client2.enableApiControl(False)

            stop_flag.clear() 

        except Exception as e:
            print("Error Occured, Canceling: ",e)
            traceback.print_exc()

            sim.client1.reset()
            sim.client1.armDisarm(False)
            sim.client1.enableApiControl(False)
            sim.client2.reset()
            sim.client2.armDisarm(False)
            sim.client2.enableApiControl(False)

        print("time elaspsed",time.time()-sim.start_time)
        print("DONE")