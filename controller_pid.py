class PIDController_x:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0

    def update(self, current_value, dt):
        error = self.setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

class PIDController_y:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0

    def update(self, current_value, dt):
        error = self.setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output
    
class PIDController_z:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.prev_error = 0
        self.integral = 0

    def update(self, current_value, dt):
        error = self.setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

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


import matplotlib.animation as animation

lead = "Drone_L"

if __name__ == "__main__":


    # connect to the AirSim simulator
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.reset()

    client.enableApiControl(True,lead)
    client.armDisarm(True, lead)
    client.takeoffAsync(20.0, lead).join()

    total = client.getMultirotorState(lead).kinematics_estimated
    current_pos = [client.getMultirotorState(lead).kinematics_estimated.position.x_val, client.getMultirotorState(lead).kinematics_estimated.position.y_val, client.getMultirotorState(lead).kinematics_estimated.position.z_val]
    current_velocity = client.getMultirotorState(lead).kinematics_estimated.linear_velocity
    acel = client.getMultirotorState(lead).kinematics_estimated.linear_acceleration
    print("kinematics",total)
    # print(f"1: {pos}, 2:{vel}, 3:{acel}")
 
    count = 0
    totalcount = 1000

    # Initialize PID controller with desired gains and setpoint
    print("current pos",current_pos)

    # Z Kp=8, Ki=0.0, Kd=12.0
    # X: Kp=25, Ki=0.0, Kd=0,  testing: kd = 0.5 too much , 0.2 ok 
    # Y: Kp=25, Ki=0.0, Kd=0
    pid_controller_x = PIDController_x(Kp=0.0, Ki=0.0, Kd=0, setpoint=10.0)
    pid_controller_y = PIDController_y(Kp=0.0, Ki=0.0, Kd=0, setpoint=10.0)
    pid_controller_z = PIDController_z(Kp=2, Ki=0.0, Kd=4, setpoint=10.0)
    start_time = time.time()
    last_time = start_time
    # Simulation loop
    timestep = 0.01  # Time step
    dt = 0.1

    position_history = []
   

    # Create a figure and axis
    fig, ax = plt.subplots()
    line, = ax.plot([], [])  # Empty line for initial plot

    sliding_window_size = 600
    time_window = np.arange(0,sliding_window_size,1)
    position_window = [0] * sliding_window_size

try:

    while True:
        current_velocity = [client.getMultirotorState(lead).kinematics_estimated.linear_velocity.x_val, client.getMultirotorState(lead).kinematics_estimated.linear_velocity.y_val,client.getMultirotorState(lead).kinematics_estimated.linear_velocity.z_val]
        
        current_pos = [client.getMultirotorState(lead).kinematics_estimated.position.x_val, client.getMultirotorState(lead).kinematics_estimated.position.y_val, client.getMultirotorState(lead).kinematics_estimated.position.z_val]
   
        # Compute control signal using PID controller
        control_signal = [
        pid_controller_x.update(current_pos[0], dt),
        pid_controller_y.update(current_pos[1], dt),
        pid_controller_z.update(current_pos[2], dt)
        ]

        # Update quadrotor velocity using control signal
        current_velocity[0] += control_signal[0] * dt
        current_velocity[1] += control_signal[1] * dt
        current_velocity[2] += control_signal[2] * dt
    
        # client.moveByVelocityZBodyFrameAsync(current_velocity[0],current_velocity[1],10, timestep, vehicle_name = lead)
        client.moveByVelocityAsync(current_velocity[0],current_velocity[1],current_velocity[2], timestep, vehicle_name = lead)
        

        count += 1
        curr_time = time.time()
        print(f"Total simulation time: {round(curr_time-start_time,4)} sec")
        time.sleep(timestep)

        if count == totalcount:
            break

        position_history.append(current_pos)

        # Update the plot
        time_window =np.append(time_window[1:],(time_window[-1]+1))
        position_window = np.append(position_window[1:],current_pos[2])
        line.set_data(time_window, position_window)
        ax.relim()
    
        ax.autoscale_view(True, True, True)
        plt.draw()
        plt.pause(0.01)  # Pause to allow the plot to update, adjust the value as needed

    print("Finished")
    client.reset()
    client.armDisarm(False)

    # that's enough fun for now. let's quit cleanly
    client.enableApiControl(False)

    position_history=np.array(position_history)
    times = np.arange(0,position_history.shape[0])*timestep
    fig, (posx,posy,posz) = plt.subplots(3, 1, figsize=(14, 10))
    posx.plot(times, position_history[:,0], label = "Pos x")
    # posx.plot(times, GT_state_history[:,0], label = "GT Pos x")
    posx.legend()
    posy.plot(times, position_history[:,1], label = "Pos y")
    # velx.plot(times[1:], velocity_GT_x, label = "GT Vel x")
    # velx.legend()
    posz.plot(times, position_history[:,2], label = "Pos z")    
    # accelx.plot(times[2:], accel_GT_x, label = "GT accel x")
    # accelx.legend()
    plt.show()

    print("Finished")

except Exception as e:
    print("Error Occured, Canceling: ",e)
    traceback.print_exc()

    client.reset()
    client.armDisarm(False)

    # that's enough fun for now. let's quit cleanly
    client.enableApiControl(False)
