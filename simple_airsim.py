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

        print("lead pose: ",client.simGetObjectPose(lead))
        print("chase pose: ",client.simGetObjectPose(chase))
        # client.moveByRollPitchYawThrottleAsync(roll=0, pitch=30, yaw=0, throttle=0.5, duration=2, vehicle_name = lead)
        # time.sleep(0.2)
        # print("object pose: ",client.simGetObjectPose(lead))
        # # print(f"1: {pos}, 2:{vel}, 3:{acel}")
        # time.sleep(4)
        # Define the positions
        leader_pos = np.array([5000.0, 0.0, 1500.0])  # Leader position
        chaser_pos = np.array([0, 0.0, 0.0])  # Chaser position
        leader_quat = [1, 0, 0, 0]  # w, x, y, z for the leader
        chaser_quat = [1, 0, 0, 0]

        transformation_matrix = vis.get_transform(leader_pos, leader_quat, chaser_pos, chaser_quat)
        print("Trans",transformation_matrix)
        vis.get_image(transformation_matrix)


        client.reset()
    
    #     count = 0
    #     totalcount = 1000

    #     # Initialize PID controller with desired gains and setpoint
    #     print("current pos",current_pos)
    #     step_target = [current_pos[0]+1,current_pos[1]+1,current_pos[2]+1]
    #     print("step target ",step_target)
    #     world = client.simGetObjectPose(lead)
    #     print("world pose, ",world  )
    #     # Z Kp=2, Ki=0.0, Kd=4.0
    #     # X: Kp=25, Ki=0.0, Kd=0,  testing: kd = 0.5 too much , 0.2 ok 
    #     # Y: Kp=25, Ki=0.0, Kd=0
    #     pid_controller_x = PIDController_x(Kp=25, Ki=0.0, Kd=50.0, setpoint=current_pos[0]+1)
    #     pid_controller_y = PIDController_y(Kp=25, Ki=0.0, Kd=50.0, setpoint=current_pos[1]+1)
    #     pid_controller_z = PIDController_z(Kp=2, Ki=0.0, Kd=4, setpoint=current_pos[2]+1)
    #     start_time = time.time()
    #     last_time = start_time
    #     # Simulation loop
    #     timestep = 0.01  # Time step
    #     dt = 0.1

    #     position_history = []
    

    #     # Create a figure and axis
    #     fig, ax = plt.subplots()
    #     line, = ax.plot([], [])  # Empty line for initial plot

    #     sliding_window_size = 600
    #     time_window = np.arange(0,sliding_window_size,1)
    #     position_window = [0] * sliding_window_size

    # try:

    #     while True:
    #         current_velocity = [client.getMultirotorState(lead).kinematics_estimated.linear_velocity.x_val, client.getMultirotorState(lead).kinematics_estimated.linear_velocity.y_val,client.getMultirotorState(lead).kinematics_estimated.linear_velocity.z_val]
            
    #         current_pos = [client.getMultirotorState(lead).kinematics_estimated.position.x_val, client.getMultirotorState(lead).kinematics_estimated.position.y_val, client.getMultirotorState(lead).kinematics_estimated.position.z_val]
    
    #         # Compute control signal using PID controller
    #         control_signal = [
    #         pid_controller_x.update(current_pos[0], dt),
    #         pid_controller_y.update(current_pos[1], dt),
    #         pid_controller_z.update(current_pos[2], dt)
    #         ]

    #         # Update quadrotor velocity using control signal
    #         current_velocity[0] += control_signal[0] * dt
    #         current_velocity[1] += control_signal[1] * dt
    #         current_velocity[2] += control_signal[2] * dt

    #         # client.moveByVelocityAsync(current_velocity[0],current_velocity[1],current_velocity[2], timestep, vehicle_name = lead)
    #         client.moveToPositionAsync(step_target[0],step_target[1],step_target[2],velocity=1,timeout_sec=timestep, vehicle_name=lead)
            


    #         count += 1
    #         curr_time = time.time()
    #         print(f"Total simulation time: {round(curr_time-start_time,4)} sec")
    #         time.sleep(timestep)

    #         if count == totalcount:
    #             break

    #         position_history.append(current_pos)

    #         # Update the plot
    #         time_window =np.append(time_window[1:],(time_window[-1]+1))
    #         position_window = np.append(position_window[1:],current_pos[0])
    #         line.set_data(time_window, position_window)
    #         ax.relim()
        
    #         ax.autoscale_view(True, True, True)
    #         plt.draw()
    #         plt.pause(0.01)  # Pause to allow the plot to update, adjust the value as needed

    #     print("Finished")
    #     world = client.simGetObjectPose(lead)
    #     print("world pose, ",world  )
    #     client.reset()
    #     client.armDisarm(False)
    #     setpoint=1.0
    #     # that's enough fun for now. let's quit cleanly
    #     client.enableApiControl(False)

    #     position_history=np.array(position_history)
        


    #     times = np.arange(0,position_history.shape[0])*timestep
    #     fig, (posx,posy,posz) = plt.subplots(3, 1, figsize=(14, 10))

    #     posx.plot(times, position_history[:,0], label = "Pos x")
    #     posx.legend()
    #     posy.plot(times, position_history[:,1], label = "Pos y")
    #     posy.legend()
    #     posz.plot(times, position_history[:,2], label = "Pos z")    
    #     posy.legend()

    #     # overshoot, settling_time, rise_time = calculate_performance_metrics(position_history, setpoint, 2, timestep)
    #     # print("Overshoot:", overshoot, "%")
    #     # print("Settling Time:", settling_time, "seconds")
    #     # print("Rise Time:", rise_time, "seconds")

    #     # posx.axhline(y=setpoint, color='k', linestyle='--', label="Setpoint")  # setpoint line
    #     # posx.axhline(y=setpoint * (1 + overshoot / 100), color='r', linestyle='--', label="Overshoot")  # Mark overshoot
    #     # posx.axvline(x=rise_time, color='g', linestyle='--', label="Rise Time")  # Mark rise time
    #     # posx.axvline(x=settling_time, color='b', linestyle='--', label="Settling Time")  # Mark settling time
    #     # posx.legend()



    #     plt.show()

    #     print("Finished")

    except Exception as e:
        print("Error Occured, Canceling: ",e)
        traceback.print_exc()

        client.reset()
        client.armDisarm(False)

        # that's enough fun for now. let's quit cleanly
        client.enableApiControl(False)