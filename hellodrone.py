# ready to run example: PythonClient/multirotor/hello_drone.py
# note: async methods take a long time to execute, add join() to wait for it finish 
# NED coordinates: +x:forward +y:right +z:down

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

from controller_pid import PIDController

event = threading.Event()
lock = threading.Lock()

count = 0
class simulation():
    def __init__(self, totalcount=1000):
        self.lead = "Drone_L"
        self.chase = "Drone_C"

        # connect to the AirSim simulator
        self.client1 = airsim.MultirotorClient()
        self.client1.confirmConnection()

        self.client2 = airsim.MultirotorClient()
        self.client2.confirmConnection()

        self.client1.enableApiControl(True,self.lead)
        self.client1.armDisarm(True, self.lead)
        self.client1.takeoffAsync(30.0, self.lead).join()

        self.client1.enableApiControl(True,self.chase)
        self.client1.armDisarm(True, self.chase)
        self.client1.takeoffAsync(30.0, self.chase).join()

        chase_kinematics = self.client1.getMultirotorState(self.chase).kinematics_estimated
        print("KINEMATICS",chase_kinematics)
        
        # Find Difference between global to NED coordinate frames
        lead_pose = self.client1.simGetObjectPose(self.lead).position
        lead_global = [lead_pose.x_val, lead_pose.y_val,lead_pose.z_val]
        lead_pose = self.client1.simGetVehiclePose(self.lead).position
        lead_NED = [lead_pose.x_val, lead_pose.y_val,lead_pose.z_val]
        self.lead_coord_diff = np.array(lead_NED) - np.array(lead_global)

        chase_pose = self.client1.simGetObjectPose(self.chase).position
        chase_global = [chase_pose.x_val, chase_pose.y_val,chase_pose.z_val]
        chase_pose = self.client1.simGetVehiclePose(self.chase).position
        chase_NED = [chase_pose.x_val, chase_pose.y_val,chase_pose.z_val]
        self.chase_coord_diff = np.array(chase_NED) - np.array(chase_global)

        print(lead_pose)

        self.mcl = RunParticle(starting_state=lead_global)    

        # Initialize mcl Position
        self.est_states = np.zeros((len(self.mcl.ref_traj) ,6)) # x y z vx vy vz
        self.gt_states  = np.zeros((len(self.mcl.ref_traj) ,16))

        pose_est_history_x = []
        pose_est_history_y = []
        pose_est_history_z = []
        velocity_est_history_x = []
        velocity_est_history_y =[]
        velocity_est_history_z = []
        self.PF_history_x = []
        self.PF_history_y = []
        self.PF_history_z = []
        self.PF_history_x.append(np.array(self.mcl.filter.particles['position'][:,0]).flatten())
        self.PF_history_y.append(np.array(self.mcl.filter.particles['position'][:,1]).flatten())
        self.PF_history_z.append(np.array(self.mcl.filter.particles['position'][:,2]).flatten())
 
        self.velocity_GT = []
        self.accel_GT = []

        self.global_state_history_L=[]
        self.global_state_history_C=[]
        self.particle_state_est=[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]

        # Assume constant time step between trajectory stepping
        self.timestep = 0.01
        self.totalcount = totalcount
        self.start_time = time.time()

        # Initialize PID
        gain_x = [20, 0, 80.0]
        gain_y = [20, 0, 80.0]
        gain_z = [2,  0, 20.0]
        self.pid = PIDController(gain_x=gain_x, gain_y=gain_y, gain_z=gain_z)
        # self.pid.update_setpoint([1,1,35])

    def global2NED(self,pose_global,vehicle_name):
        if vehicle_name == "Drone_C":
            return pose_global+self.lead_coord_diff
        else:
            return pose_global+self.chase_coord_diff
        
    def random_traj(self, i,total_count):
        x= 2* np.sin(i* 2*np.pi/total_count)
        y= np.cos(i*2*np.pi/total_count)
        z= 0.5*np.sin(i* 2*np.pi/total_count)
        return x,y,z


    def move_lead(self):
        global count 
        client = self.client2
        print("enter self.lead")
        while True:
            print("LEADER LOOP")
            effortx,efforty,effortz = self.random_traj(count,self.totalcount)
            
            client.moveByVelocityBodyFrameAsync(0,efforty,0, self.timestep, vehicle_name = self.lead)
            
            curr_time = time.time()
            print(f"Total simulation time: {round(curr_time-self.start_time,4)} sec")
            
            # time.sleep(self.timestep)
            
            with lock:
                count += 1
                if count >= self.totalcount:
                    break

            time.sleep(self.timestep)

    def move_chase(self):
        global count 
        client = self.client1

        def rotation_2d_z(angle, vector):
            #angle is radian
            R = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            return R@vector
        
        def angle_between(v1, v2):
            dot_product = np.dot(v1, v2)
            magnitude_v1 = np.linalg.norm(v1)
            magnitude_v2 = np.linalg.norm(v2)
            angle_rad = np.arccos(dot_product / (magnitude_v1 * magnitude_v2))
            angle_deg = np.degrees(angle_rad)
            print("degrees diff ",angle_deg)
            return angle_rad
        
        def movePID(chase_kinematics, lead_kinematics, target_point):
            # Compute control signal using PID controller
            dt = 0.1
            current_pos = np.array([chase_kinematics.position.x_val, chase_kinematics.position.y_val, chase_kinematics.position.z_val])
            current_velocity = np.array([chase_kinematics.linear_velocity.x_val, chase_kinematics.linear_velocity.y_val, chase_kinematics.linear_velocity.z_val])


            target_point_adjusted = np.array(target_point)-np.array([1,0,0])
            print("TARGET POINT1,",target_point_adjusted)

            lead_current_pos = np.array([lead_kinematics.position.x_val, lead_kinematics.position.y_val, lead_kinematics.position.z_val])
            target_point_adjusted = lead_current_pos-np.array([1,0,0])
            print("TARGET POINT2,",target_point_adjusted)

            self.pid.update_setpoint(target_point_adjusted)
            
            control_signal = self.pid.update(current_pos, dt)
            
            # Update quadrotor velocity using control signal
            current_velocity[0] += control_signal[0] * dt
            current_velocity[1] += control_signal[1] * dt
            current_velocity[2] += control_signal[2] * dt

            client.moveByVelocityAsync(current_velocity[0],current_velocity[1],current_velocity[2], self.timestep, vehicle_name = self.chase)
        
        while True:
            print("CHASER LOOP")
            lead_pose = [client.simGetObjectPose(self.lead).position.x_val, client.simGetObjectPose(self.lead).position.y_val,client.simGetObjectPose(self.lead).position.z_val]
            state_est = self.mcl.rgb_run(current_pose=lead_pose, past_states = self.particle_state_est, time_step=self.timestep)   
            
            
            chase_pose = [client.simGetObjectPose(self.chase).position.x_val,client.simGetObjectPose(self.chase).position.y_val,client.simGetObjectPose(self.chase).position.z_val]
        
            lead_kinematics = client.getMultirotorState(self.lead).kinematics_estimated
            chase_kinematics = client.getMultirotorState(self.chase).kinematics_estimated

            orient = rotation_2d_z(-chase_kinematics.orientation.z_val, np.array([1,0]))
            vector2 = np.array([lead_pose[0]-chase_pose[0],lead_pose[1]-chase_pose[1]])
            print(f"orient: {orient}, vecotr2: {vector2}")
            yaw_chase = -1*angle_between(orient, vector2)

            target_state = self.global2NED(state_est[:3],self.chase)
            print("MCL RERTURN",state_est[:3],target_state)
            # print("TARGET STATE ",target_state)

            # temp target state using GT
            # world_L = np.array([client.simGetObjectPose(self.lead).position.x_val,client.simGetObjectPose(self.lead).position.y_val,client.simGetObjectPose(self.lead).position.z_val])
            # world_C = np.array([client.simGetObjectPose(self.chase).position.x_val,client.simGetObjectPose(self.chase).position.y_val,client.simGetObjectPose(self.chase).position.z_val])
            # temptarget = world_L-world_C #- np.array([1,0,0])
            # print("temptarget",temptarget)
            movePID(chase_kinematics, lead_kinematics,target_state )
            # client.moveToPositionAsync(0, target_state[1], target_state[2], velocity=2, timeout_sec=self.timestep, yaw_mode=airsim.YawMode(False, yaw_chase),vehicle_name=self.chase)

            self.global_state_history_L.append(lead_pose)
            self.global_state_history_C.append(chase_pose)
            self.particle_state_est.append(state_est)
            self.velocity_GT.append([lead_kinematics.linear_velocity.x_val, 
                                lead_kinematics.linear_velocity.y_val,
                                lead_kinematics.linear_velocity.z_val])  
            self.accel_GT.append([lead_kinematics.linear_acceleration.x_val,
                                lead_kinematics.linear_acceleration.y_val,
                                lead_kinematics.linear_acceleration.z_val])
            self.PF_history_x.append(np.array(self.mcl.filter.particles['position'][:,0]).flatten())
            self.PF_history_y.append(np.array(self.mcl.filter.particles['position'][:,1]).flatten())
            self.PF_history_z.append(np.array(self.mcl.filter.particles['position'][:,2]).flatten())

            with lock:
                if count >= self.totalcount:
                    break
            
            time.sleep(self.timestep)

    def processing(self):
        self.global_state_history_L = np.array(self.global_state_history_L)
        self.global_state_history_C = np.array(self.global_state_history_C)
        self.particle_state_est = np.array(self.particle_state_est)
        
        self.PF_history_x = np.array(self.PF_history_x)
        self.PF_history_y = np.array(self.PF_history_y)
        self.PF_history_z = np.array(self.PF_history_z)

        self.velocity_GT= np.array(self.velocity_GT)
        self.accel_GT = np.array(self.accel_GT)

        times = np.arange(0,self.particle_state_est.shape[0]-2)*self.timestep


        # fig, (posx,posy,posz) = plt.subplots(3, 1, figsize=(14, 10))
        # posx.plot(times, particle_state_est[2:,0], label = "Filter Pos x")
        # posx.plot(times, global_state_history_L[:,0], label = "GT Pos x")
        # posx.legend()
        # posy.plot(times, particle_state_est[2:,1], label = "Filter Pos y")    
        # posy.plot(times, global_state_history_L[:,1], label = "GT Pos y")
        # posy.legend()
        # posz.plot(times, particle_state_est[2:,2], label = "Filter Pos z")
        # posz.plot(times, global_state_history_L[:,2], label = "GT Pos z")
        # posz.legend()

        # fig, (velx,vely,velz) = plt.subplots(3, 1, figsize=(14, 10))
        # velx.plot(times, particle_state_est[2:,3], label = "Filter Vel x")
        # velx.plot(times, velocity_GT[:,0], label = "GT Vel x")
        # # velx.set_ylim(-1,2)
        # velx.legend()
        # vely.plot(times, particle_state_est[2:,4], label = "Filter Vel y")    
        # vely.plot(times, velocity_GT[:,1], label = "GT Vel y")
        # vely.legend()
        # velz.plot(times, particle_state_est[2:,5], label = "Filter Vel z")
        # velz.plot(times, velocity_GT[:,2], label = "GT Vel z")
        # velz.legend()

        # fig, (posx,velx,accelx) = plt.subplots(3, 1, figsize=(14, 10))
        # posx.plot(times, particle_state_est[2:,0], label = "Filter Accel x")
        # posx.plot(times, global_state_history_L[:,0], label = "GT Accel x")
        # posx.legend()
        # velx.plot(times, particle_state_est[2:,3], label = "Filter Accel y")
        # velx.plot(times, velocity_GT[:,0], label = "GT Accel y")
        # velx.legend()
        # accelx.plot(times, particle_state_est[2:,6], label = "Filter Accel z")    
        # accelx.plot(times, accel_GT[:,0], label = "GT Accel z")
        # accelx.legend()

        # plt.show()


        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.global_state_history_C[:,0],self.global_state_history_C[:,1],self.global_state_history_C[:,2], color='b')
        ax.plot(self.particle_state_est[2:,0],self.particle_state_est[2:,1],self.particle_state_est[2:,2],'o',color='red')
        ax.plot(self.global_state_history_L[:,0],self.global_state_history_L[:,1],self.global_state_history_L[:,2], '*',color = 'g')
        plt.axis('equal')
        plt.legend()
        plt.show()



if __name__ == "__main__":
    sim = simulation()
    threadL = threading.Thread(target=sim.move_lead, name='Thread lead')
    threadC = threading.Thread(target=sim.move_chase, name='Thread Chase')

    try:
        print("###################################################################################### STARTING SIMULATION ##########################################################################")
        # Start the threads
        threadL.start()
        threadC.start()

        # Wait for both threads to finish
        threadL.join()
        threadC.join()

        sim.processing()

        print("Finished")       
        
        lead = "Drone_C"
        world = [sim.client1.simGetObjectPose(lead).position.x_val,sim.client1.simGetObjectPose(lead).position.y_val,sim.client1.simGetObjectPose(lead).position.z_val] 
        print("world pose, ",world  )
        chase_kinematics = sim.client1.getMultirotorState(lead).kinematics_estimated
        print("rel: ",chase_kinematics.position.x_val,chase_kinematics.position.y_val,chase_kinematics.position.z_val )
        lead = "Drone_L"
        world = [sim.client1.simGetObjectPose(lead).position.x_val,sim.client1.simGetObjectPose(lead).position.y_val,sim.client1.simGetObjectPose(lead).position.z_val] 
        print("world pose, ",world  )
        chase_kinematics = sim.client1.getMultirotorState(lead).kinematics_estimated
        print("rel: ",chase_kinematics.position.x_val,chase_kinematics.position.y_val,chase_kinematics.position.z_val )

        sim.client1.reset()
        sim.client1.armDisarm(False)
        sim.client1.enableApiControl(False)
        sim.client2.reset()
        sim.client2.armDisarm(False)
        sim.client2.enableApiControl(False)



    except Exception as e:
        print("Error Occured, Canceling: ",e)
        traceback.print_exc()

        sim.client1.reset()
        sim.client1.armDisarm(False)
        sim.client1.enableApiControl(False)
        sim.client2.reset()
        sim.client2.armDisarm(False)
        sim.client2.enableApiControl(False)





# def circle_traj(i,total_count):
#     radius = 10
#     start=lead_pose1
#     t = np.linspace(0,2*np.pi,totalcount)
#     x = lead_pose1[0]-radius - radius * np.cos(i* 2*np.pi/total_count)
#     y = lead_pose1[1]-radius * np.sin(i* 2*np.pi/total_count)
#     z= lead_pose1[2]
#     return x,y,z