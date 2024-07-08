# ready to run example: PythonClient/multirotor/hello_drone.py
# note: async methods take a long time to execute, add join() to wait for it finish 
# NED coordinates: +x:forward +y:right +z:down

# Particle filter with torch enabled PF. 
import airsim
import os
import time
import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from particle_filter_torch.particle_main import RunParticle
import traceback
import random
from controller_m.gen_traj import Generate
# from perception.perception import Perception # YK
from simple_excitation import excitation 
import threading
from pyvista_visualiser import Perception_simulation
from controller_pid import PIDController
# from graphing import graphing
import ctypes
from MC_Prediction import Prediction
import pickle
error_event = threading.Event()
lock = threading.Lock()

timelast = 0
timeinterval = 0.02
leadkinematiclast = None

def prepare_GT_NED_states(NED_state_history_L):
    GT_NED_states_L = []

    for j in range(len(NED_state_history_L)):
        posx_L = NED_state_history_L[j].position.x_val
        posy_L = NED_state_history_L[j].position.y_val
        posz_L = NED_state_history_L[j].position.z_val
        velx_L = NED_state_history_L[j].linear_velocity.x_val
        vely_L = NED_state_history_L[j].linear_velocity.y_val
        velz_L = NED_state_history_L[j].linear_velocity.z_val
        aclx_L = NED_state_history_L[j].linear_acceleration.x_val
        acly_L = NED_state_history_L[j].linear_acceleration.y_val
        aclz_L = NED_state_history_L[j].linear_acceleration.z_val
   

        GT_NED_states_L.append([posx_L,posy_L,posz_L,velx_L,vely_L,velz_L,aclx_L,acly_L,aclz_L])
       

    GT_NED_states_L = np.array(GT_NED_states_L)
 

    return GT_NED_states_L

with open('data_global_lead.pkl', 'rb') as file:
    global_state_history_L = pickle.load(file)
with open('data_NED_lead.pkl', 'rb') as file:
    NED_state_history_L = pickle.load(file)

GT_global = np.array(global_state_history_L)
GT_NED = prepare_GT_NED_states(NED_state_history_L)

positions = [np.array([0,0,0])]
for i in range(GT_NED.shape[0]):
    positions.append(positions[-1] + 0.01 *np.array([GT_NED[i,3],GT_NED[i,4],GT_NED[i,5]]))
positions = np.array(positions[1:])

times = np.arange(0, GT_global.shape[0])*0.01
fig, axs = plt.subplots(3, 2, figsize=(14, 10))

axs[0,0].plot(times, positions[:,0], label = "GT Position x")
axs[0,0].legend()
axs[1,0].plot(times, positions[:,1], label = "GT Position y")  
axs[1,0].legend()
axs[2,0].plot(times, positions[:,2], label = "GT Position z")
axs[2,0].legend()


axs[0,1].plot(times, GT_NED[:,3], label = "GT Position vx")
axs[0,1].legend()
axs[1,1].plot(times, GT_NED[:,4], label = "GT Position vy")  
axs[1,1].legend()
axs[2,1].plot(times, GT_NED[:,5], label = "GT Position vz")
axs[2,1].legend()
plt.show()


class simulation():
    def __init__(self, totalcount=1000, num_particles = 900):
        global leadkinematiclast, positions

        self.count = 0

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

        # self.client1.moveToPositionAsync(10, 0, -10, 5, vehicle_name=self.lead)
        # self.client1.moveToPositionAsync(5, 0, -10, 5, vehicle_name=self.chase).join()

        chase_kinematics = self.client1.getMultirotorState(self.chase).kinematics_estimated
        lead_kinematics = self.client1.getMultirotorState(self.lead).kinematics_estimated

        leadkinematiclast = lead_kinematics

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

        lead_global = [lead_kinematics.position.x_val, lead_kinematics.position.y_val, lead_kinematics.position.z_val]

        lead_global = positions[0]
        self.mcl = RunParticle(starting_state=lead_global, num_particles=num_particles)   
        self.prediction = Prediction()


        pose_est_history_x = []
        pose_est_history_y = []
        pose_est_history_z = []
        velocity_est_history_x = []
        velocity_est_history_y =[]
        velocity_est_history_z = []
 
        self.velocity_GT = []
        self.accel_GT = []

        self.global_state_history_L=[]
        self.global_state_history_C=[]
        self.NED_state_history_L=[]
        self.NED_state_history_C=[]
        self.particle_state_est=[np.array([0,0,0,0,0,0]),np.array([0,0,0,0,0,0])]

        self.variance_history = []
        self.prediction_history = []
        self.PF_history = []
        A = np.array(self.mcl.filter.particles['position'].cpu())
        B = np.array(self.mcl.filter.particles['velocity'].cpu())
        self.PF_history.append((A,B))

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


    # Lead Drone Movement in Figure 8
    # def xyzARG(args, real_t):
    #     period, sizex, sizey = args
    #     # print('period/sizex/sizey: ', period, sizex, sizey)
    #     if not period:
    #         period = real_t[-1]
    #     t = real_t / period * 2 * np.pi
    #     x = np.sqrt(2) * np.cos(t) / (1 + np.sin(t) ** 2)
    #     y = x * np.sin(t)
    #     x = sizex * x
    #     y = sizey * y
    #     z = np.ones_like(x) * 1.5
    #     return x, y, z


    def move_lead(self):
        client = self.client2

        print("enter self.lead")
        mode = "simple_vel"

        try:
            time.sleep(2)
            while not error_event.is_set():
                if mode == "simple_vel":
                    # print("LEADER LOOP")
                    effortx,efforty,effortz = self.random_traj(self.count,self.totalcount)

                    # client.moveOnPathAsync(lead_waypoints, 5, 20 ,airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False,0), 1, 1, vehicle_name = self.lead)

                    # client.moveByVelocityBodyFrameAsync(0,efforty,0, self.timestep, vehicle_name = self.lead)
                    client.moveByVelocityBodyFrameAsync(0,3,0, self.timestep, vehicle_name = self.lead)
                    curr_time = time.time()
                    # print(f"Total simulation time: {round(curr_time-self.start_time,4)} sec")
                    
                    # time.sleep(self.timestep)
                    
                    with lock:
                        self.count += 1
                        if self.count >= self.totalcount:
                            break

                    time.sleep(self.timestep)

                if mode == "waypoint":
                    center = airsim.Vector3r(0, 0, 34.27 ) 
                    waypoints = []
                    waypoints.append(center)
                    for cnt in range(300):
                        period=100
                        sizex=2.5
                        sizey=2.5
                        t = cnt / period * 2 * np.pi
                        x = np.sqrt(2) * np.cos(t) / (1 + np.sin(t) ** 2)
                        y = x * np.sin(t)
                        x = sizex * x
                        y = sizey * y
                        # z = np.ones_like(x) * 1.5
                        waypoint = airsim.Vector3r(x, y, 34.27) 
                        waypoints.append(waypoint) 
                    client.moveOnPathAsync(waypoints, 1, 60 ,airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False,0), 1, 1, vehicle_name = self.lead)                 

        except Exception as e:
            print(f"Exception in thread 1: {e}")
            traceback.print_exc()
            error_event.set()  # Signal the error event
            
        # x=0-5
        # y=0 # YK
        # z=34.27  
        # center = airsim.Vector3r(0, y, z) 
        # radius = 10 
        # num_waypoints = 30
        # waypoints = []
        # for i in range(num_waypoints):
        #     angle = 2 * np.pi * (i / (num_waypoints - 1))
        #     x = center.x_val + radius * np.cos(angle)
        #     y = center.y_val + radius * np.sin(angle)
        #     z = center.z_val  # Maintain same altitude
        #     waypoint = airsim.Vector3r(x, y, z)
        #     waypoints.append(waypoint)
        # client.moveOnPathAsync(waypoints, 1, 90 ,airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False,0), 1, 1, vehicle_name = self.lead)
       
    def move_chase(self):
        global timelast, timeinterval, leadkinematiclast, global_state_history_L, positions
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
            # print("degrees diff ",angle_deg)
            return angle_rad
        
        def movePID(chase_kinematics, lead_kinematics, target_point):
            # Compute control signal using PID controller
            dt = 0.1
            current_pos = np.array([chase_kinematics.position.x_val, chase_kinematics.position.y_val, chase_kinematics.position.z_val])
            current_velocity = np.array([chase_kinematics.linear_velocity.x_val, chase_kinematics.linear_velocity.y_val, chase_kinematics.linear_velocity.z_val])


            target_point_adjusted = np.array(target_point)-np.array([1,0,0])
            # print("TARGET POINT1,",target_point_adjusted)

            lead_current_pos = np.array([lead_kinematics.position.x_val, lead_kinematics.position.y_val, lead_kinematics.position.z_val])
            target_point_adjusted = lead_current_pos-np.array([3,0,0])
            # print("TARGET POINT2,",target_point_adjusted)

            self.pid.update_setpoint(target_point_adjusted)
            
            control_signal = self.pid.update(current_pos, dt)
            
            # Update quadrotor velocity using control signal
            current_velocity[0] += control_signal[0] * dt
            current_velocity[1] += control_signal[1] * dt
            current_velocity[2] += control_signal[2] * dt
            # print(current_velocity)

            client.moveByVelocityAsync(current_velocity[0],current_velocity[1],current_velocity[2], self.timestep, airsim.DrivetrainType.ForwardOnly, vehicle_name = self.chase)
        
        try:
            for jj in range(1,positions.shape[0]):
                # print("CHASER LOOP")
                global tik
                fax="tik" in globals()
                if(fax):
                    tik+=1
                else:
                    tik=0
                    print("START" , client.simGetObjectPose(self.lead).position.x_val, client.simGetObjectPose(self.lead).position.y_val,client.simGetObjectPose(self.lead).position.z_val )

                tmp=False
                if(tik>10): 
                    tmp=True
                use_Perception = False
                if use_Perception == False:
                    # lead_pose = [client.simGetObjectPose(self.lead).position.x_val, client.simGetObjectPose(self.lead).position.y_val,client.simGetObjectPose(self.lead).position.z_val]
                    
                    # lead_kinematics = self.client1.getMultirotorState(self.lead).kinematics_estimated
                    # lead_pose = [lead_kinematics.position.x_val, lead_kinematics.position.y_val, lead_kinematics.position.z_val]

                    lead_pose = positions[jj]
                    print("leadpose, ",lead_pose)
                # else:
                #     lead_pose = client.simGetObjectPose(self.lead)
                #     chase_pose = client.simGetObjectPose(self.chase)
                    
                #     leader_pos = np.array([lead_pose.position.x_val*1000, lead_pose.position.y_val*1000, lead_pose.position.z_val*1000 ])  # Leader position
                #     chaser_pos = np.array([chase_pose.position.x_val*1000, chase_pose.position.y_val*1000, chase_pose.position.z_val*1000 ])  # Chaser position
                #     leader_quat = np.array([lead_pose.orientation.w_val, lead_pose.orientation.x_val, lead_pose.orientation.y_val, -1 * lead_pose.orientation.z_val ])  # w, x, y, z for the leader
                #     chaser_quat = np.array([chase_pose.orientation.w_val, chase_pose.orientation.x_val, chase_pose.orientation.y_val, -1 * chase_pose.orientation.z_val ])
                    
                #     client.simPause(tmp)
                #     self.vis = Perception_simulation() # YK
                #     transformation_matrix = self.vis.get_transform(leader_pos, leader_quat, chaser_pos, chaser_quat)
                #     difference = self.vis.get_image(transformation_matrix)
                #     client.simPause(False)

                #     difference = np.array(difference)/1000
                #     # lead_pose = ....'change all to relative in MCL'
                #     lead_pose = [chase_pose.position.x_val + difference[0], chase_pose.position.y_val + difference[1], chase_pose.position.z_val + difference[2]] # Is this right # YK

                # Perform Particle Filter 
                timenow = time.time()
                timeinterval = timenow-timelast
                print("time now", timeinterval)
                timelast = timenow

                debug_lead_kinematics = client.getMultirotorState(self.lead).kinematics_estimated
                debug_lead_vel = np.array([debug_lead_kinematics.linear_velocity.x_val, debug_lead_kinematics.linear_velocity.y_val, debug_lead_kinematics.linear_velocity.z_val])
                state_est, variance = self.mcl.rgb_run(current_pose=lead_pose, past_states1=self.particle_state_est[-1], past_states2=self.particle_state_est[-2], time_step=self.timestep, debug_vel=debug_lead_vel, debug_time = timeinterval, debug_lead_kinematics= debug_lead_kinematics, debug_lead_kinematics_last = leadkinematiclast )   

                state_est_np = state_est.cpu().numpy()
                variance = variance.cpu()

                leadkinematiclast = debug_lead_kinematics

                # Perform Monte Carlo Prediction
                chase_pose = [client.simGetObjectPose(self.chase).position.x_val,client.simGetObjectPose(self.chase).position.y_val,client.simGetObjectPose(self.chase).position.z_val]
                # boxes, trajectories, prediction_trajectory, variances = self.prediction.find_prediction(state_est, ego_state=chase_pose, timestep=self.timestep, accel_range=3)

                # self.prediction_history.append((boxes, trajectories, prediction_trajectory, variances))
            
                
                lead_kinematics = client.getMultirotorState(self.lead).kinematics_estimated
                chase_kinematics = client.getMultirotorState(self.chase).kinematics_estimated

                orient = rotation_2d_z(-chase_kinematics.orientation.z_val, np.array([1,0]))
                vector2 = np.array([lead_pose[0]-chase_pose[0],lead_pose[1]-chase_pose[1]])
                yaw_chase = -1*angle_between(orient, vector2)

                target_state = self.global2NED(state_est_np[:3],self.chase)
                movePID(chase_kinematics, lead_kinematics, target_state )
                

                self.variance_history.append(variance)
                self.global_state_history_L.append(lead_pose)
                self.global_state_history_C.append(chase_pose)
                self.NED_state_history_L.append(lead_kinematics)
                self.NED_state_history_C.append(chase_kinematics)
                self.particle_state_est.append(state_est_np)

                A = np.array(self.mcl.filter.particles['position'].cpu())
                B = np.array(self.mcl.filter.particles['velocity'].cpu())
                self.PF_history.append((A,B))
                
                # self.count += 1
                # if self.count >= self.totalcount:
                #     break
                    
                time.sleep(self.timestep)

        except Exception as e:
            print(f"Exception in thread 2: {e}")
            traceback.print_exc()
            

if __name__ == "__main__":

    totaliteration = 920
    num_particles = 3000
  

    save = False

    
    sim = simulation(totalcount=totaliteration, num_particles=num_particles)
    threadL = threading.Thread(target=sim.move_lead, name='Thread lead')
    threadC = threading.Thread(target=sim.move_chase, name='Thread Chase')
    global retX
    retX = []
    global retY
    retY = []
    f = open("RAD10.txt", "w")
    f.close()

    try:
        print("###################################################################################### STARTING SIMULATION ##########################################################################")
        sim.move_chase()

        state_est = np.array(sim.particle_state_est)
        state_est = state_est[1:,]

        times = np.arange(0, GT_global.shape[0])*0.01
        fig, axs = plt.subplots(3, 2, figsize=(14, 10))

        axs[0,0].plot(times, state_est[:,0], label = "PF Position x")
        axs[0,0].plot(times, positions[:,0], label = "GT Position x")
        axs[0,0].legend()
        axs[1,0].plot(times, state_est[:,1], label = "PF Position y") 
        axs[1,0].plot(times, positions[:,1], label = "GT Position y")
        axs[1,0].legend()
        axs[2,0].plot(times, state_est[:,2], label = "PF Position z")
        axs[2,0].plot(times, positions[:,2], label = "GT Position z")
        axs[2,0].legend()


        axs[0,1].plot(times,state_est[:,3], label = "PF Position vx")
        axs[0,1].plot(times, GT_NED[:,3], label = "GT Position vx")
        axs[0,1].legend()
        axs[1,1].plot(times,state_est[:,4], label = "PF Position vy")  
        axs[1,1].plot(times, GT_NED[:,4], label = "GT Position vy")  
        axs[1,1].legend()
        axs[2,1].plot(times,state_est[:,5], label = "PF Position vz")
        axs[2,1].plot(times, GT_NED[:,5], label = "GT Position vz")
        axs[2,1].legend()
        plt.show()

        # sim.processing()

        sim.client1.reset()
        sim.client1.armDisarm(False)
        sim.client1.enableApiControl(False)
        sim.client2.reset()
        sim.client2.armDisarm(False)
        sim.client2.enableApiControl(False)

        if save:
            # Save the list to a pickle file
            with open('data/data_prediction.pkl', 'wb') as file:
                pickle.dump(sim.prediction_history, file)

            with open('data/data_variance.pkl', 'wb') as file:
                pickle.dump(sim.variance_history, file)

            with open('data/data_variance_choice.pkl', 'wb') as file:
                pickle.dump(sim.mcl.filter.choice_var, file)
            
            with open('data/data_global_lead.pkl', 'wb') as file:
                pickle.dump(sim.global_state_history_L, file)

            with open('data/data_NED_lead.pkl', 'wb') as file:
                pickle.dump(sim.NED_state_history_L, file)
        
            with open('data/data_NED_chase.pkl', 'wb') as file:
                pickle.dump(sim.NED_state_history_C, file)

            with open('data/PF_history.pkl', 'wb') as file:
                pickle.dump(sim.PF_history, file)  

            with open('data/PF_mean.pkl', 'wb') as file:
                pickle.dump(sim.particle_state_est, file)  

    except Exception as e:
        print("Error Occured, Canceling: ",e)
        traceback.print_exc()

        sim.client1.reset()
        sim.client1.armDisarm(False)
        sim.client1.enableApiControl(False)
        sim.client2.reset()
        sim.client2.armDisarm(False)
        sim.client2.enableApiControl(False)
