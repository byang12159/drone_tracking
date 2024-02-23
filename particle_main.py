import numpy as np
import scipy
import cv2
import time
from numpy import cos, sin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from particle_filter import ParticleFilter
# from controller import Controller
from scipy.spatial.transform import Rotation as R
from scipy.integrate import odeint
import os
import torch 
from scipy.interpolate import UnivariateSpline
from pathlib import Path
import yaml
import matplotlib.pyplot as plt 
from torchvision.utils import save_image
from scipy.spatial.transform import Rotation
import copy

class RunParticle():
    def __init__(self, trajectory,starting_state, width=320, height=320, fov=50, batch_size=32):

        self.inital_state = starting_state
        ####################### Import camera path trajectory json #######################
        with open(trajectory, 'r') as f:
            data = json.load(f)

        tks_x = (data['x']['0'],data['x']['1'],data['x']['2'])
        tks_y = (data['y']['0'],data['y']['1'],data['y']['2'])
        tks_z = (data['z']['0'],data['z']['1'],data['z']['2'])

        spline_x = UnivariateSpline._from_tck(tks_x)
        spline_y = UnivariateSpline._from_tck(tks_y)
        spline_z = UnivariateSpline._from_tck(tks_z)

        self.ref_traj_spline = [spline_x, spline_y, spline_z]
        t = np.linspace(0, 32, 1000)

        initialize_velocity_vector = np.zeros((1000,3))
        # [x y z]
        self.ref_traj = np.array([list(self.ref_traj_spline[0](t)), list(self.ref_traj_spline[1](t)), list(self.ref_traj_spline[2](t))]).T
        # [x y z vx vy vz]
        # self.ref_traj = np.hstack((self.ref_traj,initialize_velocity_vector))
        # print(self.ref_traj.shape)
  
        # fig = plt.figure(1)
        # ax = fig.add_subplot(111, projection='3d')
        # # ax.plot(x, y, z, color='b')
        # t = np.linspace(0, 32, 1000)
        # x = self.ref_traj[0](t)
        # y = self.ref_traj[1](t)
        # z = self.ref_traj[2](t)
        # plt.figure(1)
        # ax.plot(x,y,z, color = 'b')
        # plt.show()

        ####################### Initialize Variables #######################

        self.format_particle_size = 0
        # bounds for particle initialization, meters + degrees
        self.num_particle_states = 6
        self.min_bounds = {'px':-0.5,'py':-0.5,'pz':-0.5,'rz':-2.5,'ry':-179.0,'rx':-2.5,'pVx':-0.5,'pVy':-0.5,'pVz':-0.5}
        self.max_bounds = {'px':0.5,'py':0.5,'pz':0.5,'rz':2.5,'ry':179.0,'rx':2.5,      'pVx':0.5, 'pVy':0.5, 'pVz':0.5}

        self.num_particles = 300
        
        self.state_est_history = []

        self.use_convergence_protection = True
        self.convergence_noise = 0.2

        self.sampling_strategy = 'random'
        self.num_updates =0
        # self.control = Controller()

        ####################### Generate Initial Particles #######################
        self.get_initial_distribution()

        # add initial pose estimate before 1st update step
        position_est = self.filter.compute_simple_position_average()
        velocity_est = self.filter.compute_simple_velocity_average()
        state_est = np.concatenate((position_est, velocity_est))

        self.state_est_history.append(state_est)

        print("state",state_est)


    def mat3d(self, x,y,z):
        # Create a 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(x,y,z,'*')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_xlim(-40, 40)  # Set X-axis limits
        ax.set_ylim(-40, 40)  # Set Y-axis limits
        ax.set_zlim(-40, 40)  # Set Z-axis limits
        # Show the plot
        plt.show()

    def get_initial_distribution(self):
        # get distribution of particles from user, generate np.array of (num_particles, 6)
        self.initial_particles_noise = np.random.uniform(
            np.array([self.min_bounds['px'], self.min_bounds['py'], self.min_bounds['pz'],self.min_bounds['pVx'], self.min_bounds['pVy'], self.min_bounds['pVz']]),
            np.array([self.max_bounds['px'], self.max_bounds['py'], self.max_bounds['pz'],self.max_bounds['pVx'], self.max_bounds['pVy'], self.max_bounds['pVz']]),
            size = (self.num_particles, self.num_particle_states))
        
        # Dict of position + rotation, with position as np.array(300x6)
        self.initial_particles = self.set_initial_particles()
        
        # Create a 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(self.initial_particles.get('position')[:,0],self.initial_particles.get('position')[:,1],self.initial_particles.get('position')[:,2],'*')
        ax.scatter(self.inital_state[0],self.inital_state[1],self.inital_state[2],'*')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        # ax.set_xlim(-4, 4)  # Set X-axis limits
        # ax.set_ylim(-4, 4)  # Set Y-axis limits
        # ax.set_zlim(-4, 4)  # Set Z-axis limits
        # Show the plot
        plt.show()

        # self.mat3d(self.initial_particles.get('position')[:,0],self.initial_particles.get('position')[:,1],self.initial_particles.get('position')[:,2])

        # Initiailize particle filter class with inital particles
        self.filter = ParticleFilter(self.initial_particles)


    def set_initial_particles(self):
        initial_positions = np.zeros((self.num_particles, 3))
        initial_velocities = np.zeros((self.num_particles, 3))
        
        for index, particle in enumerate(self.initial_particles_noise):
            # Initialize at origin location
            # i = self.ref_traj[0]
            i = self.inital_state
            x = i[0] + particle[0]
            y = i[1] + particle[1]
            z = i[2] + particle[2]
            Vx = particle[3]
            Vy = particle[4]
            Vz = particle[5]

            # set positions
            initial_positions[index,:] = [x, y, z]
            initial_velocities[index,:] = [Vx, Vy, Vz]

        return  {'position':initial_positions, 'velocity':initial_velocities}

    def u(self, x, goal):
        yaw = x[10]
        err = [goal[0],0,0,0, goal[1],0,0,0, goal[2],0] - x[:10]
        err_pos = err[[0,4,8]]

        err_pos = np.linalg.inv(np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0,0,1]
        ]))@err_pos

        err[[0,4,8]] = err_pos
        u_pos = self.K.dot(err) + [0, 0, g / kT]
        u_ori = (goal[3]-yaw)*1+(0-x[11])*1.0
        if abs(goal[3]-yaw)>1:
            print('stop')
        return np.concatenate((u_pos, [u_ori]))
    
    ######################## The closed_loop system #######################
    def cl_nonlinear(self, x, t, x_est, goal):
        x = np.array(x)
        dot_x = dynamics(x, self.u(x_est, goal))
        return dot_x

    # simulate
    def simulate(self, x, x_est, goal, dt):
        curr_position = np.array(x)[[0, 4, 8]]
        goal_pos = goal[:3]
        error = goal_pos - curr_position
        distance = np.sqrt((error**2).sum())
        if distance > 1:
            goal[:3] = curr_position + error / distance
        return odeint(self.cl_nonlinear, x, [0, dt], args=(x_est, goal,))[-1]
    def move(self, x0=np.zeros(12), goal=np.zeros(12), dt=0.1):
        # integrate dynamics
        movement = self.control.simulate(x0, goal, dt)
        pass

    def odometry_update(self,curr_state_est):
        # Use current estimate of x,y,z,Vx,Vy,Vz and dynamics model to compute most probable system propagation
        
        system_time_interval = 0.2
        offset = curr_state_est[:3] + system_time_interval*curr_state_est[3:]

        for i in range(self.num_particles):
            self.filter.particles['position'][i] += offset
    
    def get_loss(self, current_pose, particle_poses):
        losses = []

        for i, particle in enumerate(particle_poses):
            loss = np.sqrt((current_pose[0]-particle[0])**2 + (current_pose[1]-particle[1])**2 + (current_pose[2]-particle[2])**2)
            losses.append(loss)
                   
        return losses

    def rgb_run(self,current_pose):
        start_time = time.time()

        # Update velocity with newest observation:
        timestep = 0.1
        self.filter.update_vel(current_pose,timestep)

        # make copies to prevent mutations
        particles_position_before_update = np.copy(self.filter.particles['position'])
        particles_velocity_before_update = np.copy(self.filter.particles['velocity'])

        losses = self.get_loss(current_pose, particles_position_before_update)

        temp = 1
        for index, particle in enumerate(particles_position_before_update):
            self.filter.weights[index] = 1/(losses[index]+temp)

        # Resample Weights
        self.filter.update()
        self.num_updates += 1

        position_est = self.filter.compute_weighted_position_average()
        velocity_est = self.filter.compute_weighted_velocity_average()
        state_est = np.concatenate((position_est, velocity_est))

        self.state_est_history.append(state_est)

        # Update odometry step
        print("state est:",state_est)
        self.odometry_update(state_est) 

        print(f"Update # {self.num_updates}, Iteration runtime: {time.time() - start_time}")

        return state_est
    
    # def step(self, cur_state_estimated, initial_condition, time_step, goal_state):
    #     goal_pos = [
    #         self.ref_traj_spline[0](goal_state[0]),
    #         self.ref_traj_spline[1](goal_state[0]),
    #         self.ref_traj_spline[2](goal_state[0]),
    #     ]
    #     goal_yaw = np.arctan2(
    #         self.ref_traj_spline[1](goal_state[0]+0.001)-self.ref_traj_spline[1](goal_state[0]),
    #         self.ref_traj_spline[0](goal_state[0]+0.001)-self.ref_traj_spline[0](goal_state[0])
    #     ) 
    #     goal_yaw = goal_yaw%(np.pi*2)
    #     if goal_yaw > np.pi/2:
    #         goal_yaw -= 2*np.pi
    #     goal = goal_pos + [goal_yaw]
    #     sol = self.simulate(initial_condition, cur_state_estimated, goal, time_step)
    #     self.last_ref_yaw = goal_yaw
    #     return sol



if __name__ == "__main__":

    mcl = RunParticle(trajectory="camera_path_spline.json")    

 
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

    # cam_init = np.array([[0,0,0,mcl.ref_traj[0][0]],
    #                      [0,0,0,mcl.ref_traj[0][1]],
    #                      [0,0,0,mcl.ref_traj[0][2]]])
    
    # cam_init_pos = cam_init[0:3, 3]
    # cam_rpy = R.from_matrix(cam_init[0:3, 0:3]).as_euler('xyz')
    # drone_init = np.array([
    #     cam_init_pos[0], 0, cam_rpy[0]-np.pi/2, 0, 
    #     cam_init_pos[1], 0, cam_rpy[1], 0, 
    #     cam_init_pos[2], 0, cam_rpy[2]+np.pi/2, 0, 
    # ])

    # ref_init = np.array([0,1])

    # state = drone_init 
    # ref = ref_init 
    # traj = np.concatenate(([0], drone_init, drone_init, ref_init)).reshape((1,-1))
    

    # Assume constant time step between trajectory stepping
    time_step = 1
    for iter in range(500):
        
        state_est = mcl.rgb_run(current_pose= mcl.ref_traj[iter])   
        pose_est_history_x.append(state_est[0])
        pose_est_history_y.append(state_est[1])
        pose_est_history_z.append(state_est[2])
        velocity_est_history_x.append(state_est[3])
        velocity_est_history_y.append(state_est[4])
        velocity_est_history_z.append(state_est[5])

        PF_history_x.append(np.array(mcl.filter.particles['position'][:,0]).flatten())
        PF_history_y.append(np.array(mcl.filter.particles['position'][:,1]).flatten())
        PF_history_z.append(np.array(mcl.filter.particles['position'][:,2]).flatten())
    
    PF_history_x = np.array(PF_history_x)
    PF_history_y = np.array(PF_history_y)
    PF_history_z = np.array(PF_history_z)

    times = np.arange(0,time_step*len(pose_est_history_x),time_step)
    velocity_GT = (mcl.ref_traj[1:]-mcl.ref_traj[:-1])/time_step


    fig, (vel) = plt.subplots(1, 1, figsize=(14, 10))
    vel.plot(times, velocity_GT[:len(times),0], label = "GT Vel x")
    vel.plot(times, velocity_GT[:len(times),1], label = "GT Vel y")
    vel.plot(times, velocity_GT[:len(times),2], label = "GT Vel z")
    vel.legend()
    plt.show()

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    # ax.plot(x, y, z, color='b')
    t = np.linspace(0, 32, 1000)
    x = mcl.ref_traj[:,0]
    y = mcl.ref_traj[:,1]
    z = mcl.ref_traj[:,2]
    plt.figure(1)
    ax.plot(x,y,z, color = 'b')
    ax.plot(pose_est_history_x,pose_est_history_y,pose_est_history_z, color = 'g')
    plt.show()

    # SIM_TIME = 40.0 
    # DT = SIM_TIME/len(pose_est_history_x)  # time tick [s]
    # print("DT is ",DT)
    # time = 0.0
    # show_animation = False
    # count = 0

    # # Initialize a 3D plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # Simulation loop
    # while SIM_TIME >= time:
    #     time += DT
        

    #     if show_animation:
    #         ax.cla()  # Clear the current axis

    #         # For stopping simulation with the esc key.
    #         fig.canvas.mpl_connect('key_release_event',
    #                             lambda event: [exit(0) if event.key == 'escape' else None])

    #         # Plot the trajectory up to the current count in 3D
    #         ax.plot(mcl.ref_traj[:count, 0], mcl.ref_traj[:count, 1], mcl.ref_traj[:count, 2], "*k")
    #         ax.plot(pose_est_history_x[count], pose_est_history_y[count], pose_est_history_z[count], "*r" )
    #         # ax.plot(PF_history_x[count],PF_history_y[count],PF_history_y[count], 'o',color='blue', alpha=0.5)
    #         # Additional plotting commands can be added here
            
    #         ax.set_xlabel('X')
    #         ax.set_ylabel('Y')
    #         ax.set_zlabel('Z')
    #         ax.set_xlim(-40, 40)  # Set X-axis limits
    #         ax.set_ylim(-40, 40)  # Set Y-axis limits
    #         ax.set_zlim(-40, 40)  # Set Z-axis limits

    #         ax.axis("equal")
    #         ax.set_title('3D Trajectory Animation')
    #         plt.grid(True)
    #         plt.pause(0.001)
    #     count += 1  # Increment count to update the trajectory being plotted

    # # Show the final plot after the simulation ends
    # plt.show()


    print("FINISHED CODE")