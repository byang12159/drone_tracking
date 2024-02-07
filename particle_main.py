import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from particle_filter import ParticleFilter
# from controller import Controller
from scipy.spatial.transform import Rotation as R
import os
import torch 
from scipy.interpolate import UnivariateSpline
from pathlib import Path
import yaml
import matplotlib.pyplot as plt 
from torchvision.utils import save_image
from scipy.spatial.transform import Rotation
import copy


class DroneAgent():
    def __init__(self, trajectory, width=320, height=320, fov=50, batch_size=32):

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
        # [x y z]
        self.ref_traj = np.array([list(self.ref_traj_spline[0](t)), list(self.ref_traj_spline[1](t)), list(self.ref_traj_spline[2](t))]).T

        self.ref_traj *= 10

        ## ADD: decoupled YAW control

  
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

        # bounds for particle initialization, meters + degrees
        self.min_bounds = {'px':-0.5,'py':-0.5,'pz':-0.5,'rz':-2.5,'ry':-179.0,'rx':-2.5}
        self.max_bounds = {'px':0.5,'py':0.5,'pz':0.5,'rz':2.5,'ry':179.0,'rx':2.5}

        self.num_particles = 300
        
        self.obs_img_pose = None
        self.center_about_true_pose = False
        self.all_pose_est = []
        
        self.rgb_input_count = 0

        self.use_convergence_protection = True
        self.convergence_noise = 0.2

        self.number_convergence_particles = 10 #number of particles to distribute

        self.sampling_strategy = 'random'
        self.photometric_loss = 'rgb'
        self.num_updates =0
        # self.control = Controller()

        ####################### Generate Initial Particles #######################
        self.get_initial_distribution()

        # add initial pose estimate before 1st update step
        position_est = self.filter.compute_weighted_position_average()
        rot_est = np.eye(3)
        # rot_est = self.filter.compute_simple_rotation_average()
        pose_est = np.eye(4)  # Initialize as identity matrix
        pose_est[:3, :3] = rot_est  # Set the upper-left 3x3 submatrix as the rotation matrix
        pose_est[:3, 3] = position_est  # Set the rightmost column as the translation vector
        self.all_pose_est.append(pose_est)

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
        # NOTE for now assuming everything stays in NeRF coordinates (x right, y up, z inward)

        # get distribution of particles from user, generate np.array of (num_particles, 6)
        self.initial_particles_noise = np.random.uniform(np.array(
            [self.min_bounds['px'], self.min_bounds['py'], self.min_bounds['pz']]),
            np.array([self.max_bounds['px'], self.max_bounds['py'], self.max_bounds['pz']]),
            size = (self.num_particles, 3))
        
        # Dict of position + rotation, with position as np.array(300x6)
        self.initial_particles = self.set_initial_particles()
        
        # Create a 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(self.initial_particles.get('position')[:,0],self.initial_particles.get('position')[:,1],self.initial_particles.get('position')[:,2],'*')
        ax.scatter(self.ref_traj[:,0],self.ref_traj[:,1],self.ref_traj[:,2],'*')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_xlim(-10, 10)  # Set X-axis limits
        ax.set_ylim(-10, 10)  # Set Y-axis limits
        ax.set_zlim(-10, 10)  # Set Z-axis limits
        # Show the plot
        plt.show()

        # self.mat3d(self.initial_particles.get('position')[:,0],self.initial_particles.get('position')[:,1],self.initial_particles.get('position')[:,2])

        # Initiailize particle filter class with inital particles
        self.filter = ParticleFilter(self.initial_particles)


    def set_initial_particles(self):
        initial_positions = np.zeros((self.num_particles, 3))
        initial_rotations = np.zeros((self.num_particles, 3))
        for index, particle in enumerate(self.initial_particles_noise):
            # Initialize at origin location
            i = self.ref_traj[0]
      
            x = i[0] + particle[0]
            y = i[1] + particle[1]
            z = i[2] + particle[2]

            # set positions
            initial_positions[index,:] = [x,y,z]
            
        return  {'position':initial_positions, 'rotation':initial_rotations}

    

    def move(self, x0=np.zeros(12), goal=np.zeros(12), dt=0.1):
        # integrate dynamics
        movement = self.control.simulate(x0, goal, dt)
        pass

    def publish_pose_est(self, pose_est, img_timestamp = None):
        # print("Pose Est",pose_est.shape)
        pose_est = self.move()
 
        position_est = pose_est[:3, 3]
        rot_est = R.as_quat(pose_est[:3, :3])

        # populate msg with pose information
        pose_est.pose.pose.position.x = position_est[0]
        pose_est.pose.pose.position.y = position_est[1]
        pose_est.pose.pose.position.z = position_est[2]
        pose_est.pose.pose.orientation.w = rot_est[0]
        pose_est.pose.pose.orientation.x = rot_est[1]
        pose_est.pose.pose.orientation.y = rot_est[2]
        pose_est.pose.pose.orientation.z = rot_est[3]
        # print(pose_est_gtsam.rotation().ypr())

        # publish pose
        self.pose_pub.publish(pose_est)

    def odometry_update(self,state0, state1):
        state_difference = state1-state0
        # print("statediff",state_difference)
        # rot0 = R.from_matrix(state0[:3,:3])
        # rot1 = R.from_matrix(state1[:3,:3])
        # eul0 = rot0.as_euler('xyz')
        # eul1 = rot1.as_euler('xyz')
        # diffeul = eul1-eul0
        for i in range(self.num_particles):
            self.filter.particles['position'][i] += [state_difference[0], state_difference[1], state_difference[2]]

            # peul = self.filter.particles['rotation'][i].as_euler('xyz')
            # peul += diffeul
            # prot = R.from_euler('xyz',peul)
            # self.filter.particles['rotation'][i] = prot
        
        print("Finish odometry update")
    
    def get_loss(self, current_pose, particle_poses, iter):
        losses = []

        start_time = time.time()

        for i, particle in enumerate(particle_poses):
            loss = np.sqrt((current_pose[0]-particle[0])**2 + (current_pose[1]-particle[1])**2 + (current_pose[2]-particle[2])**2)
            losses.append(loss)

        nerf_time = time.time() - start_time
                   
        return losses, nerf_time

    def rgb_run(self,iter):
        print("processing image")
        start_time = time.time()
        self.rgb_input_count += 1

        # make copies to prevent mutations
        particles_position_before_update = np.copy(self.filter.particles['position'])
        # particles_rotation_before_update = [i.as_matrix() for i in self.filter.particles['rotation']]

        total_nerf_time = 0

        # if self.sampling_strategy == 'random':
        # From the meshgrid of image, find Batch# of points to randomly sample and compare, list of 2d coordinatesg

        current_pose = self.ref_traj[iter]
        losses, nerf_time = self.get_loss(current_pose, particles_position_before_update, iter=iter)
        print("Pass losses")
        # print(losses)
        temp = 1
        for index, particle in enumerate(particles_position_before_update):
            self.filter.weights[index] = 1/(losses[index]+temp)

        total_nerf_time += nerf_time

        # Resample Weights
        self.filter.update()
        self.num_updates += 1
        
        position_est = self.filter.compute_weighted_position_average()
        # rot_est = self.filter.compute_simple_rotation_average()s
        pose_est = np.eye(4)  # Initialize as identity matrix
        # pose_est[:3, :3] = rot_est  # Set the upper-left 3x3 submatrix as the rotation matrix
        pose_est[:3, 3] = position_est  # Set the rightmost column as the translation vector
        self.all_pose_est.append(pose_est)
        
        # Update odometry step
        current_state = self.ref_traj[iter]
        next_state = self.ref_traj[iter+1]
        self.odometry_update(current_state,next_state)
        # self.publish_pose_est(pose_est)


        update_time = time.time() - start_time
        print("forward passes took:", total_nerf_time, "out of total", update_time, "for update step")

        return pose_est
    
    def TC_simulate(self, initial_condition, time_horizon, time_step, lane_map = None):
        time_steps = np.arange(0, time_horizon+time_step/2, time_step)

        state = np.array(initial_condition)
        trajectory = copy.deepcopy(state)
        trajectory = np.insert(trajectory, 0, time_steps[0])
        trajectory = np.reshape(trajectory, (1, -1))
        for i in range(1, len(time_steps)):
            x_ground_truth = state[:12]
            x_estimate = state[12:24]
            ref_state = state[24:]
            x_next = self.step(x_estimate, x_ground_truth, time_step, ref_state)
            x_next[10] = x_next[10]%(np.pi*2)
            if x_next[10] > np.pi/2:
                x_next[10] = x_next[10]-np.pi*2
            ref_next = self.run_ref(ref_state, time_step)
            state = np.concatenate((x_next, x_estimate, ref_next))
            tmp = np.insert(state, 0, time_steps[i])
            tmp = np.reshape(tmp, (1,-1))
            trajectory = np.vstack((trajectory, tmp))

        return trajectory



if __name__ == "__main__":

    drone = DroneAgent(trajectory="camera_path_spline.json")    

 
    # Initialize Drone Position
    est_states = np.zeros((len(drone.ref_traj) ,3))
    gt_states  = np.zeros((len(drone.ref_traj) ,16))
    iteration_count = np.arange(0,len(drone.ref_traj) , 1, dtype=int)

    start_time = time.time()

    pose_est_history_x = []
    pose_est_history_y = []
    pose_est_history_z = []
    PF_history_x = []
    PF_history_y = []
    PF_history_z = []
    # Assume constant time step between trajectory stepping
    for iter in range(500):
        state_now = drone.ref_traj[iter]
        state_future = drone.ref_traj[iter+1]
        i = state_now
        future = state_future
        
        pose_est = drone.rgb_run(iter)   
        pose_est_history_x.append(pose_est[0,3])
        pose_est_history_y.append(pose_est[1,3])
        pose_est_history_z.append(pose_est[2,3])

        PF_history_x.append(np.array(drone.filter.particles['position'][:,0]).flatten())
        PF_history_y.append(np.array(drone.filter.particles['position'][:,1]).flatten())
        PF_history_z.append(np.array(drone.filter.particles['position'][:,2]).flatten())
    
    PF_history_x = np.array(PF_history_x)
    PF_history_y = np.array(PF_history_y)
    PF_history_z = np.array(PF_history_z)
    print("shsss",PF_history_x.shape)

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    # ax.plot(x, y, z, color='b')
    t = np.linspace(0, 32, 1000)
    x = drone.ref_traj[:,0]
    y = drone.ref_traj[:,1]
    z = drone.ref_traj[:,2]
    plt.figure(1)
    ax.plot(x,y,z, color = 'b')
    ax.plot(pose_est_history_x,pose_est_history_y,pose_est_history_z, color = 'g')
    plt.show()

    SIM_TIME = 40.0 
    DT = SIM_TIME/len(pose_est_history_x)  # time tick [s]
    print("DT is ",DT)
    time = 0.0
    show_animation = False
    count = 0

    # Initialize a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Simulation loop
    while SIM_TIME >= time:
        time += DT
        

        if show_animation:
            ax.cla()  # Clear the current axis

            # For stopping simulation with the esc key.
            fig.canvas.mpl_connect('key_release_event',
                                lambda event: [exit(0) if event.key == 'escape' else None])

            # Plot the trajectory up to the current count in 3D
            ax.plot(drone.ref_traj[:count, 0], drone.ref_traj[:count, 1], drone.ref_traj[:count, 2], "*k")
            ax.plot(pose_est_history_x[count], pose_est_history_y[count], pose_est_history_z[count], "*r" )
            # ax.plot(PF_history_x[count],PF_history_y[count],PF_history_y[count], 'o',color='blue', alpha=0.5)
            # Additional plotting commands can be added here
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim(-40, 40)  # Set X-axis limits
            ax.set_ylim(-40, 40)  # Set Y-axis limits
            ax.set_zlim(-40, 40)  # Set Z-axis limits

            ax.axis("equal")
            ax.set_title('3D Trajectory Animation')
            plt.grid(True)
            plt.pause(0.001)
        count += 1  # Increment count to update the trajectory being plotted

    # Show the final plot after the simulation ends
    plt.show()


    print("FINISHED CODE")