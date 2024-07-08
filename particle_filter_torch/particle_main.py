import numpy as np
import scipy
import time
from numpy import cos, sin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from particle_filter_torch.particle_filter import ParticleFilter
from scipy.integrate import odeint
import os
from pathlib import Path
from scipy.spatial.transform import Rotation
import copy
import torch

class RunParticle():
    def __init__(self,starting_state, num_particles = 900, width=320, height=320, fov=50, batch_size=32):

        self.inital_state = starting_state

        ####################### Initialize Variables #######################
        self.mode_accel = False

        self.format_particle_size = 0
        # bounds for particle initialization, meters + degrees
        self.filter_dimension = 3
        self.min_bounds = {'px':-0.5,'py':-0.5,'pz':-0.5,'rz':-2.5,'ry':-179.0,'rx':-2.5,'pVx':-0.3,'pVy':-1.2,'pVz':-0.3,'Ax':-0.5,'Ay':-0.5,'Az':-0.5}
        self.max_bounds = {'px':0.5,'py':0.5,'pz':0.5,'rz':2.5,'ry':179.0,'rx':2.5,      'pVx':0.3, 'pVy':1.2, 'pVz':0.3, 'Ax':0.5,'Ay':0.5,'Az':0.5}

        self.num_particles = num_particles
        
        self.state_est_history = []

        self.use_convergence_protection = True
        self.convergence_noise = 0.2

        self.sampling_strategy = 'random'
        self.num_updates =0
        # self.control = Controller()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

        ####################### Generate Initial Particles #######################
        self.get_initial_distribution()
        
        # add initial pose estimate before 1st update step
        if self.mode_accel:
            position_est = self.filter.compute_simple_position_average()
            velocity_est = self.filter.compute_simple_velocity_average()
            acceleration_est = self.filter.compute_simple_accel_average()
            state_est = torch.cat((position_est, velocity_est, acceleration_est))
        else:
            position_est = self.filter.compute_simple_position_average()
            velocity_est = self.filter.compute_simple_velocity_average()
            state_est = torch.cat((position_est, velocity_est))

        self.state_est_history.append(state_est)

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
        if self.mode_accel:
            min_bounds = torch.tensor([self.min_bounds['px'], self.min_bounds['py'], self.min_bounds['pz'], 
                            self.min_bounds['pVx'], self.min_bounds['pVy'], self.min_bounds['pVz'],
                            self.min_bounds['Ax'], self.min_bounds['Ay'], self.min_bounds['Az']], dtype=torch.float32)
            max_bounds = torch.tensor([self.max_bounds['px'], self.max_bounds['py'], self.max_bounds['pz'], 
                                    self.max_bounds['pVx'], self.max_bounds['pVy'], self.max_bounds['pVz'],
                                    self.max_bounds['Ax'], self.max_bounds['Ay'], self.max_bounds['Az']], dtype=torch.float32)

            min_bounds = min_bounds.to(self.device)
            max_bounds = max_bounds.to(self.device)

            # Generate the initial particles noise using uniform distribution
            self.initial_particles_noise = torch.rand((self.num_particles, 9), device=self.device) * (max_bounds - min_bounds) + min_bounds
                    
            # Dict of position + rotation, with position as np.array(300x6)
            self.initial_particles = self.set_initial_particles()
            
            # Initiailize particle filter class with inital particles
            self.filter = ParticleFilter(self.initial_particles, self.device, self.mode_accel)

        else:
            min_bounds = torch.tensor([self.min_bounds['px'], self.min_bounds['py'], self.min_bounds['pz'], 
                           self.min_bounds['pVx'], self.min_bounds['pVy'], self.min_bounds['pVz']], dtype=torch.float32)
            max_bounds = torch.tensor([self.max_bounds['px'], self.max_bounds['py'], self.max_bounds['pz'], 
                                    self.max_bounds['pVx'], self.max_bounds['pVy'], self.max_bounds['pVz']], dtype=torch.float32)

            min_bounds = min_bounds.to(self.device)
            max_bounds = max_bounds.to(self.device)

            # Generate the initial particles noise using uniform distribution
            self.initial_particles_noise = torch.rand((self.num_particles, 6), device=self.device) * (max_bounds - min_bounds) + min_bounds
                    
            # Dict of position + rotation, with position as np.array(300x6)
            self.initial_particles = self.set_initial_particles()
            
            # Initiailize particle filter class with inital particles
            self.filter = ParticleFilter(self.initial_particles, self.device, self.mode_accel)

        # # Create a 3D figure
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(self.initial_particles.get('position')[:,0],self.initial_particles.get('position')[:,1],self.initial_particles.get('position')[:,2],'*')
        # ax.scatter(self.inital_state[0],self.inital_state[1],self.inital_state[2],'*')
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # # ax.set_xlim(-4, 4)  # Set X-axis limits
        # # ax.set_ylim(-4, 4)  # Set Y-axis limits
        # # ax.set_zlim(-4, 4)  # Set Z-axis limits
        # # Show the plot
        # plt.show()

    def set_initial_particles(self):
        if self.mode_accel:
            initial_positions = torch.zeros((self.num_particles, self.filter_dimension), dtype=torch.float32, device=self.device)
            initial_velocities = torch.zeros((self.num_particles, self.filter_dimension), dtype=torch.float32, device=self.device)
            initial_accelerations = torch.zeros((self.num_particles, self.filter_dimension), dtype=torch.float32, device=self.device)
            
            for index, particle_noise in enumerate(self.initial_particles_noise):
                x = self.inital_state[0] + particle_noise[0]
                y = self.inital_state[1] + particle_noise[1]
                z = self.inital_state[2] + particle_noise[2]
                Vx = particle_noise[3]
                Vy = particle_noise[4]
                Vz = particle_noise[5]
                Ax = particle_noise[6]
                Ay = particle_noise[7]
                Az = particle_noise[8]

                initial_positions[index,:] = torch.tensor([x, y, z], device=self.device)
                initial_velocities[index,:] = torch.tensor([Vx, Vy, Vz], device=self.device)
                initial_accelerations[index,:] = torch.tensor([Ax, Ay, Az], device=self.device)

            return {'position': initial_positions, 'velocity': initial_velocities, 'acceleration': initial_accelerations}
        
        else:
            initial_positions = torch.zeros((self.num_particles, self.filter_dimension), dtype=torch.float32, device=self.device)
            initial_velocities = torch.zeros((self.num_particles, self.filter_dimension), dtype=torch.float32, device=self.device)
            
            for index, particle_noise in enumerate(self.initial_particles_noise):
                x = self.inital_state[0] + particle_noise[0]
                y = self.inital_state[1] + particle_noise[1]
                z = self.inital_state[2] + particle_noise[2]
                Vx = particle_noise[3]
                Vy = particle_noise[4]
                Vz = particle_noise[5]

                initial_positions[index,:] = torch.tensor([x, y, z], device=self.device)
                initial_velocities[index,:] = torch.tensor([Vx, Vy, Vz], device=self.device)

            return {'position': initial_positions, 'velocity': initial_velocities}


    def odometry_update(self, system_time_interval):
        # if self.mode_accel:
        #     offset = self.filter.particles['velocity']  + system_time_interval*self.filter.particles['acceleration']
        #     self.filter.particles['position'] += system_time_interval*offset
        # else:
        self.filter.particles['position'] += system_time_interval*self.filter.particles['velocity']
    
    def get_loss(self, current_pose, current_vel, particle_poses, particle_vel):
        
        position_loss = torch.sum((particle_poses -current_pose) ** 2, dim=1) * 1
        velocity_loss = torch.sum((particle_vel   -current_vel)  ** 2, dim=1) * 0.8

        losses = torch.sqrt((position_loss + velocity_loss)/self.num_particles)
        # losses = torch.sqrt(position_loss/self.num_particles)

        return losses

    def get_loss_accel(self, current_pose, current_vel, current_accel, particle_poses, particle_vel, particle_accel):
        position_loss = torch.sum((particle_poses -current_pose) ** 2, dim=1) * 1
        velocity_loss = torch.sum((particle_vel   -current_vel)  ** 2, dim=1) * 0.7
        acceleration_loss = torch.sum((particle_accel   -current_accel)  ** 2, dim=1) * 0.55

        losses = torch.sqrt(position_loss + velocity_loss + acceleration_loss)
        
        return losses
    
    def rgb_run(self,current_pose, past_states1, past_states2, time_step, debug_vel, debug_time, debug_lead_kinematics, debug_lead_kinematics_last):
        start_time = time.time() 

        current_pose = torch.tensor(current_pose).to(self.device)
        past_states1 = torch.tensor(past_states1).to(self.device)
        past_states2 = torch.tensor(past_states2).to(self.device)

        # print(current_pose.is_cuda, past_states1.is_cuda, past_states2.is_cuda)
        # print(current_pose)
        # print(past_states1)
        # print(past_states2)
        
        self.odometry_update(0.01) 
        # self.odometry_update(debug_time) 

        # make copies to prevent mutations
        if self.mode_accel:
            particles_position_before_update = self.filter.particles['position'].clone().detach()
            particles_velocity_before_update = self.filter.particles['velocity'].clone().detach()
            particles_acceleration_before_update = self.filter.particles['acceleration'].clone().detach()

            current_velocity  = (current_pose-past_states1[:3])/time_step
            current_acceleration = (past_states1[3:6]-past_states2[3:6])/time_step

            losses = self.get_loss_accel(current_pose, current_velocity, current_acceleration, particles_position_before_update, particles_velocity_before_update, particles_acceleration_before_update)

            offset_val = 1
            self.filter.weights = 1/(losses+offset_val)

            # Resample Weights
            self.filter.update()
            self.num_updates += 1

            position_est = self.filter.compute_weighted_position_average()
            velocity_est = self.filter.compute_weighted_velocity_average()
            acceleration_est = self.filter.compute_weighted_accel_average()
            state_est = torch.cat((position_est, velocity_est, acceleration_est))


        else:
            particles_position_before_update = self.filter.particles['position'].clone().detach()
            particles_velocity_before_update = self.filter.particles['velocity'].clone().detach()

            raw_diff = current_pose-past_states1[:3]

            current_velocity  = (current_pose-past_states1[:3])/0.01

            current_velocity_debug = torch.tensor(debug_vel).to(self.device)
            GT_POS_LAST = torch.tensor([debug_lead_kinematics_last.position.x_val, debug_lead_kinematics_last.position.y_val, debug_lead_kinematics_last.position.z_val]).to(self.device)
            GT_POS_CURR = torch.tensor([debug_lead_kinematics.position.x_val, debug_lead_kinematics.position.y_val, debug_lead_kinematics.position.z_val]).to(self.device)
            current_velocity_debug2 = (GT_POS_CURR-GT_POS_LAST)/debug_time
            
            # print(f"calculated Velocity: finitediff:{current_velocity}, GT vel: {current_velocity_debug}")
            losses = self.get_loss(current_pose, current_velocity, particles_position_before_update, particles_velocity_before_update)

            offset_val = 1
            self.filter.weights = 1/(losses+offset_val)

            # Resample Weights
            self.filter.update()
            self.num_updates += 1

            # if self.num_updates >= 100:
            #     print("leadpose",current_pose)
            #     print(f"GT POS: {GT_POS_LAST},{GT_POS_CURR}")
            #     print(f"GT DIFF VEL: {(GT_POS_CURR-GT_POS_LAST)/debug_time}")
            #     print("time stamp:  ",time.time())
            #     print("1", raw_diff)
            #     print("2",current_velocity)
            #     print("2.5 ",current_velocity_debug2)
            #     print("3",current_velocity_debug)
            #     print("debug")

            position_est = self.filter.compute_weighted_position_average()
            velocity_est = self.filter.compute_weighted_velocity_average()
            state_est = torch.cat((position_est, velocity_est))

        
        self.state_est_history.append(state_est)
        print(f"Update # {self.num_updates}, Iteration runtime: {time.time() - start_time}")

        # # Update velocity with newest observation:
        # self.filter.update_vel(current_pose,timestep)
        # Update velocity with newest observation:
        # self.filter.update_vel(particles_position_before_update,current_pose,position_est, lastpose,time_step)
        variance = self.filter.compute_var()

        return state_est, variance
    
#######################################################################################################################################
if __name__ == "__main__":

    simple_trajx = np.arange(0,1,0.03)
    simple_trajx = simple_trajx.reshape(simple_trajx.shape[0],1)
    simple_trajx = np.concatenate((np.zeros((15,1)), simple_trajx, np.ones((15,1))), axis=0)
    simple_traj = np.hstack((simple_trajx, np.ones_like(simple_trajx), np.zeros_like(simple_trajx)))

    mcl = RunParticle(starting_state=simple_traj[0])    

    start_time = time.time()
    
    particle_state_est=[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]
    variance_history = []
    PF_history = [np.array(mcl.filter.particles['position'].cpu())]
    prediction_history = []
    
    # Assume constant time step between trajectory stepping
    time_step = 0.03

    for iter in range(1,simple_traj.shape[0]):
        
        state_est, variance = mcl.rgb_run(current_pose= simple_traj[iter], past_states1=particle_state_est[-1], time_step=time_step )   

        particle_state_est.append(state_est.cpu().numpy())
        variance_history.append(variance)
        PF_history.append(np.array(mcl.filter.particles['position'].cpu()))
    


    particle_state_est = particle_state_est[2:]
    particle_state_est = np.array(particle_state_est)
    
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    t = np.linspace(0, 32, 1000)
    ax.plot(simple_traj[:,0],simple_traj[:,1],simple_traj[:,2], color = 'b')
    ax.plot(particle_state_est[:,0], particle_state_est[:,1], particle_state_est[:,2], color = 'g')
    plt.show()

    times = np.arange(0,particle_state_est.shape[0]*0.03, 0.03)
    y_min = -0.5
    y_max = 1.5
    fig, (posx,posy,posz,velx,vely,velz) = plt.subplots(6, 1, figsize=(16, 10))
    posx.plot(times, particle_state_est[:,0], label = "Est Pos x")
    posx.plot(times, simple_traj[1:,0], label = "GT Pos x")
    posx.set_ylim(y_min, y_max)
    posx.legend() 
    posy.plot(times, particle_state_est[:,1], label = "Est Pos y")
    posy.plot(times, simple_traj[1:,1], label = "GT Pos y")
    posy.set_ylim(y_min, y_max)
    posy.legend()
    posz.plot(times, particle_state_est[:,2], label = "Est Pos z")
    posz.plot(times, simple_traj[1:,2], label = "GT Pos z")
    posz.set_ylim(y_min, y_max)
    posz.legend()
    velx.plot(times, particle_state_est[:,3], label = "GT Vel x")
    velx.legend() 
    vely.plot(times, particle_state_est[:,4], label = "GT Vel y")
    vely.legend()
    velz.plot(times, particle_state_est[:,5], label = "GT Vel z")
    velz.legend()
    plt.tight_layout()
    plt.show()


    # Particle Viewer
    # for i in range(len(PF_history)):
    #     fig = plt.figure(1)
    #     ax = fig.add_subplot(111, projection='3d')
    #     t = np.linspace(0, 32, 1000)
    #     ax.plot(simple_traj[:,0],simple_traj[:,1],simple_traj[:,2], color = 'b')
    #     ax.scatter(particle_state_est[i,0], particle_state_est[i,1], particle_state_est[i,2], c='r', s=100)
    #     ax.scatter(PF_history[i][:,0], PF_history[i][:,1], PF_history[i][:,2], c='g', alpha=0.15)
    #     plt.show()

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

