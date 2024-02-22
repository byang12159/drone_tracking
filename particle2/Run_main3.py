import numpy as np
import cv2
import time
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import time
import json
from mpl_toolkits.mplot3d import Axes3D
from particle_filter import ParticleFilter
# from full_filter import NeRF
# from nerf_image import Nerf_image
# from controller import Controller
from scipy.spatial.transform import Rotation as R
import os
import torch 
import pickle 
import copy 

def get_pose(phi, theta, psi, x, y, z, obs_img_pose, center_about_true_pose):
    if center_about_true_pose:
        # print("recentering")
        # print(obs_img_pose)
        pose = trans_t(x, y, z) @ rot_phi(phi/180.*np.pi) @ rot_theta(theta/180.*np.pi) @ rot_psi(psi/180.*np.pi)  @ obs_img_pose
    else:
        pose = trans_t(x, y, z) @ rot_phi(phi/180.*np.pi) @ rot_theta(theta/180.*np.pi) @ rot_psi(psi/180.*np.pi)
        
    return pose

class Run():
    def __init__(self, trajectory, width = 320, height = 320, fov = 50):

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

        self.initialization_center = [self.ref_traj[0,0], self.ref_traj[0,1], self.ref_traj[0,2]]

        # self.ref_traj *= 10

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

        self.format_particle_size = 0
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

        self.sampling_strategy = 'random'
        self.photometric_loss = 'rgb'
        self.num_updates =0

        ####################### Generate Initial Particles #######################
        self.get_initial_distribution()

        # add initial pose estimate before 1st update step
        position_est = self.filter.compute_simple_position_average()
        rot_est = self.filter.compute_simple_rotation_average()
        pose_est = np.zeros(3+4)  # Initialize as identity matrix
        pose_est[:3] = position_est 
        pose_est[3:] = rot_est
        self.all_pose_est.append(pose_est)

        self.last_state = None
        
    def center_euler(self, euler_angles):
        # Ensure the differences are within the range of -pi to pi
        yaw_diff = (yaw_diff + np.pi) % (2 * np.pi) - np.pi
        pitch_diff = (pitch_diff + np.pi) % (2 * np.pi) - np.pi
        roll_diff = (roll_diff + np.pi) % (2 * np.pi) - np.pi

        return [yaw_diff, pitch_diff, roll_diff]

    def mat3d(self):
        traj = np.zeros((len(self.self.ref_traj),3))
        for i in range(len(self.self.ref_traj)):
            traj[i][0] = self.self.ref_traj[i][4]
            traj[i][1] = self.self.ref_traj[i][7]
            traj[i][2] = self.self.ref_traj[i][11]

        # Create a 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(traj[:,0],traj[:,1],traj[:,2],'*')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_xlim(-5, 5)  # Set X-axis limits
        ax.set_ylim(-5, 5)  # Set Y-axis limits
        ax.set_zlim(-5, 5)  # Set Z-axis limits
        # Show the plot
        plt.show()

    def get_initial_distribution(self):
        # NOTE for now assuming everything stays in NeRF coordinates (x right, y up, z inward)

        # get distribution of particles from user, generate np.array of (num_particles, 6)
        self.initial_particles_noise = np.random.uniform(np.array(
            [self.min_bounds['px'], self.min_bounds['py'], self.min_bounds['pz'], self.min_bounds['rz'], self.min_bounds['ry'], self.min_bounds['rx']]),
            np.array([self.max_bounds['px'], self.max_bounds['py'], self.max_bounds['pz'], self.max_bounds['rz'], self.max_bounds['ry'], self.max_bounds['rx']]),
            size = (self.num_particles, 6))
        

        # Dict of position + rotation, with position as np.array(300x6)
        self.initial_particles = self.set_initial_particles()
        
        # Initiailize particle filter class with inital particles
        self.filter = ParticleFilter(self.initial_particles)


    def set_initial_particles(self):
        initial_positions = np.zeros((self.num_particles, 3))
        rots = []

        for index, particle in enumerate(self.initial_particles_noise):
            # # # For Testing: Initialize at camera location
            # i = self.self.ref_traj[0]
            # x = i[3]
            # y = i[7]
            # z = i[11]
            # rot1 = i.reshape(4,4)
            # rot = rot1[:3,:3]
            # gt_rotation_obj  = R.from_matrix(rot)
            # gt_euler  =  gt_rotation_obj.as_euler('xyz')
            # phi = gt_euler[0]
            # theta = gt_euler[1]
            # psi = gt_euler[2]



            # # For Testing: Initialize at camera location
            # i = self.self.ref_traj[0]
            # x = i[3]+particle[0]
            # y = i[7]+particle[1]
            # z = i[11]+particle[2]
            # rot1 = i.reshape(4,4)
            # rot = rot1[:3,:3]
            # gt_rotation  = R.from_matrix(rot)
            # gt_euler  =  gt_rotation.as_euler('xyz')
            # phi = gt_euler[0]+ np.pi/4
            # theta = gt_euler[1]
            # psi = gt_euler[2]
            # if index < 10:
            #     print("ROTS1",phi,theta,psi)
            # gt_rotation_obj = R.from_euler('xyz',[phi,theta,psi])

            # For random particles within given bound
            x = self.initialization_center[0] + particle[0]
            y = self.initialization_center[1] + particle[1]
            z = self.initialization_center[2] + particle[2]
            phi   =  particle[3]
            theta =  particle[4]
            psi   =  particle[5]
            gt_rotation_obj = R.from_euler('xyz',[phi,theta,psi])

            # # For random particles within given bound
            # x = particle[0]+self.initialization_center[0]
            # y = particle[1]+self.initialization_center[1]
            # z = particle[2]+self.initialization_center[2]
            # phi = particle[3]
            # theta = particle[4]
            # psi = particle[5]
            # gt_rotation_obj = R.from_euler('xyz',[phi,theta,psi])

            # set positions
            initial_positions[index,:] = [x,y,z]
            # set orientations, create rotation object
            rots.append(gt_rotation_obj)
    
        
        # # Create a 3D plot
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # for index, particle in enumerate(self.initial_particles_noise):
        #     Rotating = rots[index].as_matrix()
        #     vector = np.dot(Rotating, np.array([1, 0, 0]))  # Unit vector along the x-axis
        #     ax.quiver(initial_positions[index][0], initial_positions[index][1], initial_positions[index][2], vector[0], vector[1], vector[2])

        # # Add camera initialization
        # initial_cam = self.self.ref_traj[0].reshape(4,4)
        # vector = np.dot(initial_cam[:3,:3], np.array([1, 0, 0]))  # Unit vector along the x-axis
        # ax.quiver(initial_cam[0][3], initial_cam[1][3], initial_cam[2][3], vector[0], vector[1], vector[2], color='r')

        # # Set axis labels
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')

        # # Set axis limits
        # ax.set_xlim([-2, 2])
        # ax.set_ylim([-2, 2])
        # ax.set_zlim([-2, 2])

        # # Add a legend
        # ax.legend()

        # # Show the 3D plot
        # plt.show()

        print("INITIAL POSITION ", initial_positions)
        print("INITIAL ROT", rots)
        return {'position':initial_positions, 'rotation':np.array(rots)}



    # def vector_visualization(self, )
    def odometry_update(self,state0, state1):
        state_difference = state1-state0
        rot0 = R.from_matrix(state0[:3,:3])
        rot1 = R.from_matrix(state1[:3,:3])
        eul0 = rot0.as_euler('xyz')
        eul1 = rot1.as_euler('xyz')
        diffeul = eul1-eul0

        odometry_particle_noise = np.random.uniform(np.array(
            [self.min_bounds['px'], self.min_bounds['py'], self.min_bounds['pz'], self.min_bounds['rz'], self.min_bounds['ry'], self.min_bounds['rx']]),
            np.array([self.max_bounds['px'], self.max_bounds['py'], self.max_bounds['pz'], self.max_bounds['rz'], self.max_bounds['ry'], self.max_bounds['rx']]),
            size = (self.num_particles, 6))
        

        for i in range(self.num_particles):
            odometry_particle_noise_translation = np.random.normal(0.0, 0.001,3)
            odometry_particle_noise_rotation = np.random.normal(0.0, 0.001,3)

            self.filter.particles['position'][i] += [state_difference[0][3]+odometry_particle_noise_translation[0], state_difference[1][3]+odometry_particle_noise_translation[0], state_difference[2][3]+odometry_particle_noise_translation[0]]

            peul = self.filter.particles['rotation'][i].as_euler('xyz')
            peul = peul + diffeul + odometry_particle_noise_rotation

            # Ensure the differences are within the range of -pi to pi
            peul[0] = (peul[0] + np.pi) % (2 * np.pi) - np.pi
            peul[1] = (peul[1] + np.pi) % (2 * np.pi) - np.pi
            peul[2] = (peul[2] + np.pi) % (2 * np.pi) - np.pi

            prot = R.from_euler('xyz',peul)
            self.filter.particles['rotation'][i] = prot
        
        print("Finish odometry update")

    def rgb_run(self,iter, img, current_state, msg=None, get_rays_fn=None, render_full_image=False):
        self.odometry_update(self.last_state,current_state)
        print("processing image")
        start_time = time.time()

        # make copies to prevent mutations
        particles_position_before_update = np.copy(self.filter.particles['position'])
        particles_rotation_before_update = np.copy(self.filter.particles['rotation'])

        # resize input image so it matches the scale that NeRF expects
        img = cv2.resize(img, (int(self.nerfW), int(self.nerfH)))
        self.nerf.obs_img = img
        show_true = self.view_debug_image_iteration != 0 and self.num_updates == self.view_debug_image_iteration-1
        #Create a grid to sample image points for comparison
        self.nerf.get_poi_interest_regions(show_true, self.sampling_strategy)
        # plt.imshow(self.nerf.obs_img)
        # plt.show()

        total_nerf_time = 0

        # if self.sampling_strategy == 'random':
        # From the meshgrid of image, find Batch# of points to randomly sample and compare, list of 2d coordinates
        rand_inds = np.random.choice(self.nerf.coords.shape[0], size=self.nerf.batch_size, replace=False)
        batch = self.nerf.coords[rand_inds]

        loss_poses = []
        for index, particle in enumerate(particles_position_before_update):
            loss_pose = np.zeros((4,4))
            rot = particles_rotation_before_update[index].as_matrix()
            loss_pose[0:3, 0:3] = rot
            loss_pose[0:3,3] = particle[0:3]
            loss_pose[3,3] = 1.0
            loss_poses.append(loss_pose)

        losses, nerf_time = self.nerf.get_loss(loss_poses, batch, img, iter=iter)
        print("Pass losses")
        # print("Loss Values" ,losses)
        temp = 0
        for index, particle in enumerate(particles_position_before_update):
            self.filter.weights[index] = 1/(losses[index]+temp)

        total_nerf_time += nerf_time

        # Resample Weights
        
        position_est = self.filter.compute_simple_position_average()
        quat_est = self.filter.compute_simple_rotation_average()
        pose_est = np.zeros(3+4)  # Initialize as identity matrix
        pose_est[:3] = position_est 
        pose_est[3:] = quat_est 
        self.all_pose_est.append(pose_est)

        self.filter.update()
        self.num_updates += 1

        # # Create a 3D plot
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # for index in range(self.num_particles):
        #     Rotating = self.filter.particles['rotation'][index].as_matrix()
        #     vector = np.dot(Rotating, np.array([1, 0, 0]))  # Unit vector along the x-axis
        #     ax.quiver(self.filter.particles['position'][index][0], self.filter.particles['position'][index][1], self.filter.particles['position'][index][2], vector[0], vector[1], vector[2])

        # # Add camera 
        # initial_cam = self.self.ref_traj[iter].reshape(4,4)
        # vector = np.dot(initial_cam[:3,:3], np.array([1, 0, 0]))  # Unit vector along the x-axis
        # ax.quiver(initial_cam[0][3], initial_cam[1][3], initial_cam[2][3], vector[0], vector[1], vector[2], color='r')

        # # Pose est 
        # estimated_rot = R.from_quat(pose_est[3:])
        # vector = np.dot(estimated_rot.as_matrix(), np.array([1, 0, 0]))  # Unit vector along the x-axis
        # ax.quiver(pose_est[0], pose_est[1], pose_est[2], vector[0], vector[1], vector[2], color='g')

        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # # ax.set_xlim([-2, 2])
        # # ax.set_ylim([-2, 2])
        # # ax.set_zlim([-2, 2])
        # ax.legend()
        # plt.show()

        # Update odometry step
        # current_state = self.self.ref_traj[iter].reshape(4,4)
        # next_state = self.self.ref_traj[iter+1].reshape(4,4)
        # self.publish_pose_est(pose_est)


        update_time = time.time() - start_time
        print("forward passes took:", total_nerf_time, "out of total", update_time, "for update step")

        return pose_est

    def step(self, state, iterationnum, fog_parameter, dark_parameter, flag = False):
        cam2world = np.zeros((3,4))
        cam2world[:,3] = state[:3]
        if not flag:
            rot_mat = R.from_euler('xyz',[state[3]+np.pi/2, state[4], state[5]-np.pi/2]).as_matrix()
        else:
            rot_mat = R.from_euler('xyz',[state[3], state[4], state[5]]).as_matrix()
        cam2world[:3,:3] = rot_mat

        base_img = self.nerf.render_Nerf_image_base(cam2world, fog_parameter, dark_parameter, save=False, save_name = "base", iter=iter, particle_number=None)
        


        tmp = np.vstack((cam2world, np.array([[0,0,0,1]])))
        if self.last_state is None:
            self.last_state = copy.deepcopy(tmp)
        pose_est = self.rgb_run(iter, base_img, tmp) 
        self.last_state = copy.deepcopy(tmp)
        pos = pose_est[:3]
        rpy = R.from_quat(pose_est[3:]).as_euler('xyz')
        if not flag:
            res = np.array([pos[0], pos[1], pos[2], rpy[0]-np.pi/2, rpy[1], rpy[2]+np.pi/2])
        else:
            res = np.array([pos[0], pos[1], pos[2], rpy[0], rpy[1], rpy[2]])
            
        return res
 
if __name__ == "__main__":

    mcl = Run(trajectory="camera_path_spline.json")    

    # mcl.mat3d()
    # Initialize Drone Position
    est_states = np.zeros((len(mcl.ref_traj) ,3))
    gt_states  = np.zeros((len(mcl.ref_traj) ,16))
    est_euler  = np.zeros((len(mcl.ref_traj) ,3))  
    gt_euler   = np.zeros((len(mcl.ref_traj) ,3))  
    iteration_count = np.arange(0,len(mcl.ref_traj) , 1, dtype=int)
    
    # Assume constant time step between trajectory stepping
    for iter in range(len(mcl.ref_traj)-1):
        
        mcl.ref_traj = np.array(mcl.mcl.ref_traj[iter]).reshape((4,4))
        rpy = R.from_matrix(mcl.ref_traj[0:3, 0:3]).as_euler('xyz')
        state = np.concatenate((mcl.ref_traj[:,3], rpy))
        # base_img = mcl.nerf.render_Nerf_image_simple(mcl.mcl.ref_traj[iter],mcl.mcl.ref_traj[iter+1],save=False, save_name = "base", iter=iter, particle_number=None)
        # # cv2.imshow("img ",base_img)
        # # cv2.waitKey(0)
        # # cv2.destroyAllWindows()

    
        # pose_est = mcl.rgb_run(iter, base_img)   
        pose_est = mcl.step(state, flag = True)

        ########################## Error Visualization ##########################
        est_states[iter] = pose_est[0:3]
        gt_states[iter] = mcl.mcl.ref_traj[iter]

        est_rotation_obj = R.from_euler('xyz', pose_est[3:])
        est_euler[iter] = est_rotation_obj.as_euler('xyz', degrees=True)

        gt_matrix = gt_states[iter].reshape(4,4)
        gt_rotation_obj  = mcl.nerf.base_rotations[iter]
        gt_euler[iter]  =  gt_rotation_obj.as_euler('xyz', degrees=True)
           
    
        # Create a figure with six subplots (2 rows, 3 columns)
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 3, 1)
        plt.plot(iteration_count[:iter+1], np.abs(gt_states[:iter+1,3]-est_states[:iter+1,0]))
        plt.title('X error')

        plt.subplot(2, 3, 2)
        plt.plot(iteration_count[:iter+1], np.abs( gt_states[:iter+1,7]-est_states[:iter+1,1]) )
        plt.title('Y error')

        # Plot the third graph in the third subplot
        plt.subplot(2, 3, 3)
        plt.plot(iteration_count[:iter+1],np.abs(gt_states[:iter+1,11]-est_states[:iter+1,2]) )
        plt.title('Z error')

        # Plot the fourth graph in the fourth subplot
        plt.subplot(2, 3, 4)
        plt.plot(iteration_count[:iter+1],np.abs(est_euler[:iter+1,0]-gt_euler[:iter+1,0]) )
        plt.title('Yaw error')

        # Plot the fifth graph in the fifth subplot
        plt.subplot(2, 3, 5)
        plt.plot(iteration_count[:iter+1],np.abs(est_euler[:iter+1,1]-gt_euler[:iter+1,1]))
        plt.title('Pitch error')

        # Plot the sixth graph in the sixth subplot
        plt.subplot(2, 3, 6)
        plt.plot(iteration_count[:iter+1],np.abs(est_euler[:iter+1,2]-gt_euler[:iter+1,2]))
        plt.title('Roll error')

        plt.tight_layout()
        file_path = f'./NeRF_UAV_simulation/Plots/plot{iter}.png'
        plt.savefig(file_path)
        plt.close()
        
        print(f'mcl.ref_traj iteration {iter}:\n',mcl.mcl.ref_traj[iter])
        print(f'pose est iteration {iter}:\n',pose_est)

    print("########################Done########################")
    fig = plt.figure(0)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(gt_states[:iter+1,3], gt_states[:iter+1,7], gt_states[:iter+1,11], 'g')
    ax.scatter(gt_states[:iter+1,3], gt_states[:iter+1,7], gt_states[:iter+1,11], 'm')
    ax.plot(est_states[:iter+1,0], est_states[:iter+1,1], est_states[:iter+1,2], 'r')
    ax.scatter(est_states[:iter+1,0], est_states[:iter+1,1], est_states[:iter+1,2], 'm')
    plt.show()
