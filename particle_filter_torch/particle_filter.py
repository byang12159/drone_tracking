import numpy as np
from scipy.linalg import logm,expm
from scipy.spatial.transform import Rotation as R
from multiprocessing import Lock
import torch

class ParticleFilter:
    def __init__(self, initial_particles, device, mode_accel):
        self.num_particles=len(initial_particles['position'])
        self.particles = initial_particles
        self.device = device
        self.weights = torch.ones((self.num_particles,1), dtype=torch.float32, device=self.device)
        self.particle_lock = Lock()
        self.mode_accel = mode_accel

        self.choice_var = []

    def reduce_num_particles(self, num_particles):
        self.particle_lock.acquire()
        self.num_particles = num_particles
        self.weights = self.weights[0:num_particles]
        self.particles['position'] = self.particles['position'][0:num_particles]
        self.particles['rotation'] = self.particles['rotation'][0:num_particles]
        self.particle_lock.release()

    def predict_no_motion(self, p_x, p_y, p_z, r_x, r_y, r_z):
        self.particle_lock.acquire()
        self.particles['position'][:,0] += p_x * np.random.normal(size = (self.particles['position'].shape[0]))
        self.particles['position'][:,1] += p_y * np.random.normal(size = (self.particles['position'].shape[0]))
        self.particles['position'][:,2] += p_z * np.random.normal(size = (self.particles['position'].shape[0]))

        # TODO see if this can be made faster
        for i in range(len(self.particles['rotation'])):
            n1 = r_x * np.random.normal()
            n2 = r_y * np.random.normal()
            n3 = r_z * np.random.normal()
            self.particles['rotation'][i] = self.particles['rotation'][i].retract(np.array([n1, n2, n3]))
        self.particle_lock.release()

    def predict_with_delta_pose(self, delta_pose, p_x, p_y, p_z, r_x, r_y, r_z):

        # TODO see if this can be made faster
        delta_rot_t_tp1= delta_pose.rotation()
        for i in range(len(self.particles['rotation'])):
            # TODO do rotation in gtsam without casting to matrix
            pose = np.eye(4)  
            pose[:3, :3] = self.particles['rotation'][i] 
            pose[:3, 3] = self.particles['position'][i]

            new_pose = pose @ delta_pose
            new_position = new_pose.translation()
            self.particles['position'][i][0] = new_position[0]
            self.particles['position'][i][1] = new_position[1]
            self.particles['position'][i][2] = new_position[2]
            self.particles['rotation'][i] = new_pose.rotation()

            n1 = r_x * np.random.normal()
            n2 = r_y * np.random.normal()
            n3 = r_z * np.random.normal()
    
            # self.particles['rotation'][i] = gtsam.Rot3(self.particles['rotation'][i].retract(np.array([n1, n2, n3])).matrix())

        self.particles['position'][:,0] += (p_x * np.random.normal(size = (self.particles['position'].shape[0])))
        self.particles['position'][:,1] += (p_y * np.random.normal(size = (self.particles['position'].shape[0])))
        self.particles['position'][:,2] += (p_z * np.random.normal(size = (self.particles['position'].shape[0])))


    def update(self):
        # use fourth power
        self.weights = torch.pow(self.weights, 4)

        # normalize weights
        sum_weights=torch.sum(self.weights)
        self.weights=self.weights / sum_weights
    
        #resample
        choice = torch.multinomial(self.weights, self.num_particles, replacement=True)

        self.choice_var.append(torch.var(self.particles['position'].clone().detach()[choice,:], dim=0))
        #CHECK DIFF WAY OF COMPUTING VAR
        if self.mode_accel:
            noise_level = 0.3
            outlier_level = 0.8
            num_outliers = int(self.num_particles*0.08)
            
            random_noise = torch.rand(self.num_particles - num_outliers, 3).to(self.device) * 2 * noise_level - noise_level
            outlier_noise = torch.rand(num_outliers, 3).to(self.device) * 2 * outlier_level - outlier_level
            total_noise = torch.cat((random_noise, outlier_noise), dim=0)

            vel_noise_level = 0.7
            vel_noise_outlier = 1.6
            vel_num_outliers = int(self.num_particles*0.08)

            vel_random_noise = torch.rand(self.num_particles - num_outliers, 3).to(self.device) * 2 * vel_noise_level - vel_noise_level
            vel_outlier_noise = torch.rand(num_outliers, 3).to(self.device) * 2 * vel_noise_outlier - vel_noise_outlier
            vel_total_noise = torch.cat((vel_random_noise, vel_outlier_noise), dim=0)

            accel_noise_level = 0.1
            accel_noise = torch.rand(self.num_particles, 3).to(self.device) * 2 * accel_noise_level - accel_noise_level

            self.particles = {
                            'position':(self.particles['position'].clone().detach())[choice,:] +total_noise, 
                            'velocity':(self.particles['velocity'].clone().detach())[choice,:] +vel_total_noise,
                            'acceleration':(self.particles['acceleration'].clone().detach())[choice,:] +accel_noise
                            }
        
        else:
            noise_level = 0.3
            outlier_level = 0.8
            num_outliers = int(self.num_particles*0.08)
            
            random_noise = torch.rand(self.num_particles - num_outliers, 3).to(self.device) * 2 * noise_level - noise_level
            outlier_noise = torch.rand(num_outliers, 3).to(self.device) * 2 * outlier_level - outlier_level
            total_noise = torch.cat((random_noise, outlier_noise), dim=0)

            vel_noise_level = 0.2
            vel_noise_outlier = 0.9
            vel_num_outliers = int(self.num_particles*0.08)

            vel_random_noise = torch.rand(self.num_particles - num_outliers, 3).to(self.device) * 2 * vel_noise_level - vel_noise_level
            vel_outlier_noise = torch.rand(num_outliers, 3).to(self.device) * 2 * vel_noise_outlier - vel_noise_outlier
            vel_total_noise = torch.cat((vel_random_noise, vel_outlier_noise), dim=0)

            self.particles = {
                            'position':(self.particles['position'].clone().detach())[choice,:] +total_noise, 
                            'velocity':(self.particles['velocity'].clone().detach())[choice,:] +vel_total_noise
                            }


    def update_vel(self, particle_pose, curr_obs, curr_est,last_est, timestep):
        vel_noise_level = 0.3
        vel_noise = np.random.uniform(-vel_noise_level, vel_noise_level, size=(self.num_particles, 3)) 
        for i in range(self.num_particles):
            # est_particle_velocity = (self.particles['position'][i]-previous_state[i]) / timestep
            est_particle_velocity = (curr_est-last_est) / timestep
            
            self.particles['velocity'][i] = est_particle_velocity + vel_noise[i]

    def compute_simple_position_average(self):
        # Simple averaging does not use weighted average or k means.
        avg_pose = torch.mean(self.particles['position'], dim=0)
        return avg_pose
    
    def compute_simple_velocity_average(self):
        # Simple averaging does not use weighted average or k means. 
        avg_velocity = torch.mean(self.particles['velocity'], dim=0)
        return avg_velocity
    
    def compute_simple_accel_average(self):
        # Simple averaging does not use weighted average or k means.
        avg_accel = torch.mean(self.particles['acceleration'], dim=0)
        return avg_accel

    def compute_weighted_position_average(self):
        avg_pose = torch.sum(self.particles['position']*self.weights.view(self.weights.shape[0],1), dim=0) /torch.sum(self.weights)
        return avg_pose
    
    def compute_weighted_velocity_average(self):
        avg_velocity = torch.sum(self.particles['velocity']*self.weights.view(self.weights.shape[0],1), dim=0) /torch.sum(self.weights)
        return avg_velocity
    
    def compute_weighted_accel_average(self):
        avg_accel = torch.sum(self.particles['acceleration']*self.weights.view(self.weights.shape[0],1), dim=0) /torch.sum(self.weights)
        return avg_accel
    
    def compute_simple_rotation_average(self):
        # Simple averaging does not use weighted average or k means.
        # https://users.cecs.anu.edu.au/~hartley/Papers/PDF/Hartley-Trumpf:Rotation-averaging:IJCV.pdf section 5.3 Algorithm 1
        
        epsilon = 0.00001
        max_iters = 10
        rotations = self.particles['rotation']

        R = rotations[0].as_matrix()
        for i in range(max_iters):
            rot_sum = np.zeros((3))
            for rot in rotations:
                rot_sum = rot_sum  + logm(R.T @ rot.as_matrix())

            r = rot_sum / len(rotations)
            if np.linalg.norm(r) < epsilon:
                # print("rotation averaging converged at iteration: ", i)
                # print("average rotation: ", R)
                return R
            else:
                # TODO do the matrix math in gtsam to avoid all the type casting
                R = R @ expm(r)

    # def odometry_update(self,curr_state_est):
    #     system_time_interval = 0.001
    #     offset = system_time_interval*curr_state_est[3:]

    #     for i in range(self.num_particles):
    #         self.particles['position'][i] += offset
    #         self.particles['velocity'][i] +=0
        
    #     print("Finish odometry update")
        
    def compute_var(self):
        variance = torch.var(self.particles['position'], dim=0)
        return variance
    