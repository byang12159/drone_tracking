import numpy as np 
from numpy import cos, sin
import copy 
import matplotlib.pyplot as plt 
import scipy
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import os 
import json 
from scipy.interpolate import UnivariateSpline
from scipy.spatial.transform import Rotation as R
import torch 

import cv2
from pathlib import Path
script_dir = os.path.dirname(os.path.realpath(__file__))
from particle_main import RunParticle

import logging
from datetime import datetime




# quadrotor physical constants
g = 9.81; d0 = 10; d1 = 8; n0 = 10; kT = 0.91

# non-linear dynamics
def dynamics(state, u):
    x, vx, theta_x, omega_x, y, vy, theta_y, omega_y, z, vz, theta_z, omega_z = state.reshape(-1).tolist()
    ax, ay, F, az = u.reshape(-1).tolist()
    dot_x = np.array([
     cos(theta_z)*vx-sin(theta_z)*vy,
     g * np.tan(theta_x),
     -d1 * theta_x + omega_x,
     -d0 * theta_x + n0 * ax,
     sin(theta_z)*vx+cos(theta_z)*vy,
     g * np.tan(theta_y),
     -d1 * theta_y + omega_y,
     -d0 * theta_y + n0 * ay,
     vz,
     kT * F - g,
     omega_z,
     n0 * az])
    return dot_x

# linearization
# The state variables are x, y, z, vx, vy, vz, theta_x, theta_y, omega_x, omega_y
A = np.zeros([10, 10])
A[0, 1] = 1.
A[1, 2] = g
A[2, 2] = -d1
A[2, 3] = 1
A[3, 2] = -d0
A[4, 5] = 1.
A[5, 6] = g
A[6, 6] = -d1
A[6, 7] = 1
A[7, 6] = -d0
A[8, 9] = 1.

B = np.zeros([10, 3])
B[3, 0] = n0
B[7, 1] = n0
B[9, 2] = kT

class DroneAgent():
    def __init__(self, ref_spline = 'camera_path_spline'):

        with open(ref_spline, 'r') as f:
            data = json.load(f)

        tks_x = (data['x']['0'],data['x']['1'],data['x']['2'])
        tks_y = (data['y']['0'],data['y']['1'],data['y']['2'])
        tks_z = (data['z']['0'],data['z']['1'],data['z']['2'])

        spline_x = UnivariateSpline._from_tck(tks_x)
        spline_y = UnivariateSpline._from_tck(tks_y)
        spline_z = UnivariateSpline._from_tck(tks_z)

        self.ref_traj = [spline_x, spline_y, spline_z]

        t = np.linspace(0, 32, 1000)
        self.ref_traj_discrete = np.array([list(self.ref_traj[0](t)), list(self.ref_traj[1](t)), list(self.ref_traj[2](t))]).T

        def lqr(A, B, Q, R):
            """Solve the continuous time lqr controller.
            dx/dt = A x + B u
            cost = integral x.T*Q*x + u.T*R*u
            """
            # http://www.mwm.im/lqr-controllers-with-python/
            # ref Bertsekas, p.151

            # first, try to solve the ricatti equation
            X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

            # compute the LQR gain
            K = np.matrix(scipy.linalg.inv(R) * (B.T * X))

            eigVals, eigVecs = scipy.linalg.eig(A - B * K)

            return np.asarray(K), np.asarray(X), np.asarray(eigVals)
        
        ####################### solve LQR #######################
        n = A.shape[0]
        m = B.shape[1]
        Q = np.eye(n)
        Q[0, 0] = 100.
        Q[4, 4] = 100.
        Q[8, 8] = 100.
        # Q[11,11] = 0.01
        R = np.diag([1., 1., 1.])
        self.K, _, _ = lqr(A, B, Q, R)

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
        u_ori = (goal[3]-yaw)*2+(0-x[11])*1.0
        # if abs(goal[3]-yaw)>1:
        #     print('stop')
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
    
    def step(self, cur_state_estimated, initial_condition, time_step, goal_state):
        goal_pos = [
            self.ref_traj[0](goal_state[0]),
            self.ref_traj[1](goal_state[0]),
            self.ref_traj[2](goal_state[0]),
        ]
        goal_yaw = np.arctan2(
            self.ref_traj[1](goal_state[0]+0.001)-self.ref_traj[1](goal_state[0]),
            self.ref_traj[0](goal_state[0]+0.001)-self.ref_traj[0](goal_state[0])
        ) 
        goal_yaw = goal_yaw%(np.pi*2)
        if goal_yaw > np.pi/2:
            goal_yaw -= 2*np.pi
        goal = goal_pos + [goal_yaw]
        sol = self.simulate(initial_condition, cur_state_estimated, goal, time_step)
        self.last_ref_yaw = goal_yaw
        return sol

    def run_ref(self, ref_state, time_step):
        ref_pos = ref_state[0]
        ref_v = ref_state[1]
        return np.array([ref_pos+ref_v*time_step, ref_v])

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
    
    spline_fn = "camera_path_spline.json"
    drone_agent = DroneAgent(ref_spline=spline_fn)

    mcl = RunParticle(trajectory="camera_path_spline.json")

    cam_init = np.array([[1,0,0,drone_agent.ref_traj_discrete[0][0]],
                        [0,1,0,drone_agent.ref_traj_discrete[0][1]],
                        [0,0,1,drone_agent.ref_traj_discrete[0][2]]])
    cam_init_pos = cam_init[0:3, 3]

    # cam_rpy = [roll, pitch, yaw]
    cam_rpy = R.from_matrix(cam_init[0:3, 0:3]).as_euler('xyz')
    print("Starting Euler",cam_rpy)
    
    # drone init = [x,vx,rot_x,rot_x_dot, y,vy,rot_y,rot_y_dot, z,vx,rot_z,rot_z_dot,]
    drone_init = np.array([
        cam_init_pos[0], 0, cam_rpy[0]-np.pi/2, 0, 
        cam_init_pos[1], 0, cam_rpy[1], 0, 
        cam_init_pos[2], 0, cam_rpy[2]+np.pi/2, 0, 
    ])

    ref_init = np.array([0,4])

    state = drone_init 
    ref = ref_init 
    traj = np.concatenate(([0], drone_init, drone_init, ref_init)).reshape((1,-1))
     
    est = []

    PF_pose_est_history_x = []
    PF_pose_est_history_y = []
    PF_pose_est_history_z = []

    state_history = []
    for i in range(30):
        # gt_state = [x y z vx vy vz]
        gt_state = np.array([state[0], state[4], state[8], state[2], state[6], state[10]])
        est_state = mcl.rgb_run(current_pose= gt_state)  
        est.append(est_state)
        est_state[5] = est_state[5]%(2*np.pi)

        if est_state[5] > np.pi/2:
            est_state[5] -= np.pi*2
        # NEED CHANGE EST_STATE format from return
        est_state_full = np.array([
            est_state[0], #Estimate x
            state[1], #Drone Vx
            est_state[3], #Estimate Vx
            state[3], #Drone rot_x_dot
            est_state[1],#Estimate y
            state[5], #Drone Vy
            est_state[4], #Estimate Vy
            state[7], #Drone rot_y_dot
            est_state[2], #Estimate z
            state[9], #Drone Vz
            est_state[5], #Estimate Vx
            state[11], #Drone rot_z_dot
        ])
        init = np.concatenate((state, est_state_full, ref))
  
        trace = drone_agent.TC_simulate(init, time_horizon=0.1, time_step=0.01)
        state = trace[-1,1:13]
        ref = trace[-1, 25:]
        
        lstate = trace[-1,1:]
        ltime = i*0.1
        lstate = np.insert(lstate, 0, ltime).reshape((1,-1))
        traj = np.vstack((traj,lstate))

        state_history.append(state)
        # mcl.rgb_run()
        print("I",i)
    

    state_history = np.array(state_history)
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    t = np.linspace(0, 32, 1000)
    x = drone_agent.ref_traj[0](t)
    y = drone_agent.ref_traj[1](t)
    z = drone_agent.ref_traj[2](t)
    ax.plot(x, y, z, color='b')
    ax.plot(state_history[:,0], state_history[:,4], state_history[:,8], color='r')
    plt.show()



    # plt.figure(1)
    # # ax.plot(x,y,z, color = 'b')
    # # ax.plot(traj[:,1], traj[:,5], traj[:,9], color = 'r')
    # yaw_ref = []
    # yaw_act = []
    # for i in range(len(traj)):
    #     x, y, z = traj[i, 1], traj[i, 5], traj[i, 9]
    #     yaw = traj[i, 11]
    #     offset_x = 0.1*np.cos(yaw)
    #     offset_y = 0.1*np.sin(yaw)
    #     plt.figure(1)
    #     # ax.plot([x, x+offset_x], [y, y+offset_y], [z, z], 'g')
    #     yaw_act.append(yaw)

    #     t = traj[i, 25]
    #     x = drone_agent.ref_traj[0](t)
    #     y = drone_agent.ref_traj[1](t)
    #     z = drone_agent.ref_traj[2](t)

    #     xn = drone_agent.ref_traj[0](t+0.01)
    #     yn = drone_agent.ref_traj[1](t+0.01)
    #     yaw = np.arctan2(yn-y, xn-x)
    #     yaw = yaw%(np.pi*2)
    #     if yaw > np.pi/2:
    #         yaw -= 2*np.piq

    #     yaw_ref.append(yaw)
    #     offset_x = 0.1*np.cos(yaw)
    #     offset_y = 0.1*np.sin(yaw)
    #     plt.figure(1)
    #     ax.plot([x, x+offset_x], [y, y+offset_y], [z, z], 'teal')
    #     # ax.scatter([x],[y],[z],color = 'm')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # # ax.legend()

    # # plt.figure(2)
    # # plt.plot(traj[1:,15], label='est')
    # # plt.plot(traj[:-1,3], label='act')
    # # plt.title('roll')
    # # plt.legend()

    # # plt.figure(3)
    # # plt.plot(traj[1:,19], label='est')
    # # plt.plot(traj[:-1,7], label='act')
    # # plt.title('pitch')
    # # plt.legend()

    # # plt.figure(4)
    # # plt.plot(traj[1:,23], label='est')
    # # plt.plot(traj[:-1,11], label='act')
    # # plt.title('yaw')
    # # plt.legend()

    # plt.show()
