from __future__ import print_function, division
import controller_m.quadrocoptertrajectory as quadtraj
# import quadrocoptertrajectory as quadtraj

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


class Generate:
    def __init__(self):
        # Define the input limits:
        self.fmin = 5  #[m/s**2]
        self.fmax = 25 #[m/s**2]
        self.wmax = 20 #[rad/s]
        self.minTimeSec = 0.02 #[s]

        # Define how gravity lies:
        self.gravity = [0,0,-9.81]
        

    def generate_traj(self,starting_state, goal_state, Tf=1, vis = False):
        # duration Tf
        # trajectory starting state & end state convention:
        # [x, y, z, vx, vy, vz, ax, ay, az]

        # trajectory starting state
        pos0 = starting_state[:3]
        vel0 = starting_state[3:6]
        acc0 = starting_state[6:9]

        # trajectory goal state:
        posf = goal_state[:3]
        velf = goal_state[3:6]
        accf = goal_state[6:9]

        traj = quadtraj.RapidTrajectory(pos0, vel0, acc0, self.gravity)
        traj.set_goal_position(posf)
        traj.set_goal_velocity(velf)
        traj.set_goal_acceleration(accf)
    
        # Note: if you'd like to leave some states free, there are two options to 
        # encode this. As example, we will be leaving the velocity in `x` (axis 0)
        # free:
        #
        # Option 1: 
        # traj.set_goal_velocity_in_axis(1,velf_y);
        # traj.set_goal_velocity_in_axis(2,velf_z);
        # 
        # Option 2:
        # traj.set_goal_velocity([None, velf_y, velf_z])
        
        # Run the algorithm, and generate the trajectory.
        traj.generate(Tf)

        # Test input feasibility
        inputsFeasible = traj.check_input_feasibility(self.fmin, self.fmax, self.wmax, self.minTimeSec)

        # Test whether we fly into the floor
        floorPoint  = [0,0,0]  # a point on the floor
        floorNormal = [0,0,1]  # we want to be in this direction of the point (upwards)
        positionFeasible = traj.check_position_feasibility(floorPoint, floorNormal)
        
        # for i in range(3):
        #     print("Axis #" , i)
        #     print("\talpha = " ,traj.get_param_alpha(i), "\tbeta = "  ,traj.get_param_beta(i), "\tgamma = " ,traj.get_param_gamma(i))
        # print("Total cost = " , traj.get_cost())
        # print("Input feasibility result: ",    quadtraj.InputFeasibilityResult.to_string(inputsFeasible),   "(", inputsFeasible, ")")
        # print("Position feasibility result: ", quadtraj.StateFeasibilityResult.to_string(positionFeasible), "(", positionFeasible, ")")

        numPlotPoints = 100
        time = np.linspace(0, Tf, numPlotPoints)
        position = np.zeros([numPlotPoints, 3])
        velocity = np.zeros([numPlotPoints, 3])
        acceleration = np.zeros([numPlotPoints, 3])
        thrust = np.zeros([numPlotPoints, 1])
        ratesMagn = np.zeros([numPlotPoints,1])

        for i in range(numPlotPoints):
            t = time[i]
            position[i, :] = traj.get_position(t)
            velocity[i, :] = traj.get_velocity(t)
            acceleration[i, :] = traj.get_acceleration(t)
            thrust[i] = traj.get_thrust(t)
            ratesMagn[i] = np.linalg.norm(traj.get_body_rates(t))

        
    
        if vis == True:
            
            ###########################################
            # Plot the trajectories, and their inputs #
            ###########################################

            # Create a figure and a 3D axis
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Plot the 3D surface
            ax.plot(position[:,0],position[:,1],position[:,2])

            # Set labels and title
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            ax.set_title('3D Trajectory Plot')
            # Show the plot
            plt.show()


            figStates, axes = plt.subplots(3,1,sharex=True)
            gs = gridspec.GridSpec(6, 2)
            axPos = plt.subplot(gs[0:2, 0])
            axVel = plt.subplot(gs[2:4, 0])
            axAcc = plt.subplot(gs[4:6, 0])

            for ax,yvals in zip([axPos, axVel, axAcc], [position,velocity,acceleration]):
                cols = ['r','g','b']
                labs = ['x','y','z']
                for i in range(3):
                    ax.plot(time,yvals[:,i],cols[i],label=labs[i])

            axPos.set_ylabel('Pos [m]')
            axVel.set_ylabel('Vel [m/s]')
            axAcc.set_ylabel('Acc [m/s^2]')
            axAcc.set_xlabel('Time [s]')
            axPos.legend()
            axPos.set_title('States')

            infeasibleAreaColour = [1,0.5,0.5]
            axThrust = plt.subplot(gs[0:3, 1])
            axOmega  = plt.subplot(gs[3:6, 1])
            axThrust.plot(time,thrust,'k', label='command')
            axThrust.plot([0,Tf],[self.fmin,self.fmin],'r--', label='fmin')
            axThrust.fill_between([0,Tf],[self.fmin,self.fmin],-1000,facecolor=infeasibleAreaColour, color=infeasibleAreaColour)
            axThrust.fill_between([0,Tf],[self.fmax,self.fmax], 1000,facecolor=infeasibleAreaColour, color=infeasibleAreaColour)
            axThrust.plot([0,Tf],[self.fmax,self.fmax],'r-.', label='fmax')

            axThrust.set_ylabel('Thrust [m/s^2]')
            axThrust.legend()

            axOmega.plot(time, ratesMagn,'k',label='command magnitude')
            axOmega.plot([0,Tf],[self.wmax,self.wmax],'r--', label='wmax')
            axOmega.fill_between([0,Tf],[self.wmax,self.wmax], 1000,facecolor=infeasibleAreaColour, color=infeasibleAreaColour)
            axOmega.set_xlabel('Time [s]')
            axOmega.set_ylabel('Body rates [rad/s]')
            axOmega.legend()

            axThrust.set_title('Inputs')

            #make the limits pretty:
            axThrust.set_ylim([min(self.fmin-1,min(thrust)), max(self.fmax+1,max(thrust))])
            axOmega.set_ylim([0, max(self.wmax+1,max(ratesMagn))])

            plt.show()

        return position, velocity

    
    

     

if __name__ == "__main__":
    # pos0 = [0, 0, 2] #position
    # vel0 = [0, 0, 0] #velocity
    # acc0 = [0, 0, 0] #acceleration

    # # trajectory goal state:
    # posf = [1, 0, 1]  # position
    # velf = [0, 0, 1]  # velocity
    # accf = [0, 9.81, 0]  # acceleration

    start_state = [0,0,2,0,0,0,0,0,0]
    goal_state = [1,0,1,0,0,1,0,9.81,0]

    d1 = Generate()
    d1.generate_traj(starting_state=start_state, goal_state= goal_state, vis=True)