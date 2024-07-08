import csv
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np

def graphing_states(timestep):        

    variance_history=np.array(variance_history)
    fig, (xval, yval, zval) = plt.subplots(3, 1, figsize=(14, 10))
    times = np.arange(0,variance_history.shape[0])*timestep
    xval.plot(times, variance_history[:,0], label = "Var x")
    xval.plot(times, GT_NED_states_L[:,0], label = "GT Pos x")
    xval.plot(times, GT_NED_states_L[:,3], label = "GT Vel x")
    xval.plot(times, GT_NED_states_L[:,6], label = "GT Accel x")
    xval.legend()
    yval.plot(times, variance_history[:,1], label = "Var y")  
    yval.plot(times, GT_NED_states_L[:,1], label = "GT Pos y")
    yval.plot(times, GT_NED_states_L[:,4], label = "GT Vel y")
    yval.plot(times, GT_NED_states_L[:,7], label = "GT Accel y")
    yval.legend()
    zval.plot(times, variance_history[:,2], label = "Var z")
    zval.plot(times, GT_NED_states_L[:,2], label = "GT Pos z")
    zval.plot(times, GT_NED_states_L[:,5], label = "GT Vel z")
    zval.plot(times, GT_NED_states_L[:,8], label = "GT Accel z")
    zval.legend()
    plt.show()

    # times = np.arange(0,particle_state_est.shape[0]-2)*timestep


    fig, (posx,posy,posz,velx,vely,velz) = plt.subplots(6, 1, figsize=(16, 10))
    posx.plot(times, GT_NED_states_L[:,0], label = "GT Pos x")
    posx.legend() 
    posy.plot(times, GT_NED_states_L[:,1], label = "GT Pos y")
    posy.legend()
    posz.plot(times, GT_NED_states_L[:,2], label = "GT Pos z")
    posz.legend()
    velx.plot(times, GT_NED_states_L[:,3], label = "GT Vel x")
    velx.legend() 
    vely.plot(times, GT_NED_states_L[:,4], label = "GT Vel y")
    vely.legend()
    velz.plot(times, GT_NED_states_L[:,5], label = "GT Vel z")
    velz.legend()
    # aclx.plot(times, GT_NED_states_L[:,6], label = "GT Accel x")
    # aclx.legend() 
    # acly.plot(times, GT_NED_states_L[:,7], label = "GT Accel y")
    # acly.legend()
    # aclz.plot(times, GT_NED_states_L[:,8], label = "GT Accel z")
    # aclz.legend()
    plt.tight_layout()

    # times = np.arange(0,particle_state_est.shape[0]-2)*timestep


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

    plt.show()


    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(global_state_history_C[:,0],global_state_history_C[:,1],global_state_history_C[:,2], color='b')
    ax.plot(particle_state_est[2:,0],particle_state_est[2:,1],particle_state_est[2:,2],'o',color='red')
    ax.plot(global_state_history_L[:,0],global_state_history_L[:,1],global_state_history_L[:,2], '*',color = 'g')
    plt.axis('equal')
    plt.legend()
    plt.show()

def PF_mean(GT_global, GT_NED, estimates, timestep=0.01):
    if not isinstance(estimates, np.ndarray):
        estimates = np.array(estimates)
   
    times = np.arange(0, estimates.shape[0])*timestep

    fig, axs = plt.subplots(3, 2, figsize=(14, 10))

    axs[0,0].plot(times, estimates[:,0], label = "PF Estimated x")
    axs[0,0].plot(times, GT_global[:,0], label = "GT Position x")
    axs[0,0].legend()
    axs[1,0].plot(times, estimates[:,1], label = "PF Estimated y")  
    axs[1,0].plot(times, GT_global[:,1], label = "GT Position y")  
    axs[1,0].legend()
    axs[2,0].plot(times, estimates[:,2], label = "PF Estimated z")
    axs[2,0].plot(times, GT_global[:,2], label = "GT Position z")
    axs[2,0].legend()
    
    axs[0,1].plot(times, estimates[:,3], label = "PF Estimated vx")
    axs[0,1].plot(times, GT_NED[:,3], label = "GT Position vx")
    axs[0,1].legend()
    axs[1,1].plot(times, estimates[:,4], label = "PF Estimated vy")  
    axs[1,1].plot(times, GT_NED[:,4], label = "GT Position vy")  
    axs[1,1].legend()
    axs[2,1].plot(times, estimates[:,5], label = "PF Estimated vz")
    axs[2,1].plot(times, GT_NED[:,5], label = "GT Position vz")
    axs[2,1].legend()
    plt.show()

def PF_mean_error(estimates,states, timestep=0.01):
    if not isinstance(estimates, np.ndarray):
        estimates = np.array(estimates)
    if not isinstance(states, np.ndarray):
        states = np.array(states)

    error = estimates - states

    times = np.arange(0, error.shape[0]*timestep, timestep)

    fig, (errx, erry, errz) = plt.subplots(3, 1, figsize=(14, 10))

    errx.plot(times, error[:,0], label = "Error x")
    errx.legend()
    erry.plot(times, error[:,1], label = "Error y")  
    erry.legend()
    errz.plot(times, error[:,2], label = "Error z")
    errz.legend()
    plt.show()

def prepare_GT_NED_states(NED_state_history_L, NED_state_history_C):
    GT_NED_states_L = []
    GT_NED_states_C = []
    for j in range(len(NED_state_history_C)):
        posx_L = NED_state_history_L[j].position.x_val
        posy_L = NED_state_history_L[j].position.y_val
        posz_L = NED_state_history_L[j].position.z_val
        velx_L = NED_state_history_L[j].linear_velocity.x_val
        vely_L = NED_state_history_L[j].linear_velocity.y_val
        velz_L = NED_state_history_L[j].linear_velocity.z_val
        aclx_L = NED_state_history_L[j].linear_acceleration.x_val
        acly_L = NED_state_history_L[j].linear_acceleration.y_val
        aclz_L = NED_state_history_L[j].linear_acceleration.z_val
        
        posx_C = NED_state_history_C[j].position.x_val
        posy_C = NED_state_history_C[j].position.y_val
        posz_C = NED_state_history_C[j].position.z_val
        velx_C = NED_state_history_C[j].linear_velocity.x_val
        vely_C = NED_state_history_C[j].linear_velocity.y_val
        velz_C = NED_state_history_C[j].linear_velocity.z_val
        aclx_C = NED_state_history_C[j].linear_acceleration.x_val
        acly_C = NED_state_history_C[j].linear_acceleration.y_val
        aclz_C = NED_state_history_C[j].linear_acceleration.z_val

        GT_NED_states_L.append([posx_L,posy_L,posz_L,velx_L,vely_L,velz_L,aclx_L,acly_L,aclz_L])
        GT_NED_states_C.append([posx_C,posy_C,posz_C,velx_C,vely_C,velz_C,aclx_C,acly_C,aclz_C])

    GT_NED_states_L = np.array(GT_NED_states_L)
    GT_NED_states_C = np.array(GT_NED_states_C)

    return GT_NED_states_L, GT_NED_states_C

def PF_variance(variance_history, timestep):
    # Plots variance obtained from np.var(particle['position'])
    times = np.arange(0, variance_history.shape[0])*timestep

    fig, (x,y,z) = plt.subplots(3, 1, figsize=(14, 10))
    x.plot(times, variance_history[:,0], label = "Variance x")
    x.legend()
    y.plot(times, variance_history[:,1], label = "Variance y")  
    y.legend()
    z.plot(times, variance_history[:,2], label = "Variance z")
    z.legend()
    plt.tight_layout()
    plt.show()

def PF_vel_particles(particles, GT_global, GT_NED, estimates, timestep=0.01):
    if not isinstance(estimates, np.ndarray):
        estimates = np.array(estimates)
   
    times = np.arange(0, estimates.shape[0])*timestep
    try:
        for j in range(0, len(times), 40):
            fig, axs = plt.subplots(3, 2, figsize=(14, 10))

            t = np.ones(particles[j].shape[0]) * times[j]

            axs[0,0].plot(times, estimates[:,0], label = "PF Estimated x")
            axs[0,0].plot(times, GT_global[:,0], label = "GT pos x")
            axs[0,0].plot(t,particles[j][:,0],'*','g')
            axs[0,0].legend()
            axs[1,0].plot(times, estimates[:,1], label = "PF Estimated y")  
            axs[1,0].plot(times, GT_global[:,1], label = "GT pos y")  
            axs[1,0].plot(t,particles[j][:,1],'*','g')
            axs[1,0].legend()
            axs[2,0].plot(times, estimates[:,2], label = "PF Estimated z")
            axs[2,0].plot(times, GT_global[:,2], label = "GT pos z")
            axs[2,0].plot(t,particles[j][:,2],'*','g')
            axs[2,0].legend()


            axs[0,1].plot(times, estimates[:,3], label = "PF Estimated vx")
            axs[0,1].plot(times, GT_NED[:,3], label = "GT Vel x")
            axs[0,1].plot(t,particles[j][:,3],'*','g')
            axs[0,1].legend()
            axs[1,1].plot(times, estimates[:,4], label = "PF Estimated vy")  
            axs[1,1].plot(times, GT_NED[:,4], label = "GT Vel y")  
            axs[1,1].plot(t,particles[j][:,4],'*','g')
            axs[1,1].legend()
            axs[2,1].plot(times, estimates[:,5], label = "PF Estimated vz")
            axs[2,1].plot(times, GT_NED[:,5], label = "GT Vel z")
            axs[2,1].plot(t,particles[j][:,5],'*','g')
            axs[2,1].legend()

            plt.tight_layout()
            plt.show()
    except Exception as e:
            print(f"Exception: {e}")

def perpare_PF_history(PF_history):
    particles_history = []
    for i in range(len(PF_history)):
        particles_history.append(np.hstack((PF_history[i][0], PF_history[i][1])))

    return particles_history



if __name__ == '__main__':
    timestep = 0.01

    with open('data/data_prediction.pkl', 'rb') as file:
        prediction_data = pickle.load(file)

    with open('data/data_variance.pkl', 'rb') as file:
        variance_data = pickle.load(file)
    
    with open('data/data_variance_choice.pkl', 'rb') as file:
        variance_data2 = pickle.load(file)
    
    with open('data_global_lead.pkl', 'rb') as file:
        global_state_history_L = pickle.load(file)

    with open('data_NED_lead.pkl', 'rb') as file:
        NED_state_history_L = pickle.load(file)

    with open('data/data_NED_chase.pkl', 'rb') as file:
        NED_state_history_C = pickle.load(file)

    with open('data/PF_history.pkl', 'rb') as file:
        PF_history = pickle.load(file)  

    with open('data/PF_mean.pkl', 'rb') as file:
        PF_means = pickle.load(file)
        PF_means = PF_means[2:]

    # variance_history = []
    # for i in range(len(prediction_data)):
    #     variance_history.append(prediction_data[i][3])

    # variance_history = []
    # for i in range(len(prediction_data)):
    #     variance_history.append(variance_data[i])
    variance_data = np.array(variance_data)

    var2 = []
    for i in range(len(variance_data2)):
        var2.append(variance_data2[i].cpu().numpy())
    var2 = np.array(var2)

    global_state_history_L = np.array(global_state_history_L)

    GT_NED_states_L, GT_NED_states_C = prepare_GT_NED_states(NED_state_history_L, NED_state_history_C)

    # particles_history = perpare_PF_history(PF_history)


    # PF_mean(global_state_history_L, GT_NED_states_L, PF_means, timestep=timestep)
    
    # PF_vel_particles(particles_history, global_state_history_L, GT_NED_states_L, PF_means, timestep=timestep)

    var_y_max = np.argmax(variance_data[:,1])
    for i in range(GT_NED_states_L.shape[0]):
        if GT_NED_states_L[i,4] >= 0.1:
            vel_init_index = i
            break 
    times = np.arange(0, variance_data.shape[0])*timestep
    fig, (x,y,z) = plt.subplots(3, 1, figsize=(14, 10))
    x.plot(times, variance_data[:,0], label = "Variance x")
    x.plot(times, var2[:,0], label = "Variance x2")
    x.plot(times, GT_NED_states_L[:,3], label = "GT vel x")
    x.legend()
    y.plot(times, variance_data[:,1], label = "Variance y")  
    y.plot(times, var2[:,1], label = "Variance y2")  
    y.plot(times, GT_NED_states_L[:,4], label = "GT vel y")
    y.axvline(times[var_y_max], color='r', linestyle='--', label=f'Max at t={times[var_y_max]:.2f}')
    y.axvline(times[vel_init_index], color='g', linestyle='--', label=f'Vel Init t={times[vel_init_index]:.2f}')
    y.legend()
    z.plot(times, variance_data[:,2], label = "Variance z")
    z.plot(times, var2[:,2], label = "Variance z2")
    z.plot(times, GT_NED_states_L[:,5], label = "GT vel z")
    z.legend()
    plt.tight_layout()
    plt.show()

    # PF_variance(variance_data, timestep=timestep)

    

    # PF_mean_error(GT_NED_states_L, PF_means)

    # steps = np.arange(0,len(prediction_data[0][3]),1)
    # for i in range(len(prediction_data)):
    #     plt.plot(GT_NED_states_L[i][4], variance_history[i][-1],'x')
    # plt.title('Variance Plot')
    # plt.xlabel('Velocity Y')
    # plt.ylabel('Max Variance Value')

    # # Show the plot
    # plt.show()

    # for i in range(len(GT_NED_states_L)):
    #     plt.plot(steps, variance_history[i])
    # plt.title('Variance Plot')
    # plt.xlabel('Prediction Steps')
    # plt.ylabel('Variance Value')

    # # Show the plot
    # plt.show()