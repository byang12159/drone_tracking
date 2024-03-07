# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# def compute_cam_bound(depth):
#     # Given FOV of pinhole camera and distance from camera, computes the rectangle range of observable image
#     fov_h = 100 #degrees
#     fov_d = 138 #degrees

#     rec_width = 2 * (np.tan(np.deg2rad(fov_h/2)) * depth )
#     b = 2 * (np.tan(np.deg2rad(fov_d/2)) * depth )
#     rec_height = np.sqrt(b**2 - rec_width**2)

#     return rec_width,rec_height

# print(compute_cam_bound(0.5))

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import lti, step

# # Define the transfer function parameters
# numerator = [1]
# denominator = [1, 2, 1]  # Example second-order system with no damping (overshoot)
# system = lti(numerator, denominator)

# # Define the time points for simulation
# t = np.linspace(0, 10, 1000)

# # Simulate the step response
# t, response = step(system, T=t)

# # Plot the step response
# plt.plot(t, response)
# plt.xlabel('Time')
# plt.ylabel('Response')
# plt.title('Step Response')
# plt.grid(True)
# plt.show()
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import TransferFunction, step

# # Define desired rise time and settling time
# tr_desired = 1.0  # Desired rise time in seconds

# # Calculate the desired dominant poles using rise time
# zeta = 0.707  # Desired damping ratio (for a critically damped system)
# wn = 2.2 / tr_desired  # Desired natural frequency
# poles = np.roots([1, 2 * zeta * wn, wn ** 2])  # Calculate the poles

# # Create the transfer function
# numerator = [1]  # Assuming unity gain
# denominator = [1, 2 * zeta * wn, wn ** 2]  # Second-order system
# system = TransferFunction(numerator, denominator)

# # Define the time points for simulation
# t = np.linspace(0, 5, 1000)

# # Simulate the step response
# t, response = step(system, T=t)

# # Plot the step response
# plt.plot(t, response)
# plt.xlabel('Time (s)')
# plt.ylabel('Response')
# plt.title('Step Response')
# plt.grid(True)
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import TransferFunction, step

# # Define desired rise time and settling time
# tr_desired = 2.0  # Desired rise time in seconds

# # Calculate the desired dominant poles using rise time
# zeta = 0.707  # Desired damping ratio (for a critically damped system)
# wn = 2.2 / tr_desired  # Desired natural frequency
# poles = np.roots([1, 2 * zeta * wn, wn ** 2])  # Calculate the poles

# # Create the transfer function
# numerator = [1]  # Assuming unity gain
# denominator = [1, 2 * zeta * wn, wn ** 2]  # Second-order system
# system = TransferFunction(numerator, denominator)

# # Define the time points for simulation
# t = np.linspace(0, 10, 1000)

# # Simulate the step response
# t, response = step(system, T=t)

# # Plot the step response
# plt.plot(t, response)
# plt.xlabel('Time (s)')
# plt.ylabel('Response')
# plt.title('Step Response')
# plt.grid(True)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

def excitation(initial_V, acceleration,visualize = False):
    # Sample acceleration vs. time data
    t = np.linspace(0, 10, 100)  # Time points
    a = np.zeros(100)  # Example acceleration function (you should replace this with your data)
    a[40:50] = acceleration

    # Numerical integration to obtain velocity
    dt = t[1] - t[0]
    v = np.zeros_like(a)
    v[0] = initial_V
    for i in range(1, len(t)):
        v[i] = v[i-1] + 0.5 * (a[i] + a[i-1]) * dt

    dt = t[1] - t[0]
    x = np.zeros_like(v)
    for i in range(1, len(t)):
        x[i] = x[i-1] + 0.5 * (v[i] + v[i-1]) * dt

    if visualize:
        # Plot acceleration vs. time and velocity vs. time graphs
        plt.figure(figsize=(10, 8))
        plt.subplot(3, 1, 1)
        plt.plot(t, a, 'b-')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (m/s^2)')
        plt.title('Acceleration vs. Time')

        plt.subplot(3, 1, 2)
        plt.plot(t, v, 'r-')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.title('Velocity vs. Time')

        plt.subplot(3, 1, 3)
        plt.plot(t, x, 'r-')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.title('Position vs. Time')

        plt.tight_layout()
        plt.show()

    return v

while __name__ == "__main__":
    excitation(2)

