import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def circle(totalcount):
    radius = 2
    start=[30,20]
    t = np.linspace(0,2*np.pi,totalcount)
    x = 30-radius - radius * np.cos(t)
    y = 20-radius * np.sin(t)

    plt.plot(x, y)
    plt.plot(30,20,'x',color='red')

    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Sine Wave')

    # Show the plot
    plt.show()

circle(200)