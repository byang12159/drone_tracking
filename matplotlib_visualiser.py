import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
from matplotlib.patches import Circle

import math
import cv2 as cv

#from utils import transformations
def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True

    """
    import numpy
    import math
    q = numpy.array(quaternion[:4], dtype=numpy.float64, copy=True)
    nq = numpy.dot(q, q)
    _EPS = numpy.finfo(float).eps * 4.0
    if nq < _EPS:
        return numpy.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = numpy.outer(q, q)
    return numpy.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=numpy.float64)
def interpolate(p_from, p_to, num):
    direction = (p_to - p_from) / np.linalg.norm(p_to - p_from)
    distance = np.linalg.norm(p_to - p_from) / (num - 1)

    ret_vec = []

    for i in range(0, num):
        ret_vec.append(p_from + direction * distance * i)

    return np.array(ret_vec)

def plotImage(ax, img, R, t, size=np.array((1/4, 1/4)), img_scale=1):
    img_size = (np.array((img.shape[0], img.shape[1])) / img_scale).astype('int32')
    img = cv.resize(img, ((img_size[1], img_size[0])))

    corners = np.array(([0., 0, 0], [0, size[0], 0],
                        [size[1], 0, 0], [size[1], size[0], 0]))

    corners += t
    corners = corners @ R
    xx = np.zeros((img_size[0], img_size[1]))
    yy = np.zeros((img_size[0], img_size[1]))
    zz = np.zeros((img_size[0], img_size[1]))
    l1 = interpolate(corners[0], corners[2], img_size[0])
    xx[:, 0] = l1[:, 0]
    yy[:, 0] = l1[:, 1]
    zz[:, 0] = l1[:, 2]
    l1 = interpolate(corners[1], corners[3], img_size[0])
    xx[:, img_size[1] - 1] = l1[:, 0]
    yy[:, img_size[1] - 1] = l1[:, 1]
    zz[:, img_size[1] - 1] = l1[:, 2]

    for idx in range(0, img_size[0]):
        p_from = np.array((xx[idx, 0], yy[idx, 0], zz[idx, 0]))
        p_to = np.array((xx[idx, img_size[1] - 1], yy[idx, img_size[1] - 1], zz[idx, img_size[1] - 1]))
        l1 = interpolate(p_from, p_to, img_size[1])
        xx[idx, :] = l1[:, 0]
        yy[idx, :] = l1[:, 1]
        zz[idx, :] = l1[:, 2]

    ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, facecolors=img / 255, shade=False)
    return None

marker_GT = [0.19199954223632812, 0.027038570404052734, 0.4160528869628906, 0.00018454967439174652, -0.000207060843706131, -0.00027632442116737365, 0.007449865818023682, -0.010549224853515626, -0.014547050476074219, 3.1265869140625, 0.0030364990234375, -0.02341461181640625, 0.00458526611328125, -0.0071258544921875, 0.9996871948242188, 1709134152.8482578]
img = cv.imread("aruco2.png")

with open("datacollect1.pkl",'rb') as handle:
    data = pickle.load(handle)
marker_GT = data[0][1]

print(len(data))
idx_spotted = []
Ts_full = []
snapstate_full = []
target_ID = 0

for j in range(len(data)):
    if len(data[j][3][0]) != 0:
        for a in range(len(data[j][3][1])):
            if data[j][3][1][a] == target_ID:
                # print(data[j][2][0][a])
                idx_spotted.append(j)
                Ts_full.append(data[j][3][0][a])
                snapstate_full.append(data[j][2])

snapstate = snapstate_full[3]
Ts = Ts_full[3]
# Unit vectors along axes
xaxis = np.array([1, 0, 0])
yaxis = np.array([0, 1, 0])
zaxis = np.array([0, 0, 1])

q = marker_GT[11:15] # x, y, z, w
T_WM = quaternion_matrix(q)
T_WM[:3, 3] = marker_GT[:3]

scale = 1
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot(0,0,0, 'x',color='red',label='Vicon Zero')
ax.plot(snapstate[0],snapstate[1],snapstate[2],'x',color='blue',label='Drone Ground Truth')
ax.plot(marker_GT[0],marker_GT[1],marker_GT[2],'x',color='cyan',label='Marker Ground Truth')
ax.plot([0,xaxis[0]],[0,xaxis[1]],[0,xaxis[2]],color='red')
ax.plot([0,yaxis[0]],[0,yaxis[1]],[0,yaxis[2]],color='green')
ax.plot([0,zaxis[0]],[0,zaxis[1]],[0,zaxis[2]],color='blue')
########### PLOT for Marker Poses ###########
xaxis_h = np.array([1,0,0,1])
yaxis_h = np.array([0,1,0,1])
zaxis_h = np.array([0,0,1,1])

marker_center = T_WM[:4,3]
marker_tail_x = T_WM@xaxis_h
marker_tail_y = T_WM@yaxis_h
marker_tail_z = T_WM@zaxis_h

R_rad = np.array((0.0, 90.0, 0.0)) * math.pi / 180
R = cv.Rodrigues(R_rad)[0]

print(R)

t = np.array(([R[0][0]*marker_GT[0]+R[0][1]*marker_GT[1]+R[0][2]*marker_GT[2], R[1][0]*marker_GT[0]+R[1][1]*marker_GT[1]+R[1][2]*marker_GT[2], R[2][0]*marker_GT[0]+R[2][1]*marker_GT[1]+R[2][2]*marker_GT[2]]))

plotImage(ax, img,R,t)
# ########### PLOT for Camera Poses ###########
translation = Ts[:3,3]
rotation = np.linalg.inv(Ts[:3,:3])
Ts_inv = np.linalg.inv(Ts)

aruco_head =   T_WM@Ts_inv.dot(np.array([0,0,0,1]))
aruco_tail_x = T_WM@Ts_inv.dot(xaxis_h*scale)
aruco_tail_y = T_WM@Ts_inv.dot(yaxis_h*scale)
aruco_tail_z = T_WM@Ts_inv.dot(zaxis_h*scale)

ax.plot(aruco_head[0],aruco_head[1],aruco_head[2],'x',color='green',label='Calculated Camera Pose')



local_x_axis = [aruco_tail_x[0] - aruco_head[0], aruco_tail_x[1] - aruco_head[1], aruco_tail_x[2] - aruco_head[2]]
local_y_axis = [aruco_tail_y[0] - aruco_head[0], aruco_tail_y[1] - aruco_head[1], aruco_tail_y[2] - aruco_head[2]]

# Normalize the local axes to ensure they are of unit length
norm_local_x_axis = local_x_axis / np.linalg.norm(local_x_axis)
norm_local_y_axis = local_y_axis / np.linalg.norm(local_y_axis)


side_length_square = 0.075  # New, smaller size for the square
side_length_triangle = 0.1  # Side length for the triangle (adjust as needed)
d = side_length_square / 2  # Half the side length of the square

# Define the vertices of the square such that its center is at the drone_head
local_square = [(d, d), (d, -d), (-d, -d), (-d, d)]

# Define the vertices of the triangle such that one corner touches the middle of one side of the square
# and the triangle extends outward
offset_triangle = d  # The distance from the center of the square to the touching point on the square
local_triangle = [(d, 0),  # The corner touching the middle of the square's side
                  (d + side_length_triangle * np.sqrt(3) / 2, -side_length_triangle / 2),  # Other vertices inverted
                  (d + side_length_triangle * np.sqrt(3) / 2, side_length_triangle / 2)]

# Continue with the transformation and plotting as previously described
def transform_to_global(local_vertices, drone_head, norm_local_x_axis, norm_local_y_axis):
    global_vertices = []
    for vertex in local_vertices:
        global_x = drone_head[0] + vertex[0] * norm_local_x_axis[0] + vertex[1] * norm_local_y_axis[0]
        global_y = drone_head[1] + vertex[0] * norm_local_x_axis[1] + vertex[1] * norm_local_y_axis[1]
        global_z = drone_head[2] + vertex[0] * norm_local_x_axis[2] + vertex[1] * norm_local_y_axis[2]
        global_vertices.append((global_x, global_y, global_z))
    return global_vertices

# Transform and plot the square
global_square = transform_to_global(local_square, aruco_head, norm_local_x_axis, norm_local_y_axis)
square_xs, square_ys, square_zs = zip(*global_square)
square_xs += (square_xs[0],)
square_ys += (square_ys[0],)
square_zs += (square_zs[0],)
ax.plot(square_xs, square_ys, square_zs, color='red')

# Transform and plot the triangle
global_triangle = transform_to_global(local_triangle, aruco_head, norm_local_x_axis, norm_local_y_axis)
triangle_xs, triangle_ys, triangle_zs = zip(*global_triangle)
triangle_xs += (triangle_xs[0],)
triangle_ys += (triangle_ys[0],)
triangle_zs += (triangle_zs[0],)
ax.plot(triangle_xs, triangle_ys, triangle_zs, color='red')


########### PLOT for Drone Poses ###########
T_WD = quaternion_matrix(snapstate[11:15])
T_WD[:3,3] = snapstate[:3]
drone_axes_scale = 1

drone_head = T_WD[:3,3]
drone_axes_tip_x = T_WD@xaxis_h
drone_axes_tip_y = T_WD@yaxis_h
drone_axes_tip_z = T_WD@zaxis_h


local_x_axis = [drone_axes_tip_x[0] - drone_head[0], drone_axes_tip_x[1] - drone_head[1], drone_axes_tip_x[2] - drone_head[2]]
local_y_axis = [drone_axes_tip_y[0] - drone_head[0], drone_axes_tip_y[1] - drone_head[1], drone_axes_tip_y[2] - drone_head[2]]

# Normalize the local axes to ensure they are of unit length
norm_local_x_axis = local_x_axis / np.linalg.norm(local_x_axis)
norm_local_y_axis = local_y_axis / np.linalg.norm(local_y_axis)

# Parameters for the circles
num_points = 100  # Number of points that make up each circle
radius = 0.03  # Radius of each circle
d = 0.1  # Distance between the centers of the circles

# Define the centers of the four circles in local space (relative to drone_head)
local_centers = [(-d/2 - 0.05, -d/2), (-d/2 - 0.05, d/2), (d/2 + 0.05, -d/2), (d/2 + 0.05, d/2)]

theta = np.linspace(0, 2*np.pi, num_points)

for center in local_centers:
    # Calculate the circle points in local space
    x_local_circle = center[0] + radius * np.cos(theta)
    y_local_circle = center[1] + radius * np.sin(theta)

    # Transform circle points to global space
    global_circle_x = drone_head[0] + x_local_circle * norm_local_x_axis[0] + y_local_circle * norm_local_y_axis[0]
    global_circle_y = drone_head[1] + x_local_circle * norm_local_x_axis[1] + y_local_circle * norm_local_y_axis[1]
    global_circle_z = drone_head[2] + x_local_circle * norm_local_x_axis[2] + y_local_circle * norm_local_y_axis[2]

    # Plot each circle in global space
    ax.plot(global_circle_x, global_circle_y, global_circle_z, color='blue')


ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(elev=13, azim=123, roll=0)

plt.savefig('sampleFileName4.png')
print(snapstate[0]-aruco_head[0],snapstate[1]-aruco_head[1],snapstate[2]-aruco_head[2])
# plt.legend()
# plt.show()


