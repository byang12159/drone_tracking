import numpy as np
import pyvista as pv
# print(pv.Report())
from PIL import Image
import cv2
import math
from scipy.spatial.transform import Rotation as R
import cv2.aruco as aruco
import sys, os
import pickle
import time
from utils.transformations import euler_from_matrix
def quaternion_to_euler(w, x, y, z):
    # Calculate roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # Calculate pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp) # use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)
    # Calculate yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

def detect_aruco(frame, save="output1.png", visualize=True, marker_size=62):
    # ret, frame = cap.read()
    scale = 1
    # frame = cv2.imread("screenshot.png")
    # width = int(frame.shape[1] * scale)
    # height = int(frame.shape[0] * scale)
    # dim = (width, height)
    # frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    # parameters = aruco.DetectorParameters_create()
    # markerCorners, markerIds, rejectedCandidates= aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dictionary, parameters)
    markerCorners, markerIds, rejectedCandidates= detector.detectMarkers(gray)
    frame = aruco.drawDetectedMarkers( frame, markerCorners, markerIds )
    Ts = []
    ids = []
    camera_mtx = np.array([[4000.0,  0.,         1024/2],
                        [  0. ,        4000.0, 768/2],
                        [  0.   ,        0.         ,  1.        ]])
    # camera_mtx = np.identity(3)
    # camera_mtx = np.array([[1.95553717e+04, 0.00000000e+00, 5.14662975e+02],
    #                       [0.00000000e+00, 5.61599540e+04, 3.33162595e+02],
    #                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    # camera_mtx = np.array([[886.73363353,   0.        , 511.5],
    #                       [  0.        , 665.10751011, 383.5],
    #                       [  0.        ,   0.        ,   1. ]])
    #distortion_param = np.array([[-3.69778027e+03, -1.23141160e-01,  1.46877989e+01, -7.97192259e-02, -3.28441832e-06]])
    # distortion_param = np.array([0, 0, 0, 0, 0])
    distortion_param = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
    if markerIds is not None:
        rvecs, tvecs, objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, markerLength=marker_size, cameraMatrix=camera_mtx, distCoeffs=distortion_param)
        # rvecs, tvecs, objpoints = aruco.estimatePoseSingleMarkers(markerCorners, marker_size, , )
        for i in range(len(markerIds)):
            # Unpacking the compounded list structure
            rvec = rvecs[i][0]
            tvec = tvecs[i][0]
            # print("Rvec",rvec)
            # print("Tvec",tvec)
            ids.append(markerIds[i][0])
            if save or visualize:
                print("a")
                frame = cv2.drawFrameAxes(frame, camera_mtx, distortion_param, rvec, tvec,length = 100, thickness=6)
            rotation_mtx, jacobian = cv2.Rodrigues(rvec)
            translation = tvec
            T = np.identity(4)
            T[:3, :3] = rotation_mtx
            T[:3, 3] = translation 
            Ts.append(T)
    if save:
        cv2.imwrite(save, frame)
    if visualize:
        cv2.imshow("camera view", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # print(markerIds)
    # Multiple ids in 1D list
    # Mutiple Ts, select first marker 2d array by Ts[0]
    # print(Ts)
    return Ts, ids,tvecs


###################################################################################################################
# Creating the visualization

image = cv2.imread('aruco_0.png')  # Change 'example.jpg' to the path of your image
# aruco_0: shape (388,388,3), marker itself is (378x378), with addition of 5 pixels of white buffer space on every side

# Create a PyVista texture
texture = pv.numpy_to_texture(image)
# Create a plane surface mesh
plane = pv.Plane(i_size=image.shape[0], j_size=image.shape[1])
# Apply the texture to the surface mesh
plane.texture_map_to_plane(inplace=True)
# Apply the transformation
# plane.transform(transformation_matrix)

camera = pv.Camera()


plotter = pv.Plotter()

# Mesh plotted at point (0,0,0)
plotter.add_mesh(plane, texture=texture)

# Plot a random point
# origin = np.array([[0.0, 0.0, 0.0]]) 
# origin_poly = pv.PolyData(origin)
# plotter.add_mesh(origin_poly, color="red", point_size=10, render_points_as_spheres=True)


# plotter.view_xy()
# plotter.camera = camera
# plotter.camera.position = (0,0,3000)
# plotter.camera.focal_point = (0.0, 0.0, 0.0)
# plotter.camera.view_angle = 60.0

plotter.camera.up = (0,0,1)
print("camera up",plotter.camera.up)
x = 0
y = 0
z = 2000
yaw = 0
pitch = 0
roll = 0
plotter.camera_position = [(x, y, z), (0,0,0), (yaw, pitch, roll)]
plotter.camera.roll = 45.0
plotter.camera.azimuth = 45.0
plotter.camera.elevation = 45.0

# plotter.renderer.ResetCameraClippingRange()

# print("camera up",plotter.camera.up)
img_plot = plotter.show(screenshot="screenshot_vista.png", return_img=True) 
# note for windows: Please use the q-key to close the plotter as some operating systems (namely Windows) will experience issues saving a screenshot if the exit button in the GUI is pressed.

# plotter.show(screenshot='airplane2.png')

print(img_plot.shape)

# frame = cv2.imread("screenshot_vista.png")
# Ts, ids, tvecs = detect_aruco(frame)
# # cv2.imshow("img",img_plot)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# print("ID found: ",ids)
# print("Returned TS: ",Ts[0])
# print("Tvec", tvecs)

# euler = np.array(euler_from_matrix(Ts[0][:3,:3])) * 180/np.pi
# print("euler, ", euler)