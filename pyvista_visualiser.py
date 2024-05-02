import numpy as np
import pyvista as pv
from PIL import Image
import cv2
import math
from scipy.spatial.transform import Rotation as R
import cv2.aruco as aruco
import sys, os

import pickle
import time



class Perception_simulation:
    def __init__(self):

        # Load the image
        image = Image.open('inverted_aruco2.png')  # Change 'example.jpg' to the path of your image

        # Convert the image to a numpy array
        image_array = np.array(image)

        # Create a PyVista texture
        self.texture = pv.numpy_to_texture(image_array)

        # Create a plane surface mesh
        self.plane = pv.Plane(i_size=image.width, j_size=image.height)

        # Apply the texture to the surface mesh
        self.plane.texture_map_to_plane(inplace=True)
    
    def get_transform(self, leader_pos, leader_quat, chaser_pos, chaser_quat):
        
        angles_l = np.array(self.quaternion_to_euler(*leader_quat))
        angles_c = np.array(self.quaternion_to_euler(*chaser_quat))
        angles = angles_l- angles_c
        angles_rel = [-angles[1],angles[2],angles[0]]
        # Step 1: Compute relative position
        relative_pos = leader_pos - chaser_pos

        # Step 2: Apply the inverse rotation of the chaser to the relative position
        # Create a rotation object from the chaser's quaternion (note scipy uses scalar-last convention by default)
        chaser_rotation = R.from_quat([chaser_quat[1], chaser_quat[2], chaser_quat[3], chaser_quat[0]])

        # Inverse of chaser's rotation
        chaser_rotation_inv = chaser_rotation.inv()

        # Rotate the relative position
        rotated_pos = chaser_rotation_inv.apply(relative_pos)


        camera_frame_pos = [-rotated_pos[1], rotated_pos[2], rotated_pos[0]]

        R_rad = np.array(angles_rel) * math.pi / 180
        rotation_matrix = cv2.Rodrigues(R_rad)[0]
        # Your translation vector
        translation_vector = np.array(camera_frame_pos)  # Replace with your translation vector
        # print(translation_vector)
        # print(angles_rel)
        # Create a 4x4 transformation matrix
        transformation_matrix = np.eye(4)  # Initialize as identity matrix
        transformation_matrix[:3, :3] = rotation_matrix  # Set the top-left 3x3 to your rotation matrix
        transformation_matrix[:3, 3] = translation_vector  # Set the top-right 3x1 to your translation vector
        
        # print("GT difference: ",leader_pos-chaser_pos)
        return transformation_matrix
    
    def get_image(self, transformation_matrix):
        # Apply the transformation
        self.plane.transform(transformation_matrix)

        # Create a PyVista plotter
        plotter = pv.Plotter(off_screen=True)

        # Add the transformed plane with texture to the plotter
        plotter.add_mesh(self.plane, texture=self.texture)


        # origin = np.array([[0, 00, 10000]])  # Coordinates of the origin point
        # origin_poly = pv.PolyData(origin)
        # plotter.add_mesh(origin_poly, color="red", point_size=10, render_points_as_spheres=True)


        # Adjust the camera settings as needed
        plotter.view_xy()
        plotter.set_position([0000, 0, 000])
        plotter.camera.focal_point = [0, 000, 1000]
        plotter.camera.view_angle = 60.0

        # Display the plot
        # plotter.show(screenshot=True)

        # Save the screenshot
        virtual_img = plotter.screenshot('screenshot.png')
        plotter.close()
        # print("Arriced Get IMage")
        Ts, ids = self.detect_aruco()

 
        Ts = Ts[0]

        # print("Estimated difference: ",Ts[2][3],Ts[0][3], -Ts[1][3] )
        return [Ts[2][3],Ts[0][3], -Ts[1][3]]

    def quaternion_to_euler(self,w, x, y, z):
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

    def detect_aruco(self, cap=None, save="output1.png", visualize=True, marker_size=750):


        # ret, frame = cap.read()
        scale = 1
        frame = cv2.imread("screenshot.png")
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        dim = (width, height)
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

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
        #camera_mtx = np.array([[489.53842117,  0.,         307.82908611],
                            # [  0. ,        489.98143193, 244.48380801],
                            # [  0.   ,        0.         ,  1.        ]])
        #camera_mtx = np.identity(3)
        # camera_mtx = np.array([[1.95553717e+04, 0.00000000e+00, 5.14662975e+02],
        #                       [0.00000000e+00, 5.61599540e+04, 3.33162595e+02],
        #                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        camera_mtx = np.array([[665.10751011,   0.        , 511.5],
                            [  0.        , 665.10751011, 383.5],
                            [  0.        ,   0.        ,   1. ]])
    #     camera_mtx = np.array([[678.57920529 ,  0.      ,   482.18115694],
    #  [  0.     ,    685.91554947, 375.83301486],
    #  [  0.   ,        0.    ,       1.        ]])
        #distortion_param = np.array([[-3.69778027e+03, -1.23141160e-01,  1.46877989e+01, -7.97192259e-02, -3.28441832e-06]])
        # distortion_param = np.array([[2.48770555e+00,  1.22911439e-02 , 2.98116458e-01, -3.75310299e-03, 1.86549660e-04]])
        #distortion_param = np.array([[-0.04312776 , 0.32870159 ,-0.01099838, -0.01273789, -0.56414927]])
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

                    frame = cv2.drawFrameAxes(frame, camera_mtx, distortion_param, rvec, tvec,length = 100, thickness=6)
            
                rotation_mtx, jacobian = cv2.Rodrigues(rvec)
                translation = tvec
                
                T = np.identity(4)
                T[:3, :3] = rotation_mtx
                T[:3, 3] = translation 
                # print(T)
                Ts.append(T)

        #if save:
            #cv2.imwrite(save, frame)

        if visualize:
            # print("IF VIS")
            cv2.imshow("camera view", frame)
            cv2.waitKey(1)
            # print("AFTER 0")

            #cv2.destroyAllWindows()

        return Ts, ids

    
if __name__ == "__main__":
    #cv2.namedWindow("camera view", cv2.WINDOW_NORMAL)
    # Define the positions
    leader_pos = np.array([5000.0, 0.0,1000.0])  # Leader position
    chaser_pos = np.array([0, 0.0, 0.0])  # Chaser position
    leader_quat = [  0.8838835, 0.3061862, 0.1767767, -0.3061862 ]  # w, x, y, z for the leader
    chaser_quat = [1, 0, 0, 0]

    
    for i in range(10000):
        per = Perception_simulation()
        leader_pos[1] = leader_pos[1] + 20
        
        transformation_matrix = per.get_transform(leader_pos, leader_quat, chaser_pos, chaser_quat)

        per.get_image(transformation_matrix)





    
