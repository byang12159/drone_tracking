import cv2 
import numpy as np
import cv2.aruco as aruco

def detect_aruco(frame, save="output1.png", visualize=True, marker_size=750):
   
    scale = 1
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dim = (width, height)
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
    #camera_mtx = np.array([[489.53842117,  0.,         307.82908611],
                        # [  0. ,        489.98143193, 244.48380801],
                        # [  0.   ,        0.         ,  1.        ]])
    camera_mtx = np.identity(3)
    # camera_mtx = np.array([[1.95553717e+04, 0.00000000e+00, 5.14662975e+02],
    #                       [0.00000000e+00, 5.61599540e+04, 3.33162595e+02],
    #                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    camera_mtx = np.array([[886.73363353,   0.        , 511.5],
                          [  0.        , 665.10751011, 383.5],
                          [  0.        ,   0.        ,   1. ]])
    #distortion_param = np.array([[-3.69778027e+03, -1.23141160e-01,  1.46877989e+01, -7.97192259e-02, -3.28441832e-06]])
    distortion_param = np.array([[2.48770555e+00,  1.22911439e-02 , 2.98116458e-01, -3.75310299e-03, 1.86549660e-04]])
    #distortion_param = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
    if markerIds is not None:
        rvecs, tvecs, objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, markerLength=marker_size, cameraMatrix=camera_mtx, distCoeffs=distortion_param)
        # rvecs, tvecs, objpoints = aruco.estimatePoseSingleMarkers(markerCorners, marker_size, , )
        print("R",rvecs)
        print("T",tvecs)
        print("P",objPoints)
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
            T[:3, 3] = translation / 1000.
            Ts.append(T)
    if save:
        cv2.imwrite(save, frame)
    if visualize:
        print("a")
        cv2.imshow("camera view", frame)
    print(markerIds)
    # Multiple ids in 1D list
    # Mutiple Ts, select first marker 2d array by Ts[0]
    print(Ts)
    return Ts, ids

img = cv2.imread("screenshot_vista.png")
print("img shape: ",img.shape)

# img = img[12:400,16:404,:]
light = []
count = 0
for i in range(img.shape[0]):
    light.append(img[i,50,0])
    if img[i,50,0] == 0:
        count +=1
print(light)
print("inner size ",count)

print("img shape", img.shape)
# cv2.imwrite("aruco_0.png",img)

cv2.circle(img,[505,400],4,(255,0,0),2,2)
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


