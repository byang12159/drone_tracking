#!/usr/bin/env python
import pickle
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge
import cv2
import os
bridge = CvBridge()
information = []
information2 = []

def callback(data):
    #rospy.loginfo("Received Odometry data:\n%s", data.pose.pose)
    # pose_information = np.array([data.pose.pose.position.x])
    information.append(data)

def image_callback_1(msg):
    idcheck = msg.header.stamp
    rospy.loginfo(idcheck)
    #rospy.loginfo("Image 1 recieving {}".format(msg.header.frame_id)
    #rospy.loginfo(msg)
    cv_image = bridge.imgmsg_to_cv2(msg,"passthrough")
    
    image1_filename ="Fisheye1/Image1_{}.png".format(idcheck)
    cv2.imwrite(image1_filename, cv_image)
 
def image_callback_2(msg):
    idcheck = msg.header.stamp
    rospy.loginfo(idcheck)  
    #rospy.loginfo(msg)
    cv_image = bridge.imgmsg_to_cv2(msg,"passthrough")
    
    image2_filename ="Fisheye2/Image2_{}.png".format(idcheck)
    cv2.imwrite(image2_filename, cv_image)
       

def odom_subscriber():
    rospy.init_node('odom_subscriber', anonymous=True)
    
    rospy.Subscriber("/camera/odom/sample", Odometry, callback)

    rospy.Subscriber("/camera/fisheye1/image_raw", Image, image_callback_1)    
    rospy.Subscriber("/camera/fisheye2/image_raw", Image, image_callback_2) 
    
    rospy.spin()

if __name__ == '__main__':
    odom_subscriber()
    
    file_path = "camdata.pkl"
    file_path2 = "imgdata.pkl"
    with open(file_path, 'wb') as file:
        pickle.dump(information,file )
    #with open(file_path2, 'wb') as file:
    #    pickle.dump(information2,file)
    print("Saved Data")

