#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

# roslaunch turtlebot3_gazebo turtlebot3_world.launch
# roslaunch turtlebot3_gazebo turtlebot3_gazebo_rviz.launch -> vizuelizacija
# image topic path -> /camera/depth/image_raw
bridge = CvBridge()

def init_node():
    rospy.init_node("depth_image_process_node")
    rospy.logdebug_once("Depth image node init done!")
    rate = rospy.Rate(1) #1 Hz
    sub = rospy.Subscriber("/camera/depth/image_raw", Image, queue_size=10, callback=print_depth_data_callback)
    rospy.spin()

    return sub

def save_image_values_to_txt(data):
    np.set_printoptions(threshold=np.inf)
    np.savetxt("image.txt", data, fmt="%f")

    return None

def print_depth_data_callback(msg: Image):
    rospy.loginfo("Received an image!")
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
    save_image_values_to_txt(cv_image)

if __name__ == "__main__":
    init_node()