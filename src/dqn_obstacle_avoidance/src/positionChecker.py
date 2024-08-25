#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from math import sqrt

# Target position (change these values to your desired coordinates)
target_x = 1.0
target_y = 1.0
tolerance = 0.1  # Allowable error range

def check_position(data: Odometry):
    # Get current position from Odometry message
    current_x = data.pose.pose.position.x
    current_y = data.pose.pose.position.y
    print(current_x)
    print(current_y)


def listener():
    # Initialize the ROS node
    rospy.init_node('check_turtlebot_position', anonymous=True)

    # Subscribe to the /odom topic to get TurtleBot3's position
    rospy.Subscriber("/odom", Odometry, check_position)

    # Keep the node running
    rospy.spin()

if __name__ == '__main__':
    listener()
