#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist

"""
Publisher node for mobile robot control
"""

def init_node(topic: str, data_class, queue_size):
    rospy.init_node("control_node")
    rospy.loginfo("Control node: START")

    pub = rospy.Publisher(topic, data_class, queue_size=queue_size)
    
    return pub

def send_command(publisher: rospy.Publisher, move_forward: float, rotate: float):

    msg = Twist()

    msg.linear.x = move_forward
    msg.angular.z = rotate

    publisher.publish(msg)

    return True

if __name__ == "__main__":
    pub = init_node("cmd_vel", Twist, 10)
    cnt = 0
    while not rospy.is_shutdown():
        cnt += 0.22
        send_command(pub, cnt, 0)