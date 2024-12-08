#!/usr/bin/env python3

"""
Publisher for commands and subscriber for depth info.
"""

import sys
import cv2

# GPIO interface 
import gpio_interface
import dqn_inference_interface as dqn_inference

# ROS specific imports
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class MainRobotProcess():

    def __init__(self):
        self.bridge = CvBridge()
        self.engine_interface = dqn_inference.DQN_Inference()
        self.depth_image_subscriber = None
        [self.p_left, self.p_right] = gpio_interface.setup(25)

        rospy.init_node("Main_robot_process")

    def init_node(self):
        """
        Init and run the main process with this function call
        """
        self.depth_image_subscriber = self.create_depth_image_subscriber()
        rospy.spin()

    def create_depth_image_subscriber(self):
        """
        Create the depth image subscriber to pass the image to the DQN network.
        """

        try:
            rospy.loginfo("Creating the depth image subscriber...")
            depth_image_subscriber = rospy.Subscriber(name="depth_subscriber", data_class=Image, queue_size=1, callback=self.depth_image_callback)
            return depth_image_subscriber
        except:
            rospy.logerr("Could not create the depth image subscriber. Exiting...")
            sys.exit()

    def depth_image_callback(self, msg: Image) -> None:
        """
        Callback function for the depth_subscriber node. It gets executed if a new image is on the topic.
        """

        # get the input from depthNet
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        resize_image = cv2.resize(cv_image, (256, 256))

        # pass the resized image to the DQN net and move the motors accordingly
        output_command = self.dqn_process_output(resize_image)
        self.apply_robot_movement_command(output_command)

        return None

    def dqn_process_output(self, input_depth_image) -> int:
        """
        When a new image is received from the callback, forward it to the network to get the predicted action.
        """
        out_command = self.engine_interface.run_dqn_inference(input_image=input_depth_image)

        return out_command
        
    def apply_robot_movement_command(self, command : int) -> None:
        """
        With a predicted action, the action is forwarded to the GPIO interface for motor execution
        """
        gpio_interface.handle_dqn_input(command)

        return None
    
    def clear_motor_gpio(self):
        gpio_interface.stop()
        self.p_left.stop()
        self.p_right.stop()
        gpio_interface.gpio.cleanup()

if __name__ == "__main__":
    main_process_object = MainRobotProcess()

    try:
        main_process_object.init_node() # will perform rospy.spin()
    except KeyboardInterrupt:
        main_process_object.clear_motor_gpio()