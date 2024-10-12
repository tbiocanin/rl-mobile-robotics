#!/usr/bin/env python3
from gymnasium import spaces
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Twist, Pose, Point, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge, CvBridgeError
import cv2
import rospy
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
import multiprocessing
import random
import tf
import tf.transformations
from typing import Tuple, Dict, Any

"""
Description: Gymnasium wrapper to be used with stable-baselines3 within ROS
"""

class MobileRobot(gym.Env):

    def __init__(self, verbose: int) -> None:
        super(MobileRobot, self).__init__()

        # ROS specific params
        rospy.init_node("DQN_control_node", anonymous=True)
        self.initial_state = ModelState()
        self.bridge = CvBridge()
        self.command_publisher = None
        self.image_subscriber = None
        self.scan_subscriber = None
        self.state = None
        self.log_level = 0
        self.step_per_ep = 400

        # hardcoded for now
        self.init_node()
        self.distance = 0.0
        self.reward = 0.0
        self.info = self._get_info()
        self.step_counter = 0
        self.step_counter_limit = 0

        # gym specific attributes
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low = 0, high= 255, shape=(1, 256, 256), dtype=np.float32)
        self.done = False
        self.truncted = False

        self.min_region_values = []

    def timer_callback(self, event):
        rospy.loginfo("Did not crash untill the end of the episode!")
        self.done = True

    def step(self, action):
        
        if self.log_level == 1:
            rospy.loginfo("Doing a step")

        self.step_counter += 1
        self.step_counter_limit += 1
        if (self.step_counter == self.step_per_ep):
            self.step_counter = 0
            self.done = True
            return self.state, self.reward, self.done, self.truncted, self.info
        
        self.update_velocity(action)
        self.reward = self._compute_reward()
        self.info = self._get_info()
        return self.state, self.reward, self.done, self.truncted, self.info
    
    def update_velocity(self, action):
        if self.log_level == 1:
            rospy.loginfo("Velocity update.")
        msg = Twist()
        # diskretizovano stanje, zavisno od akcije onda ce biti inkrement poslat
        if action == 0:
            msg.linear.x = 0.3
            msg.angular.z = 0.0
        elif action == 1:
            msg.linear.x = 0.0
            msg.angular.z = 0.3
            self.reward -= .75
        elif action == 2:
            msg.linear.x = 0.0
            msg.angular.z = -0.3
            self.reward -= .75

        self.command_publisher.publish(msg)

        return None

    def _compute_reward(self):
        
        # movement reward
        self.reward += .5

        if self.truncted:
            rospy.logwarn("Crash detected, asigning negative reward")
            self.reward -= 1000

        if self.done and not self.truncted:
            self.reward += 300

        return self.reward

    def reset(self, seed = None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        rospy.loginfo("Reseting robot position for the next episode...")

        # generating random positions for next episodes
        rand_x_position = random.uniform(-1.5, -2)
        rand_y_position = random.uniform(-1, 1)

        self.initial_state = ModelState()
        self.initial_state.model_name = 'turtlebot3_waffle'
        set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        orientation = self._generate_random_orientation()

        self.initial_state.pose = Pose(
            position = Point(rand_x_position, rand_y_position, 0),
            orientation=Quaternion(orientation[0], orientation[1], orientation[2], orientation[3])
        )

        self.initial_state.reference_frame = 'world'
        self.info = self._get_info()
        set_model_state(self.initial_state)
        obs = self.state
        self.done = False
        self.truncted = False
        rospy.loginfo("At the end of this episode the reward was: " + str(self.reward))
        self.reward = 0
        return obs, self.info

    def create_depth_image_sub(self):
        rospy.loginfo_once("Creating depth image subscriber!")
        rate = rospy.Rate(100)
        self.image_subscriber = rospy.Subscriber("/camera/depth/image_raw", Image, queue_size=5, callback=self.update_state_callback)
        return self.image_subscriber

    def update_state_callback(self, msg: Image) -> None:

        if self.log_level == 1:
            rospy.loginfo("Next state received.")
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

        # (1080, 1920) -> original image shape

        resized_image = cv2.resize(cv_image, (1024, 1024))
        self.state = self._get_region_minimum(resized_image, 4)

        self.state = np.expand_dims(self.state, axis=0)
        return None

    def create_control_pub(self):
        rospy.loginfo_once("Creating control node publisher")
        self.command_publisher = rospy.Publisher(name="cmd_vel", data_class=Twist, queue_size=10)

        return self.command_publisher

    def scan_front_face(self, scan_data):

        front_range = scan_data.ranges[20:110]
        min_distance = min(front_range)

        if min_distance < .25:
            rospy.logwarn("Reseting robot position, episode is finished due to a crash.")
            self.truncted = True
        elif min_distance > .31 and min_distance < .5:
            self.reward += 0.02

    def create_scan_node(self):
        rospy.loginfo("Creating Lidar node...")
        rate = rospy.Rate(100)
        scan_node_sub = rospy.Subscriber(name='/scan', data_class=LaserScan, queue_size=10, callback=self.scan_front_face)

        return scan_node_sub
    
    def init_node(self):
        self.command_publisher = self.create_control_pub()
        self.image_subscriber = self.create_depth_image_sub()
        self.scan_subscriber = self.create_scan_node()
    
    def run_node(self):
        rospy.spin()

    def _get_info(self):
        
        info = {
            'cumulative_reward'  : self.reward
        }

        return info

    def _get_region_minimum(self, input_image, block_size):
        
        region_mins = np.zeros((input_image.shape[0] // block_size, input_image.shape[1] // block_size), dtype=np.float32)
        for y in range(0, input_image.shape[0], block_size):
            for x in range(0, input_image.shape[1], block_size):
                
                current_block = input_image[y:y+block_size, x:x+block_size]
                min_val = np.min(current_block)
                if np.isnan(min_val):
                    min_val = 4.1

                region_mins[y // block_size, x // block_size] = self._scale_values(min_val)
        
        # NOTE: Uncomment this block for debug purposes
        # np.savetxt('array_full.txt', region_mins, fmt='%f') # shows the pixel values positioned like on the image 
        # cv2.imshow("Resize", region_mins) # shows the new minimized pixel value image 
        # cv2.waitKey(1) 

        return region_mins
    
    def _scale_values(self, value: float) -> float:
        return (value - 0.3)/(4 - 0.3)
    
    def _generate_random_orientation(self):
        """
        Generating parameters for random position and orientation
        """
        rand_yaw = random.uniform(np.pi/2, -np.pi/2)
        new_quaternion = tf.transformations.quaternion_from_euler(0, 0, rand_yaw)

        return new_quaternion

if __name__ == "__main__":
    pass
