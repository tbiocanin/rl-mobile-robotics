#!/usr/bin/env python3
from gymnasium import spaces
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Twist, Pose, Point, Quaternion
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
import cv2
import rospy
import gymnasium as gym
import numpy as np
import random
import tf
import tf.transformations
from typing import Tuple, Dict, Any
from multiprocessing import Lock

"""
Description: Gymnasium wrapper to be used with stable-baselines3 within ROS
"""

class MobileRobot(gym.Env):

    def __init__(self, verbose : int, start_learning_at) -> None:
        super(MobileRobot, self).__init__()

        # ROS specific attributes
        rospy.init_node("DQN_control_node", anonymous=True)
        self.initial_state = ModelState()
        self.bridge = CvBridge()
        self.command_publisher = None
        self.image_subscriber = None
        self.scan_subscriber = None
        self.state = None
        self.log_level = 0

        self.step_counter = 0
        self.step_counter_limit = 150
        self.start_learning_at = start_learning_at
        self.learning_counter = 0


        self.init_node()
        self.distance = 0.0
        self.reward = 0.0
        self.info = self._get_info()


        # gym specific attributes
        self.action_space = spaces.Discrete(n=3, seed=10, start=0)
        self.observation_space = spaces.Box(low = 0.3, high= 4, shape=(1, 256, 256), dtype=np.float32)
        self.done = False
        self.truncted = False
        self.prev_action = None

        # thread lock objects for the lidar and camera 
        self.image_lock = Lock()
        self.lidar_lock = Lock()
        

    def step(self, action):

        if self.log_level == 1:
            rospy.loginfo("Doing a step")

        # Terminating block
        self.step_counter += 1
        if (self.step_counter == self.step_counter_limit):
            self.step_counter = 0
            self.done = True
        
        self.update_velocity(action)
        return_reward, self.truncted = self._compute_reward(action)
        self.reward += return_reward
        self.info = self._get_info()

        # tranformation based on the env_checker.py
        obs = self.state
        
        if obs is not None:
            # expand it to be (1, 256, 256)
            obs = np.expand_dims(obs, axis=0)
        else:
            # if the sim is not ready, obs can be None so we have to handle that with dummy data
            obs = np.zeros((1, 256, 256), dtype=np.float32)
            obs = obs.clip(0.3, 4)

        self.prev_action = action

        return obs, return_reward, self.done, self.truncted, self.info
    
    def update_velocity(self, action):
        if self.log_level == 1:
            rospy.loginfo("Velocity update.")
        msg = Twist()
        rate = rospy.Rate(1000)

        # diskretizovano stanje, zavisno od akcije onda ce biti inkrement poslat
        if action == 0:
            msg.linear.x = 0.3
            msg.angular.z = 0.0
            self.command_publisher.publish(msg)
            rate.sleep()
        elif action == 1:
            msg.linear.x = 0.0
            msg.angular.z = 0.45
            self.command_publisher.publish(msg)
            rate.sleep()
        elif action == 2:
            msg.linear.x = 0.0
            msg.angular.z = -0.45
            self.command_publisher.publish(msg)
            rate.sleep()
        return None

    def _compute_reward(self, action):
        
        curret_step_reward = 0
        self.learning_counter += 1
        if self.learning_counter < self.start_learning_at:
            if self.distance < .25:
                rospy.logwarn("Reseting robot position, episode is finished due to a crash.")
                self.truncted = True

            return 0, self.truncted

        # distance based reward
        if self.distance < .25:
            rospy.logwarn("Reseting robot position, episode is finished due to a crash.")
            self.truncted = True
            self.step_counter = 0
            curret_step_reward -= 15
            return curret_step_reward, self.truncted
        elif self.distance > .35 and self.distance < .5:
            curret_step_reward -= 0.0075
        
        if self.distance < .5 and action == 0:
            curret_step_reward -= 0.8
        
        if self.distance < .5 and (self.prev_action == 1 and action == 1):
            curret_step_reward += 0.25
        
        if self.distance < .5 and (self.prev_action == 2 and action == 2):
            curret_step_reward += 0.25
        
        if self.distance > .51 and action == 0:
            curret_step_reward += 0.45

        # action based reward
        if action == 0:
            curret_step_reward += 0.06 # 2 it/s
        elif action == 1:
            curret_step_reward -= 0.05
        elif action == 2:
            curret_step_reward -= 0.05

        if (action == 1 or action == 2) and self.distance > .51:
            curret_step_reward -= 0.01

        if self.done and not self.truncted:
            curret_step_reward += 20

        return curret_step_reward, self.truncted

    def reset(self, seed = 10, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:

        super().reset(seed=seed)
        rospy.loginfo("Reseting robot position for the next episode...")

        # generating random positions for next episodes

        # This is for the default map
        rand_x_position = random.uniform(-1.75, -2)
        rand_y_position = random.uniform(-1, 1)

        # This is for the warehouse map, TODO: don't do this here
        # rand_x_position = random.uniform(-2, -3)
        # rand_y_position = random.uniform(.5, .75)

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
        
        # tranformation based on the env_checker.py, same logic like in step()
        obs = self.state
        if obs is not None:
            obs = np.expand_dims(obs, axis=0)
        else:
            obs = np.zeros((1, 256, 256), dtype=np.float32)
            obs = obs.clip(0.3, 4)
        
        self.done = False
        self.truncted = False
        rospy.loginfo("At the end of this episode the reward was: " + str(self.reward))
        self.reward = 0

        return obs, self.info

    def create_depth_image_sub(self):
        rospy.loginfo_once("Creating depth image subscriber!")
        self.image_subscriber = rospy.Subscriber("/camera/depth/image_raw", Image, queue_size=1, callback=self.update_state_callback)
        return self.image_subscriber

    def update_state_callback(self, msg: Image) -> None:
        
        self.image_lock.acquire()

        if self.log_level == 1:
            rospy.loginfo("Next state received.")
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

        # (1080, 1920) -> original image shape

        resized_image = cv2.resize(cv_image, (256, 256))
        self.state = self._get_region_minimum(resized_image, 1)

        self.image_lock.release()

        return None

    def create_control_pub(self):
        rospy.loginfo_once("Creating control node publisher")
        self.command_publisher = rospy.Publisher(name="cmd_vel", data_class=Twist, queue_size=1)

        return self.command_publisher

    def scan_front_face(self, scan_data):

        self.lidar_lock.acquire()

        front_range = []
        front_range_left = list(scan_data.ranges[330:360])
        front_range_right = list(scan_data.ranges[0:30])

        # a nasty workaround...
        min_distance_left = min(front_range_left)
        min_distance_right = min(front_range_right)
        front_range = [min_distance_left, min_distance_right]
        self.distance = min(front_range)

        self.lidar_lock.release()

        return None

    def create_scan_node(self):
        rospy.loginfo("Creating Lidar node...")
        scan_node_sub = rospy.Subscriber(name='/scan', data_class=LaserScan, queue_size=1, callback=self.scan_front_face)

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
                # min_val = np.min(current_block)
                if np.isnan(current_block):
                    current_block = 0.3

                region_mins[y // block_size, x // block_size] = current_block
        
        # NOTE: Uncomment this block for debug purposes
        # np.savetxt('array_full.txt', region_mins, fmt='%f') # shows the pixel values positioned like on the image 
        # cv2.imshow("Resize", region_mins) # shows the new minimized pixel value image 
        # cv2.waitKey(1) 

        return region_mins
    
    def _generate_random_orientation(self):
        """
        Generating parameters for random position and orientation
        """
        rand_yaw = random.uniform(0, 2*np.pi)
        new_quaternion = tf.transformations.quaternion_from_euler(0, 0, 0)

        return new_quaternion

if __name__ == "__main__":
    pass
