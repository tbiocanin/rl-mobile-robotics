#!/usr/bin/env python3
from gymnasium import spaces
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Twist, Pose, Point, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import rospy
import gymnasium as gym
import numpy as np
from math import sqrt
from stable_baselines3 import DQN
import multiprocessing
import random

"""
Description: Gymnasium wrapper to be used with stable-baselines3 within ROS
"""
class MobileRobot(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self) -> None:
        super(MobileRobot, self).__init__()

        # ROS specific params
        rospy.init_node("DQN_control_node", anonymous=True)
        self.initial_state = ModelState()
        self.bridge = CvBridge()
        self.command_publisher = None
        self.image_subscriber = None
        self.state = None
        # hardcoded for now
        self.goal_x = 1.5
        self.goal_y = 1.5
        self.timer = rospy.Timer(rospy.Duration(10), self.timer_callback)
        self.init_node()
        self.distance = 0.0
        self.reward = 0.0
        self.info = self._get_info()

        # gym specific attributes
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low = 0, high= 255, shape=(1, 128, 128), dtype=np.float32)
        self.done = False

        self.min_region_values = []

    def timer_callback(self, event):
        self.done = True

    def step(self, action):
        rospy.logdebug("Doing a step")
        self.update_velocity(action)
        self.reward = self._compute_reward()
        self.info = self._get_info()
        return self.state, self.reward, self.done, False, self.info
    
    def update_velocity(self, action):
        rospy.logdebug("Velocity update.")
        msg = Twist()
        # diskretizovano stanje, zavisno od akcije onda ce biti inkrement poslat
        if action == 0:
            msg.linear.x = 0.25
            msg.angular.z = 0.0
        elif action == 1:
            msg.linear.x = 0.0
            msg.angular.z = 0.3
        elif action == 2:
            msg.linear.x = 0.0
            msg.angular.z = -0.3

        self.command_publisher.publish(msg)

        return None

    def _compute_reward(self):
        
        self.reward = 0.1

        # TODO: reward hadnling with the ROI being bellow the treshhold
        print(np.min(self.state))
        if np.min(self.state) < 0.4:
            print("ADDING NEGATIVE REWARD FOR DISTANCE")
            self.reward -= 2

        return self.reward


    def is_done_clbk(self, data: Odometry):
        # check based on the location
        
        rospy.logdebug("Checking if done")
        current_x = data.pose.pose.position.x
        current_y = data.pose.pose.position.y
        self.distance = sqrt((self.goal_x - current_x) ** 2 + (self.goal_y - current_y) ** 2)

    def reset(self, seed = None, options=None):
        super().reset(seed=seed)
        rospy.loginfo("Reseting robot position for the next episode...")

        # generating random positions for next episodes
        rand_x_position = random.randint(-2, 2)
        rand_y_position = random.uniform(-0.2, -0.7)

        self.initial_state = ModelState()
        self.initial_state.model_name = 'turtlebot3_waffle'
        set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        self.initial_state.pose = Pose(
            position = Point(rand_x_position, rand_y_position, 0),
            orientation=Quaternion(0, 0, 0, 1)
        )

        self.initial_state.reference_frame = 'world'
        self.info = self._get_info()
        set_model_state(self.initial_state)
        obs = np.expand_dims(self.state, axis=0)
        self.done = False

        return obs, self.info

    def create_depth_image_sub(self):
        rospy.loginfo_once("Creating depth image subscriber!")
        rate = rospy.Rate(1)
        self.image_subscriber = rospy.Subscriber("/camera/depth/image_raw", Image, queue_size=10, callback=self.update_state_callback)
        return self.image_subscriber

    def update_state_callback(self, msg: Image) -> None:

        rospy.logdebug("Next state received.")
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

        # TODO: pre-process the image and send the discrete min version of the image to self.state
        # (1080, 1920) -> original image shape

        resized_image = cv2.resize(cv_image, (1024, 1024))
        self.state = self._get_region_minimum(resized_image, 8)
        if np.min(self.state)< 0.4:
            print("TOO CLOSE")
            self.done = True
            
        return None

    def create_control_pub(self):
        rospy.loginfo_once("Creating control node publisher")
        self.command_publisher = rospy.Publisher(name="cmd_vel", data_class=Twist, queue_size=10)

        return self.command_publisher
    
    def create_localization_sub(self):
        rospy.loginfo_once("Localization node init!")
        rospy.Rate(1)
        rospy.Subscriber("/odom", Odometry, self.is_done_clbk)

    def init_node(self):
        self.command_publisher = self.create_control_pub()
        self.image_subscriber = self.create_depth_image_sub()
        self.create_localization_sub()

    def run_node(self):
        rospy.spin()

    def _get_info(self):
        data = Odometry()
        current_x = data.pose.pose.position.x
        current_y = data.pose.pose.position.y
        
        distance = sqrt((self.goal_x - current_x) ** 2 + (self.goal_y - current_y) ** 2)
        info = {
            'distance_to_target' : distance,
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
                    min_val = float('inf')
                region_mins[y // block_size, x // block_size] = min_val

        # self.min_region_values = region_mins
        # resized_image = cv2.resize(region_mins, (input_image.shape[1], input_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        return region_mins

if __name__ == "__main__":
    try:
        gymWrapper = MobileRobot()
    except rospy.ROSInterruptException:
        rospy.logerr("Nodes not initialized!")

    ros_process = multiprocessing.Process(target=gymWrapper.run_node)

    ros_process.start()

    model = DQN(
        "CnnPolicy",
        env=gymWrapper,
        policy_kwargs = dict(normalize_images=False, net_arch=[128, 128]),
        learning_rate=5e-4,
        buffer_size=100,
        learning_starts=10,
        batch_size=64,
        gamma=0.99,
        tensorboard_log="dqn_log/",
        device='cpu'
    )

    model.learn(1e3, progress_bar=True)
    model.save("dqn_log/model")
    for _ in range(100):
        done = False
        observation, info = gymWrapper.reset()
        gymWrapper.timer = rospy.get_rostime()
        observation = np.expand_dims(observation, axis=0)
        while not done:
            action = model.predict(observation, deterministic=True)
            obs, reward, done, truncted, info = gymWrapper.step(action)
            if done:
                gymWrapper.reset()

# NOTE : obs je ok da bude isto sto i stanje
# NOTE : treba videti kako cemo prosledjivati sliku, napraviti da bude odnos isti
# NOTE : diskretizacija; podeliti sliku u grid i za svaki od tih polja u gridu uzmi najmanju distancu (jer je to najrizicnije) i da to bude stanje