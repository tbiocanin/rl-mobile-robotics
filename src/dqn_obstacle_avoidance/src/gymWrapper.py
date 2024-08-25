#!/usr/bin/env python3
from gymnasium import spaces
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import rospy
import gymnasium as gym
import numpy as np
from math import sqrt
from stable_baselines3 import DQN
"""
Description: Gymnasium wrapper to be used within stable-baselines3
"""
class MobileRobot(gym.Env):

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

        self.init_node()

        # gym specific attributes
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low = 0, high= 255, shape=(1, 64, 64), dtype=np.uint8)
        self.done = False
        self.reward = 0.0

    def step(self, action):
        # update velocity ce biti pozvan ode
        rospy.logdebug("Doing a step")
        self.update_velocity(action)

        self.reward = self.compute_reward()

        return self.state, self.reward, self.done, {}
    
    def update_velocity(self, action):
        rospy.logdebug("Velocity update.")
        msg = Twist()
        # diskretizovano stanje, zavisno od akcije onda ce biti inkrement poslat
        print(action)
        if action == 0:
            msg.linear.x = 0.2
            msg.angular.z = 0.0
        elif action == 1:
            msg.linear.x = 0.0
            msg.angular.z = 0.2
        elif action == 2:
            msg.linear.x = 0.0
            msg.angular.z = -0.2

        self.command_publisher.publish(msg)

        return None

    def compute_reward(self):
        
        if self.done == True:
            self.reward += 10
        else:
            self.reward -= 0.1

        return self.reward


    def is_done(self, data: Odometry):
        # check based on the location
        # TODO: to be more dynamic
        rospy.logdebug("Checking if done")
        current_x = data.pose.pose.position.x
        current_y = data.pose.pose.position.y
        
        distance = sqrt((self.goal_x - current_x) ** 2 + (self.goal_y - current_y) ** 2)

        if distance < 0.2:
            return True
        else:
            return False

    def reset(self, seed = None, options=None):
        # actually the init method of the env
        rospy.loginfo_once("Reseting robot position for the next episode...")
        self.initial_state.pose.position.x = -2
        self.initial_state.pose.position.y = -0.5
        self.initial_state.pose.position.z = 0
        self.initial_state.pose.orientation.x = 0
        self.initial_state.pose.orientation.z = 0
        self.initial_state.pose.orientation.w = 0
        self.initial_state.twist.linear.x = 0
        self.initial_state.twist.linear.z = 0
        self.initial_state.twist.angular.x = 0
        self.initial_state.twist.angular.z = 0

        return self.state, {}

    def create_depth_image_sub(self):
        rospy.loginfo_once("Creating depth image subscriber!")
        self.image_subscriber = rospy.Subscriber("/camera/depth/image_raw", Image, queue_size=10, callback=self.update_state_callback)

        return self.image_subscriber

    def update_state_callback(self, msg: Image) -> None:

        rospy.logdebug("Next state received.")
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        self.state = cv2.resize(cv_image, (64, 64))

        return None

    def create_control_pub(self):
        rospy.loginfo_once("Creating control node publisher")
        self.command_publisher = rospy.Publisher(name="cmd_vel", data_class=Twist, queue_size=10)

        return self.command_publisher
    
    def create_localization_sub(self):
        rospy.loginfo_once("Localization node init!")
        rospy.Subscriber("/odom", Odometry, self.is_done)

    def init_node(self):
        self.command_publisher = self.create_control_pub()
        self.image_subscriber = self.create_depth_image_sub()
        self.create_localization_sub()

    def run_node(self):
        rospy.spin()

if __name__ == "__main__":
    # train loop init
    # NOTE neki bag je ovde u inicijalizaciji 
    try:
        gymWrapper = MobileRobot()
        gymWrapper.run_node()
    except rospy.ROSInterruptException:
        pass
    # model = DQN(
    #     "CnnPolicy",
    #     gymWrapper,
    #     policy_kwargs = dict(normalize_images=False, net_arch=[124, 124]),
    #     learning_rate=5e-4,
    #     buffer_size=100,
    #     learning_starts=10,
    #     batch_size=10,
    #     gamma=0.99,
    #     tensorboard_log="dqn_log/"
    # )

    # model.learn(10, progress_bar=True)
    # model.save("dqn_log/model")
    # for _ in range(100):
    #     done = False
    #     observation = gymWrapper.reset()
    #     while not done:
    #         action = model.predict(observation)
    #         obs, reward, done, info = gymWrapper.step(action)

    #         if done:
    #             gymWrapper.reset()

    for _ in range(100):
        action = gymWrapper.action_space.sample()
        obs, reward, done, info = gymWrapper.step(action)
        if done:
            obs = gymWrapper.reset()