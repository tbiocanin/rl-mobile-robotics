#!/usr/bin/env python3
from gymnasium import spaces
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
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
import multiprocessing
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
        self.timer = 0
        self.init_node()

        self.reward = 0.0
        self.info = self._get_info()

        # gym specific attributes
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low = 0, high= 255, shape=(1, 64, 64), dtype=np.uint8)
        self.done = False

    def step(self, action):
        # update velocity ce biti pozvan ode
        rospy.logdebug("Doing a step")
        self.update_velocity(action)
        self.timer = rospy.get_time()
        self.reward = self.compute_reward()
        self.info = self._get_info()
        return self.state, self.reward, self.done, self.timer, self.info
    
    def update_velocity(self, action):
        rospy.logdebug("Velocity update.")
        msg = Twist()
        # diskretizovano stanje, zavisno od akcije onda ce biti inkrement poslat
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
        # self.timer = rospy.get_time()

        # if self.timer > 100:
        #     self.timer = 0
        #     self.done = True
        #     return self.done
        
        rospy.logdebug("Checking if done")
        current_x = data.pose.pose.position.x
        current_y = data.pose.pose.position.y
        
        distance = sqrt((self.goal_x - current_x) ** 2 + (self.goal_y - current_y) ** 2)

        if distance < 0.2:
            self.done = True
            return self.done
        else:
            return self.done

    def reset(self, seed = None, options=None):
        # actually the init method of the env
        rospy.loginfo_once("Reseting robot position for the next episode...")
        set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.initial_state.pose.position.x = 0
        self.initial_state.pose.position.y = 0
        self.initial_state.pose.position.z = 0
        self.initial_state.pose.orientation.x = 0
        self.initial_state.pose.orientation.z = 0
        self.initial_state.pose.orientation.w = 0
        self.initial_state.twist.linear.x = 0
        self.initial_state.twist.linear.z = 0
        self.initial_state.twist.angular.x = 0
        self.initial_state.twist.angular.z = 0
        self.initial_state.reference_frame = 'world'
        self.info = self._get_info()
        set_model_state(self.initial_state)
        return self.state, self.info

    def create_depth_image_sub(self):
        rospy.loginfo_once("Creating depth image subscriber!")
        rate = rospy.Rate(1)
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

    def _get_obs(self):
        pass

if __name__ == "__main__":
    try:
        gymWrapper = MobileRobot()
    except rospy.ROSInterruptException:
        rospy.logerr("Nodes not initialized!")

    ros_process = multiprocessing.Process(target=gymWrapper.run_node)

    ros_process.start()

    model = DQN(
        "CnnPolicy",
        gymWrapper,
        policy_kwargs = dict(normalize_images=False, net_arch=[64, 64]),
        learning_rate=5e-4,
        buffer_size=100,
        learning_starts=10,
        batch_size=10,
        gamma=0.99,
        tensorboard_log="dqn_log/"
    )

    model.learn(10, progress_bar=True)
    gymWrapper.reset()
    model.save("dqn_log/model")
    for _ in range(100):
        done = False
        observation, info = gymWrapper.reset()
        observation = np.expand_dims(observation, axis=0)
        while not done:
            action, _states = model.predict(observation)
            obs, reward, done, truncted, info = gymWrapper.step(action)
            if done:
                gymWrapper.reset()