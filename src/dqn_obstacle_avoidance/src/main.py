#!/usr/bin/env python3
from gymWrapper import MobileRobot
from stable_baselines3 import DQN
from sb3_contrib import QRDQN
import multiprocessing
import rospy
import numpy as np
from cnn import MobileRobotCNN

# print option for numpy
np.set_printoptions(threshold=np.inf)

"Main script that will be run for training the model"

if __name__ == "__main__":
    try:
        gymWrapper = MobileRobot(0, 50000)
    except rospy.ROSInterruptException:
        rospy.logerr("Nodes not initialized!")

    ros_process = multiprocessing.Process(target=gymWrapper.run_node)

    ros_process.start()

    # integrating the custom network via the policy_kwargs argument
    policy_kwargs_custom = dict(
        features_extractor_class = MobileRobotCNN,
        features_extractor_kwargs=dict(features_dim=256),
        normalize_images = False
    )

    model = DQN(
        "CnnPolicy",
        env=gymWrapper,
        policy_kwargs = policy_kwargs_custom,
        learning_rate=3e-4,
        exploration_initial_eps=1,
        exploration_final_eps=0.4,
        exploration_fraction=0.55,
        buffer_size=6000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        tensorboard_log="dqn_log/",
        device='cuda',
        target_update_interval=100,
        train_freq=350,
        verbose=1,
        seed=42
    )

    # first lear
    # model = model.learn(7e3, progress_bar=True, log_interval=1)
    # model.save("dqn_log/model1")
    # model = DQN.load("dqn_log/model1.zip", env=gymWrapper, device="cuda")
    
    # second learn
    # model.learning_rate = 1e-4
    # model.exploration_final_eps=0.25
    # model = model.learn(7e3, progress_bar=True, log_interval=1) 
    # model.save("dqn_log/model2")
    # model = DQN.load("dqn_log/model2.zip", env=gymWrapper, device="cuda")

    # third learn
    # model.learning_rate = 3e-4
    # model.exploration_final_eps=0.4
    # model.target_update_interval = 700
    # model = model.learn(7e3, progress_bar=True, log_interval=1) 
    # model.save("dqn_log/model3")
    # model = DQN.load("dqn_log/model3.zip", env=gymWrapper, device="cuda")

    # fourth learn
    # model.exploration_final_eps=0.85
    # model.target_update_interval = 700
    # model = model.learn(7e3, progress_bar=True, log_interval=1) 
    # model.save("dqn_log/model7")
    model = DQN.load("dqn_log/model7.zip", env=gymWrapper, device="cuda")
    
    gymWrapper.step_counter_limit = float('inf')
    for _ in range(100):
        done = truncted = False
        observation, info = gymWrapper.reset()
        while not done:
            action, _ = model.predict(observation)
            obs, reward, done, truncted, info = gymWrapper.step(action)
            if done or truncted:
                gymWrapper.reset()
                