#!/usr/bin/env python3
from gymWrapper import MobileRobot
from stable_baselines3 import DQN
import multiprocessing
import rospy
import numpy as np
from cnn import MobileRobotCNN

# DEBUG: print option for numpy
np.set_printoptions(threshold=np.inf)

"Main script that will be run for training the model"

if __name__ == "__main__":
    try:
        gymWrapper = MobileRobot(verbose=0, start_learning_at=250)
    except rospy.ROSInterruptException:
        rospy.logerr("Nodes not initialized!")

    ros_process = multiprocessing.Process(target=gymWrapper.run_node)

    ros_process.start()

    # integrating the custom network via the policy_kwargs argument
    policy_kwargs_custom = dict(
        features_extractor_class = MobileRobotCNN,
        features_extractor_kwargs=dict(features_dim=512),
        normalize_images = False
    )

    model = DQN(
        "CnnPolicy",
        env=gymWrapper,
        policy_kwargs = policy_kwargs_custom,
        learning_rate=1e-6,
        exploration_initial_eps=1,
        exploration_final_eps=0.1,
        exploration_fraction=0.4,
        buffer_size=10000,
        learning_starts=64,
        batch_size=64,
        gamma=0.99,
        tensorboard_log="dqn_log/",
        device='cuda',
        target_update_interval=20,
        train_freq=1,
        verbose=1,
        seed=10,
        gradient_steps=1,
        tau=0.01
    )

    # model = model.learn(10e3, progress_bar=True, log_interval=1) 
    # model.save("dqn_log/model1")
    del model
    model = DQN.load("dqn_log/model1", env=gymWrapper, device="cuda") # model 2 had the best results on the new experimental reward space

    gymWrapper.step_counter_limit = 500
    for _ in range(100):
        done = truncted = False
        observation, info = gymWrapper.reset()
        while not done:
            action, _ = model.predict(observation)
            observation, reward, done, truncted, info = gymWrapper.step(action)
            observation = np.expand_dims(observation, axis=0)
            if done or truncted:
                gymWrapper.reset()
                