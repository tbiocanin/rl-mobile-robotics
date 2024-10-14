#!/usr/bin/env python3
from gymWrapper import MobileRobot
from stable_baselines3 import DQN
import multiprocessing
import rospy
import numpy as np
from cnn import MobileRobotCNN

# print option for numpy
np.set_printoptions(threshold=np.inf)

"Main script that will be run for training the model"

if __name__ == "__main__":
    try:
        gymWrapper = MobileRobot(0)
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
        learning_rate=1e-4,
        exploration_initial_eps=1,
        exploration_final_eps=0.7,
        exploration_fraction=0.5,
        buffer_size=20000,
        learning_starts=10000,
        batch_size=128,
        gamma=0.9,
        tensorboard_log="dqn_log/",
        device='cuda',
        target_update_interval=10000,
        train_freq=(5000, "step"),
        verbose=1
    )


    gymWrapper.reset()
    model.learn(10e4, progress_bar=True, log_interval=1)
    model.save("dqn_log/model3")
    # model.load("dqn_log/model3")
    # gymWrapper.step_per_ep = 1000
    # for _ in range(100):
    #     done = truncted = False
    #     observation, info = gymWrapper.reset()
    #     gymWrapper.timer = rospy.get_rostime()
    #     observation = np.expand_dims(observation, axis=0)
    #     while not done:
    #         action, _ = model.predict(observation)
    #         # print(action)
    #         obs, reward, done, truncted, info = gymWrapper.step(action[0])
    #         if done or truncted:
    #             gymWrapper.reset()
                