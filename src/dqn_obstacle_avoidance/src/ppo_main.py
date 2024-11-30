#!/usr/bin/env python3
from gymWrapper import MobileRobot
from stable_baselines3 import PPO
import multiprocessing
import rospy
import numpy as np
from cnn import MobileRobotCNN

from stable_baselines3.common.evaluation import evaluate_policy
from customTensorboard import SumRewardCallback

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

    model = PPO(
        policy="CnnPolicy",
        env=gymWrapper,
        learning_rate=1e-6,
        policy_kwargs=policy_kwargs_custom,
        tensorboard_log="QRdqn_log/",
        device="cuda",
        seed=10,
        batch_size=32,
        n_steps=512,
        ent_coef=0.005,
        vf_coef=0.5 
    )


    customCallbackObject = SumRewardCallback()

    model = model.learn(25e3, progress_bar=True, log_interval=1) 
    model.save("QRdqn_log/model1")
    # del model
    # model = PPO.load("PPO_log/model1", env=gymWrapper, device="cuda") # model 2 had the best results on the new experimental reward space

    gymWrapper.step_counter_limit = 500
    for _ in range(100):
        done = truncted = False
        observation, info = gymWrapper.reset()
        while not done:
            action, _ = model.predict(observation)
            # print(action)
            observation, reward, done, truncted, info = gymWrapper.step(action)
            observation = np.expand_dims(observation, axis=0)
            if done or truncted:
                gymWrapper.reset()
