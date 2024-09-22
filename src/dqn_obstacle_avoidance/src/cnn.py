#!/usr/bin/env python3
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

"""
Custom CNN that will be used instead of the default one within the library
"""
class MobileRobotCNN(BaseFeaturesExtractor):


    """
    __init__(self, observation_space, features_dim)
        observation_space -> self explananotary, the observation space of an agent during training and eval
        features_dim -> dimensions of the features, default value of 256
    """
    def __init__(self, observation_space, features_dim=256):
        super(MobileRobotCNN, self).__init__(observation_space, features_dim)

        # Neural network achitecture definition
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=observation_space.shape[0], out_channels=32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    """forward(self, observation)
        forward pass through the network (cnn then a fully connected network)
    """
    def forward(self, observation):
        cnn_out = self.cnn(observation)

        return self.fc(cnn_out)