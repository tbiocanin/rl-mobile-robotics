import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import DQN
import gym
import numpy as np
from gymnasium import spaces

class MobileRobotCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space : spaces.Box, features_dim=512):
        super(MobileRobotCNN, self).__init__(observation_space, features_dim)

        input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
        # First convolutional layer
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            
            # Second convolutional layer
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            
            # Second convolutional layer
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.ReLU(),

            # Third convolutional layer
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        # Compute the output dimension after convolutions to initialize fully connected layers
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        
        # Fully connected layers
        self.fc1 = nn.Linear(n_flatten, features_dim)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.cnn(x)
        x = self.relu4(self.fc1(x))

        return x 