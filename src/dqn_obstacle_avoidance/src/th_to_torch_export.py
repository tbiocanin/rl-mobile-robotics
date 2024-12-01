#!/usr/bin/env python3

"""
Export script that converts from .th (pytorch) file format to .onnx to be used with jetson-nano
"""
from gymWrapper import MobileRobot
from stable_baselines3 import DQN
import torch as th
import rospy as rp

rp.loginfo("Loading the model")
gymWrapper = MobileRobot(verbose=0, start_learning_at=250)
model = DQN.load("dqn_log/model1.zip", env=gymWrapper, device="cuda")

onnxable_model = model.policy
onnxable_model.to("cuda")

obs = th.zeros((1, 1, 256, 256), dtype=th.float32, device="cuda")

onnx_path = "model1.onnx"

rp.loginfo("Exporting the model...")
th.onnx.export(
    onnxable_model,
    obs,
    onnx_path,
    opset_version=10
)

rp.loginfo("Export done!")