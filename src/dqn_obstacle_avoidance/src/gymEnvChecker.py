"""
Verification script provided by stablebaselines3 for validation 
of the env implementation. This is a simple python script so do not
run it with ROS. Just simply run it with python3 gymEnvChecker.py

This is from the docs 'Using Custom Environments'
"""
from gymWrapper import MobileRobot
from stable_baselines3.common.env_checker import check_env

if __name__ == "__main__":
    env = MobileRobot(0, 0)
    #check the env
    check_env(env)