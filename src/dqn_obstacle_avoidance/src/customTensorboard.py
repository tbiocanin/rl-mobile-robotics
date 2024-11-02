import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class SumRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(SumRewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_reward_sum = 0
        self.step_counter = 0

    def _on_step(self) -> bool:
        # Accumulate rewards at every step
        self.episode_reward_sum += self.locals["rewards"]
        self.step_counter += 1

        # Check if the episode is done
        if self.locals["dones"]:
            self.episode_rewards.append(self.episode_reward_sum)
            # Log the sum of rewards and print steps for the current episode
            self.logger.record('sum_reward_per_episode', self.episode_reward_sum)
            self.logger.record('step_cnt_curr_ep', self.step_counter)
            # Reset reward sum and steps for the next episode
            self.episode_reward_sum = 0
            self.step_counter = 0

        return True
