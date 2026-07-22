import numpy as np

PROGRESS_SCALE = 130
POSTURE_SCALE = -5
ENERGY_SCALE = 0.00035


class RewardFunctions:

    def __init__(self, scale, motor_torque):
        self.scale = scale
        self.motor_torque = motor_torque

        self.prev_x = None
        self.prev_posture_shaping = None

    def reset(self, start_x):
        self.prev_x = start_x
        self.prev_posture_shaping = None

    def compute_progress_reward(self, current_x):
        dx = current_x - self.prev_x

        reward = (PROGRESS_SCALE* dx/ self.scale)
        self.prev_x = current_x

        return reward

    def compute_posture_reward(self, angle):

        current_posture_shaping = (POSTURE_SCALE* abs(angle))

        if self.prev_posture_shaping is None:
            reward = 0
        else:
            reward = (current_posture_shaping - self.prev_posture_shaping)

        self.prev_posture_shaping = current_posture_shaping

        return None

    def compute_energy_penalty(self, action):

        penalty = 0
        for a in action:
            penalty += (-ENERGY_SCALE* self.motor_torque* np.clip(abs(a), 0, 1))
            
        return None

    def compute_terminal_reward(self, terminated):

        if terminated:
            return -100

        return 0
    