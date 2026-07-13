'''
This is to compare that E00_rewards_sep and the orignal 
base_bipedalwalker.py generate the same reward.

'''
import numpy as np 
from envs.base_bipedalwalker import BipedalWalker as BaseWalker
from envs.E00_reward_sep import BipedalWalker as RewardSepWalker

env1=BaseWalker()
env2=RewardSepWalker()

obs1,_ = env1.reset(seed = 42)# seed is added to make sure both env generate the same terrain and intial conditions 
obs2,_ = env2.reset(seed = 42) 

for step in range (100):
    action = env1.action_space.sample()
    obs1,reward1,terminated1,truncated1,info1 = env1.step(action)
    obs2,reward2,terminated2,truncated2,info2 = env2.step(action)
    
    difference = abs(reward1-reward2)
    print(
    f"{step:03d} | "
    f"Base = {reward1:.6f} | "
    f"Refactored = {reward2:.6f} | "
    f"Diff = {difference:.8f}"
    )
    if difference > 1e-6:
        print("Mismatch Detected")
        break
        
    
    if terminated1 or terminated2 or truncated1 or truncated2:
        break
