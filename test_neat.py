import pickle
import gymnasium as gym
import neat
import numpy as np

env_name="BipedalWalker-v3"
config_path="neat_config.txt"

def get_x(env):
    return env.unwrapped.hull.position.x

config=neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path
)

with open("models/neat/best_genome.pkl","rb") as f:
    genome=pickle.load(f)

net=neat.nn.FeedForwardNetwork.create(genome,config)

env=gym.make(env_name,render_mode="human")

obs,info=env.reset()
start_x=get_x(env)
max_x=start_x
total_reward=0
steps=0
done=False

while not done and steps<1200:
    action=net.activate(obs)
    action=np.array(action,dtype=np.float32)
    action=np.clip(action,-1,1)

    obs,reward,terminated,truncated,info=env.step(action)

    max_x=max(max_x,get_x(env))
    total_reward+=reward
    steps+=1
    done=terminated or truncated

final_x=get_x(env)
distance=max_x-start_x
crossed=1 if final_x>=88 else 0

env.close()

print("reward:",round(total_reward,2))
print("steps:",steps)
print("distance:",round(distance,2))
print("final_x:",round(final_x,2))
print("crossed:",crossed)