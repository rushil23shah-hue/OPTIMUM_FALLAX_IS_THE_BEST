import os
import pickle
import gymnasium as gym
import neat
import numpy as np

env_name="BipedalWalker-v3"
config_path="neat_config.txt"
generations=100
max_steps=1200

os.makedirs("models/neat",exist_ok=True)

def get_x(env):
    return env.unwrapped.hull.position.x

def eval_genomes(genomes,config):
    env=gym.make(env_name)

    for genome_id,genome in genomes:
        net=neat.nn.FeedForwardNetwork.create(genome,config)

        obs,info=env.reset()
        start_x=get_x(env)
        max_x=start_x
        total_reward=0
        steps=0
        done=False

        while not done and steps<max_steps:
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

        genome.fitness=total_reward+(distance*8)-(steps*0.1)+(crossed*700)

    env.close()

config=neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path
)

population=neat.Population(config)
population.add_reporter(neat.StdOutReporter(True))
population.add_reporter(neat.Checkpointer(10,filename_prefix="models/neat/checkpoint-"))

winner=population.run(eval_genomes,generations)

with open("models/neat/best_genome.pkl","wb") as f:
    pickle.dump(winner,f)

print("training completed")
print("best genome saved")
print("best fitness:",round(winner.fitness,2))