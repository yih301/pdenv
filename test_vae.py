import gym
import gym_panda
from gym_panda.wrapper_env.VAE import VAE
import torch
import tensorflow as tf
import numpy as np
import copy
import pickle
import gym_panda
from gym_panda.wrapper_env.wrapper import *
import pdb

seed = 3
env = gym.make("disabledpanda-v0")
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
#vae_path='C:\\Users\\Yilun\\Desktop\\Robot\\logs\\data\\baseline\\model\\baseline100_3.pt'
vae_path='C:\\Users\\Yilun\\Desktop\\Robot\\logs\\debug\\work!!\\3_normal.pt'
model = infeasibleVAEExpert(vae_path)

reward = []
success = []
for i in range(10):
    done = False
    state = env.reset()['ee_position']
    print(state)
    while not done:
        ns = model.get_next_states(state)
        action = 50*(ns-state)
        state, rewards, done, info = env.step(action)
        state = state['ee_position']
    reward.append(rewards)
    success.append(rewards>0)

print(reward)
print(success)