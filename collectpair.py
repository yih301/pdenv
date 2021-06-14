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
pairs = []
for i in range(1000):
    done = False
    state = env.reset()['ee_position']
    #print(state)
    while not done:
        prev_state = state
        action = env.action_space.sample()
        state, rewards, done, info = env.step(action)
        state = state['ee_position']
        #print(prevstate,state,action)
        tosave = np.concatenate((prev_state,state,action),axis=0)
        pairs.append(tosave)
        #print(tosave)
        #pdb.set_trace()
#pickle.dump(pairs, open('..\\logs\\data\\pairs100.pkl'), 'wb')
pickle.dump(pairs, open('..\\logs\\data\\baseline\\pairs1000_3.pkl', 'wb'))
