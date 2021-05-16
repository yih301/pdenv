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

'''#env = gym.make("panda-v0")
vae_path='C:\\Users\\Yilun\\Desktop\\Robot\\logs\\debug\\pandaenv-random-v0_0.005_vae.pt'
model = infeasibleVAEExpert(vae_path)
#target_point = np.array([0.56-self.obj_len, 0, 0.35])
#target_point2 = np.array([0.56+self.obj_len, 0, 0.35])
#target_point3 = np.array([0.56+self.obj_len+0.1, 0, 0.1])
prev_state = [ 2.89214939e-01, -6.08537960e-12,  1.61080852e-01, 0.41,0,0.35,0.71,0,0.35,  0.81,0,0.1]
done = False
i=0
while i<50:
    expect_state = model.get_next_states(prev_state)
    prev_state = expect_state
    i=i+1
    print(expect_state)'''

env = gym.make("panda-v0")
vae_path='C:\\Users\\Yilun\\Desktop\\Robot\\logs\\debug\\pandaenv-random-v0_0.005_vae.pt'
model = infeasibleVAEExpert(vae_path)

done = False
state = env.reset()
state = [ 2.89214939e-01, -6.08537960e-12,  1.61080852e-01, 0.41,0,0.35,0.71,0,0.35,  0.81,0,0.1]
while not done:
    action = model.get_next_states(state)[:3]-state[:3]
    state, rewards, done, info = env.step(action)
    #print(state)
    state = state['ee_position']
    state = np.concatenate((state, np.array([0.41,0,0.35,0.71,0,0.35,  0.81,0,0.1])))