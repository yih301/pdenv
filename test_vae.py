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

'''env = gym.make("disabledpanda-v0")
state = env.reset()
done=False
while not done:
    action = np.array([0,10,0])
    state, rewards, done, info = env.step(action)
    state = state['joint_position']
    print(state)'''


env = gym.make("panda-v0")
vae_path='C:\\Users\\Yilun\\Desktop\\Robot\\logs\\debug\\pandaenv-random-v0_0.005_vae.pt'
model = infeasibleVAEExpert(vae_path)
#expert_path = 'C:\\Users\\Yilun\\Desktop\\Robot\\logs\\data\\infeasible_traj_9_1_kongxi2.pkl'
expert_path = 'C:\\Users\\Yilun\\Desktop\\Robot\\logs\\data\\infeasible_traj_9_1_0523_full.pkl'
expert_traj_raw = list(pickle.load(open(expert_path, "rb")))
#pdb.set_trace()

for i in range(len(expert_traj_raw)):
    done = False
    state = env.reset()
    #state =expert_traj_raw[i][0]
    jointposition = np.concatenate((expert_traj_raw[i][0][:9],np.array([0.03,0.03])),axis=None)
    env.panda._reset_robot(jointposition)
    state=env.panda.state['ee_position']
    #pdb.set_trace()
    while not done:
        ns = model.get_next_states(state)
        action = 10*(ns-state)
        #print(ns)
        state, rewards, done, info = env.step(action)
        state = state['ee_position']

'''vae_path='C:\\Users\\Yilun\\Desktop\\Robot\\logs\\debug\\pandaenv-random-v0_0.005_vae.pt'
model = infeasibleVAEExpert(vae_path)

state = [0.24137686, -0.03, 0.12]
#print(model.get_next_states(state))
count=0
while count<200:
    state =model.get_next_states(state)
    print(state)
    count+=1'''

'''env = gym.make("panda-v0")
#env = collectDemonsWrapper(env)  
expert_traj_raw = list(pickle.load(open("C:\\Users\\Yilun\\Desktop\\Robot\\logs\\data\\infeasible_traj_9_1_fixorifixzkongxi.pkl", "rb")))
state = expert_traj_raw[0][0]
done = False
i=1
while not done:
    pos = expert_traj_raw[0][i]
    state, r, done, info = 100*env.step(pos-state)
    i+=1'''