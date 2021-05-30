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



'''env = gym.make("feasibilitypanda-v0")
vae_path='C:\\Users\\Yilun\\Desktop\\Robot\\logs\\debug\\pandaenv-random-v0_0.005_vae.pt'
model = infeasibleVAEExpert(vae_path)
#expert_path = 'C:\\Users\\Yilun\\Desktop\\Robot\\logs\\data\\infeasible_traj_9_1_kongxi2.pkl'
expert_path = 'C:\\Users\\Yilun\\Desktop\\Robot\\logs\\data\\infeasible_traj_9_1_0524_full.pkl'
expert_path2 = 'C:\\Users\\Yilun\\Desktop\\Robot\\logs\\data\\infeasible_traj_9_1_0528_full.pkl'
expert_traj_raw = list(pickle.load(open(expert_path, "rb")))
expert_traj_raw2 = list(pickle.load(open(expert_path2, "rb")))
#pdb.set_trace()

for i in range(len(expert_traj_raw)-18):
    done = False
    state = env.reset(i+18,"dis")
    #state =expert_traj_raw[i][0]
    #jointposition = np.concatenate((expert_traj_raw[i+18][0][:9],np.array([0.03,0.03])),axis=None)
    #env.panda._reset_robot(jointposition)
    #state=env.panda.state['ee_position']
    #pdb.set_trace()
    while not done:
        ns = model.get_next_states(state)
        action = 50*(ns-state)
        #print(ns)
        state, rewards, done, info = env.step(action)
        #state = state['ee_position']
for i in range(len(expert_traj_raw2)):
    done = False
    state = env.reset(i,"normal")
    while not done:
        ns = model.get_next_states(state)
        action = 50*(ns-state)
        #print(ns)
        state, rewards, done, info = env.step(action)'''



env = gym.make("disabledpanda-v0")
vae_path='C:\\Users\\Yilun\\Desktop\\Robot\\logs\\debug\\feasibility!!\\pandaenv-random-v0_0.005_vae.pt'
#vae_path='C:\\Users\\Yilun\\Desktop\\Robot\\logs\\debug\\pandaenv-random-v0_0.005_vae.pt'
model = infeasibleVAEExpert(vae_path)
expert_path = 'C:\\Users\\Yilun\\Desktop\\Robot\\logs\\data\\infeasible_traj_9_1_50.pkl'
#expert_traj_raw = list(pickle.load(open(expert_path, "rb")))
#pdb.set_trace()

reward = []
success = []
for i in range(10):
    done = False
    state = env.reset()['ee_position']
    print(state)
    #state =expert_traj_raw[0][0]
    #state =[ 2.93779939e-01, -1.82864940e-04,  1.69315249e-01]
    #env.panda._set_start(position=state)
    #pdb.set_trace()
    while not done:
        ns = model.get_next_states(state)
        #print(ns)
        action = 50*(ns-state)
        state, rewards, done, info = env.step(action)
        state = state['ee_position']
    reward.append(rewards)
    success.append(rewards>0)

print(reward)
print(success)