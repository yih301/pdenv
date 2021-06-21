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
import pybullet as p


vae_paths = ['C:\\Users\\Yilun\\Desktop\\Robot\\logs\\debug\\work!!\\3_fea.pt',
'C:\\Users\\Yilun\\Desktop\\Robot\\logs\\debug\\work!!\\6_fea.pt',
'C:\\Users\\Yilun\\Desktop\\Robot\\logs\\debug\\work!!\\22_fea.pt',
'C:\\Users\\Yilun\\Desktop\\Robot\\logs\\debug\\work!!\\66_fea.pt',
'C:\\Users\\Yilun\\Desktop\\Robot\\logs\\debug\\work!!\\3333_fea.pt']

'''vae_paths = ['C:\\Users\\Yilun\\Desktop\\Robot\\logs\\debug\\work!!\\3_normal.pt',
'C:\\Users\\Yilun\\Desktop\\Robot\\logs\\debug\\work!!\\6_normal.pt',
'C:\\Users\\Yilun\\Desktop\\Robot\\logs\\debug\\work!!\\22_normal.pt',
'C:\\Users\\Yilun\\Desktop\\Robot\\logs\\debug\\work!!\\66_normal.pt',
'C:\\Users\\Yilun\\Desktop\\Robot\\logs\\debug\\work!!\\3333_normal.pt']'''
'''vae_paths = ['C:\\Users\\Yilun\\Desktop\\Robot\\logs\\data\\baseline\\model\\baseline1000_3.pt',
'C:\\Users\\Yilun\\Desktop\\Robot\\logs\\data\\baseline\\model\\baseline1000_6.pt',
'C:\\Users\\Yilun\\Desktop\\Robot\\logs\\data\\baseline\\model\\baseline1000_33.pt',
'C:\\Users\\Yilun\\Desktop\\Robot\\logs\\data\\baseline\\model\\baseline1000_66.pt',
'C:\\Users\\Yilun\\Desktop\\Robot\\logs\\data\\baseline\\model\\baseline1000_666.pt']'''
seed_list = [3,6,22,66,3333]
#seed_list = [3,6,33,66,666]
env = gym.make("disabledpanda-v0")
reward = []
success = []
cstate = []
nstate = []
for i in range(1):
    print(seed_list[i])
    seed = seed_list[i]
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    #vae_path='C:\\Users\\Yilun\\Desktop\\Robot\\logs\\data\\baseline\\model\\baseline100_3.pt'
    #vae_path='C:\\Users\\Yilun\\Desktop\\Robot\\logs\\debug\\work!!\\3_normal.pt'
    vae_path=vae_paths[i]
    print(vae_path)
    model = infeasibleVAEExpert(vae_path)
    for i in range(10):
        done = False
        state = env.reset()['ee_position']
        print(state)
        while not done:
            ns = model.get_next_states(state)
            #cstate.append(state)
            prev = state
            action = 50*(ns-state)
            state, rewards, done, info = env.step(action)
            state = state['ee_position']
            #nstate.append(state)
            p.addUserDebugLine(prev, state, [1, 0, 0], lineWidth=3., lifeTime=0)
        reward.append(rewards)
        success.append(int(rewards>0))
pdb.set_trace()
#pickle.dump(reward, open('..\\logs\\plotdata\\id_reward.pkl', 'wb'))
#pickle.dump(success, open('..\\logs\\plotdata\\id_success.pkl', 'wb'))
