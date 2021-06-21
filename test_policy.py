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
import argparse
from itertools import count
import scipy.optimize
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *
#from gen_dem import demo
import pickle
import random
import torch.nn as nn
import torch
from SAIL.utils.math import *
from torch.distributions import Normal
from SAIL.models.sac_models import weights_init_
from torch.distributions.categorical import Categorical
from SAIL.models.ppo_models import Value, Policy, DiscretePolicy
import pybullet as p


env = gym.make("disabledpanda-v0")

seed_list = [0,2345,3456,4567,5678]
dict_list = ["disabledpanda-v0_default_simulated_robot_1.0_0.005_models_seed_0.pt",
"disabledpanda-v0_default_simulated_robot_1.0_0.005_models_seed_2345.pt",
"disabledpanda-v0_default_simulated_robot_1.0_0.005_models_seed_3456.pt",
"disabledpanda-v0_default_simulated_robot_1.0_0.005_models_seed_4567.pt",
"disabledpanda-v0_default_simulated_robot_1.0_0.005_models_seed_5678.pt"]

reward = []
success = []
for i in range(1):
    seed = seed_list[i]
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    policy_indices = list(range(num_inputs))
    #state_dict = torch.load("disabledpanda-v0_default_simulated_robot_1.0_0.005_models_seed_0.pt")
    policy_net = Policy(num_inputs, 3, log_std=-5)
    policy_net.load_state_dict(torch.load(dict_list[i])[1])
    print(i,dict_list[i])
    for i in range(1):
        done = False
        state =  torch.from_numpy(env.reset()['ee_position'])
        while not done:
            #pdb.set_trace()
            mean_action, action_log_std, action_std = policy_net.forward(state.float().reshape((1,3)))
            prev = state.detach().numpy()
            mean_action = mean_action.detach().numpy() 
            #pdb.set_trace()
            state, rewards, done, info = env.step(np.array(mean_action[0]))
            state = torch.from_numpy(state['ee_position'])
            p.addUserDebugLine(prev, state, [1, 0, 0], lineWidth=3., lifeTime=0)
        pdb.set_trace()
        reward.append(rewards)
        success.append(int(rewards>0))

pickle.dump(reward, open('..\\logs\\plotdata\\sail_reward.pkl', 'wb'))
pickle.dump(success, open('..\\logs\\plotdata\\sail_success.pkl', 'wb'))