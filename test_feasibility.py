import argparse
from itertools import count

import gym
import gym_panda
from gym_panda.wrapper_env.wrapper import *
import scipy.optimize

import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *

env_name = "disabledpanda-v0"
env = gym.make(env_name)
env = infeasibleWrapper(env)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
policy_indices = list(range(num_inputs))

#env.seed(args.seed)
#torch.manual_seed(args.seed)
policy_net = Policy(len(policy_indices), num_actions)
policy_net.load_state_dict(torch.load("C:\\Users\\Yilun\\Desktop\\Robot\\logs\\pytorchmodel"))
#pdb.set_trace()

gt_data =  pickle.load(open('..\\logs\\data\\infeasible_traj_9_1_0523_full.pkl', 'rb'))
done = False
state = env.reset()
jointposition = np.concatenate((gt_data[18][0][:9],np.array([0.03,0.03])),axis=None)
env.panda._reset_robot(jointposition)
state = env.panda.state['ee_position']
while True:
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state).float())
    state, rewards, dones, info = env.step(action_mean)
