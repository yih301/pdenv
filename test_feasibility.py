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

env_name = "feasibilitypanda-v0"
env = gym.make(env_name)
#env = infeasibleWrapper(env)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
policy_indices = list(range(num_inputs))

seed=3    #3 6 22 66 3333
env.seed(seed)
torch.manual_seed(seed)
policy_net = Policy(len(policy_indices), num_actions)
policy_net.load_state_dict(torch.load("C:\\Users\\Yilun\\Desktop\\Robot\\logs\\good\\dis\\dis5_3model"))
#torch.save(policy_net, "..\\logs\\goodfullmodel\\normal\\normal3333model")
#policy_net.load_state_dict(torch.load("C:\\Users\\Yilun\\Desktop\\Robot\\logs\\good\\normal\\normal48_543model"))
#policy_net.load_state_dict(torch.load("C:\\Users\\Yilun\\Desktop\\Robot\\logs\\dis22_5model"))
#policy_net.load_state_dict(torch.load("C:\\Users\\Yilun\\Desktop\\Robot\\logs\\normal48_3333_12model"))
#pdb.set_trace()


#gt_data =  pickle.load(open('..\\logs\\data\\infeasible_traj_9_1_0524_full.pkl', 'rb'))
gt_data =  pickle.load(open('..\\logs\\data\\infeasible_traj_9_1_5dis_full.pkl', 'rb'))
#gt_data =  pickle.load(open('..\\logs\\data\\infeasible_traj_9_1_0528_full.pkl', 'rb'))
done = False
state = env.reset()
jointposition = np.concatenate((gt_data[1][0][:9],np.array([0.03,0.03])),axis=None)
env.panda._reset_robot(jointposition)
#state = np.concatenate((env.panda.state['joint_position'],env.panda.state['ee_position']),axis=None)
state = env.panda.state['ee_position']
reward=0
i =0
while True:
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state).float())
    state, rewards, dones, info = env.step(action_mean)
    reward = reward +rewards #* (0.9**i)
    i+=1
    if(i==gt_data[1].shape[0]):
        print(reward)
    #state = np.concatenate((env.panda.state['joint_position'],env.panda.state['ee_position']),axis=None)
print(reward)
