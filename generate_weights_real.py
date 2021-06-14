import os

import gym
#import gym_circle_move
import numpy as np
import matplotlib.pyplot as plt
import time

import gym_panda
from gym_panda.wrapper_env.wrapper import *
import argparse
from itertools import count

import scipy.optimize

import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *

from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import DDPG,PPO2, TRPO

def pickdemo():
    folderlist =['lb','lu','rb','ru','up']
    demo = []
    env = gym.make("feasibilitypanda-v0")
    state = env.reset()
    for folder in folderlist:
        print(folder)
        count=0
        for i in range(10):
            #print(i)
            if not (folder=='lb' and i==9):
                singledemo = pickle.load(open('..\\logs\\demo\\'+folder+'\\demo_'+str(i)+'.pkl', 'rb'))
                jointposition = np.concatenate((singledemo[0],np.array([0,0,0.03,0.03])),axis=None)
                env.panda._reset_robot(jointposition)
                state = env.panda.state['ee_position']
                if not (0.27<=state[2]<=0.33 or (state[2]<=0.27 and -0.005<state[1]<0.005) or (state[2]>=0.27 and -0.0035<state[1]<-0.003) or (state[2]>=0.33 and 0.003<state[1]<0.0035)):
                    count+=1
                    demostates = []
                    for i in range(len(singledemo)):
                        jointposition = np.concatenate((singledemo[i],np.array([0,0,0.03,0.03])),axis=None)
                        env.panda._reset_robot(jointposition)
                        state = env.panda.state['ee_position']
                        demostates.append(state)
                    demo.append(np.array(demostates))
        print(count)
    print(len(demo))
    pickle.dump(demo, open('..\\logs\\demo\\fulldemo.pkl', 'wb'))
    #pdb.set_trace()

def generate_weight(DISMODEL_PATH,NORMALMODEL_PATH, DEMONS,DIS_DEMONS, NORMAL_DEMONS, SAVE_PATH, seed):
    env = gym.make("feasibilitypanda-v0")
    env.seed(seed)
    torch.manual_seed(seed)
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    policy_indices = list(range(num_inputs))
    dispolicy_net = Policy(len(policy_indices), num_actions)
    normalpolicy_net = Policy(len(policy_indices), num_actions)
    dispolicy_net.load_state_dict(torch.load(DISMODEL_PATH))
    normalpolicy_net.load_state_dict(torch.load(NORMALMODEL_PATH))

    # get data
    data =  np.array(pickle.load(open(DEMONS, 'rb')))   #new demo data
    disabled =  pickle.load(open(DIS_DEMONS, 'rb'))   #normal panda data
    normal =  pickle.load(open(NORMAL_DEMONS, 'rb'))   #normal panda data
    #pdb.set_trace()
    discount = 0.9

    weights = np.zeros(data.shape[0])  #shape is num of traj
    #pdb.set_trace()
    for j in range(data.shape[0]-3):
        print("demo",j)
        trajweight = np.zeros(data[j].shape[0])  #shape is length of current traj
        state = env.reset(j,"demo") #will set to the jth init state of normal(it doesn't matter if it is random)
        i = 0
        done = False
        step = normal[j].shape[0]
        demostep = data[j].shape[0]
        waitstep = int(step/demostep)
        #print("wa",waitstep)
        count=0
        while not done:
            prestate = torch.from_numpy(state).unsqueeze(0)
            action_mean, _, action_std = normalpolicy_net(Variable(prestate).float())
            state, rewards, done, info = env.step(action_mean)
            if count==0 and i<data[j].shape[0]-1:
                #print("record")
                prdemo = data[j][i]
                nxtdemo = data[j][i+1]
                #pdb.set_trace()
                transit = state-(prestate.numpy()-prdemo)
                dis = np.linalg.norm(nxtdemo - transit)
                trajweight[i] = dis*discount**(i)
                i = i + 1
            count = count+1
            if(count==waitstep):
                count=0
        #print(trajweight)
        weights[j] = -np.sum(trajweight) #update weight traj by traj
    for j in range(3):
        print("dis",j)
        num = data.shape[0]-3
        trajweight = np.zeros(data[j+num].shape[0])  #shape is length of current traj
        state = env.reset(j,"dis") #will set to the jth init state of dis(it doesn't matter if it is random)
        i = 0
        done = False
        step = disabled[j].shape[0]
        #pdb.set_trace()
        demostep = data[j+num].shape[0]
        waitstep = int(step/demostep)
        #print("wa",waitstep)
        count=0
        while not done:
            prestate = torch.from_numpy(state).unsqueeze(0)
            action_mean, _, action_std = dispolicy_net(Variable(prestate).float())
            state, rewards, done, info = env.step(action_mean)
            if count==0 and i<data[j+num].shape[0]-1:
                #print("record")
                prdemo = data[j+num][i]
                nxtdemo = data[j+num][i+1]
                #pdb.set_trace()
                transit = state-(prestate.numpy()-prdemo)
                dis = np.linalg.norm(nxtdemo - transit)
                trajweight[i] = dis*discount**(i)
                i = i + 1
            count = count+1
            if(count==waitstep):
                count=0
        #print(trajweight)
        weights[j+num] = -np.sum(trajweight) #update weight traj by traj
    #pdb.set_trace()    
    print(weights)
    print("average:",np.mean(weights))
    # rescale weights
    maxweight = np.max(weights)
    low = weights.shape[0]/10
    high = weights.shape[0]*9/10
    #pdb.set_trace()
    alpha = np.abs(np.mean(weights[int(low):int(high)])-maxweight)/10
    weights = np.exp((weights -maxweight) /alpha)
    print(weights)
    #pickle.dump(weights, open(SAVE_PATH, 'wb'))
            

if __name__ == "__main__":
    #pickdemo()
    generate_weight("C:\\Users\\Yilun\\Desktop\\Robot\\logs\\good\\dis\\dis5_3333model",
                    "C:\\Users\\Yilun\\Desktop\\Robot\\logs\\good\\normal\\normal48_3333model",  
                    '..\\logs\\demo\\fulldemo_no_u.pkl',
                    '..\\logs\\data\\infeasible_traj_9_1_5dis.pkl',
                    '..\\logs\\data\\infeasible_traj_9_1_0528.pkl',
                    '..\\logs\\data\\seeds\\demo_weights_3333.pkl',
                    3333)
#3 6 22 66(2) 3333