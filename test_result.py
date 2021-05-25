import os

import gym
#import gym_circle_move
import numpy as np
import matplotlib.pyplot as plt
import time

import gym_panda
from gym_panda.wrapper_env.wrapper import *

from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import DDPG,PPO2, TRPO


def test_model(MODEL_PATH):
    env = gym.make("feasibilitypanda-v0")
    #env = infeasibleWrapper(env)
    model = TRPO.load(MODEL_PATH)
    gt_data =  pickle.load(open('..\\logs\\data\\infeasible_traj_9_1_0523_full.pkl', 'rb'))


    done = False
    state = env.reset()
    #pdb.set_trace()
    #print(gt_data[1][0])
    jointposition = np.concatenate((gt_data[18][0][:9],np.array([0.03,0.03])),axis=None)
    env.panda._reset_robot(jointposition)
    state = np.concatenate((env.panda.state['joint_position'],env.panda.state['ee_position']),axis=None)
    #pdb.set_trace()
    while True:
        action, _states = model.predict(state)
        state, rewards, dones, info = env.step(0.5*action)
        #print(state[27:30])
        #env.render()
            

if __name__ == "__main__":
    test_model('C:\\Users\\Yilun\\Desktop\\Robot\\logs\\models\\sb-trpo-joint-target-diffdynamics-2021-05-24\\best_model.zip')

