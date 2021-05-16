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


def generate_weight(MODEL_PATH, DEMONS_PATH, SAVE_PATH):
    env = gym.make("panda-v0")
    env = infeasibleWrapper(env)
    model = TRPO.load(MODEL_PATH)

    # generate weights
    gt_data =  pickle.load(open(DEMONS_PATH, 'rb'))
    weights = np.zeros((gt_data.shape[0], gt_data.shape[1]))
    for j in range(gt_data.shape[0]):
        obs = env.reset(j)
        state = env.panda.state['ee_position']
        i = 0
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            state = env.panda.state['ee_position']
            dis = info['dis']
            weights[j,i] = dis
            i = i + 1
    sum_weights = np.sum(weights, axis=1)
    print(sum_weights.shape)
    for i in range(gt_data.shape[0]):
        print(sum_weights[i])
    
    # rescale weights
    sum_weights = 1/sum_weights
    sum_weights = sum_weights / np.max(sum_weights)
    for i in range(gt_data.shape[0]):
        print(sum_weights[i])

    n = gt_data.shape[0]
    t = gt_data.shape[1]
    dim = gt_data.shape[2]
    raw_data = np.ones((n,t ,dim+1))
    raw_data[:,:,:dim] = gt_data
    for i in range(n):
        raw_data[i,:,dim] = sum_weights[i]*raw_data[i,:,dim]
    pickle.dump(raw_data, open(SAVE_PATH, 'wb'))
            

if __name__ == "__main__":
    generate_weight('..\\logs\\models\\sb-trpo-joint-target-diffdynamics-2021-05-11/best_model.zip', 
                    '..\\logs\\data\\infeasible_traj_demons.pkl',
                    '..\\logs\\data\\infeasible_traj_demons_with_weights.pkl')

