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
    env = gym.make("panda-v0")
    env = SkipStepsWrapperVAE(env)
    model = TRPO.load(MODEL_PATH)

    done = False
    state = env.reset()
    print(env.panda.state['ee_position'])
    while not done:
        action, _states = model.predict(state)
        state, rewards, done, info = env.step(action)
        
        #print(env.panda.state['ee_position'])
    print(env.panda.state['ee_position'])
    env.close()
            

if __name__ == "__main__":
    test_model('C:\\Users\\Yilun\\Desktop\\Robot\\logs\\models\\sb-trpo-joint-target-diffdynamics-2021-05-11\\best_model.zip')

