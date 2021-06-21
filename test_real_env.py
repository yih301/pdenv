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


class imitationExpert(object):
  """Expert to across a block from one side to another side.""" 
  def __init__(self):
    self.kp = 0.3
    self.kd = 0
    self.ierror = np.zeros(3)
    self.obj_len = 0.15

  def get_next_states(self, state):
    #print("State is:",state)
    target_point = np.array([0.55, 0, 0.5])
    target_point2 = np.array([0, 0, 2.0])
    dpos = target_point - state
    if state[2] >= 0.5:
      #print("Switch to target 2!")
      target_point = target_point2
      dpos = target_point - state
    next_pos = self.kp * dpos + self.kd * self.ierror
    self.ierror += dpos
    return next_pos

class ralExpert(object):
  """Expert to across a block from one side to another side.""" 
  def __init__(self):
    self.kp = 0.5
    self.kd = 0
    self.ierror = np.zeros(3)
    self.obj_len = 0.15

  def get_next_states(self, state):
    #print("State is:",state)
    target_point = np.array([0.1, 0, 0.1])
    dpos = target_point - state
    next_pos = self.kp * dpos + self.kd * self.ierror
    self.ierror += dpos
    return next_pos

class sailExpert(object):
  """Expert to across a block from one side to another side.""" 
  def __init__(self):
    self.kp = 1.0
    self.kd = 0
    self.ierror = np.zeros(3)
    self.obj_len = 0.15

  def get_next_states(self, state):
    #print("State is:",state)
    target_point = np.array([0.55, 0, 0.5])
    target_point2 = np.array([0, 0, 2.0])
    dpos = target_point - state
    if state[2] >= 0.5:
      #print("Switch to target 2!")
      target_point = target_point2
      dpos = target_point - state
    next_pos = self.kp * dpos + self.kd * self.ierror
    self.ierror += dpos
    return next_pos

class ourExpert(object):
  """Expert to across a block from one side to another side.""" 
  def __init__(self):
    self.kp = 0.8
    self.kd = 0
    self.ierror = np.zeros(3)
    self.obj_len = 0.15

  def get_next_states(self, state):
    #print("State is:",state)
    target_point = np.array([0.42, 0, 0.4])
    target_point2 = np.array([0.68, 0, 0.4])
    target_point3 = np.array([1.0, 0, 0.0])
    dpos = target_point - state
    #self.kp = 1.3
    #pdb.set_trace()
    if state[0] >= 0.42:
      #print("Switch to target 2!")
      target_point = target_point2
      dpos = target_point - state
    if state[0] >= 0.68:
      #print("Switch to target 3!")
      target_point = target_point3
      dpos = target_point - state
    next_pos = self.kp * dpos + self.kd * self.ierror
    #pdb.set_trace()
    self.ierror += dpos
 
    return next_pos

env = gym.make("realpanda-v0")
state = env.reset()
'''pdb.set_trace()
expert = ourExpert()   #imitationExpert ralExpert sailExpert ourExpert
done = False
while not done:
    pos = expert.get_next_states(state['ee_position'])
    state, r, done, info = env.step(pos)
    print(state['ee_position'])
print(r)
'''
for i in range(100):
  p.addUserDebugLine([i*0.1,0,0.1], [(i+1)*0.1,0,0.1], [1, 0, 0], lineWidth=3., lifeTime=0)
