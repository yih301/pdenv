import argparse
import sys
import os
import inspect

import gym
from gym import wrappers, logger
'''current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
from wrapper_env.wrapper import collectDemonsWrapper, infeasibleWrapper'''
import gym_panda
from gym_panda.wrapper_env.wrapper import collectDemonsWrapper, infeasibleWrapper
import time
import pickle
import numpy as np
from env_1 import SimpleEnv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

log_dir = 'C:\\Users\\Yilun\\Desktop\\Robot\\logs\\data'
#log_dir = '/iliad/u/yilunhao/logs/data'   #uncomment this when running in cluster

class CircleExpert(object):
    """ Expert to draw a circle centered at (cx, cy) with a radius of 0.2."""
    def __init__(self, env=None):
        self.r = 0.2
        self.step = 0.1 * np.pi #0.2 * np.pi
        self.cx = 0.35
        self.cy = 0.52

    
    def get_next_states(self, state):
        cur_pos = np.array([state[0] - self.cx, state[2] - self.cy])
        cur_the = np.arctan2(cur_pos[1], cur_pos[0])
   
        next_the = cur_the
        # if abs(np.sqrt(np.sum(np.square(cur_the))) - self.r) < 0.1:
        #     # if on the circle, move forward by one step
        next_the += self.step      

        next_pos = np.array([self.cx + self.r * np.cos(next_the), 0, self.cy + self.r * np.sin(next_the)])
        return next_pos

class SinExpert(object):
  """Expert to draw a sin line. A very bad tuned and non-robust controller!"""
  
  def __init__(self):
    self.w = np.pi / 100
    self.magnitude = 0.05
    self.step = -0.1

    self.t = 0

    self.kpx = 1
    self.kpy = 100

    self.e_x = 0
    self.e_y = 0

  def get_next_states(self, state):

    cur_x = state[0]
    cur_y = state[1]
    cur_z = state[2]

    if self.t == 0:
      self.init_x = cur_x
      self.init_y = cur_y
    
    self.t += 1

    t_x = self.init_x + self.step * self.t
    t_y = self.init_y + self.magnitude * np.sin(self.w * self.t)

    next_x = cur_x + self.kpx * (t_x - cur_x)
    next_y = cur_y + self.kpy * (t_y - cur_y)
    
    next_z = cur_z

    

    next_pos = np.array([next_x, next_y, next_z])
    return next_pos

class targetExpert(object):
  """Expert to draw the robot arm to a target position.""" 
  def __init__(self):
    self.target_point = np.array([0.5, 0, 0.7])
    self.kp = 10
    self.kd = 0
    self.ierror = np.zeros(3)

  def get_next_states(self, state):
    dpos =  self.target_point - state
    if np.linalg.norm(dpos) < 0.01:
      return np.zeros(3)
    next_pos = self.kp * dpos + self.kd * self.ierror
    self.ierror += dpos
 
    return next_pos

    


def recordInfeasibleTraj(log_dir, traj_num=20):
  env = gym.make("panda-v0")
  env = collectDemonsWrapper(env)  
  expert = targetExpert()

  data = []
  for j in range(traj_num):  
    state=env.reset()
    traj=[state]
    done = False
    while not done:     
      state, r, done, info = env.step(expert.get_next_states(state)-state, mode=1)
      traj.append(state)
    data.append(np.array(traj))
  data = np.array(data)
  print(data.shape)

  pickle.dump(data, open(os.path.join(log_dir, 'infeasible_traj_demons.pkl'), 'wb'))

def plotDemons(pickle_path):
  data = pickle.load(open(pickle_path, 'rb'))

  traj_num = data.shape[0]
  ax = plt.axes(projection='3d')
  for j in range(traj_num):
    ax.scatter3D(data[j,:, 0], data[j,:, 1], data[j,:, 2])
  ax.scatter3D(data[:,0, 0], data[:,0, 1], data[:,0, 2])
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  plt.show()
  


if __name__ == '__main__':
    
  recordInfeasibleTraj(log_dir)
  pickle_path = os.path.join(log_dir, 'infeasible_traj_demons.pkl')
  plotDemons(pickle_path)

  
 

  


