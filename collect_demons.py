import argparse
import sys
import os
import inspect

import gym
from gym import wrappers, logger
import gym_panda
from gym_panda.wrapper_env.wrapper import collectDemonsWrapper, infeasibleWrapper
import time
import pickle
import numpy as np
#from env_1 import SimpleEnv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pdb


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

class CrossBlockExpert(object):
  """Expert to across a block from one side to another side.""" 
  def __init__(self):
    self.kp = 0.5
    self.kd = 0
    self.ierror = np.zeros(3)
    self.obj_len = 0.15

  def get_next_states(self, state):
    #print("State is:",state)
    target_point = np.array([0.56-self.obj_len, 0, 0.35])
    target_point2 = np.array([0.56+self.obj_len, 0, 0.35])
    target_point3 = np.array([0.56+self.obj_len+0.1, 0, 0.1])
    dpos = target_point - state
    #self.kp = 1.3
    #pdb.set_trace()
    if state[0] >= 0.41:
      #print("Switch to target 2!")
      target_point = target_point2
      dpos = target_point - state
    if state[0] >= 0.56+0.15:
      #print("Switch to target 3!")
      target_point = target_point3
      dpos = target_point - state
    next_pos = self.kp * dpos + self.kd * self.ierror
    #pdb.set_trace()
    self.ierror += dpos
 
    return next_pos

class CrossBlockExpertNormal(object):
  """Expert to across a block from one side to another side.""" 
  def __init__(self,state):
    self.kp = 0.5
    self.kd = 0
    self.ierror = np.zeros(3)
    self.obj_len = 0.15
    if(state[1]>0.0005 and state[2] <0.135): #left down
      self.yvalue = np.random.randint(15,30)/100
      #self.zvalue = np.random.randint(10,15)/100
      self.zvalue = state[2]
    elif(state[1]<-0.0005 and state[2]<0.135): #right down
      self.yvalue = np.random.randint(15,30)/100*-1
      #self.zvalue = np.random.randint(10,15)/100
      self.zvalue = state[2]
    elif(state[1]>0.0015 and state[2] >0.165): #left upper
      self.yvalue = np.random.randint(15,30)/100
      #self.zvalue = np.random.randint(15,35)/100
      self.zvalue = state[2]
    elif(state[1]<-0.0015 and state[2] >0.165): #right upper
      self.yvalue = np.random.randint(15,30)/100*-1
      #self.zvalue = np.random.randint(15,35)/100
      self.zvalue = state[2]
    elif(-0.001<state[1]<0.001 and state[2] >0.165): #middle up
      self.yvalue = 0
      self.zvalue = 0.35
    else:
      self.yvalue = 0
      self.zvalue = 0.35
    print(self.yvalue)
    print(self.zvalue)

  def get_next_states(self, state):
    #print("State is:",state)
    #target_point = np.array([0.56-self.obj_len, self.yvalue,state[2]])
    #target_point2 = np.array([0.56+self.obj_len, self.yvalue,state[2]])
    target_point = np.array([0.56-self.obj_len, self.yvalue,self.zvalue])
    target_point2 = np.array([0.56+self.obj_len, self.yvalue,self.zvalue])
    target_point3 = np.array([0.56+self.obj_len+0.1, 0, 0.1])
    dpos = target_point - state
    #self.kp = 1.3
    #pdb.set_trace()
    if state[0] >= 0.41:
      #print("Switch to target 2!")
      target_point = target_point2
      dpos = target_point - state
    if state[0] >= 0.56+0.15:
      #print("Switch to target 3!")
      target_point = target_point3
      dpos = target_point - state
    next_pos = self.kp * dpos + self.kd * self.ierror
    #pdb.set_trace()
    self.ierror += dpos
 
    return next_pos

'''class HalfCircleExpert(object):
    """ Expert to draw a circle centered at (cx, cy) with a radius of 0.2."""
    def __init__(self, env=None):
        self.r = 0.38
        self.step = 0.3 * np.pi #0.2 * np.pi
        self.cx = 0.56
        self.cy = 0.1
    def get_next_states(self, state):
        #print(state)
        cur_pos = np.array([state[0] - self.cx, state[2] - self.cy])
        cur_the = np.arctan2(cur_pos[1], cur_pos[0])
   
        next_the = cur_the
        next_the += self.step
        #print()
        next_pos = np.array([self.cx + self.r * np.cos(next_the), 0, self.cy + self.r * np.sin(next_the)])
        pdb.set_trace()
        #print("nextpos is:",next_pos)
        return next_pos

class HalfCircleExpertNormal(object):
    """ Expert to draw a circle centered at (cx, cy) with a radius of 0.2."""
    def __init__(self, env=None):
        self.r = 0.38
        self.step = 0.3 * np.pi #0.2 * np.pi
        self.cx = 0.56
        self.cy = 0.0
        self.cz = 0.1
        self.degree = np.random.randint(180)*np.pi/180
        print("degree",self.degree)

    def get_next_states(self, state, init):
        cur_pos = np.array([state[0] - self.cx, state[1] - self.cy, state[2] - self.cz])
        cur_the = np.arctan2(cur_pos[2], cur_pos[0])
   
        next_the = cur_the
        next_the += self.step
        #self.degree += self.step
        oriZ = self.cy + self.r * np.sin(next_the)
        next_pos = 1000*np.array([self.cx + self.r * np.cos(next_the), oriZ*np.cos(self.degree), oriZ* np.sin(self.degree)])
        #pdb.set_trace()
        #print("action:",next_pos-state)
        #print("nextpos is:",next_pos)
        return next_pos'''

'''
def testExpert():
  env = gym.make("panda-v0")
  #env = gym.make("disabledpanda-v0")
  env = collectDemonsWrapper(env)
  #pdb.set_trace()
  #expert = CrossBlockExpert()
  expert = CrossBlockExpertNormal()
  
  
  for i in range(30):
    state=env.reset()
    print(i,state)
  done = False
  while not done:
    #action = expert.get_next_states(state, init)-state
    #action = np.array([0.0,1000.0,0.0])
    #print(action)
    #state, r, done, info = env.step(action, mode=1)
    state, r, done, info = env.step((expert.get_next_states(state)))
    if(done is True):
      print("done!!!!!!!!!!!!")'''

def recordInfeasibleTraj(log_dir, traj_numdis=0, traj_num =1):
  env = gym.make("panda-v0")
  env = collectDemonsWrapper(env)  

  data = []
  countd = 0
  countu = 0
  #statelist = [[0.24137686, -0.015, 0.12], [0.24872114, 0.015, 0.13]]
  for j in range(traj_num):
    state=env.reset()
    pdb.set_trace()
    #state = statelist[j]
    #env.panda._set_start(position=state)
    #expert = CrossBlockExpertNormal(state)
    while((state[2]<0.135 and countd==9) or (state[2]>0.165 and countu==9) or 0.135<=state[2]<=0.165 or (state[2]<=0.135 and -0.005<state[1]<0.005) or (state[2]>=0.165 and -0.0015<state[1]<-0.001) or (state[2]>=0.165 and 0.001<state[1]<0.0015)):
      state=env.reset()
    expert = CrossBlockExpertNormal(state)
    if(state[2] <0.135):
      countd+=1
    else:
      countu+=1
    print(countd,countu)
    traj=[np.array(state)]
    done = False
    while not done:
      pos = expert.get_next_states(state)
      state, r, done, info = env.step(pos)
      traj.append(np.array(state))
    data.append(np.array(traj))
  env.close()
  
  env = gym.make("disabledpanda-v0")
  env = collectDemonsWrapper(env)  
  for j in range(traj_numdis):  
    expert = CrossBlockExpert()
    state=env.reset()
    traj=[np.array(state)]
    done = False
    while not done:     
      pos = expert.get_next_states(state)
      state, r, done, info = env.step(pos)
      traj.append(np.array(state))
    data.append(np.array(traj))
  env.close()
  data = np.array(data)
  #pdb.set_trace()
  pickle.dump(data, open(os.path.join(log_dir, 'infeasible_traj_9_1_final.pkl'), 'wb'))

def plotDemons(pickle_path):
  #pdb.set_trace()
  data = pickle.load(open(pickle_path, 'rb'))
  traj_num = data.shape[0]
  ax = plt.axes(projection='3d')
  for j in range(traj_num):
    ax.scatter3D(data[j][:,0], data[j][:,1], data[j][:,2])
  #ax.scatter3D(data[:][0,0], data[:][0,1], data[:][0,2])
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  plt.show()
  

def rundemon(pickle_path):
  #pdb.set_trace()
  data = pickle.load(open(pickle_path, 'rb'))
  env = gym.make("panda-v0")
  env = collectDemonsWrapper(env)  
  state = env.reset()
  state = data[2][0]
  expert = CrossBlockExpertNormal(state)  
  done = False
  while not done:
    pos = expert.get_next_states(state)
    state, r, done, info = env.step(pos)
  env.close()

if __name__ == '__main__':
  #testExpert()
  recordInfeasibleTraj(log_dir)
  #pickle_path = os.path.join(log_dir, 'infeasible_traj_9_1_final.pkl')
  #plotDemons(pickle_path)
  #rundemon(pickle_path)
  
 

  


