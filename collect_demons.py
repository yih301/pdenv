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
    elif(state[1]>0.0035 and state[2] >0.165): #left upper
      self.yvalue = np.random.randint(15,30)/100
      #self.zvalue = np.random.randint(15,35)/100
      self.zvalue = state[2]
    elif(state[1]<-0.0035 and state[2] >0.165): #right upper
      self.yvalue = np.random.randint(15,30)/100*-1
      #self.zvalue = np.random.randint(15,35)/100
      self.zvalue = state[2]
    elif(-0.003<state[1]<0.003 and state[2] >0.165): #middle up
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

def recordInfeasibleTraj(log_dir, traj_numdis=2, traj_num =18):
  env = gym.make("panda-v0")
  env = collectDemonsWrapper(env)  

  data = []
  bigdata=[]
  countd = 0
  countu = 0
  #statelist = [[0.24137686, -0.015, 0.12], [0.24872114, 0.015, 0.13]]
  for j in range(traj_num):
    state=env.reset()
    while((state[2]<0.135 and countd==9) or (state[2]>0.165 and countu==9) or 0.135<=state[2]<=0.165 or (state[2]<=0.135 and -0.005<state[1]<0.005) or (state[2]>=0.165 and -0.0035<state[1]<-0.003) or (state[2]>=0.165 and 0.003<state[1]<0.0035)):
      state=env.reset()
    expert = CrossBlockExpertNormal(state)
    if(state[2] <0.135):
      countd+=1
    else:
      countu+=1
    print(countd,countu)
    bigstate =  np.concatenate((
            env.panda.state['joint_position'],# 5
            env.panda.state['joint_velocity'],# 5
            env.panda.state['joint_torque'],# 5
            env.panda.state['ee_position'],# 3
            env.panda.state['ee_quaternion'],
            env.panda.state['ee_euler'], # 3
            env.panda.state['gripper_contact'],
            ), axis=None)
    #pdb.set_trace()
    bigtraj = [bigstate]
    traj=[np.array(state)]
    done = False
    while not done:
      pos = expert.get_next_states(state)
      state, r, done, info = env.step(pos)
      traj.append(np.array(state))
      bigstate =  np.concatenate((
            env.panda.state['joint_position'],# 5
            env.panda.state['joint_velocity'],# 5
            env.panda.state['joint_torque'],# 5
            env.panda.state['ee_position'],# 3
            env.panda.state['ee_quaternion'],
            env.panda.state['ee_euler'], # 3
            env.panda.state['gripper_contact'],
            ), axis=None)
      bigtraj.append(bigstate)
    data.append(np.array(traj))
    bigdata.append(np.array(bigtraj))
  env.close()
  
  env = gym.make("disabledpanda-v0")
  env = collectDemonsWrapper(env)  
  for j in range(traj_numdis):  
    expert = CrossBlockExpert()
    state=env.reset()
    bigstate =  np.concatenate((
            env.panda.state['joint_position'],# 9
            env.panda.state['joint_velocity'],# 9
            env.panda.state['joint_torque'],# 9
            env.panda.state['ee_position'],# 3
            env.panda.state['ee_quaternion'], #4
            env.panda.state['ee_euler'], # 3
            env.panda.state['gripper_contact'],
            ), axis=None)
    bigtraj = [bigstate]
    traj=[np.array(state)]
    done = False
    while not done:     
      pos = expert.get_next_states(state)
      state, r, done, info = env.step(pos)
      traj.append(np.array(state))
      bigstate =  np.concatenate((
            env.panda.state['joint_position'],# 5
            env.panda.state['joint_velocity'],# 5
            env.panda.state['joint_torque'],# 5
            env.panda.state['ee_position'],# 3
            env.panda.state['ee_quaternion'],
            env.panda.state['ee_euler'], # 3
            env.panda.state['gripper_contact'],
            ), axis=None)
      bigtraj.append(bigstate)
    data.append(np.array(traj))
    bigdata.append(np.array(bigtraj))
  env.close()
  data = np.array(data)
  bigdata = np.array(bigdata)
  #pdb.set_trace()
  pickle.dump(data, open(os.path.join(log_dir, 'infeasible_traj_9_1_0524.pkl'), 'wb'))
  pickle.dump(bigdata, open(os.path.join(log_dir, 'infeasible_traj_9_1_0524_full.pkl'), 'wb'))


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
  

def rundemon():
  #pdb.set_trace()
  #data = pickle.load(open(pickle_path, 'rb'))
  expert_path = 'C:\\Users\\Yilun\\Desktop\\Robot\\logs\\data\\infeasible_traj_9_1_0524.pkl'
  expert_traj_raw = list(pickle.load(open(expert_path, "rb")))
  pdb.set_trace()
  env = gym.make("panda-v0")
  env = collectDemonsWrapper(env)
  for i in range(len(expert_traj_raw)):  
    state = env.reset()
    state = expert_traj_raw[i][0]
    env.panda._set_start(position=np.array(state))
    expert = CrossBlockExpertNormal(state)  
    done = False
    while not done:
      pos = expert.get_next_states(state)
      state, r, done, info = env.step(pos)
  env.close()

if __name__ == '__main__':
  #testExpert()
  recordInfeasibleTraj(log_dir)
  pickle_path = os.path.join(log_dir, 'infeasible_traj_9_1_0524.pkl')
  plotDemons(pickle_path)
  #rundemon()
  
 

  


