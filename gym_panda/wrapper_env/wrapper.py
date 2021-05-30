"""Wrappers for intergating imitation reward function."""
import gym
import gym_panda
from gym_panda.wrapper_env.VAE import VAE
import torch
import numpy as np
import copy
import pickle
import time
import pdb

class collectDemonsWrapper(gym.Wrapper):
    """ Wrapper used to collect demonstrations."""
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.env = env
        self.time_step = 0
        self.eps_len = 8000
        

    def reset(self):
        self.env.reset()
        state = self.env.panda.state['ee_position']
        #print(state)
        self.time_step = 0
        return state


    def step(self, action):
        self.env.step(action)       
        state = self.env.panda.state['ee_position']
        self.time_step += 1
        self.prev_state = copy.deepcopy(state)
        done = (self.time_step >= self.eps_len - 1) or np.linalg.norm(np.array([state[0] - 0.81, state[2] - 0.1])) < 0.028
        #done = False
        reward = 0
        info = {}      
        return state, reward, done, info


    def close(self):
        self.env.close()

class infeasibleWrapper(gym.Wrapper):
    """ Wrapper used to train reinforcement learning agent to track multiple trajectories."""
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.env = env
        self.time_step = 0
        self.eps_len = 8000
        self.gt_data =  pickle.load(open('..\\logs\\data\\infeasible_traj_9_1_0523_full.pkl', 'rb'))
        

    def _random_select(self, state, idx=None):
        #print("random select")
        # random pick start pos
        if idx is None:
            #self.gt_num = np.random.choice(self.gt_data.shape[0]-2)
            self.gt_num = np.random.choice(2)+18
        else:
            self.gt_num = idx
        self.gt = self.gt_data[self.gt_num][:][:]
        
       
        pos = self.gt[0][27:30]
        jointposition = np.concatenate((self.gt[0][:9],np.array([0.03,0.03])),axis=None)
        # pos[1] = 0
        self.env.panda._reset_robot(jointposition)
        #pdb.set_trace()
        self.eps_len = self.gt.shape[0]
        print(self.env.panda.state['ee_position'], pos)

    def reset(self,idx=None):
        print("wrapper")
        self.env.reset()
        state = self.env.panda.state['ee_position']
        self._random_select(state,idx)

        '''state =  np.concatenate((
            self.env.panda.state['joint_position'],# 5
            self.env.panda.state['ee_position'],# 3
            ), axis=None)'''
        state = self.env.panda.state['ee_position']
        self.time_step = 0
        
        return state


    def step(self, action):
        #print("step in wrapper")
        #pdb.set_trace()
        #print("action is:" ,action)
        self.env.step(action)       

        state = self.env.panda.state['ee_position']
        self.time_step += 1
        done = (self.time_step >= self.eps_len - 1)
        dis = np.linalg.norm(state - self.gt[self.time_step][27:30])
        #print(state, self.gt[self.time_step,:], dis)
        reward = -dis
        info = {}
        self.prev_state = copy.deepcopy(state)
        '''full_state =  np.concatenate((
            self.env.panda.state['joint_position'],# 9
            self.env.panda.state['ee_position'],# 3
            ), axis=None)'''
        full_state = self.env.panda.state['ee_position']
        info['dis'] = dis
        
        return full_state, reward, done, info


    def close(self):
        self.env.close()

class infeasibleVAEExpert():
    def __init__(self, vae_path):
        print("wrapper")
        self.model = VAE(3)
        model_dict = torch.load(vae_path, map_location='cpu')
        self.model.load_state_dict(model_dict)
        self.model.eval
        self.time_step = 0
        self.eps_len = 8000

    def get_next_states(self, state):
        #pdb.set_trace()
        expect_state = self.model.get_next_states(torch.FloatTensor(state)).detach().numpy()
        return expect_state
    
    def step(self, action):
        print("step")
        self.env.step(action)       
        state = self.env.panda.state['ee_position']
        self.time_step += 1
        self.prev_state = copy.deepcopy(state)
        print(np.linalg.norm(np.array([state[0] - 0.81, state[2] - 0.1])))
        done = (self.time_step >= self.eps_len - 1) or np.linalg.norm(np.array([state[0] - 0.81, state[2] - 0.1])) < 0.028
        reward = 0
        info = {}      
        return state, reward, done, info


class SkipStepsWrapperVAE(gym.Wrapper):
    """ Wrapper to use CircleExpert for rewards, with max episode length = 40000."""
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        #gym.Wrapper.__init__ does not have (self, env, path) so I pull it out
        vae_path='C:\\Users\\Yilun\\Desktop\\Robot\\logs\\debug\\kongxilargem!!\\pandaenv-random-v0_0.005_vae.pt'
        #need to change path to run in cluster
        #vae_path = '/iliad/u/yilunhao/logs/pandaenv-random-v0_0.005_vae.pt'
        self.env = env
        self.time_step = 0
        self.eps_len = 500 # TODO: tune it for different task (circle, sin, or infeasible).
        self.model = infeasibleVAEExpert(vae_path)

    def reset(self):
        # state = self.env.reset()['ee_position']
        self.env.reset()
        self.prev_state = self.env.panda.state['ee_position']
        self.expect_state = self.model.get_next_states(self.prev_state)
        state =  np.concatenate((
            self.env.panda.state['joint_position'],# 5
            self.env.panda.state['joint_velocity'],# 5
            self.env.panda.state['joint_torque'],# 5
            self.env.panda.state['ee_position'],# 3
            self.env.panda.state['ee_euler'], # 3
            ), axis=None)
        self.time_step = 0
        return state


    def step(self, action):
        self.env.step(action)
        self.time_step += 1

        state = self.env.panda.state['ee_position']
        done = (self.time_step > self.eps_len)
        dis = 10 * np.linalg.norm(state - self.expect_state)
        reward = - (dis**2)
        self.expect_state = self.model.get_next_states(state)
        info = {}
        self.prev_state = copy.deepcopy(state)
        full_state =  np.concatenate((
            self.env.panda.state['joint_position'],# 5
            self.env.panda.state['joint_velocity'],# 5
            self.env.panda.state['joint_torque'],# 5
            self.env.panda.state['ee_position'],# 3
            self.env.panda.state['ee_euler'], # 3
            ), axis=None)
        return full_state, reward, done, info


    def close(self):
        self.env.close()