"""Wrappers for intergating imitation reward function."""
import gym
import gym_panda
from gym_panda.wrapper_env.VAE import VAE
import torch
import numpy as np
import copy
import pickle
import time

class collectDemonsWrapper(gym.Wrapper):
    """ Wrapper used to collect demonstrations."""
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.env = env
        self.time_step = 0
        self.eps_len = 8000
        

    def _random_reset(self):
        """ Random pick init start positions.
        Note: you can customize the range, but be sure to check the reachibility of the robot arm.
        """
        theta = np.random.uniform(low=0.0, high=2*np.pi)
        phi = np.random.uniform(low=0.0, high=2*np.pi)
        r = np.random.uniform(low=0.1, high=0.1)
        dpos = [0.15+r*np.sin(theta)*np.cos(phi),r*np.sin(theta)*np.sin(phi),0.2+r*np.cos(theta)]
        self.env.panda._set_start(position=dpos)

    def reset(self):
        self.env.reset()
        #self._random_reset()
        #self.env.resetobj()
        state = self.env.panda.state['ee_position']
        print(state)
        self.time_step = 0
        return state


    def step(self, action):
        #time.sleep(0.05)
        #print("wrapper")
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
        self.eps_len = 500
        self.gt_data =  pickle.load(open('..\\logs\\data\\infeasible_traj_demons.pkl', 'rb'))
        

    def _random_select(self, state, idx=None):
        # random pick start pos
        if idx is None:
            self.gt_num = np.random.choice(self.gt_data.shape[0])
        else:
            self.gt_num = idx
        self.gt = self.gt_data[self.gt_num, :, :]
       
        pos = self.gt[0, :]
        # pos[1] = 0
        self.env.panda._set_start(position=pos)
        print(self.env.panda.state['ee_position'], pos)

    def reset(self,idx=None):
        self.env.reset()
        state = self.env.panda.state['ee_position']
        self._random_select(state,idx)

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

        state = self.env.panda.state['ee_position']
        self.time_step += 1
        done = (self.time_step >= self.eps_len - 1)
        dis = 10 * np.linalg.norm(state - self.gt[self.time_step,:])
        reward = - (dis**2)
        info = {}
        self.prev_state = copy.deepcopy(state)
        full_state =  np.concatenate((
            self.env.panda.state['joint_position'],# 5
            self.env.panda.state['joint_velocity'],# 5
            self.env.panda.state['joint_torque'],# 5
            self.env.panda.state['ee_position'],# 3
            self.env.panda.state['ee_euler'], # 3
            ), axis=None)
        info['dis'] = dis/10
        
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
        expect_state = self.model.get_next_states(torch.FloatTensor(state)).detach().numpy()
        return expect_state
    
    '''def step(self, action):
        print("step")
        self.env.step(action)       
        state = self.env.panda.state['ee_position']
        self.time_step += 1
        self.prev_state = copy.deepcopy(state)
        print(np.linalg.norm(np.array([state[0] - 0.81, state[2] - 0.1])))
        done = (self.time_step >= self.eps_len - 1) or np.linalg.norm(np.array([state[0] - 0.81, state[2] - 0.1])) < 0.028
        reward = 0
        info = {}      
        return state, reward, done, info'''


class SkipStepsWrapperVAE(gym.Wrapper):
    """ Wrapper to use CircleExpert for rewards, with max episode length = 40000."""
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        #gym.Wrapper.__init__ does not have (self, env, path) so I pull it out
        vae_path='C:\\Users\\Yilun\\Desktop\\Robot\\logs\\debug\\pandaenv-random-v0_0.005_vae.pt'
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