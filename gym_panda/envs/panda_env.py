import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_panda.panda_bullet.panda import Panda, DisabledPanda, FeasibilityPanda
from gym_panda.panda_bullet.objects import YCBObject, InteractiveObj, RBOObject
import os
import numpy as np
import pybullet as p
import pybullet_data
import pdb
import copy
import pickle

class PandaEnv(gym.Env):
  metadata = {'render.modes': ['human']}
  def __init__(self, panda_type=Panda):
    # create simulation (GUI)
    self.urdfRootPath = pybullet_data.getDataPath()
    #p.connect(p.DIRECT)
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)

    # set up camera
    self._set_camera()

    # load some scene objects
    p.loadURDF(os.path.join(self.urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.65])
    p.loadURDF(os.path.join(self.urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.65])

    # example YCB object
    obj1 = YCBObject('czj_large_box')
    obj1.load()
    p.resetBasePositionAndOrientation(obj1.body_id, [0.56, 0., 0.1], [0, 0, 0, 1])
    #obj2 = YCBObject('007_tuna_fish_can')
    #obj2.load()
    #p.resetBasePositionAndOrientation(obj2.body_id, [0.2, 0, 0], [0, 0, 0, 1])



    self.panda = panda_type()
    self.arm_id = self.panda.panda
    self.obj_id = obj1.body_id

    self.n = 3
    self.action_space = spaces.Box(low=-1.0, high=+1.0, shape=(self.n, ), dtype=np.float32)
    self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(3,), dtype=np.float32)  #change this for different dimension (shape33)

    # load a panda robot
    #self.panda = Panda()

  def reset(self):
    print("env")
    self.panda.reset()
    print( self.panda.state['ee_position'])
    #p.resetBasePositionAndOrientation(self.obj_id, self.panda.state['ee_position'], [0, 0, 0, 1])
    self.reward = 0
    self.reachgoal = False
    return self.panda.state

  def resetobj(self):
    #print("Hi!!!!!!!!!!!")
    print( self.panda.state['ee_position']) #+np.array([0,0,0.1])
    p.resetBasePositionAndOrientation(self.obj_id, self.panda.state['ee_position']+np.array([0,0.002,0.05]), [0, 0, 0, 1])
    return self.panda.state

  def close(self):
    p.disconnect()

  def step(self, action, verbose=False):
    """ mode = 1, controlling by end effector dposition
        mode = 0, controlling by joint action
    """
    # get current state
    state = self.panda.state
    
    '''if mode == 1:
      action /=1000
      self.panda.step(mode=mode,  dposition=action)
    else:
      action /= 500
      self.panda.step(mode=mode,  djoint=action)'''

    self.panda.step(dposition=action)

    # take simulation step
    p.stepSimulation()

    # return next_state, reward, done, info
    next_state = self.panda.state
    self.reward -=1
    state = next_state['ee_position']
    #collision
    self.reachgoal = np.linalg.norm(np.array([state[0] - 0.81, state[1],state[2] - 0.1])) < 0.028
    done =False
    if(self.reachgoal):
      self.reward+=5000
      done = True
    closest_point = p.getClosestPoints(self.arm_id, self.obj_id, 100)
    close_points = [[point[5], point[6]] for point in closest_point]
    min_distance = 100
    for point_pair in close_points:
      dist = (point_pair[0][0]-point_pair[1][0])**2 + (point_pair[0][1]-point_pair[1][1])**2 + (point_pair[0][2]-point_pair[1][2])**2
      if dist < min_distance:
        min_distance = dist
      if verbose:
        print(next_state['ee_position'], min_distance)
      if min_distance < 0.0001:
        self.reward -= 10000
        done = True
    info = {}
    return next_state, self.reward, done, info

  def render(self, mode='human', close=False):
    (width, height, pxl, depth, segmentation) = p.getCameraImage(width=self.camera_width,
                                                                    height=self.camera_height,
                                                                    viewMatrix=self.view_matrix,
                                                                    projectionMatrix=self.proj_matrix)
    rgb_array = np.array(pxl, dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (self.camera_height, self.camera_width, 4))
    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def _set_camera(self):
    self.camera_width = 256
    self.camera_height = 256
    p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=0, cameraPitch=-30,
                                    cameraTargetPosition=[0.5, -0.2, 0.0])
    self.view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.5, 0, 0],
                                                            distance=1.0,
                                                            yaw=90,
                                                            pitch=-50,
                                                            roll=0,
                                                            upAxisIndex=2)
    self.proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                    aspect=float(self.camera_width) / self.camera_height,
                                                    nearVal=0.1,
                                                    farVal=100.0)

class DisabledPandaEnv(PandaEnv):

    def __init__(self, panda_type=DisabledPanda):
        super(DisabledPandaEnv, self).__init__(panda_type=panda_type)
        self.action_space = spaces.Box(low=np.array([-1., -1.]), high=np.array([1., 1.]), dtype=np.float32)

    def step(self, action,verbose=False):
        action = action.squeeze()
        action1 = np.array([0., 0., 0.])
        action1[0] = action[0]
        #pdb.set_trace()
        action1[2] = action[2]   #change to 1 for rl, 2 for collect demons
        next_state = self.panda.state
        return super(DisabledPandaEnv, self).step(action1, verbose)


class FeasibilityPandaEnv(PandaEnv):

  def __init__(self, panda_type=FeasibilityPanda):
    super(FeasibilityPandaEnv, self).__init__(panda_type=panda_type)
    self.action_space = spaces.Box(low=np.array([-1., -1.]), high=np.array([1., 1.]), dtype=np.float32)
    #self.gt_data =  pickle.load(open('..\\logs\\data\\infeasible_traj_9_1_0523_full.pkl', 'rb'))
    self.gt_data1 =  pickle.load(open('infeasible_traj_9_1_0524_full.pkl', 'rb'))
    self.gt_data2 =  pickle.load(open('infeasible_traj_9_1_0528_full.pkl', 'rb'))
    #self.gt_data = np.concatenate((self.gt_data[:9], self.gt_data[12:18]),axis=0)
    self.time_step = 0
    self.eps_len = 8000
  
  def _random_select(self, idx=None, version=None):
        # random pick start pos
        if idx is None:
            #self.gt_num = np.random.choice(self.gt_data.shape[0]-2)
            #self.gt_num = np.random.choice(2)+18
            self.gt_num = np.random.choice(self.gt_data.shape[0])
            self.gt_data = self.gt_data1
            #self.gt_data = self.gt_data2
        else:
            self.gt_num = idx
            if version == "dis":
              self.gt_data = self.gt_data1
            else:
              self.gt_data = self.gt_data2
        print(self.gt_data.shape)
        self.gt = self.gt_data[self.gt_num][:][:]
        pos = self.gt[0][27:30]
        jointposition = np.concatenate((self.gt[0][:9],np.array([0.03,0.03])),axis=None)
        self.panda._reset_robot(jointposition)
        self.eps_len = self.gt.shape[0]
        #print(self.panda.state['ee_position'], pos)
  
    
  def reset(self,idx=None, version = None):
    self.panda.reset()
    #print( self.panda.state['ee_position'])
    self._random_select(idx, version)
    self.time_step = 0
    '''state =  np.concatenate((
            self.panda.state['joint_position'],# 5
            self.panda.state['ee_position'],# 3
            ), axis=None)'''
    #p.resetBasePositionAndOrientation(self.obj_id, self.panda.state['ee_position'], [0, 0, 0, 1])
    return self.panda.state['ee_position']
    return state

  def step(self, action,verbose=False):
    #print(self.eps_len)
    action = action.squeeze()
    action1 = np.array([0., 0., 0.])
    action1[0] = action[0]
    #pdb.set_trace()
    action1[2] = action[1]
    super(FeasibilityPandaEnv, self).step(action1, verbose)
    state = self.panda.state['ee_position']
    self.time_step += 1
    done = (self.time_step >= self.eps_len - 1)
    dis = np.linalg.norm(state - self.gt[self.time_step][27:30])
    reward = -dis
    #print(reward)
    info = {}
    self.prev_state = copy.deepcopy(state)
    full_state = self.panda.state['ee_position']
    '''full_state =  np.concatenate((
            self.panda.state['joint_position'],# 9
            self.panda.state['ee_position'],# 3
            ), axis=None)'''
    info['dis'] = dis
        
    return full_state, reward, done, info

      
      