import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_panda.panda_bullet.panda import Panda, DisabledPanda
from gym_panda.panda_bullet.objects import YCBObject, InteractiveObj, RBOObject
import os
import numpy as np
import pybullet as p
import pybullet_data
import pdb


class PandaEnv(gym.Env):
  metadata = {'render.modes': ['human']}
  def __init__(self, panda_type=Panda):
    # create simulation (GUI)
    self.urdfRootPath = pybullet_data.getDataPath()
    p.connect(p.DIRECT)
    #p.connect(p.GUI)
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
    #self.arm_id = self.panda.panda
    #self.obj_id = obj2.body_id

    self.n = 9 # 3
    self.action_space = spaces.Box(low=-1.0, high=+1.0, shape=(self.n, ), dtype=np.float32)
    self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(33,), dtype=np.float32)

    # load a panda robot
    #self.panda = Panda()

  def reset(self):
    self.panda.reset()
    print( self.panda.state['ee_position'])
    #p.resetBasePositionAndOrientation(self.obj_id, self.panda.state['ee_position'], [0, 0, 0, 1])
    return self.panda.state

  def resetobj(self):
    #print("Hi!!!!!!!!!!!")
    print( self.panda.state['ee_position']) #+np.array([0,0,0.1])
    p.resetBasePositionAndOrientation(self.obj_id, self.panda.state['ee_position']+np.array([0,0.002,0.05]), [0, 0, 0, 1])
    return self.panda.state

  def close(self):
    p.disconnect()

  def step(self, action, verbose=False):
    #print("env")
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
    reward = 0.0
    done = False
    info = {}
    return next_state, reward, done, info

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
        action1[2] = action[2]
        return super(DisabledPandaEnv, self).step(action1, verbose)