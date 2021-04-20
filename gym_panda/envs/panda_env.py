import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_panda.panda_bullet.panda import Panda
from gym_panda.panda_bullet.objects import YCBObject, InteractiveObj, RBOObject
import os
import numpy as np
import pybullet as p
import pybullet_data


class PandaEnv(gym.Env):
  metadata = {'render.modes': ['human']}
  def __init__(self):
    # create simulation (GUI)
    self.urdfRootPath = pybullet_data.getDataPath()
    # p.connect(p.DIRECT)
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)

    # set up camera
    self._set_camera()

    # load some scene objects
    p.loadURDF(os.path.join(self.urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.65])
    p.loadURDF(os.path.join(self.urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.65])

    # example YCB object
    # obj1 = YCBObject('003_cracker_box')
    # obj1.load()
    # p.resetBasePositionAndOrientation(obj1.body_id, [0.7, -0.2, 0.1], [0, 0, 0, 1])

    self.n = 9 # 3
    self.action_space = spaces.Box(low=-1.0, high=+1.0, shape=(self.n, ), dtype=np.float32)
    self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(33,), dtype=np.float32)

    # load a panda robot
    self.panda = Panda()

  def reset(self):
    self.panda.reset()
    return self.panda.state

  def close(self):
    p.disconnect()

  def step(self, action, mode=0):
    """ mode = 1, controlling by end effector dposition
        mode = 0, controlling by joint action
    """
    # get current state
    state = self.panda.state
    
    if mode == 1:
      action /=1000
      self.panda.step(mode=mode,  dposition=action)
    else:
      action /= 500
      self.panda.step(mode=mode,  djoint=action)

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
    p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=30, cameraPitch=-60,
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

