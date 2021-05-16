import os

import gym
import gym
from gym import wrappers, logger
import gym_panda
from gym_panda.wrapper_env.wrapper import *
# import gym_circle_move
import numpy as np
import matplotlib.pyplot as plt


from stable_baselines import DDPG,PPO2,TRPO
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
#from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.ddpg.policies import MlpPolicy as ddpg_MlpPolicy

from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from datetime import datetime

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.latest_path = os.path.join(log_dir, 'latest_model')
        self.best_mean_reward = -np.inf
        self.reward = []

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        if self.latest_path is not None:
            os.makedirs(self.latest_path, exist_ok=True)

    def _on_step(self) -> bool:
        # print("h------------------------------------------------------g")
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)
        
        if self.n_calls % 1e4 == 0:
          self.model.save(self.latest_path)

        return True




if __name__ == "__main__":
    # make env
    env_name = "panda-v0"
    env = gym.make(env_name)
    env = SkipStepsWrapperVAE(env)
    #env = infeasibleWrapper(env)

    # Create log dir
    #log_dir = "/iliad/u/yilunhao/logs/models/sb-trpo-joint-target-diffdynamics-{}/".format(datetime.now().strftime("%Y-%m-%d"))
    log_dir = "../logs/models/sb-trpo-joint-target-diffdynamics-{}/".format(datetime.now().strftime("%Y-%m-%d"))
    tensorboard_dir = "../logs/logs"
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = Monitor(env, log_dir)
    env.reset()
    # print(env.n)
    n_actions = env.action_space.shape[-1]
    param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.01, desired_action_stddev=0.01)
    action_noise =  OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
    model = TRPO(MlpPolicy, env, verbose=1,  tensorboard_log=tensorboard_dir)
    # model = DDPG(ddpg_MlpPolicy, env, param_noise=param_noise, action_noise=action_noise, verbose=1,tensorboard_log=tensorboard_dir)#, param_noise=param_noise, action_noise=action_noise)        
    callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir)
    
    # Train the model
    time_steps = 1e8
    model.learn(total_timesteps=int(time_steps), callback=callback)
    model.save(os.path.join(log_dir, "final_model.pt"))

    # model = PPO2.load("/home/jingjia16/marl/models/best_model.zip")
    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()


    # results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "DDPG")
    # plt.savefig(os.path.join(log_dir, 'plot.png'))
