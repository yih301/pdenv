# gym_panda

## Usage: 
Git clone and pip install the repository
```python
import gym_panda
from gym_panda.wrapper_env.wrapper import *

env_name = "panda-v0"
env = gym.make(env_name)
```

### Wrappers

Three wrappers in `gym_panda/gym_panda/wrapper_env/wrapper.py`. 

`collectDemonsWrapper` used to collect the demonstrations using `gym_panda/gym_panda/panda_bullet/collect_demons.py`.

`infeasibleWrapper`: used to train a reinforcement learning agent to track multiple trajectories in demonstrations. 

`SkipStepsWrapperVAE`: used to train a reinforcement learning agent using VAE as reference.


### Tips:

#### How to change dynamics
`gym_panda/gym_panda/panda_bullet/panda.py` Line 121, you can change the disabled joint index by changing elements of ` disabled_joint_idx`

#### Disable or Enable GUI
`gym_panda/gym_panda/envs/panda_env.py` Line 17 and 18
```python
p.connect(p.DIRECT) # disable GUI
p.connect(p.GUI) # Enable GUI
```




