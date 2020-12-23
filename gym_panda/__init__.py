import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)


register(
    id='panda-v0',
    entry_point='gym_panda.envs:PandaEnv',
)