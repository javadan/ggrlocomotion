import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='RobotableEnv-v0',
    entry_point='gym_robotable.envs:RobotableEnv',
)

