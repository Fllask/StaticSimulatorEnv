from gymnasium.envs.registration import register
from static_envs.envs.static_sequential_env import SequentialDiscreteGeneric
register(
     id="static-env2d/generic-v0",
     entry_point="static_envs:SequentialDiscreteGeneric",
     max_episode_steps=300,
)