import math
from typing import Optional

import numpy as np
import pygame
from pygame import gfxdraw

import gym
from gym import spaces
from gym.utils import seeding
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec
from numba import njit, prange
from EnvGenerator import RoomGenerator

from Test_Environment import TargetSearchEnv

class DiscreteTargetSearchEnv(py_environment.PyEnvironment):
    def __init__(self):
        self._env = TargetSearchEnv()
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32,minimum=0,maximum=2,name='action')
        
    def action_spec(self):
        return self._action_spec
    def _reset(self):
        return self._env._reset()
    
    def observation_spec(self):
        return self._env.observation_spec()
    
    def _step(self,action):
        action_map = {
            0 : [0,-1],
            1 : [1,0],
            2 : [0,1]
        }
        return self._env._step(action_map[int(action)],prev_action=int(action))
    
    def render(self, mode='rgb_array'):
        return self._env.render(mode)
    def close(self):
        self._env.close()
