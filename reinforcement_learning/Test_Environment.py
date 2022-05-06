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
from tf_agents.specs import array_spec, BoundedArraySpec
from EnvGenerator import RoomGenerator

def sides(o):
    s = [
        [o[0],o[1],o[0],o[1]+o[3]], 
        [o[0]+o[2],o[1],o[2],o[1]+o[3]], 
        [o[0],o[1]+o[3],o[0]+o[2],o[1]+o[3]], 
        [o[0],o[1],o[0]+o[2],o[1]]  
    ]
    return s

def _update_robot_view(bot,obstacles,target,fov,nchannels,bounds,n_obstacles):
    bx, by, angle = bot
    low_angle = angle-(0.5*fov)
    high_angle = angle + (0.5*fov)
    angle_step = fov/nchannels
    
    view_angles = [ low_angle+(i*angle_step) for i in range(nchannels+1) ]
    view_angles.reverse()
    new_target = [ [lx-bx,ly-by,hx,hy] for [lx,ly,hx,hy] in target ]
    lx,ly,hx,hy = bounds
    new_bounds = [lx-bx,ly-by,hx,hy]

    view = [ 0 for x in range(nchannels+1) ]
    view_dists = [np.inf for x in range(nchannels+1) ]

    for i in range(0,len(view_angles)):
        angle = view_angles[i]
        direction = (np.cos(angle),np.sin(angle))
        for obj in obstacles: 
            new_obj = obj - np.array([bx,by,0,0])
            for s in sides(new_obj):
                # Ray Tracing from
                # https://github.com/000Nobody/2D-Ray-Tracing
                x1,y1,x2,y2 = s
                x3,y3,x4,y4 = [0.0,0.0,direction[0],direction[1]]
                denominator = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
                numerator = (x1-x3)*(y3-y4)-(y1-y3)*(x3-x4)
                if denominator != 0:
                    t = numerator / denominator
                    u = -((x1-x2)*(y1-y3)-(y1-y2)*(x1-x3))/denominator
                    if 1 > t > 0 and u > 0:
                        x = x1+t * (x2-x1)
                        y = y1+t * (y2-y1)
                        collidePos = [x, y]
                        vdist = np.linalg.norm(np.array(collidePos))
                        if vdist < view_dists[i]:
                            view[i] = 1
                            view_dists[i] = vdist
        for obj in new_target: 
            for s in sides(obj):
                x1,y1,x2,y2 = s
                x3,y3,x4,y4 = [0.0,0.0,direction[0],direction[1]]
                denominator = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
                numerator = (x1-x3)*(y3-y4)-(y1-y3)*(x3-x4)
                if denominator != 0:
                    t = numerator / denominator
                    u = -((x1-x2)*(y1-y3)-(y1-y2)*(x1-x3))/denominator
                    if 1 > t > 0 and u > 0:
                        x = x1+t * (x2-x1)
                        y = y1+t * (y2-y1)
                        collidePos = [x, y]
                        vdist = np.linalg.norm(np.array(collidePos))
                        if vdist < view_dists[i]:
                            view[i] = 2
                            view_dists[i] = vdist
        for s in sides(new_bounds):
            x1,y1,x2,y2 = s
            x3,y3,x4,y4 = [0.0,0.0,direction[0],direction[1]]
            denominator = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
            numerator = (x1-x3)*(y3-y4)-(y1-y3)*(x3-x4)
            if denominator != 0:
                t = numerator / denominator
                u = -((x1-x2)*(y1-y3)-(y1-y2)*(x1-x3))/denominator
                if 1 > t > 0 and u > 0:
                    x = x1+t * (x2-x1)
                    y = y1+t * (y2-y1)
                    collidePos = [x, y]
                    vdist = np.linalg.norm(np.array(collidePos))
                    if vdist < view_dists[i]:
                        view[i] = 0
                        view_dists[i] = vdist

    view_dists = np.array(view_dists)
    view_dists[np.isinf(view_dists)] = 0
    full_arr = np.zeros( len(view)+len(view_dists) )
    for i in range(0,len(view)):
        if view[i] == 2:
            full_arr[i] = 1
        else:
            full_arr[i] = 0
    for i in range(len(view),len(full_arr)):
        full_arr[i] = view_dists[i-len(view)]
    return full_arr


class TargetSearchEnv(py_environment.PyEnvironment):
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self):
        
        self.bounds = np.array([0.0,0.0,5.0,5.0])
        self.geow = self.bounds[2]-self.bounds[0]
        self.geoh = self.bounds[3]-self.bounds[1]
        self.generator = RoomGenerator(5,5,5)
        self._n_view_channels = 43 
        
        self._fov = 1.5 
        
        self.low = np.array([self.bounds[0],self.bounds[1],self.bounds[0],self.bounds[1]], dtype=np.float32)
        self.high = np.array([self.bounds[2],self.bounds[3],self.bounds[2],self.bounds[3]], dtype=np.float32)
        
        self._action_spec = BoundedArraySpec(
        shape=(2,), dtype=np.int32, minimum=[0,-1], maximum=[1,1], name='action')
        
        self._observation_max = []
        self._observation_min = []
        for i in range(self._n_view_channels+1):
            self._observation_max.append(1) 
            self._observation_min.append(0)

        diag_dist = np.linalg.norm(self.bounds[[0,1]]-self.bounds[[2,3]])
        self._observation_max.append(diag_dist)
        self._observation_min.append(0)
        for i in range(self._n_view_channels+1):
            self._observation_max.append(diag_dist)
            self._observation_min.append(0)
        for x in self.low:
            self._observation_min.append(x)
        for x in self.high:
            self._observation_max.append(x)
        self._observation_max = np.array(self._observation_max)
        self._observation_min = np.array(self._observation_min)
        self._observation_spec = BoundedArraySpec(
        shape=(((self._n_view_channels+1)*2)+5, ), dtype=np.float32, minimum=self._observation_min, maximum=self._observation_max, name='observation')
        
        self.screen = None
        self.clock = None
        self.isopen = True
        self._episode_ended = False
        
    def action_spec(self):
        return self._action_spec
    def observation_spec(self):
        return self._observation_spec
    
    def _step(self, action, prev_action=None):
        self._nSteps = self._nSteps + 1
        if self._episode_ended:
            return self.reset()
        
        move_size = 0.1
        turn_size = 0.05
        bot = self._state[0]
        bot[2] = bot[2] + (turn_size*action[1])
        bot[0] = bot[0] + (action[0]*move_size*np.cos(bot[2]))
        bot[1] = bot[1] + (action[0]*move_size*np.sin(bot[2]))
        
        for obs in self._state[1]: 
            if bot[0] > obs[0] and bot[1] > obs[1] and bot[0] < obs[0]+obs[2] and bot[1] < obs[1]+obs[3]:
                obs_x = obs[0]+(0.5*obs[2])
                obs_y = obs[1]+(0.5*obs[3])
                theta_center = np.arctan2((bot[0]-obs_y),(bot[1]-obs_x) )
                if np.abs(theta_center) > ((3*np.pi)/4): 
                    bot[0] = obs[0]
                elif theta_center > (np.pi/4): 
                    bot[1] = obs[1]+obs[3]
                elif theta_center < -(np.pi/4): 
                    bot[1] = obs[1]
                else: 
                    bot[0] = obs[0]+obs[2]
        
        self._state[0] = bot
        bot = np.array(bot)
        target_center = np.array([0.0,0.0])
        target_center[0] = self._state[2][0][0]+(0.5*self._state[2][0][2])
        target_center[1] = self._state[2][0][1]+(0.5*self._state[2][0][3])
        reward = 0
        curr_dist = np.linalg.norm(bot[[0,1]]-target_center)

        reward = -0.7
        obs_r = 0.1
        for obs in self._state[1]:
            if bot[0] > obs[0]-obs_r and bot[1] > obs[1]-obs_r:
                if bot[0] < obs[0]+obs[2]+obs_r and bot[1] < obs[1]+obs[3]+obs_r:
                    reward = -1.0
        if bot[0] < self.bounds[0]+obs_r or bot[1] < self.bounds[1]+obs_r or bot[0] > self.bounds[2]-obs_r or bot[1] > self.bounds[3]-obs_r:
                reward = -1.0
        if self.prev_dist < curr_dist:
            reward = -1.5
        if self.prev_dist > curr_dist:
            reward = -0.2
        if bot[0] < (self._state[2][0][0]+self._state[2][0][2]) and bot[0] > self._state[2][0][0] and bot[1] < (self._state[2][0][1]+self._state[2][0][3]) and bot[1] > self._state[2][0][1]:
            self._episode_ended = True
            reward = 0.0
        discount = 0.99

        self.prev_dist = curr_dist
        obs=_update_robot_view(self._state[0],np.array(self._state[1]),np.array(self._state[2]),self._fov,self._n_view_channels,self.bounds,len(self._state[1]))
        obs = np.concatenate([obs,target_center,bot[[0,1,2]]])
        if self._episode_ended:
            return ts.termination(np.array(obs, dtype=np.float32), reward=reward)
        else:            
            return ts.transition(np.array(obs, dtype=np.float32), reward=reward, discount=discount)
    
    def getView(self):
        return _update_robot_view(self._state[0],np.array(self._state[1]),np.array(self._state[2]),self._fov,self._n_view_channels,self.bounds,len(self._state[1]))
                                    
    def _reset( self ):
        self._eprew = 0
        self._previous_act = None
        self._previous_acts = [-1,-1]
        self._episode_ended = False
        self._nSteps = 0
        self._state = self.generator.gen_good_env(self.geow,self.geoh)
        self.n_obstacles = len(self._state[1])
        bot = np.array(self._state[0])
        
        for obs in self._state[1]: 
            if bot[0] > obs[0] and bot[1] > obs[1] and bot[0] < obs[0]+obs[2] and bot[1] < obs[1]+obs[3]:
                obs_x = obs[0]+(0.5*obs[2])
                obs_y = obs[1]+(0.5*obs[3])
                theta_center = np.arctan2((bot[0]-obs_y),(bot[1]-obs_x) )
                if np.abs(theta_center) > ((3*np.pi)/4): 
                    bot[0] = obs[0]
                elif theta_center > (np.pi/4): 
                    bot[1] = obs[1]+obs[3]
                elif theta_center < -(np.pi/4): 
                    bot[1] = obs[1]
                else: 
                    bot[0] = obs[0]+obs[2]
        
        self._state[0] = bot
        target_center = np.array([0.0,0.0])
        target_center[0] = self._state[2][0][0]+(0.5*self._state[2][0][2])
        target_center[1] = self._state[2][0][1]+(0.5*self._state[2][0][3])
        curr_dist = np.linalg.norm(bot[[0,1]]-target_center)
        self.prev_dist = curr_dist
        obs=_update_robot_view(self._state[0],np.array(self._state[1]),np.array(self._state[2]),self._fov,self._n_view_channels,self.bounds,len(self._state[1]))
        obs = np.concatenate([obs,target_center,bot[[0,1,2]]])
        return ts.restart(np.array(obs, dtype=np.float32))

    def render(self, mode="human"):
        screen_width = 640
        screen_height = 480

        world_width = self.bounds[2] - self.bounds[0]
        world_height = self.bounds[3] - self.bounds[1]
        scale_w = screen_width / world_width
        scale_h = screen_height / world_height
        
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))
        
        bot = np.array(self._state[0])
        bot[0] = int(bot[0]*scale_w)
        bot[1] = int(bot[1]*scale_h)
        target = np.array(self._state[2])
        for i in range(len(target)):
            target[i][0] = int(target[i][0]*scale_w)
            target[i][1] = int(target[i][1]*scale_h)
            target[i][2] = int(target[i][2]*scale_w)
            target[i][3] = int(target[i][3]*scale_h)
        obstacles = np.array(self._state[1])
        for i in range(len(obstacles)):
            obstacles[i][0] = int(obstacles[i][0]*scale_w)
            obstacles[i][1] = int(obstacles[i][1]*scale_h)
            obstacles[i][2] = int(obstacles[i][2]*scale_w)
            obstacles[i][3] = int(obstacles[i][3]*scale_h)
        for t in target:
            pygame.gfxdraw.box(self.surf, [int(x) for x in t], (100, 255, 100))
        for o in obstacles:
            pygame.gfxdraw.box(self.surf, [int(x) for x in o], (50,50,50))
        
        radius = 5
        for i in range(1,radius):
            pygame.gfxdraw.circle(self.surf, int(bot[0]), int(bot[1]), i, (100,100,255))
        
        pygame.gfxdraw.circle(self.surf, int(bot[0]), int(bot[1]), radius, (0,0,0))
        x=int(bot[0])
        y=int(bot[1])
        x2=int(x+10*np.cos(bot[2]))
        y2 = int(y+10*np.sin(bot[2]))
        pygame.gfxdraw.line(self.surf,x,y,x2,y2,(0,0,0))
        pygame.gfxdraw.line(self.surf,x+1,y+1,x2+1,y2+1,(0,0,0))
        pygame.gfxdraw.line(self.surf,x-1,y-1,x2-1,y2-1,(0,0,0))
        pygame.gfxdraw.line(self.surf,x-1,y+1,x2-1,y2+1,(0,0,0))
        pygame.gfxdraw.line(self.surf,x+1,y-1,x2+1,y2-1,(0,0,0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.isopen = False
