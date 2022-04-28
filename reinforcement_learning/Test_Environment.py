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
from numba import njit, prange
from EnvGenerator import RoomGenerator

@njit()
def sides(o):
    # get the 4 sides of a rectangle
    s = [
        [o[0],o[1],o[0],o[1]+o[3]], # left
        [o[0]+o[2],o[1],o[2],o[1]+o[3]], # right
        [o[0],o[1]+o[3],o[0]+o[2],o[1]+o[3]], # top
        [o[0],o[1],o[0]+o[2],o[1]]  # bottom
    ]
    return s

@njit(fastmath=True)
def _update_robot_view(bot,obstacles,target,fov,nchannels,bounds,n_obstacles):
    bx, by, angle = bot
    low_angle = angle-(0.5*fov)
    high_angle = angle + (0.5*fov)
    angle_step = fov/nchannels
    # build list of views from robot
    view_angles = [ low_angle+(i*angle_step) for i in range(nchannels+1) ]
    view_angles.reverse()
    # initialize view array

    # assume bot 0,0
    # transpose obstacle, target and boundary locations to match bot being at (0,0)
    # if len(obstacles) > 0:

    # new_obs = [ [lx-bx,ly-by,hx,hy] for [lx,ly,hx,hy] in obstacles ]
    new_target = [ [lx-bx,ly-by,hx,hy] for [lx,ly,hx,hy] in target ]
    lx,ly,hx,hy = bounds
    new_bounds = [lx-bx,ly-by,hx,hy]

    view = [ 0 for x in range(nchannels+1) ]
    view_dists = [np.inf for x in range(nchannels+1) ]

    for i in prange(0,len(view_angles)):
        angle = view_angles[i]
        direction = (np.cos(angle),np.sin(angle))
        for obj in obstacles: # Each object has 4 sides.
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
        for obj in new_target: # Each object has 4 sides.
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
    #full_arr = []
    #for i in range(len(view)):
    #    if view[i] == 2:
    #        full_arr.append(1)
    #    else:
    #        full_arr.append(0)
    #    full_arr.append(view_dists[i])
    full_arr = np.zeros( len(view)+len(view_dists) )
    for i in range(0,len(view)):
        if view[i] == 2:
            full_arr[i] = 1
        else:
            full_arr[i] = 0
    for i in range(len(view),len(full_arr)):
        full_arr[i] = view_dists[i-len(view)]
    # # full_arr = view
    # for d in view_dists:
        # full_arr.append(d)
    return full_arr


class TargetSearchEnv(py_environment.PyEnvironment):
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self):
        # Define left, bottom, right, and top of env
        self.bounds = np.array([0.0,0.0,5.0,5.0])
        self.geow = self.bounds[2]-self.bounds[0]
        self.geoh = self.bounds[3]-self.bounds[1]
        self.generator = RoomGenerator(5,5,5)
        self._n_view_channels = 43 # number of slices FOV cut into
        # self._fov = 2.0*np.pi # Radian Field of view for the robot
        self._fov = 1.5 # Radian Field of view for the actual robot
        
        # Observation: [ bot_x, bot_y, target_x, target_y ]
        self.low = np.array([self.bounds[0],self.bounds[1],self.bounds[0],self.bounds[1]], dtype=np.float32)
        self.high = np.array([self.bounds[2],self.bounds[3],self.bounds[2],self.bounds[3]], dtype=np.float32)
     
        # action has two floats: forward/backward and left/right
        self._action_spec = BoundedArraySpec(
        shape=(2,), dtype=np.int32, minimum=[0,-1], maximum=[1,1], name='action')
        
        self._observation_max = []
        self._observation_min = []
        for i in range(self._n_view_channels+1):
            self._observation_max.append(1) # max value for object label in view
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
        # self._reward_spec = BoundedArraySpec(shape=(1,), minimum=-2000.0, maximum=2000, dtype=np.float32, name='reward')
        
        self.screen = None
        self.clock = None
        self.isopen = True
        self._episode_ended = False
        
    def action_spec(self):
        return self._action_spec
    def observation_spec(self):
        return self._observation_spec
    # def reward_spec(self):
    #     return self._reward_spec
    
    def _step(self, action, prev_action=None):
        self._nSteps = self._nSteps + 1
        if self._episode_ended:
            return self.reset()
        # Multiplier for movement and turning
        move_size = 0.1
        turn_size = 0.05
        # Update state based on action
        bot = self._state[0]
        # turn, then move
        bot[2] = bot[2] + (turn_size*action[1])
        bot[0] = bot[0] + (action[0]*move_size*np.cos(bot[2]))
        bot[1] = bot[1] + (action[0]*move_size*np.sin(bot[2]))
        # Make it so you can't move into an obstacle
        for obs in self._state[1]: # iterate through obstacles
            if bot[0] > obs[0] and bot[1] > obs[1] and bot[0] < obs[0]+obs[2] and bot[1] < obs[1]+obs[3]:
                # Inside Obstacle: Need to handle this case to clip the bot out.
                obs_x = obs[0]+(0.5*obs[2])
                obs_y = obs[1]+(0.5*obs[3])
                
                theta_center = np.arctan2((bot[0]-obs_y),(bot[1]-obs_x) )
                # [-pi, pi]
                # Cases to find robot location in obstacle:
                if np.abs(theta_center) > ((3*np.pi)/4): # left
                    bot[0] = obs[0]
                elif theta_center > (np.pi/4): # top
                    bot[1] = obs[1]+obs[3]
                elif theta_center < -(np.pi/4): # bottom
                    bot[1] = obs[1]
                else: # right
                    bot[0] = obs[0]+obs[2]
        
        self._state[0] = bot
        bot = np.array(bot)
        # Get center of target rectangle
        target_center = np.array([0.0,0.0])
        target_center[0] = self._state[2][0][0]+(0.5*self._state[2][0][2])
        target_center[1] = self._state[2][0][1]+(0.5*self._state[2][0][3])
        # REWARD CALCULATION
        reward = 0
        curr_dist = np.linalg.norm(bot[[0,1]]-target_center)
        
        # lightly penalize not being on the target. Think of it as a move cost.
        reward = -0.5
        if self.prev_dist > curr_dist:
            reward = -0.2
        # if prev_action != None:
        #     if prev_action == 3:
        #         reward = -0.1 # softly penalize not moving.
        # if action[0] < 0.0:
        #     reward = -0.7
        # penalize moving away from target
        # if curr_dist > self.prev_dist:
        #     reward = -1
        # Heavily penalize being near an obstacle
        obs_r = 0.1 # distance from obstacle where penalties start
        for obs in self._state[1]: # iterate through obstacles
            if bot[0] > obs[0]-obs_r and bot[1] > obs[1]-obs_r:
                if bot[0] < obs[0]+obs[2]+obs_r and bot[1] < obs[1]+obs[3]+obs_r:
                    reward = -1.0
        # Penalize being near the boundary
        # if bot[0] > self.bounds[0]+obs_r and bot[1] > self.bounds[1]+obs_r:
        #     if bot[0] < obs[0]+obs[2]-obs_r and bot[1] < obs[1]-obs[3]+obs_r:
        #         reward = -1.0
        # Penalize Looping
        # if prev_action != None:
        #     if prev_action == 0 or prev_action == 2:
        #         if prev_action == 0 and self._previous_act == 2:
        #             reward = -2.0
        #         if prev_action == 2 and self._previous_act == 0:
        #             reward = -2.0
        #         self._previous_act = prev_action
        #         time_since_move = 0
        #         for i in range(len(self._previous_acts)-1,-1,-1):
        #             if self._previous_acts[i] in [1,2,3]:
        #                 break
        #             time_since_move = time_since_move + 1
        #         reward = -1*time_since_move

        #     # if prev_action != 1: # Prevent spinning
        #     #     if prev_action == self._previous_acts[-2] and prev_action == self._previous_acts[-1]:
        #     #         reward = -2.0
        #     self._previous_acts.append(prev_action)
        # Reward being inside target area
        # if curr_dist <= 0.5:
        #     reward=2  
        if self.prev_dist < curr_dist:
            reward = -1.5
        if bot[0] < (self._state[2][0][0]+self._state[2][0][2]) and bot[0] > self._state[2][0][0] and bot[1] < (self._state[2][0][1]+self._state[2][0][3]) and bot[1] > self._state[2][0][1]:
            reward = 2
        # discount.
        discount = 0.9
            
        # End states.
        # Bot outside env domain
        # if bot[0] < self.bounds[0] or bot[1] < self.bounds[1] or bot[0] > self.bounds[2] or bot[1] > self.bounds[3]:
        #     self._episode_ended = True
        if bot[0] < self.bounds[0]:
            self._state[0][0] = self.bounds[0]
        if bot[1] < self.bounds[1]:
            self._state[0][1] = self.bounds[1]
        if bot[0] > self.bounds[2]:
            self._state[0][0] = self.bounds[2]
        if bot[1] > self.bounds[3]:
            self._state[0][0] = self.bounds[3]
        # steps occured
        self._eprew = self._eprew + reward
        if self._nSteps > 1000:# or self._eprew > 200:
            self._episode_ended = True

        # update previous distance.
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
        # Randomly initialize the bot and target locations.
        self._state = self.generator.gen_good_env(self.geow,self.geoh)
        # self.generator.print()
        # print(self._state)
        self.n_obstacles = len(self._state[1])
        bot = np.array(self._state[0])
        
        # If bot spawned in an obstacle, kick it out.
        for obs in self._state[1]: # iterate through obstacles
            if bot[0] > obs[0] and bot[1] > obs[1] and bot[0] < obs[0]+obs[2] and bot[1] < obs[1]+obs[3]:
                # Inside Obstacle: Need to handle this case to clip the bot out.
                obs_x = obs[0]+(0.5*obs[2])
                obs_y = obs[1]+(0.5*obs[3])
                
                theta_center = np.arctan2((bot[0]-obs_y),(bot[1]-obs_x) )
                # Cases to find robot location in obstacle and remove it:
                if np.abs(theta_center) > ((3*np.pi)/4): # left
                    bot[0] = obs[0]
                elif theta_center > (np.pi/4): # top
                    bot[1] = obs[1]+obs[3]
                elif theta_center < -(np.pi/4): # bottom
                    bot[1] = obs[1]
                else: # right
                    bot[0] = obs[0]+obs[2]
        
        self._state[0] = bot
        
        
        target_center = np.array([0.0,0.0])
        target_center[0] = self._state[2][0][0]+(0.5*self._state[2][0][2])
        target_center[1] = self._state[2][0][1]+(0.5*self._state[2][0][3])
        curr_dist = np.linalg.norm(bot[[0,1]]-target_center)
        # Initialize location and distance holders
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
        
        # get bot and target positions, scale them as needed
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
        # Render. Bot is blue, Target Green. Set of concentric circles
        radius = 5
        for i in range(1,radius):
            pygame.gfxdraw.circle(self.surf, int(bot[0]), int(bot[1]), i, (100,100,255))
        # Draw outermost circle in black.
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
