#!/usr/bin/python3

# Generates a room with obstacles to be used for
# reinforcement learning environment.

import numpy as np
import random
import time
import sys

class RoomGenerator:
    def __init__(self, w, h, n_obstacles):
        self.w = w
        self.h = h
        self.n_obstacles = n_obstacles
        self._generated = False
        self.genEnv()
        self.sleep_len = 0.10
        self.num_to_char = {
            0: ' ',
            1: '#',
            2: '!',
            3: '@',
            4: ','
        }
        self.char_to_num = dict()
        self.verbosity = 1
        for num in self.num_to_char:
            self.char_to_num[self.num_to_char[num]] = num

    def gen_good_env(self, geow, geoh):
        self.genEnv()
        while not self.check():
            self.genEnv()
        bot, target, obs = self.env(geow, geoh)
        return [ bot, obs, [target] ]
            
    def set_cells_from_file(self,fname):
        try:
            ifh = open(fname,'r')
            chars = [ x.split('\n')[0] for x in ifh.readlines() ]
            new_cells = [ [ self.char_to_num[c] for c in row ] for row in chars ]
            return self.check_set_cells(new_cells)
        except:
            return False
        

    def check_set_cells(self, cells=None):
        if cells is None:
            cells = self.cells
        h = len(cells)
        if h < 1:
            return False
        w = len(cells[0])
        for row in cells:
            if w != len(row):
                return False
        bot_found = 0
        target_found = 0
        obs_found = 0
        for i in range(h):
            for j in range(w):
                c = cells[i][j]
                if c == 3:
                    bot_found = bot_found + 1
                    bot_loc = [j, i]
                elif c == 2:
                    target_found = target_found + 1
                    target_loc = [j, i]
                elif c == 1:
                    obs_found = obs_found + 1
        if bot_found == 0 or target_found == 0:
            return False
        g = RoomGenerator(w,h,obs_found)
        g.cells = cells
        g.bot_loc = bot_loc
        g.target_loc = target_loc
        good = g.check(verbose=self.verbosity)
        if good:
            self.w = w
            self.h = h
            self.cells = cells
            self.n_obstacles = obs_found
            self.bot_loc = bot_loc
            self.target_loc = target_loc
        return good

    def print(self,arr=None):
        if arr == None:
            arr = self.cells
        print('-' * (len(arr[0]) + 2))
        for row in arr:
            print('|', end='')
            for cell in row:
                print(self.num_to_char[cell], end='')
            print('|', end='')
            print()
        print('-' * (len(arr[0]) + 2))

    def genEnv(self):
        w = self.w
        h = self.h
        n_obstacles = self.n_obstacles
        cells = [[0 for x in range(w)] for y in range(h)]

        obs_coords = []
        for _ in range(n_obstacles):
            coord = [random.randrange(0, w), random.randrange(0, h)]
            while coord in obs_coords:
                coord = [random.randrange(0, w), random.randrange(0, h)]
            obs_coords.append(coord)
        for x, y in obs_coords:
            cells[y][x] = 1

        bot_loc = [random.randrange(0, w), random.randrange(0, h)]
        while cells[bot_loc[1]][bot_loc[0]] != 0:
            bot_loc = [random.randrange(0, w), random.randrange(0, h)]
        cells[bot_loc[1]][bot_loc[0]] = 3

        target_loc = [random.randrange(0, w), random.randrange(0, h)]
        while cells[target_loc[1]][target_loc[0]] != 0:
            target_loc = [random.randrange(0, w), random.randrange(0, h)]
        cells[target_loc[1]][target_loc[0]] = 2
        self.cells = cells
        self.bot_loc = bot_loc
        self.target_loc = target_loc
        self._generated = True

    def check(self, verbose=0):
        if not self._generated:
            return False
        cells = self.cells
        bot_loc = self.bot_loc
        target_loc = self.target_loc
        visited = [[np.array(bot_loc)]]
        move_dict = {
            0: np.array([-1, 0]),
            1: np.array([0, 1]),
            2: np.array([1, 0]),
            3: np.array([0, -1])
        }
        done = False
        search_arr = [[x for x in row] for row in cells]
        if verbose != 0:
            print("SEARCHING:")
            self.print(search_arr)
            print()
        bot_loc_revisited = 0
        while not done:
            if done:
                break
            visited.append([])
            for c in visited[-2]:
                if done:
                    break
                for m in move_dict:
                    n = c + move_dict[m]
                    if n[0] < self.w and n[1] < self.h and n[0] >= 0 and n[1] >= 0:
                        pass
                    else:
                        continue
                    f = search_arr[n[1]][n[0]]
                    if f == 2:
                        return True
                        done = True
                        break
                    elif f == 0:
                        visited[-1].append(n)
                        search_arr[n[1]][n[0]] = 4
                        if verbose == 2:
                            self.print(search_arr)
                            time.sleep(self.sleep_len)
            if verbose == 1:
                self.print(search_arr)
                time.sleep(self.sleep_len)
            if len(visited[-1]) == 0:
                return False
                break

    def env(self, geow, geoh):
        cells = self.cells
        cells.reverse()
        h = self.h
        w = self.w
        if h < 1 or w < 1:
            return None
        dw = geow / w
        dh = geoh / h
        bounds = [0.0, 0.0, geow, geoh]
        bot = []
        target = []
        obstacles = []
        for i in range(h):
            for j in range(w):
                c = cells[i][j]
                if c == 1:
                    obstacles.append([
                        j * dw, i * dh, dw, dh
                    ])
                elif c == 2:
                    target = [
                        j * dw, i * dh, dw, dh
                    ]
                elif c == 3:
                    bot = [
                        random.uniform(j * dw, (j + 1) * dw),
                        random.uniform(i * dh, (i + 1) * dh),
                        random.uniform(-np.pi, np.pi)
                    ]
                else:
                    pass
        cells.reverse()
        return bot, target, obstacles
