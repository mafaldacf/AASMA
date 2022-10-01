import copy
from dataclasses import asdict
import logging
import random
import math
from tkinter import S
import time

import numpy as np

logger = logging.getLogger(__name__)

from PIL import ImageColor
import gym
from gym import spaces
from gym.utils import seeding

from ma_gym.envs.utils.action_space import MultiAgentActionSpace
from ma_gym.envs.utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text, draw_score_board, draw_cell_outline
from ma_gym.envs.utils.observation_space import MultiAgentObservationSpace

NUM_ACTIONS = 8
DOWN, LEFT, UP, RIGHT, NOOP, STEAL, FREEZE, DELIVER = range(NUM_ACTIONS)

class EnvCaptureTheFlag(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, grid_shape=(19, 19), cell_size=26, n_blue_agents=1, n_red_agents=1, full_observable=False, penalty=-0.5, step_cost=-0.01, max_steps=200,
        n_blue_flags=3, n_red_flags=3, agent_view_mask=(8, 8), freeze_time=5, winning_points=10):

        self._grid_shape = grid_shape
        self._cell_size = cell_size
        self.n_blue_agents = n_blue_agents
        self.n_red_agents = n_red_agents
        self.n_agents = n_blue_agents + n_red_agents
        self._max_steps = max_steps
        self.viewer = None
        self.game_type = None
        self.n_blue_flags = n_blue_flags
        self.n_red_flags = n_red_flags
        self.n_flags = n_blue_flags + n_red_flags
        self.freeze_time = freeze_time
        
        self.num_stolen_flags = {}
        self.num_recovered_flags = {}
        self.num_delivered_flags = {}
        self.num_frozen_agents = {}

        self.num_stolen_flags['B'] = 0
        self.num_stolen_flags['R'] = 0

        self.num_recovered_flags['B'] = 0
        self.num_recovered_flags['R'] = 0

        self.num_delivered_flags['B'] = 0
        self.num_delivered_flags['R'] = 0

        self.num_frozen_agents['B'] = 0
        self.num_frozen_agents['R'] = 0

        self._step_count = None
        self._penalty = penalty
        self._step_cost = step_cost

        self.score = [0, 0]
        self.winning_points = winning_points

        self.full_observable = full_observable
        self.action_space = MultiAgentActionSpace([spaces.Discrete(NUM_ACTIONS) for _ in range(self.n_agents)])
        self.activated_action_space_attacker =  [DOWN, LEFT, UP, RIGHT, NOOP, STEAL, DELIVER]
        self.activated_action_space_defender = [DOWN, LEFT, UP, RIGHT, NOOP, FREEZE]
        self._agent_view_mask = agent_view_mask
        mask_size = np.prod(self._agent_view_mask)
        self._obs_high = np.array([1., 1.] + [1.] * mask_size + [1.0], dtype=np.float32)
        self._obs_low = np.array([0., 0.] + [0.] * mask_size + [0.0], dtype=np.float32)
        if self.full_observable:
            self._obs_high = np.tile(self._obs_high, self.n_agents)
            self._obs_low = np.tile(self._obs_low, self.n_agents)

        self.agent_pos = {_: None for _ in range(self.n_agents)}

        self.agent_team = {}
        for agent_i in range(self.n_blue_agents):
            self.agent_team[agent_i] = 'B'
        for agent_i in range(self.n_blue_agents, self.n_agents):
            self.agent_team[agent_i] = 'R'

        self.flag_pos = {_: None for _ in range(self.n_flags)}
        
        self.agent_has_flag = {_: None for _ in range(self.n_agents)}
        self.agent_is_frozen = {_: 0 for _ in range(self.n_agents)}

        self.areas_layout = dict()
        self.walls_layout = dict()
        self.set_map_layout()
        
        self._full_obs = self.__create_grid()

        self._agent_dones = [False for _ in range(self.n_agents)]

        self.observation_space = MultiAgentObservationSpace(
            [spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])

        self.seed()

    def set_game_type(self, game_type):
        self.game_type = game_type

    def set_num_agents(self, n_blue_agents, n_red_agents):
        self.n_blue_agents = n_blue_agents 
        self.n_red_agents = n_red_agents
        self.n_agents = n_blue_agents + n_red_agents

        self.action_space = MultiAgentActionSpace([spaces.Discrete(NUM_ACTIONS) for _ in range(self.n_agents)])

        if self.full_observable:
            self._obs_high = np.tile(self._obs_high, self.n_agents)
            self._obs_low = np.tile(self._obs_low, self.n_agents)

        self.agent_pos = {_: None for _ in range(self.n_agents)}

        self.agent_team = {}
        for agent_i in range(self.n_blue_agents):
            self.agent_team[agent_i] = 'B'
        for agent_i in range(self.n_blue_agents, self.n_agents):
            self.agent_team[agent_i] = 'R'

        self.agent_has_flag = {_: None for _ in range(self.n_agents)}
        self.agent_is_frozen = {_: 0 for _ in range(self.n_agents)}


        self.observation_space = MultiAgentObservationSpace(
            [spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])

    def reset(self):
        self.__draw_base_img()

        self.agent_pos = {}
        self.flag_pos = {}
        self.agent_has_flag = {}
        self.agent_is_frozen = {}
        self.score = [0, 0]

        self.num_stolen_flags = {}
        self.num_recovered_flags = {}
        self.num_delivered_flags = {}
        self.num_frozen_agents = {}

        self.num_stolen_flags['B'] = 0
        self.num_stolen_flags['R'] = 0

        self.num_recovered_flags['B'] = 0
        self.num_recovered_flags['R'] = 0

        self.num_delivered_flags['B'] = 0
        self.num_delivered_flags['R'] = 0

        self.num_frozen_agents['B'] = 0
        self.num_frozen_agents['R'] = 0

        self.__init_full_obs()

        self._step_count = 0
        self._agent_dones = [False for _ in range(self.n_agents)]
        return self.get_agent_obs(), self.get_areas_obs()

    def step(self, agents_action):
        self._step_count += 1

        for agent_i in range(self.n_agents):
            if self.agent_is_frozen[agent_i] > 0:
                self.agent_is_frozen[agent_i] -= 1

        for agent_i, action in enumerate(agents_action):
            if not (self._agent_dones[agent_i]) and self.agent_is_frozen[agent_i] == 0:
                self.__update_agent(agent_i, action)

        if self._step_count >= self._max_steps:
            for i in range(self.n_agents):
                self._agent_dones[i] = True

        return self.get_agent_obs(), self._agent_dones, self.score

    def get_actions_metrics(self):
        return self.num_stolen_flags, self.num_recovered_flags, self.num_delivered_flags, self.num_frozen_agents

    def render(self, mode='human'):
        img = copy.copy(self._base_img)

        for agent_i in range(self.n_agents):
            self.__draw_agent(img, agent_i, self.agent_team[agent_i], self.agent_pos[agent_i], self.agent_is_frozen[agent_i])

        for flag_i in range(self.n_blue_flags):
            self.__draw_flag(img, 'B', self.flag_pos[flag_i])

        for flag_i in range(self.n_blue_flags, self.n_flags):
            self.__draw_flag(img, 'R', self.flag_pos[flag_i])

        for agent_i in range(self.n_agents):
            self.__draw_agent_id(img, agent_i, self.agent_pos[agent_i])

        img = draw_score_board(img, self.score, board_height=self._cell_size)

        if self.game_type != None:
            write_cell_text(img, text=self.game_type, pos=[0, math.floor(self._grid_shape[1]/2)-1], cell_size=self._cell_size, fill='black', margin=0.40)

        img = np.asarray(img)
        if mode == 'rgb_array':
            return img

        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def seed(self, n=None):
        self.np_random, seed = seeding.np_random(n)
        return [seed]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    '''
    ------------------------------------------------------
    -------------- environment observation ---------------
    ------------------------------------------------------
    '''

    def get_blue_flag_respawn(self):
        if self._full_obs[FLAGS][4][1] == FLAG_IDS['none']:
            return [4, 1]
        if self._full_obs[FLAGS][9][1] == FLAG_IDS['none']:
            return [9, 1]
        if self._full_obs[FLAGS][14][1] == FLAG_IDS['none']:
            return [14, 1]
        return None

    def get_red_flag_respawn(self):
        if self._full_obs[FLAGS][4][self._grid_shape[1]-2] == FLAG_IDS['none']:
            return [4, self._grid_shape[1]-2]
        if self._full_obs[FLAGS][9][self._grid_shape[1]-2] == FLAG_IDS['none']:
            return [9, self._grid_shape[1]-2]
        if self._full_obs[FLAGS][14][self._grid_shape[1]-2] == FLAG_IDS['none']:
            return [14, self._grid_shape[1]-2]
        return None

    def get_blue_agent_respawn(self):
        while True:
            index = random.randint(0, 8)
            pos = self.areas_layout['delivery_blue_area'][index]
            if self._full_obs[ENTITIES][pos[0]][pos[1]] == ENTITY_IDS['empty']:
                return pos

    def get_red_agent_respawn(self):
        while True:
            index = random.randint(0, 8)
            pos = self.areas_layout['delivery_red_area'][index]
            if self._full_obs[ENTITIES][pos[0]][pos[1]] == ENTITY_IDS['empty']:
                return pos
    
    def __init_full_obs(self):
        self._full_obs = self.__create_grid()

        for agent_i in range(self.n_agents):
            if self.is_blue_agent(agent_i):
                respawn = self.get_blue_agent_respawn()
            else:
                respawn = self.get_red_agent_respawn()
            self.agent_pos[agent_i] = respawn
            self.agent_has_flag[agent_i] = None
            self.agent_is_frozen[agent_i] = 0
            self.__update_agent_view(agent_i)

        self.flag_pos[0] = [4, 1]
        self.__update_flag_view(0)
        self.flag_pos[1] = [9, 1]
        self.__update_flag_view(1)
        self.flag_pos[2] = [14, 1]
        self.__update_flag_view(2)

        self.flag_pos[3] = [4, self._grid_shape[1]-2]
        self.__update_flag_view(3)
        self.flag_pos[4] = [9, self._grid_shape[1]-2]
        self.__update_flag_view(4)
        self.flag_pos[5] = [14, self._grid_shape[1]-2]
        self.__update_flag_view(5)

        self.__draw_base_img()

    def __update_agent_view(self, agent_i):
        self._full_obs[ENTITIES][self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] = ENTITY_IDS['agent'] + str(agent_i)

    def __update_flag_view(self, flag_i):
        self._full_obs[FLAGS][self.flag_pos[flag_i][0]][self.flag_pos[flag_i][1]] = FLAG_IDS['flag'] + str(flag_i)

    def __create_grid(self):
        """
        Returns a 3 dimensional array. The first index defines the setting of the position:
        - ENTITY(0): indicates what entity is present (agent, wall, empty)
        - FLAG(1): indicates if flag is present (red, blue, none)
        - AREA(2): indicates if position is within a delimited area (defensive_blue, defensive_red, delivery_blue, delivery_red, free)
        """
        entities = [[ENTITY_IDS['empty'] for _ in range(self._grid_shape[1])] for _ in range(self._grid_shape[0])]
        flags = [[FLAG_IDS['none'] for _ in range(self._grid_shape[1])] for _ in range(self._grid_shape[0])]
        areas = [[AREA_IDS['free'] for _ in range(self._grid_shape[1])] for _ in range(self._grid_shape[0])]
        areas = self.__create_areas_grid(areas)
        entities = self.__create_walls_grid(entities)
        _grid = [entities, flags, areas]
        return _grid

    def __create_areas_grid(self, grid):
        for cell_i in range(len(self.areas_layout['defensive_blue_area'])):
            grid[self.areas_layout['defensive_blue_area'][cell_i][0]][self.areas_layout['defensive_blue_area'][cell_i][1]] = AREA_IDS['defensive_blue']
            grid[self.areas_layout['defensive_red_area'][cell_i][0]][self.areas_layout['defensive_red_area'][cell_i][1]] = AREA_IDS['defensive_red']

        for cell_i in range(len(self.areas_layout['delivery_blue_area'])):
            grid[self.areas_layout['delivery_blue_area'][cell_i][0]][self.areas_layout['delivery_blue_area'][cell_i][1]] = AREA_IDS['delivery_blue']
            grid[self.areas_layout['delivery_red_area'][cell_i][0]][self.areas_layout['delivery_red_area'][cell_i][1]] = AREA_IDS['delivery_red']

        return grid

    def __create_walls_grid(self, grid):
        for cell_i in range(len(self.walls_layout['wall'])):
            grid[self.walls_layout['wall'][cell_i][0]][self.walls_layout['wall'][cell_i][1]] = ENTITY_IDS['wall']
        return grid

    def simplified_features(self): # currently not used
        """"
        Return an array of the <n_agents> agents observation space.
        - Contains <n_agents> arrays with each agent observation space. Each <agent_i> array is composed of:
            - 1 array with <n_blue_agents in observation area> tuples (row, col, agent_id, has_flag, is_frozen) with positions and states of blue agents
            - 1 array with <n_red_agents in observation area> tuples (row, col, agent_id, has_flag, is_frozen) with positions and states of blue agents
            - 1 array with <n_blue_flags in observation area> tuples (row, col, blue_flag_id) with positions of blue flags
            - 1 array with <n_red_flags in observation area> tuples (row, col, red_flag_id) with positions of red flags
        """

        current_grid = np.array(self._full_obs)
        features = []
 
        blue_agent_pos = []
        for agent_id in range(0, self.n_blue_agents):
            tag = ENTITY_IDS['agent']+str(agent_id)
            row, col = np.where(current_grid[ENTITIES] == tag)
            row = row[0]
            col = col[0]
            blue_agent_pos.append((row, col, agent_id, self.agent_has_flag[agent_id], self.agent_is_frozen[agent_id]))

        red_agent_pos = []
        for agent_id in range(self.n_blue_agents, self.n_agents):
            tag = ENTITY_IDS['agent']+str(agent_id)
            row, col = np.where(current_grid[ENTITIES] == tag)
            row = row[0]
            col = col[0]
            red_agent_pos.append((row, col, agent_id, self.agent_has_flag[agent_id], self.agent_is_frozen[agent_id]))

        blue_flag_pos = []
        for blue_flag_i in range(self.n_blue_flags):
            tag = FLAG_IDS['flag']+str(blue_flag_i)
            row, col = np.where(current_grid[FLAGS] == tag)
            row = row[0]
            col = col[0]
            blue_flag_pos.append((row, col, blue_flag_i))

        red_flag_pos = []
        for red_flag_i in range(self.n_blue_flags, self.n_flags):
            tag = FLAG_IDS['flag']+str(red_flag_i)
            row, col = np.where(current_grid[FLAGS] == tag)
            row = row[0]
            col = col[0]
            red_flag_pos.append((row, col, red_flag_i))

        features.append(blue_agent_pos)
        features.append(red_agent_pos)
        features.append(blue_flag_pos)
        features.append(red_flag_pos)

        return features

    def get_areas_obs(self):
        """
        Return a dictionary with positions of all existing areas and walls
        1 dictionary with 4 keys:
            - 'delivery_blue_area'  : array of 9 tuples (row, col)
            - 'delivery_red_area'   : array of 9 tuples (row, col)
            - 'defensive_blue_area' : array of 27 tuples (row, col), 9 for each 3 areas
            - 'defensive_red_area'  : array of 27 tuples (row, col), 9 for each 3 areas
        """
        return self.areas_layout

    def get_agent_obs(self):
        """"
        Return an array of the <n_agents> agents observation space.
        - Contains <n_agents> arrays with each agent observation space. Each <agent_i> array is composed of:
            - 1 tuple (id, row, col, agent_id, has_flag, is_frozen) with the position and state of current agent
            - 1 array with <n_blue_agents in observation area> tuples (row, col, agent_id, has_flag, is_frozen) with positions and states of blue agents
            - 1 array with <n_red_agents in observation area> tuples (row, col, agent_id, has_flag, is_frozen) with positions and states of blue agents
            - 1 array with <n_blue_flags in observation area> tuples (row, col, blue_flag_id) with positions of blue flags
            - 1 array with <n_red_flags in observation area> tuples (row, col, red_flag_id) with positions of red flags
            - 1 array with <n_walls> tuples (row, col)

            has_flag -> index of flag if true, otherwise None
        """

        if self.full_observable:
            obs = []
            features = self.simplified_features()
            for agent_i in range(self.n_agents):
                agent_i_obs = [(self.agent_pos[agent_i][0], self.agent_pos[agent_i][1], agent_i, self.agent_has_flag[agent_i], self.agent_is_frozen[agent_i])]
                features_i = features
                obs.append(agent_i_obs+features_i)
            return obs

        obs = []
 
        for agent_i in range(self.n_agents):
            pos = self.agent_pos[agent_i]
            agent_i_obs = []
            blue_agent_pos = []
            red_agent_pos = []
            blue_flag_pos = []
            red_flag_pos = []
            walls = []

            for row in range(max(0, pos[0] - self._agent_view_mask[0]), min(pos[0] + self._agent_view_mask[0] + 1, self._grid_shape[0])):
                for col in range(max(0, pos[1] - self._agent_view_mask[1]), min(pos[1] + self._agent_view_mask[1] + 1, self._grid_shape[1])):
                    if ENTITY_IDS['agent'] in self._full_obs[ENTITIES][row][col]:
                        i = self.get_agent_id([row, col])
                        if i != agent_i and i < self.n_blue_agents:
                            blue_agent_pos.append((row, col, i, self.agent_has_flag[i], self.agent_is_frozen[i]))
                        elif i != agent_i:
                            red_agent_pos.append((row, col, i, self.agent_has_flag[i], self.agent_is_frozen[i]))
                    if FLAG_IDS['flag'] in self._full_obs[FLAGS][row][col]:
                        flag_id = self.get_flag_id([row, col])
                        if flag_id < self.n_blue_flags:
                            blue_flag_pos.append((row, col, flag_id))
                        else:
                            red_flag_pos.append((row, col, flag_id))
                    if ENTITY_IDS['wall'] == self._full_obs[ENTITIES][row][col]:
                        walls.append((row, col))

            # add an array with tuple containing the position and state of current agent
            agent_i_obs.append((self.agent_pos[agent_i][0], self.agent_pos[agent_i][1], agent_i, self.agent_has_flag[agent_i], self.agent_is_frozen[agent_i]))

            agent_i_obs.append(blue_agent_pos)
            agent_i_obs.append(red_agent_pos)
            agent_i_obs.append(blue_flag_pos)
            agent_i_obs.append(red_flag_pos)
            agent_i_obs.append(walls)
            
            obs.append(agent_i_obs)

        return obs

    def _neighbour_agents(self, pos): #this method is currently not used but may be useful later
        # check if agent is in neighbour
        _count = 0
        neighbours_xy = []
        if self.is_valid([pos[0] + 1, pos[1]]) and ENTITY_IDS['agent'] in self._full_obs[ENTITIES][pos[0] + 1][pos[1]]:
            _count += 1
            neighbours_xy.append([pos[0] + 1, pos[1]])
        if self.is_valid([pos[0] - 1, pos[1]]) and ENTITY_IDS['agent'] in self._full_obs[ENTITIES][pos[0] - 1][pos[1]]:
            _count += 1
            neighbours_xy.append([pos[0] - 1, pos[1]])
        if self.is_valid([pos[0], pos[1] + 1]) and ENTITY_IDS['agent'] in self._full_obs[ENTITIES][pos[0]][pos[1] + 1]:
            _count += 1
            neighbours_xy.append([pos[0], pos[1] + 1])
        if self.is_valid([pos[0], pos[1] - 1]) and ENTITY_IDS['agent'] in self._full_obs[ENTITIES][pos[0]][pos[1] - 1]:
            neighbours_xy.append([pos[0], pos[1] - 1])
            _count += 1

        agent_id = []
        for x, y in neighbours_xy:
            agent_id.append(int(self._full_obs[ENTITIES][x][y].split(ENTITY_IDS['agent'])[1]) - 1)
        return _count, agent_id

    '''
    -----------------------------------------------------------------------------
    -------------------- auxiliar observation space functions  ------------------
    -----------------------------------------------------------------------------
    '''
    def get_agent_id(self, pos):
        if self.is_valid(pos) and ENTITY_IDS['agent'] in self._full_obs[ENTITIES][pos[0]][pos[1]]:
            return int(self._full_obs[ENTITIES][pos[0]][pos[1]][-1])
        return -1

    def get_flag_id(self, pos):
        if self.is_valid(pos) and FLAG_IDS['flag'] in self._full_obs[FLAGS][pos[0]][pos[1]]:
            return int(self._full_obs[FLAGS][pos[0]][pos[1]][-1])
        return None

    def is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    def _is_cell_vacant(self, pos):
        if self.is_valid(pos):
            obs_entity = self._full_obs[ENTITIES][pos[0]][pos[1]]
            return obs_entity==ENTITY_IDS['empty']
        return False

    def _is_cell_defensive_blue_area(self, pos):
        if self.is_valid(pos):
            obs_entity = self._full_obs[AREAS][pos[0]][pos[1]]
            return obs_entity==AREA_IDS['defensive_blue_area']
        return False

    def _is_cell_defensive_red_area(self, pos):
        if self.is_valid(pos):
            obs_entity = self._full_obs[AREAS][pos[0]][pos[1]]
            return obs_entity==AREA_IDS['defensive_red_area']
        return False

    def _is_cell_delivery_blue_area(self, pos):
        if self.is_valid(pos):
            obs_entity = self._full_obs[AREAS][pos[0]][pos[1]]
            return obs_entity==AREA_IDS['delivery_blue_area']
        return False

    def _is_cell_delivery_red_area(self, pos):
        if self.is_valid(pos):
            obs_entity = self._full_obs[AREAS][pos[0]][pos[1]]
            return obs_entity==AREA_IDS['delivery_red_area']
        return False

    def _cell_has_blue_flag(self, pos):
        flag_id = self.get_flag_id(pos)
        if flag_id != None:
            return flag_id < self.n_blue_flags
        return False

    def _cell_has_red_flag(self, pos):
        flag_id = self.get_flag_id(pos)
        if flag_id != None:
            return flag_id >= self.n_blue_flags and flag_id < self.n_flags
        return False

    '''
    ------------------------------------------------------
    -------------------- agent actions  ------------------
    ------------------------------------------------------
    '''

    def get_flag_index(self, pos):
        for index in range(self.n_flags):
            if pos == self.flag_pos[index]:
                return index
        return None

    def is_blue_agent(self, agent_i):
        return agent_i >= 0 and agent_i < self.n_blue_agents and self.agent_team[agent_i] == 'B'

    def is_red_agent(self, agent_i):
        return agent_i >= self.n_blue_agents and agent_i < self.n_agents and self.agent_team[agent_i] == 'R'

    def get_nearby_positions(self, agent_i):
        positions = []
        pos = self.agent_pos[agent_i]
        if self.is_valid([pos[0]-1, pos[1]]): # up
            positions.append([pos[0]-1, pos[1]])
        if self.is_valid([pos[0], pos[1]+1]): # right
            positions.append([pos[0], pos[1]+1])
        if self.is_valid([pos[0], pos[1]-1]): # left
            positions.append([pos[0], pos[1]-1])
        if self.is_valid([pos[0]+1, pos[1]]): # bottom
            positions.append([pos[0]+1, pos[1]])
        return positions
    
    def get_frozen_positions(self, agent_i):
        nearby_positions = self.get_nearby_positions(agent_i)
        frozen_positions = []
        pos = self.agent_pos[agent_i]

        for pos in nearby_positions:
            if self.is_blue_agent(agent_i) and self._full_obs[AREAS][pos[0]][pos[1]] == AREA_IDS['defensive_blue']:
                frozen_positions.append(pos)
            elif self.is_red_agent(agent_i) and self._full_obs[AREAS][pos[0]][pos[1]] == AREA_IDS['defensive_red']:
                frozen_positions.append(pos)

        return frozen_positions

    def get_frozen_agents(self, agent_i):
        frozen_positions = self.get_frozen_positions(agent_i)
        frozen_agents = []
        
        if self.is_blue_agent(agent_i):
            for pos in frozen_positions:
                red_agent_i = self.get_agent_id(pos)
                if self.is_red_agent(red_agent_i):
                    frozen_agents.append(red_agent_i)
        else:
            for pos in frozen_positions:
                blue_agent_i = self.get_agent_id(pos)
                if self.is_blue_agent(blue_agent_i):
                    frozen_agents.append(blue_agent_i)

        return frozen_agents

    def do_deliver(self, agent_i, curr_pos):
        # blue agent delivers flag
        if self.agent_team[agent_i] == 'B':

            # deliver blue flag in blue defensive area (RECOVER)
            if self._cell_has_blue_flag(curr_pos):
                self.agent_has_flag[agent_i] = None
                self.increment_num_recovered_flags(agent_i)

            # deliver red flag in blue delivery area (DELIVER)
            if self._cell_has_red_flag(curr_pos):
                flag_index = self.get_flag_index(curr_pos)
                self._full_obs[FLAGS][curr_pos[0]][curr_pos[1]] = FLAG_IDS['none']

                respawn = self.get_red_flag_respawn()
                if respawn == None:
                    raise Exception(f'Agent {agent_i}: no free red flag respawn!')

                self.flag_pos[flag_index] = respawn
                self._full_obs[FLAGS][respawn[0]][respawn[1]] = FLAG_IDS['flag'] + str(flag_index)
                self.agent_has_flag[agent_i] = None
                self.score[0] += 1
                self.increment_num_delivered_flags(agent_i)

        # red agent delivers flag
        else:

            # deliver red flag in red defensive area (RECOVER)
            if self._cell_has_red_flag(curr_pos):
                self.agent_has_flag[agent_i] = None
                self.increment_num_recovered_flags(agent_i)

            # deliver blue flag in red delivery area (DELIVER)
            if self._cell_has_blue_flag(curr_pos):
                flag_index = self.get_flag_index(curr_pos)
                self._full_obs[FLAGS][curr_pos[0]][curr_pos[1]] = FLAG_IDS['none']

                respawn = self.get_blue_flag_respawn()
                if respawn == None:
                    raise Exception(f'Agent {agent_i}: no free blue flag respawn!')

                self.flag_pos[flag_index] = respawn
                self._full_obs[FLAGS][respawn[0]][respawn[1]] = FLAG_IDS['flag'] + str(flag_index)
                self.agent_has_flag[agent_i] = None
                self.score[1] += 1
                self.increment_num_delivered_flags(agent_i)

    def do_freeze(self, agent_i):
        frozen_agents = self.get_frozen_agents(agent_i)
        for frozen_agent_i in frozen_agents:
            self.agent_is_frozen[frozen_agent_i] = self.freeze_time

        if len(frozen_agents) != 0:
            self.increment_num_frozen_agents(agent_i)

    def are_close_positions(self, pos1: tuple, pos2: tuple):
        row_dist = abs(pos1[0]-pos2[0])
        col_dist = abs(pos1[1]-pos2[1])
        if row_dist + col_dist < 2:
            return True
        return False

    def change_flag_pos(self, enemy_agent_i, agent_i, flag_i):
        prev_pos = self.agent_pos[enemy_agent_i]
        new_pos = self.agent_pos[agent_i]
        self._full_obs[FLAGS][prev_pos[0]][prev_pos[1]] = FLAG_IDS['none']
        self._full_obs[FLAGS][new_pos[0]][new_pos[1]] = FLAG_IDS['flag'] + str(flag_i)
        self.agent_has_flag[agent_i] = self.agent_has_flag[enemy_agent_i]
        self.agent_has_flag[enemy_agent_i] = None
        self.flag_pos[flag_i] = new_pos

    def steal_flag_from_enemy_agent(self, agent_i, enemy_agent_i):
        if self.are_close_positions(self.agent_pos[agent_i], self.agent_pos[enemy_agent_i]) and self.agent_has_flag[enemy_agent_i] != None:
            self.change_flag_pos(enemy_agent_i, agent_i, self.agent_has_flag[enemy_agent_i])
            return True
        return False

    def increment_num_stolen_flags(self, agent_i):
        self.num_stolen_flags[self.agent_team[agent_i]] += 1

    def increment_num_frozen_agents(self, agent_i):
        self.num_frozen_agents[self.agent_team[agent_i]] += 1

    def increment_num_recovered_flags(self, agent_i):
        self.num_recovered_flags[self.agent_team[agent_i]] += 1

    def increment_num_delivered_flags(self, agent_i):
        self.num_delivered_flags[self.agent_team[agent_i]] += 1

    def do_steal(self, agent_i, curr_pos):
        # steal flag from defensive area: agent position = flag position
        self.agent_has_flag[agent_i] = self.get_flag_index(curr_pos)

        # got the flag
        if self.agent_has_flag[agent_i] != None:
            self.increment_num_stolen_flags(agent_i)
        
        # steal flag from nearby enemy red agent
        if self.agent_has_flag[agent_i] == None and self.is_blue_agent(agent_i):
            for enemy_agent_i in range(self.n_blue_agents, self.n_agents):
                if self.steal_flag_from_enemy_agent(agent_i, enemy_agent_i):
                    self.increment_num_stolen_flags(agent_i)
                    break

        # steal flag from nearby enemy blue agent
        elif self.agent_has_flag[agent_i] == None and self.is_red_agent(agent_i):
            for enemy_agent_i in range(self.n_blue_agents):
                if self.steal_flag_from_enemy_agent(agent_i, enemy_agent_i):
                    self.increment_num_stolen_flags(agent_i)
                    break


    def __update_agent(self, agent_i, move):
        curr_pos = copy.copy(self.agent_pos[agent_i])
        next_pos = None

        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4: # no-op
            pass
        elif move == 5: # steal
            self.do_steal(agent_i, curr_pos)
        elif move == 6: # freeze
            if self.agent_is_frozen[agent_i] == 0: # agent is not frozen -> timer is set to 0
                self.do_freeze(agent_i)
        elif move == 7: #deliver
            self.do_deliver(agent_i, curr_pos)
            
        else:
            raise Exception(f'Agent {agent_i}: action Not found!')

        if next_pos is not None:
            if self._is_cell_vacant(next_pos):
                self.agent_pos[agent_i] = next_pos
                self._full_obs[ENTITIES][curr_pos[0]][curr_pos[1]] = ENTITY_IDS['empty']
                self.__update_agent_view(agent_i)

                if self.agent_has_flag[agent_i] != None:
                    flag_index = self.agent_has_flag[agent_i]
                    self.flag_pos[flag_index] = next_pos
                    self._full_obs[FLAGS][curr_pos[0]][curr_pos[1]] = FLAG_IDS['none']
                    self.__update_flag_view(flag_index)
            return

    def get_action_meanings(self, agent_i=None):
        if agent_i is not None:
            assert agent_i <= self.n_agents
            return [ACTION_MEANING[i] for i in range(self.action_space[agent_i].n)]
        else:
            return [[ACTION_MEANING[i] for i in range(ac.n)] for ac in self.action_space]

    def action_space_sample(self):
        return [agent_action_space.sample() for agent_action_space in self.action_space]

    '''
    ------------------------------------------------------
    ---------- drawing methods for visual board ----------
    ------------------------------------------------------
    '''

    def __draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], self._cell_size, fill='white', line_color='gray')
        self.__draw_areas(self._base_img)
        self.__draw_walls(self._base_img)

    def __draw_areas(self, img):
        for cell_i in range(len(self.areas_layout['defensive_blue_area'])):
            fill_cell(img, self.areas_layout['defensive_blue_area'][cell_i], self._cell_size, fill=COLORS['blue_area'], margin=0.05)
            fill_cell(img, self.areas_layout['defensive_red_area'][cell_i], self._cell_size, fill=COLORS['red_area'], margin=0.05)
        
        for cell_i in range(len(self.areas_layout['delivery_blue_area'])):
            fill_cell(img, self.areas_layout['delivery_blue_area'][cell_i], self._cell_size, fill=COLORS['blue_area'], margin=0.05)
            fill_cell(img, self.areas_layout['delivery_red_area'][cell_i], self._cell_size, fill=COLORS['red_area'], margin=0.05)

    def __draw_walls(self, img):
        for cell_i in range(len(self.walls_layout['wall'])):
            fill_cell(img, self.walls_layout['wall'][cell_i], self._cell_size, fill='gray', margin=0.05)

    def __draw_agent(self, img, agent_id, team, pos, is_frozen):
        color = COLORS['blue_agent']
        if team == 'R': 
            color = COLORS['red_agent']
        draw_circle(img, pos, self._cell_size, fill=color, radius=0.1)
        if is_frozen:
            draw_cell_outline(img, pos, self._cell_size, fill='black')

    def __draw_agent_id(self, img, agent_id, pos):
        write_cell_text(img, str(agent_id), pos, self._cell_size, fill='white', margin=0.35)
        

    def __draw_flag(self, img, team, pos):
        color = COLORS['blue_flag']
        if team == 'R': 
            color = COLORS['red_flag']
        fill_cell(img, pos, self._cell_size, color, margin=0.35)

    def set_map_layout(self):
        """
        creates a dictionary of map layout where each key contains the type of the area 
        or a wall and the corresponding value corresponds to an array of grid positions
        """

        self.areas_layout['delivery_blue_area'] = [None]*9
        self.areas_layout['delivery_red_area'] = [None]*9
        self.areas_layout['defensive_blue_area'] = [None]*27
        self.areas_layout['defensive_red_area'] = [None]*27

        index = 0
        for j in range(3): # vertical successive squares
            for k in range(3): # horizontal successive squares
                # 1 attacking area for each team
                self.areas_layout['delivery_blue_area'][index] = (j, 8+k)
                self.areas_layout['delivery_red_area'][index] = (self._grid_shape[0]-3+j, 8+k)
                index = index + 1

        gap=5 # gap between first rows of successive defensive areas
        index = 0
        for i in range(3):
            for j in range(3): # vertical successive squares
                for k in range(3): # horizontal successive squares
                        # 3 defensive areas for each team
                        self.areas_layout['defensive_blue_area'][index] = (i*gap+j+3, k)
                        self.areas_layout['defensive_red_area'][index] = (i*gap+j+3, self._grid_shape[1]-3+k)
                        index = index + 1
            
        self.walls_layout['wall'] = [None]*24

        index = 0
        for i in range(6, 12):
            # vertical wall left side
            self.walls_layout['wall'][index] = (i, 5)
            # vertical wall right side
            self.walls_layout['wall'][index+1] = (i, self._grid_shape[1]-6)
            index = index + 2

        # rectangle wall top left
        for i in range(0, 2):
            for j in range(self._grid_shape[0]-7, self._grid_shape[0]-4):
                self.walls_layout['wall'][index] = (i, j)
                index = index + 1

        # rectangle wall bottom right
        for i in range(self._grid_shape[0]-2, self._grid_shape[0]):
            for j in range(4, 7):
                self.walls_layout['wall'][index] = (i, j)
                index = index + 1

ACTION_MEANING = {
    0: "DOWN",
    1: "LEFT",
    2: "UP",
    3: "RIGHT",
    4: "NOOP",
    5: "STEAL",
    6: "FREEZE",
    7: "DELIVER"
}

ENTITIES = 0
FLAGS = 1
AREAS = 2

ENTITY_IDS = {
    'agent': 'A',
    'wall': 'W',
    'empty': '0'
}

FLAG_IDS = {
    'flag': 'F',
    'none': '0'
}

AREA_IDS = {
    'defensive_red': 'DR',
    'defensive_blue': 'DB',
    'delivery_red': 'AR',
    'delivery_blue': 'AB',
    'free': '0'
}

COLORS = {
    'blue_area': '#86d7f0',
    'red_area': '#f08686',
    'blue_agent': '#33a8cc',
    'red_agent': '#f54242',
    'blue_flag': '#195a6e',
    'red_flag': '#6e1922'
}