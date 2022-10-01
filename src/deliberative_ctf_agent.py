from agent import Agent
import math
import numpy as np
from abc import abstractmethod

ACTIONS = 8

class DeliberativeAgent(Agent):

    '''
    DESIRES: STEAL, FREEZE, DELIVER, GO_TO_ENEMY_DEFENSIVE_AREAS, GO_TO_TEAM_DEFENSIVE_AREAS
    
    '''

    def __init__(self, agent_id: int, team: str):
        super(DeliberativeAgent, self).__init__(f"Deliberative Agent", team)
        self.agent_id = agent_id
        self.agent_pos = tuple()
        self.n_actions = ACTIONS
        self.agent_plan = list()
        self.n_plan = 0
        self.desires = list()
        self.intention = list()
        self.selected_intention = list()

        self.DOWN, self.LEFT, self.UP, self.RIGHT, self.NOOP, self.STEAL, self.FREEZE, self.DELIVER = range(ACTIONS)
        self.GO_TO_ENEMY_DEFENSIVE_AREAS = 8
        self.GO_TO_TEAM_DEFENSIVE_AREAS = 9


    def get_id(self) -> int:
        return self.agent_id


    '''
    ------------------------------------------------------
    ---------------- deliberate agent --------------------
    ------------------------------------------------------
    '''
    @abstractmethod
    def options(self):
        raise NotImplementedError()
    
    @abstractmethod
    def plan(self):
        raise NotImplementedError()

    @abstractmethod
    def reconsider(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def sound(self, action) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def succeeded(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def impossible(self) -> bool:
        raise NotImplementedError()


    def update_beliefs(self):
        """
        Update agent beliefs through its sensors and the environment observation.
        """
        self.agent_pos = (self.observation[0][0], self.observation[0][1])
        self.agent_state = self.observation[0]
        self.walls_pos = self.observation[5]

        self.flag_id = self.observation[0][3]

        if self.team == 'R':
            self.enemies_state = self.observation[1]
            self.enemy_defensive_areas_pos = self.map_layout_obs['defensive_blue_area']
            self.enemy_flags_pos = self.observation[3]

            self.team_state = self.observation[2]
            self.team_defensive_areas_pos = self.map_layout_obs['defensive_red_area']
            self.team_delivery_area_pos = self.map_layout_obs['delivery_red_area']
            self.team_flags_pos = self.observation[4]

            self.team_flag_respawns = []
            for area in range(3):
                self.team_flag_respawns.append(self.map_layout_obs['defensive_red_area'][9*area + 4])

            self.enemy_defensive_areas_pos.reverse()
            self.enemy_flags_pos.reverse()

            self.defensive_area = (4 + 9 * self.team_agent_index)%27
            
        else:
            self.enemies_state = self.observation[2]
            self.enemy_defensive_areas_pos = self.map_layout_obs['defensive_red_area']
            self.enemy_flags_pos = self.observation[4]
            self.team_state = self.observation[1]
            self.team_defensive_areas_pos = self.map_layout_obs['defensive_blue_area']
            self.team_delivery_area_pos = self.map_layout_obs['delivery_blue_area']
            self.team_flags_pos = self.observation[3]

            self.team_flag_respawns = []
            for area in range(3):
                self.team_flag_respawns.append(self.map_layout_obs['defensive_blue_area'][9*area + 4])

            self.defensive_area = (13 + 9 * self.team_agent_index)%27




    def filter(self):
        """
        Choose one of the available desires and compute the agent intention.
        The agent intention corresponds to the coordinates where the agent must move to.
        """

        if self.desires[0][0] == self.DELIVER and self.desires[0][1] == "team_delivery_area":
            self.intention = [self.DELIVER, self.team_delivery_area_pos[4], "team_delivery_area"]

        elif self.desires[0][0] == self.DELIVER and self.desires[0][1] == "team_defensive_area":
            respawn_index = self.desires[0][2]
            self.intention = [self.DELIVER, (self.team_flag_respawns[respawn_index][0],self.team_flag_respawns[respawn_index][1]), "team_defensive_area"]

        elif self.desires[0][0] == self.STEAL and self.desires[0][1] == "enemy_defensive_area":
            index = self.desires[0][2]
            self.intention = [self.STEAL, self.enemy_defensive_areas_pos[index], "enemy_defensive_area"]

        elif self.desires[0][0] == self.STEAL and self.desires[0][1] == "enemy":
            enemy_index = self.desires[0][2]
            self.intention = [self.STEAL, (self.enemies_state[enemy_index][0],self.enemies_state[enemy_index][1]), "enemy"]

        elif self.desires[0][0] == self.GO_TO_ENEMY_DEFENSIVE_AREAS:
            index = self.desires[0][1]
            self.intention = [self.GO_TO_ENEMY_DEFENSIVE_AREAS, self.enemy_defensive_areas_pos[index]]

        elif self.desires[0][0] == self.FREEZE:
            enemy_index = self.desires[0][1]
            self.intention = [self.FREEZE, (self.enemies_state[enemy_index][0],self.enemies_state[enemy_index][1]),enemy_index]

        elif self.desires[0][0] == self.GO_TO_TEAM_DEFENSIVE_AREAS:
            pos = self.desires[0][1]
            self.intention = [self.GO_TO_TEAM_DEFENSIVE_AREAS, self.team_defensive_areas_pos[pos]]

        else:
            raise Exception('Agent must have at least one desire')



    '''
    ------------------------------------------------------
    ----------- function to execute action --------------
    ------------------------------------------------------
    '''

    def action(self) -> int:
        self.update_beliefs()

        if self.agent_plan!=[] and not self.succeeded() and not self.impossible():
            if self.reconsider():
                self.desires = []
                self.intention = []
                self.options()
                self.filter()

            if not self.sound(self.agent_plan[0]): #TODO:remove the action in social_conventions and in role_agent
                self.agent_plan = []
                self.plan()

            action = self.agent_plan[0]
            self.agent_plan = self.agent_plan[1:]
            return action

        elif self.agent_plan!=[] and self.succeeded():
            action = self.agent_plan[0]
            self.agent_plan = self.agent_plan[1:]
            return action

        else:
            self.desires = []
            self.intention = []
            self.agent_plan = []
            self.options()
            self.filter()
            self.plan()

        return self.NOOP




    '''
    ------------------------------------------------------
    ------ auxiliar functions to deliberate agent --------
    ------------------------------------------------------
    '''

    def has_flag(self):
        return self.agent_state[3] != None



    def get_pos(self, state):
        return (state[0], state[1])



    def distance(self, pos1, pos2):
        return math.sqrt((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2)



    def wall_in_pos(self, pos):
        return pos in self.walls_pos




    def agent_in_pos(self, pos):
        for agent_state in self.team_state:
            if (agent_state[0], agent_state[1]) == pos:
                return True
        for enemy_state in self.enemies_state:
            if (enemy_state[0], enemy_state[1]) == pos:
                return True
        return False




    def get_free_team_flag_respawn(self):
        for respawn in self.team_flag_respawns:
            free = True
            for flag_state in self.team_flags_pos:
                if (flag_state[0], flag_state[1]) == respawn:
                    free = False

            if free:
                return respawn
        return None




    def flag_of_team(self):
        return self.agent_state[3] == self.team




    def enemy_has_team_flag(self, enemy_state):
        # Blue team: red enemy with blue flag
        if self.team == 'B' and enemy_state[3] in range(self.n_blue_flags):
            return True

        # Red team: blue enemy with red flag
        elif self.team == 'R' and enemy_state[3] in range(self.n_blue_flags, self.n_flags):
            return True

        return False




    def enemy_has_enemy_flag(self, enemy_state):
        # Blue team: red enemy with red flag
        if self.team == 'B' and enemy_state[3] in range(self.n_blue_flags, self.n_flags):
            return True

        # Red team: blue enemy with blue flag
        elif self.team == 'R' and enemy_state[3] in range(self.n_blue_flags):
            return True

        return False




    def agent_is_frozen(self, state):
        return state[4] > 0




    def is_frozen(self):
        return self.observation[0][4] > 0




    def get_remaining_freeze_time(self):
        return self.observation[0][4]




    def enemy_in_nearby_position(self, dest_pos):
        """
        Verify if an enemy agent is nearby a destination area.
        Nearby meaning a range of 2 squares for each direction
        Returns false if observation is limited for a given destination.
        """
        found_enemy = False
        for enemy_state in self.enemies_state:
            if abs(dest_pos[0]-enemy_state[0]) < 2 and abs(dest_pos[1]-enemy_state[1]) < 2:
                found_enemy = True

        return found_enemy




    def are_close_positions(self, pos1, pos2):
        row_dist = abs(pos1[0]-pos2[0])
        col_dist = abs(pos1[1]-pos2[1])
        if row_dist + col_dist < 2:
            return True




    def enemies_in_close_position(self, dest_pos):
        """
        Verify if an enemy agent is nearby a destination area.
        Nearby meaning up, left, right, down
        Returns false if observation is limited for a given destination.
        """
        found_enemy = False
        for enemy_state in self.enemies_state:
            if self.are_close_positions(dest_pos, self.get_enemy_pos(enemy_state)):
                found_enemy = True
        return found_enemy




    def enemies_in_pos(self, pos):
        for enemy_state in self.enemies_state:
            if self.get_enemy_pos(enemy_state) == pos:
                return True
        return False



    def agent_in_flag_pos(self, flag_pos):
        return self.agent_pos == flag_pos



    def get_enemy_flag_pos(self, index):
        return (self.enemy_flags_pos[index][0], self.enemy_flags_pos[index][1])



    def get_team_flag_pos(self, index):
        return (self.team_flags_pos[index][0], self.enemy_flags_pos[index][1])



    def get_flag_pos(self, flag_state):
        return (flag_state[0], flag_state[1])


        
    def get_flag_id(self, flag_state):
        return flag_state[2]



    def get_free_team_flag_respawn(self):
        for respawn in self.team_flag_respawns:
            free = True
            for flag_state in self.team_flags_pos:
                if self.get_flag_pos(flag_state) == respawn and respawn != self.agent_pos: # in case agent is already in position for delivery
                    free = False

            if free:
                return respawn
        return None



    def flag_in_pos(self, pos):
        for enemy_flag_state in self.enemy_flags_pos:
            if self.get_flag_pos(enemy_flag_state) == pos:
                return True
        for team_flag_state in self.team_flags_pos:
            if self.get_flag_pos(team_flag_state) == pos:
                return True
        return False


    
    def get_enemy_pos(self, enemy_state: list):
        return (enemy_state[0], enemy_state[1])


    
    # It is used in plan
    def advance_to_pos(self, agent_pos: tuple, dest_pos: tuple) -> int:
        def _move_vertically(distances, pos, horizontal_obstacle = False):
            down = (pos[0] + 1, pos[1])
            up = (pos[0] - 1, pos[1])

            roll = -1
            if horizontal_obstacle: # avoid case where two horizontal agents are blocking each other forever
                roll = np.random.uniform(0, 1)

            if (roll > 0.5 and roll <= 1 or distances[0] > 0) and down not in self.path and down not in self.obstacles:
                return self.DOWN, down
            elif (roll >= 0 and roll <= 0.5 or distances[0] < 0) and up not in self.path and up not in self.obstacles:
                return self.UP, up
            elif (pos[0], pos[1]) not in self.path:
                return self.NOOP, pos
            return None, None
            
        def _move_horizontally(distances, pos, vertical_obstacle = False):
            right = (pos[0], pos[1] + 1)
            left = (pos[0], pos[1] - 1)

            roll = -1
            if vertical_obstacle: # avoid case where two vertical agents are blocking each other forever
                roll = np.random.uniform(0, 1)

            if (roll > 0.5 and roll <= 1 or distances[1] > 0) and right not in self.path and right not in self.obstacles:
                return self.RIGHT, right
            elif (roll >= 0 and roll <= 0.5 or distances[1] < 0) and left not in self.path and left not in self.obstacles:
                return self.LEFT, left
            elif (pos[0], pos[1]) not in self.path:
                return self.NOOP, pos
            return None, None

        # obstacle in position except for stealing/freezing destination of target agents
        def obstacle_in_pos(pos):
            # wall blocking the position
            if self.wall_in_pos(pos):
                return True

            # enemy blocking the position
            if self.enemies_in_pos(pos):
                if not (self.are_close_positions(pos, dest_pos) and (self.intention[0] == self.FREEZE or self.intention[0] == self.STEAL and self.intention[2] == 'enemy')):
                    return True

            # flag blocking the position for any agent carrying another flag
            if self.has_flag() and agent_pos != pos and self.flag_in_pos(pos):
                return True
                
            return False

        def obstacle_in_horizontal_pos(pos, distances): # left, right
            left = (pos[0], pos[1] - 1)
            right = (pos[0], pos[1] + 1)

            if distances[1] < 0 and obstacle_in_pos(left):
                return True
            elif distances[1] > 0 and obstacle_in_pos(right):
                return True
            return False

        def obstacle_in_vertical_pos(pos, distances): # up, down
            up = (pos[0] - 1, pos[1])
            down = (pos[0] + 1, pos[1])

            if distances[0] < 0 and obstacle_in_pos(up):
                return True
            elif distances[0] > 0 and obstacle_in_pos(down):
                return True
            return False

        distance_dest = np.array(dest_pos) - np.array(agent_pos)
        abs_distances = np.absolute(distance_dest)

        horizontal_obstacle = obstacle_in_horizontal_pos(agent_pos, distance_dest)
        vertical_obstacle = obstacle_in_vertical_pos(agent_pos, distance_dest)

        if not horizontal_obstacle and (abs_distances[0] < abs_distances[1] or vertical_obstacle):
            return _move_horizontally(distance_dest, agent_pos, vertical_obstacle)

        if not vertical_obstacle and (abs_distances[0] > abs_distances[1] or horizontal_obstacle):
            return _move_vertically(distance_dest, agent_pos, horizontal_obstacle)
        
        roll = np.random.uniform(0, 1)
        return _move_horizontally(distance_dest, agent_pos) if roll > 0.5 else _move_vertically(distance_dest, agent_pos)
