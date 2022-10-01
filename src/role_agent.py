import numpy as np
from deliberative_ctf_agent import DeliberativeAgent
from scipy.spatial.distance import cityblock
import time

class RoleAgent(DeliberativeAgent):
    def __init__(self, agent_id: int, team: str, n_agents_team: int, roles: list, n_blue_flags: int, n_red_flags: int):
        super(RoleAgent, self).__init__(agent_id, team)

        self.ATTACKER, self.DEFENDER = range(2)
        self.ROLES = roles

        self.n_agents = n_agents_team
        self.n_roles = len(roles)
        self.n_blue_flags = n_blue_flags
        self.n_red_flags = n_red_flags
        self.n_flags = n_blue_flags +  n_red_flags
        self.new_role = False


        self.team_agent_index = self.agent_id
        if team == 'R':
            self.team_agent_index -= n_agents_team # index for first agent on each team should start at 0 to help defining team potential indexes

        if self.team == 'B':
            self.area = 22
        else:
            self.area = 4
        self.plus = 9

        self.team_potentials = [ [ 0 for _ in range(self.n_agents) ] for _ in range(self.n_roles)]
        self.agent_role = None

    '''
    ------------------------------------------------------
    --------------- agent role computation ---------------
    ------------------------------------------------------
    '''

    def potential_function_attacker(self, observation:list, map_layout_obs: dict) -> int:
        """
        Calculate the potencial function.
        The potencial function consists of the negative Manhattan distance between the
        'agent_pos' and the target position which can either be:
        - enemy that has the flag
        - enemy's defensive area

        :param observation: list of agent observation space
        :param map_layout_obs: dictionary with 4 keys corresponding to each area/wall of the map
        """

        if self.attacker_has_flag(observation):
            return 0 # max potential -> must be an attacker

        agent_pos = self.get_agent_pos(observation)
        enemies_state = self.get_enemies_state(self.team, observation)
        enemy_defensive_areas_pos = self.get_enemy_defensive_areas_pos(self.team, map_layout_obs)

        max_potential = -400
        for enemy_state in enemies_state:
            enemy_pos = self.get_enemy_pos(enemy_state)
            if self.enemy_has_flag(enemy_state): # enemy has a flag
                p = -cityblock(agent_pos, enemy_pos)
                max_potential = max(max_potential, p)

        for area_pos in enemy_defensive_areas_pos:
            p = -cityblock(agent_pos, area_pos)
            max_potential = max(max_potential, p)
        
        return max_potential

    def potential_function_defender(self, observation:list, map_layout_obs: dict) -> int:
        """
        Calculate the potencial function.
        The potencial function consists of the negative Manhattan distance between the
        'agent_pos' and the target position which can either be:
        - own defensive area

        :param observation: list of agent observation space
        :param map_layout_obs: dictionary with 4 keys corresponding to each area/wall of the map
        """

        agent_pos = self.get_agent_pos(observation)
        defensive_areas_pos = self.get_defensive_areas_pos(self.team, map_layout_obs)

        max_potential = -400
        for area_pos in defensive_areas_pos:
            p = -cityblock(agent_pos, area_pos)
            max_potential = max(max_potential, p)
        
        return p

    def potential_function(self, role: int, observation: list, map_layout_obs: dict) -> int:
        if role == self.ATTACKER:
            return self.potential_function_attacker(observation, map_layout_obs)
        else:
            return self.potential_function_defender(observation, map_layout_obs)
    
    def role_assignment(self):
        """
        Compute algorithm for role assignment in every agent node and assign its role.
        """

        agents_assigned = []
        team_roles = [None]*self.n_agents
    
        for role_i in range(self.n_roles):
            max_agent_i = None
            max_potential = -999
            for team_agent_i in range(self.n_agents):
                if team_agent_i not in agents_assigned and self.team_potentials[role_i][team_agent_i] > max_potential:
                    max_potential = self.team_potentials[role_i][team_agent_i]
                    max_agent_i = team_agent_i

            if max_agent_i != None:
                team_roles[max_agent_i] = self.ROLES[role_i]
                agents_assigned.append(max_agent_i)

        last_role = self.agent_role
        self.agent_role = team_roles[self.team_agent_index]

        if last_role != self.agent_role:
            self.new_role = True
    
    def assign_team_fixed_roles(self):
        for i in range(len(self.ROLES)):
            if self.team_agent_index == i:
                self.agent_role = self.ROLES[i]

    def communicate_roles(self, team: list, observation: list, map_layout_obs: dict):
        for role in self.ROLES:
            potential = self.potential_function(role, observation, map_layout_obs)
            self.broadcast(role, team, potential)

    def broadcast(self, role: int, team: list, potential: int):
        for agent in team:
            agent.receive_agent_potential(role, self.team_agent_index, potential)

    def receive_agent_potential(self, role: int, agent_i: int, potential: int):
        self.team_potentials[role][agent_i] = potential

    '''
    ------------------------------------------------------
    ------ auxiliar functions to role computation --------
    ------------------------------------------------------
    '''


    def is_attacker(self) -> bool:
        return self.agent_role == self.ATTACKER
    
    def is_defender(self) -> bool:
        return self.agent_role == self.DEFENDER

    def get_agent_pos(self, observation: dict):
        return (observation[0][0], observation[0][1])

    def attacker_has_flag(self, observation: dict):
        return observation[0][3] != None

    def get_enemies_state(self, color:str , observation: dict):
        if color == 'Red':
            return observation[1]
        return observation[2]

    def get_team_state(self, color:str , observation: dict):
        if color == 'Red':
            return observation[2]
        return observation[1]

    def get_enemy_defensive_areas_pos(self, color: str, map_layout_obs: dict):
        if color == 'Red':
            return map_layout_obs['defensive_blue_area']
        return map_layout_obs['defensive_red_area']

    def get_enemy_pos(self, enemy_state: list):
        return (enemy_state[0], enemy_state[1])

    def enemy_has_flag(self, enemy_state: list):
        return enemy_state[3] != None

    def get_defensive_areas_pos(self, color: str, map_layout_obs: dict):
        if color == 'Red':
            return map_layout_obs['defensive_red_area']
        return map_layout_obs['defensive_blue_area']
    
    '''
    ------------------------------------------------------
    ------ auxiliar functions to deliberate agent --------
    ------------------------------------------------------
    '''

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

    #FIXME would be a good idea to get respawn by order of close locality
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

    '''
    ------------------------------------------------------
    ---------------- deliberative agent ------------------
    ------------------------------------------------------
    '''
    
    def options(self):
        """
        Form agent desires from its beliefs.
        """

        if self.agent_role == self.ATTACKER:
            # Deliver current flag
            if self.has_flag():
                self.desires.append((self.DELIVER, "team_delivery_area"))
                return

            # Steal enemy flag from enemy defensive area if there is no enemy defending it
            # exception goes to when agent is already in position for stealing the flag
            for index, _ in enumerate(self.enemy_flags_pos):
                flag_pos = self.get_enemy_flag_pos(index)
                if not self.enemy_in_nearby_position(flag_pos) and flag_pos in self.enemy_defensive_areas_pos:
                    index = self.enemy_defensive_areas_pos.index(flag_pos)
                    self.desires.append((self.STEAL, "enemy_defensive_area", index))

            # Steal enemy flag from enemy that is not defending an area
            for index, enemy_state in enumerate(self.enemies_state):
                if self.enemy_has_enemy_flag(enemy_state) and not self.get_pos(enemy_state) in self.team_defensive_areas_pos:
                    self.desires.append((self.STEAL, "enemy", index))
            
            # Go to enemy defensive area if there isn't an enemy nearby
            go_to = False
            for area in range(3):
                if not self.enemy_in_nearby_position(self.enemy_defensive_areas_pos[area*9 + 4]):
                    go_to = True
                    self.desires.append((self.GO_TO_ENEMY_DEFENSIVE_AREAS, area*9 + 4))
            
            # Otherwise, if no enemy nearby, go to the middle team defensive area
            if not go_to:
                self.desires.append((self.GO_TO_ENEMY_DEFENSIVE_AREAS, self.defensive_area))

        elif self.agent_role == self.DEFENDER:
            # Deliver current flag
            if self.has_flag():
                respawn = self.get_free_team_flag_respawn()

                if respawn == None:
                    raise Exception(f"Agent {self.agent_id}: there is no team defensive area position to deliver a recovered flag")

                index = self.team_flag_respawns.index(respawn)
                self.desires.append((self.DELIVER, "team_defensive_area", index))
                return
            
            # Steal team flag from enemy
            for index, enemy_state in enumerate(self.enemies_state):
                if self.enemy_has_team_flag(enemy_state):
                    self.desires.append((self.STEAL, "enemy", index))

            # Freeze enemy if he is not already frozen and is in team defensive area
            for index, enemy_state in enumerate(self.enemies_state):
                if not self.agent_is_frozen(enemy_state) and self.get_pos(enemy_state) in self.team_defensive_areas_pos:
                    self.desires.append((self.FREEZE, index))

            # Go to team defensive area if there is an enemy nearby
            go_to = False
            for area in range(3):
                for enemy_state in self.enemies_state:
                    if self.get_pos(enemy_state) in self.team_defensive_areas_pos[area*9:(area+1)*9]:
                        self.desires.append((self.GO_TO_TEAM_DEFENSIVE_AREAS, area*9 + 4))
                        go_to = True
            
            # Otherwise, if no enemy nearby, go to the middle team defensive area
            if not go_to:
                #self.area = (self.area + 9) % 27
                self.desires.append((self.GO_TO_TEAM_DEFENSIVE_AREAS, self.area))
        else:
            raise Exception(f'Agent {self.agent_id}: agent must have a role in order to form desires')
        
    def plan(self):
        """
        Compute every action needed to perform the filtered intention.
        """

        plan = list()

        dest_pos = self.intention[1]
        location = self.agent_pos

        self.path = []
        self.obstacles = []
        
        while dest_pos != location:
            self.agent_plan = list()
            self.agent_plan.append(self.NOOP)
            
            action, location = self.advance_to_pos(location, dest_pos)

            if action == None:
                break

            self.path.append(location)
            plan.append(action)

        # agent performs no operation after arriving to destination 'go to' area
        if self.intention[0] in [self.GO_TO_ENEMY_DEFENSIVE_AREAS, self.GO_TO_TEAM_DEFENSIVE_AREAS]:
            for _ in range(3):
                plan.append(self.NOOP)

        # agent performs freeze or steal before arriving to position, when close to the enemy
        elif self.intention[0] == self.FREEZE or self.intention[0] == self.STEAL and self.intention[2] == 'enemy':
            plan[-1] = self.intention[0]

        # otherwise, last agent action correponds to its intention
        else:
            plan.append(self.intention[0])

        self.selected_intention = self.intention
        self.agent_plan = plan

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

    def succeeded(self) -> bool:
        dest = self.intention[1]

        if dest == self.agent_pos:
            return True

        # Agents with freeze or steal enemy intention only need to be close (up, down, left, right) to succeed
        if self.are_close_positions(dest, self.agent_pos):
            if self.intention[0] == self.FREEZE or self.intention[0] == self.STEAL and self.intention[2] == "enemy":
                return True

        return False

    def impossible(self) -> bool:
        dest = self.intention[1]
        action = self.agent_plan[0]

        if self.new_role:
            self.new_role = False
            return True

        if self.is_frozen():
            return True

        # agent tries to deliver in the wrong position
        if action == self.DELIVER and self.agent_pos != dest:
            return True

        # defender agent tries to freeze and agent isn't close
        if action == self.FREEZE and not self.enemies_in_close_position(dest):
            return True

        # agent with flag wants to move to position where there is already a flag
        if self.has_flag():
            if action == self.DOWN and self.flag_in_pos((self.agent_pos[0]+1, self.agent_pos[1])):
                return True
            if action == self.UP and self.flag_in_pos((self.agent_pos[0]-1, self.agent_pos[1])):
                return True
            if action == self.LEFT and self.flag_in_pos((self.agent_pos[0], self.agent_pos[1]-1)):
                return True
            if action == self.LEFT and self.flag_in_pos((self.agent_pos[0], self.agent_pos[1]+1)):
                return True
        
        return False

    def reconsider(self) -> bool:
        action = self.agent_plan[0]
        intention = self.intention[0]
        dest = self.intention[1]

        if self.agent_role == self.ATTACKER:

            # attacker agent is trying to steal a flag or going to enemy area and there is an enemy defending it
            # exception goes to when agent is already in position for stealing the flag
            if self.intention[0] == self.STEAL and self.agent_pos != self.intention[1] or self.intention[0] == self.GO_TO_ENEMY_DEFENSIVE_AREAS:
                if self.enemy_in_nearby_position(self.intention[1]):
                    return True

            # flag got stolen
            if intention == self.DELIVER and not self.has_flag():
                return True

        elif self.agent_role == self.DEFENDER:

            # defender agent planning on going to team defensive area and finds an enemy with a flag in the way
            if intention == self.GO_TO_TEAM_DEFENSIVE_AREAS:
                for enemy_state in self.enemies_state:
                    if self.enemy_has_team_flag(enemy_state):
                        return True

            # defender agent planning on delivering in a team area that already contains another flag
            if intention == self.DELIVER:
                for flag_state in self.team_flags_pos:
                    if self.get_flag_pos(flag_state) == dest and self.get_flag_id(flag_state) != self.flag_id:
                        return True
        

        # 'go to' position is already occupied
        if intention in [self.GO_TO_ENEMY_DEFENSIVE_AREAS, self.GO_TO_TEAM_DEFENSIVE_AREAS] and self.agent_in_pos(dest):
            return True

        # target agent has moved
        if intention == self.FREEZE or intention == self.STEAL and self.intention[2] == 'enemy' and not self.agent_in_pos(dest):
            return True

        # next position is occupied
        # FIXME should be in impossible actions! but it doesn't work :(
        if action == self.DOWN and self.enemies_in_pos((self.agent_pos[0]+1, self.agent_pos[1])):
            return True
        elif action == self.UP and self.enemies_in_pos((self.agent_pos[0]-1, self.agent_pos[1])):
            return True
        elif action == self.LEFT and self.enemies_in_pos((self.agent_pos[0], self.agent_pos[1]-1)):
            return True
        elif action == self.RIGHT and self.enemies_in_pos((self.agent_pos[0], self.agent_pos[1]+1)):
            return True

    def sound(self, action) -> bool:
        return self.selected_intention == self.intention