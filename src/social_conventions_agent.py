from deliberative_ctf_agent import DeliberativeAgent


class SocialConventionAgent(DeliberativeAgent):
    
    def __init__(self, agent_id: int, team: str , n_agents_team: int, agent_order: list, action_order:list, n_blue_flags: int, n_red_flags: int):
        super(SocialConventionAgent, self).__init__(agent_id, team)
        self.n_agents = n_agents_team
        self.agent_order = agent_order
        self.action_order = action_order
        self.n_blue_flags = n_blue_flags
        self.n_red_flags = n_red_flags
        self.n_flags = n_red_flags + n_blue_flags

        self.team_agent_index = self.agent_id
        self.area = 4
        self.plus = 9
        if team == 'R': 
            self.team_agent_index -= n_agents_team


    def social_convention(self):
        '''
            Social Convention with agents

        '''
        index = self.agent_order.index((self.agent_id)%self.n_agents)
        return self.action_order[index]


    '''
    ------------------------------------------------------
    ---------------- deliberate agent --------------------
    ------------------------------------------------------
    '''

    def options(self):
        # <Desire to DELIVER the flag>
        if self.has_flag() and not self.flag_of_team():
            self.desires.append((self.DELIVER, "team_delivery_area"))

        elif self.has_flag() and self.flag_of_team():
            respawn = self.get_free_team_flag_respawn()
            if respawn == None:
                raise Exception(f"Agent {self.agent_id}: there is no team defensive area position to deliver a recovered flag")

            index = self.team_flag_respawns.index(respawn)
            self.desires.append((self.DELIVER, "team_defensive_area", index))

        # <Social Conversion>
        final_action = self.social_convention()

        go_to = False

        # <Desire to STEAL>
        if final_action == self.STEAL:
            # Steal enemy flag from enemy defensive area if there is no enemy defending it
            for flag_state in self.enemy_flags_pos:
                flag_pos = (flag_state[0], flag_state[1])
                if (self.agent_pos == flag_pos or not self.enemy_in_nearby_position(flag_pos)) and flag_pos in self.enemy_defensive_areas_pos:
                    index = self.enemy_defensive_areas_pos.index(flag_pos)
                    self.desires.append((self.STEAL, "enemy_defensive_area", index))

            # Steal enemy flag from enemy that is not defending an area
            for index, enemy_state in enumerate(self.enemies_state):
                if self.enemy_has_enemy_flag(enemy_state):
                    self.desires.append((self.STEAL, "enemy", index))

            # Go to enemy defensive area if there isn't an enemy nearby
            for pos in range(3):
                if not self.enemy_in_nearby_position(self.enemy_defensive_areas_pos[pos * 9 + 4]):
                    go_to = True
           # <Desire to GO_TO_ENEMY_DEFENSIVE_AREAS>
                    self.desires.append((self.GO_TO_ENEMY_DEFENSIVE_AREAS, pos * 9 + 4))

             # Otherwise, if enemy nearby, go to one of team defensive areas


        # <Desire to FREEZE>
        else:
            # Freeze enemy if he is not already frozen and is in team defensive area
            for index, enemy_state in enumerate(self.enemies_state):
                if not self.agent_is_frozen(enemy_state)  and self.get_pos(enemy_state) in self.team_defensive_areas_pos:
                    self.desires.append((self.FREEZE, index))

            # Go to team defensive area if there is an enemy nearby
            for pos in range(3):
                for enemy_state in self.enemies_state:
                    if self.get_pos(enemy_state) in self.team_defensive_areas_pos[pos*9:(pos+1)*9]:
                        go_to = True
                        self.desires.append((self.GO_TO_TEAM_DEFENSIVE_AREAS, pos*9 + 4))
        
            # <Desire to GO_TO_TEAM_DEFENSIVE_AREAS>
            # Otherwise, if no enemy nearby, go to one of team defensive areas

        if not go_to:
            self.area = (self.area + 9) % 27
            self.desires.append((self.GO_TO_TEAM_DEFENSIVE_AREAS, self.area))



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

        if self.intention[0] == self.GO_TO_ENEMY_DEFENSIVE_AREAS:
            plan.append(self.NOOP)

        elif self.intention[0] == self.GO_TO_TEAM_DEFENSIVE_AREAS:
            for i in range(3):
                plan.append(self.NOOP)

        elif self.intention[0] == self.FREEZE or self.intention[0] == self.STEAL and self.intention[2] == 'enemy':
            plan[-1] = self.intention[0]

        else:
            plan.append(self.intention[0])

        self.agent_plan = plan
        self.selected_intention = self.intention


    def reconsider(self) -> bool:
        action = self.agent_plan[0]
        intention = self.intention[0]
        dest = self.intention[1]

        if (intention == self.STEAL and self.agent_pos != self.intention[1]) or intention == self.GO_TO_ENEMY_DEFENSIVE_AREAS or intention == self.GO_TO_TEAM_DEFENSIVE_AREAS:
            if self.enemy_in_nearby_position(self.intention[1]):
                return True

        if intention == self.DELIVER:
            if not self.has_flag():
                return True
            elif self.intention[2] == "team_defensive_area":
                for flag_state in self.team_flags_pos:
                    if self.get_flag_pos(flag_state) == dest and self.get_flag_id(flag_state) != self.flag_id:
                        return True
        
        # 'go to' position is already occupied
        if intention in [self.GO_TO_ENEMY_DEFENSIVE_AREAS, self.GO_TO_TEAM_DEFENSIVE_AREAS] and self.agent_in_pos(dest):
            return True

         # target agent has moved
        if intention == self.FREEZE or intention == self.STEAL and self.intention[2] == 'enemy' and not self.agent_in_pos(dest):
            return True

        if action == self.DOWN and self.enemies_in_pos((self.agent_pos[0]+1, self.agent_pos[1])):
            return True
        elif action == self.UP and self.enemies_in_pos((self.agent_pos[0]-1, self.agent_pos[1])):
            return True
        elif action == self.LEFT and self.enemies_in_pos((self.agent_pos[0], self.agent_pos[1]-1)):
            return True
        elif action == self.RIGHT and self.enemies_in_pos((self.agent_pos[0], self.agent_pos[1]+1)):
            return True

        return False

    def sound(self, action) -> bool:
        return self.selected_intention == self.intention


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

        # agent tries to deliver in the wrong position
        if action == self.DELIVER and self.agent_pos != dest:
            return True

        # agent tries to freeze and agent isn't close
        if action == self.FREEZE and not self.enemies_in_close_position(dest):
            self.area = (self.area - self.plus)%27
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