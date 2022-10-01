import numpy as np
from scipy.spatial.distance import cityblock
from role_agent import RoleAgent
from social_conventions_agent import SocialConventionAgent
import time

ROLE_MODE = 0
SOCIAL_CONVENTIONS_MODE = 1
SUCCESSIVE_ROLES = 10
MAX_SOCIAL_CONVENTIONS_TIMESTEP = 30

class RoleSocialConventionAgent(RoleAgent):
    def __init__(self, agent_id: int, team: str, n_agents_team: int, roles: list, agent_order: list, action_order: list, n_blue_flags: int, n_red_flags: int):
        super(RoleSocialConventionAgent, self).__init__(agent_id, team, n_agents_team, roles, n_blue_flags, n_red_flags)

        # Role with Social Convention
        self.mode = ROLE_MODE
        self.latest_roles = [] # saves previous roles
        self.social_conventions_timestep = 0
        self.agent_order = agent_order
        self.action_order = action_order

    def social_convention(self):
        '''
            Social Convention with agents

        '''
        index = self.agent_order.index((self.agent_id)%self.n_agents)
        return self.action_order[index]

    def options(self):
        if self.mode == ROLE_MODE:
            RoleAgent.options(self)
        else:
            SocialConventionAgent.options(self)

    def reconsider(self) -> bool:
        if self.mode == ROLE_MODE:
            RoleAgent.reconsider(self)
        else:
            SocialConventionAgent.reconsider(self)
    
    def role_assignment(self):
        """
        Compute algorithm for role assignment in every agent node and assign its role.
        """
        if self.mode == ROLE_MODE:
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

            # Role Agent with Social Conventions
            self.latest_roles.append(self.agent_role)

    def communicate_roles(self, team: list, observation: list, map_layout_obs: dict):
        # Role Agent with Social Conventions

        if self.mode == SOCIAL_CONVENTIONS_MODE and self.social_conventions_timestep >= MAX_SOCIAL_CONVENTIONS_TIMESTEP:
            self.mode == ROLE_MODE
            self.latest_roles = []
            self.social_conventions_timestep = 0

        elif self.mode == ROLE_MODE:
            dillema = False
            if len(self.latest_roles) > SUCCESSIVE_ROLES:
                for i in range(SUCCESSIVE_ROLES):
                    if self.latest_roles[-1-i] != self.latest_roles[-2-i]:
                        dillema = True

                if dillema == True:
                    self.social_conventions_timestep = 0
                    self.latest_roles = []
                    self.mode = SOCIAL_CONVENTIONS_MODE

        if self.mode == ROLE_MODE:
            for role in self.ROLES:
                potential = self.potential_function(role, observation, map_layout_obs)
                self.broadcast(role, team, potential)
        
        if self.mode == SOCIAL_CONVENTIONS_MODE:
            self.social_conventions_timestep += 1

    def broadcast(self, role: int, team: list, potential: int):
        for agent in team:
            agent.receive_agent_potential(role, self.team_agent_index, potential)

    def receive_agent_potential(self, role: int, agent_i: int, potential: int):
        self.team_potentials[role][agent_i] = potential