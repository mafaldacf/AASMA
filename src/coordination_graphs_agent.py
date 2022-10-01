import numpy as np

from deliberative_ctf_agent import DeliberativeAgent

class CoordinationGraphsAgent(DeliberativeAgent):

    def __init__(self, agent_id: int, team: str , n_agents_team: int, payoffs:list):
        super(CoordinationGraphsAgent, self).__init__(agent_id, team)
        self.n_agents = n_agents_team
        self.payoffS = payoffs
        self.agent_list = list(range(n_agents_team))
        self.arcs = list()

    def make_coordination_graphs(self):
        for i in range(self.n_agents-1):
            self.arcs.append((i,i+1))

    def variable_elimination(self):
        return

    def options(self):
        return

    def plan(self):
        return

    def reconsider(self) -> bool:
        return

    def sound(self, action:int) ->bool:
        return

    def succeeded(self) -> bool:
        return

    def impossible(self) -> bool:
        return
