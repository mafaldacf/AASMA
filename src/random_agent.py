import numpy as np

from agent import Agent

class RandomAgent(Agent):

    def __init__(self, n_actions: int, team: str):
        super(RandomAgent, self).__init__("Random Agent", team)
        self.n_actions = n_actions
        self.id = 0

    def get_id(self) -> int:
        return self.id

    def action(self) -> int:
        return np.random.randint(self.n_actions)