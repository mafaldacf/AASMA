import numpy as np
from abc import ABC, abstractmethod


class Agent(ABC):

    """
    Base agent class.
    Represents the concept of an autonomous agent.
    """

    def __init__(self, name: str, team: str):
        self.name = name
        self.observation = None
        self.team = team

    def see(self, observation: np.ndarray, map_layout_obs: dict):
        self.observation = observation
        self.map_layout_obs = map_layout_obs

    def get_id(self) -> int:
        return self.id

    @abstractmethod
    def action(self) -> int:
        raise NotImplementedError()

    def communicate_roles(self, team: list, observation: list, map_layout_obs: dict):
        return 0 # should not be implemented...just to ignore compiler errors while we use random agents

    def role_assignment(self, agent_i: int):
        return 0 # should not be implemented...just to ignore compiler errors while we use random agents