import argparse
import time
import numpy as np

from gym import Env
from typing import Sequence
from utils import compare_results
from env_capture_the_flag import EnvCaptureTheFlag
from agent import Agent
from random_agent import RandomAgent
from social_conventions_agent import SocialConventionAgent
from role_agent import RoleAgent
from coordination_graphs_agent import CoordinationGraphsAgent
from role_social_conventions_agent import RoleSocialConventionAgent

ACTIONS = 8
DOWN, LEFT, UP, RIGHT, NOOP, STEAL, FREEZE, DELIVER = range(ACTIONS)

FIXED_ROLE = "Fixed Role Team"
ROLE = "Role Team"
ROLE_SOCIAL_CONVENTIONS = "Role with Social Conventions Team"

ROLES = 2
ATTACKER, DEFENDER = range(ROLES)

def communicate_team_roles(team_agents: list, observations: list, map_layout_obs: dict):
    for agent in team_agents: # in parallel
        agent.communicate_roles(team_agents, observations[agent.get_id()], map_layout_obs)

    for agent in team_agents: # in parallel
        agent.role_assignment()

def communicate_roles(blue_agents: list, red_agents: list, observations: list, map_layout_obs: dict):
    communicate_team_roles(blue_agents, observations, map_layout_obs)
    communicate_team_roles(red_agents, observations, map_layout_obs)

def assign_team_fixed_roles(team_agents: list):
    for agent in team_agents:
        agent.assign_team_fixed_roles()

def run_multi_agent(environment: Env, agents: Sequence[Agent], game_type: str, blue_team_type: str, red_team_type: str, n_episodes: int, demo: bool, render_sleep_time: float) -> np.ndarray:

    results_steps = np.zeros(n_episodes)

    results_blue_stolen_flags = np.zeros(n_episodes)
    results_red_stolen_flags = np.zeros(n_episodes)

    results_blue_recovered_flags = np.zeros(n_episodes)
    results_red_recovered_flags = np.zeros(n_episodes)

    results_blue_delivered_flags = np.zeros(n_episodes)
    results_red_delivered_flags = np.zeros(n_episodes)

    results_blue_frozen_agents = np.zeros(n_episodes)
    results_red_frozen_agents = np.zeros(n_episodes)

    n_agents = environment.n_agents
    n_blue_agents = environment.n_blue_agents
    blue_agents = agents[:n_blue_agents]
    red_agents = agents[n_blue_agents:]

    blue_wins = 0
    red_wins = 0

    for episode in range(n_episodes):

        steps = 0

        terminals = [False for _ in range(n_agents)]
        observations, map_layout_obs = environment.reset()

        if blue_team_type == FIXED_ROLE:
            assign_team_fixed_roles(blue_agents)
        if red_team_type == FIXED_ROLE:
            assign_team_fixed_roles(red_agents)

        if blue_team_type in [ROLE, ROLE_SOCIAL_CONVENTIONS]:
            communicate_team_roles(blue_agents, observations, map_layout_obs)
        if red_team_type in [ROLE, ROLE_SOCIAL_CONVENTIONS]:
            communicate_team_roles(red_agents, observations, map_layout_obs)


        if demo:
            print(f"[{game_type}] Episode {episode}")
            environment.render()
            time.sleep(render_sleep_time)

        while not all (terminals):
            steps += 1
            actions = []
            for i in range(n_agents):
                agents[i].see(observations[i],map_layout_obs)
                actions.append(agents[i].action())
            observations, terminals, score = environment.step(actions)

            if blue_team_type == ROLE_SOCIAL_CONVENTIONS:
                communicate_team_roles(blue_agents, observations, map_layout_obs)
            if red_team_type == ROLE_SOCIAL_CONVENTIONS:
                communicate_team_roles(red_agents, observations, map_layout_obs)

            if demo:
                print(f"[{game_type}] Timestep {steps}")
                environment.render()
                time.sleep(render_sleep_time)

            if score[0] == environment.winning_points:
                if demo:
                    print(f"Team BLUE won! Score: BLUE ({score[0]}) - RED ({score[1]}). Steps = {steps}")
                blue_wins += 1
                break
            if score[1] == environment.winning_points:
                if demo:
                    print(f"Team RED won! Score: BLUE ({score[0]}) - RED ({score[1]}). Steps = {steps}")
                red_wins += 1
                break

        stolen_flags, recovered_flags, delivered_flags, frozen_agents = environment.get_actions_metrics()
        

        results_steps[episode] = steps

        results_blue_stolen_flags[episode] = stolen_flags['B']
        results_red_stolen_flags[episode] = stolen_flags['R']

        results_blue_recovered_flags[episode] = recovered_flags['B']
        results_red_recovered_flags[episode] = recovered_flags['R']

        results_blue_delivered_flags[episode] = delivered_flags['B']
        results_red_delivered_flags[episode] = delivered_flags['R']

        results_blue_frozen_agents[episode] = frozen_agents['B']
        results_red_frozen_agents[episode] = frozen_agents['R']

        environment.close()

    # Arrays with one element
    results_blue_wins = np.array([blue_wins])
    results_red_wins = np.array([red_wins])

    results_wins = [results_blue_wins, results_red_wins]
    results_stolen_flags = [results_blue_stolen_flags, results_red_stolen_flags]
    results_recovered_flags = [results_blue_recovered_flags, results_red_recovered_flags]
    results_delivered_flags = [results_blue_delivered_flags, results_red_delivered_flags]
    results_frozen_agents = [results_blue_frozen_agents, results_red_frozen_agents]

    return [results_steps, results_wins, results_stolen_flags, results_recovered_flags, results_delivered_flags, results_frozen_agents]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--demo", type=bool, default=False)
    parser.add_argument("--render-sleep-time", type=float, default=0.05)
    opt = parser.parse_args()

    # 1 - Setup the environment
    environment = EnvCaptureTheFlag(grid_shape=(19, 19), cell_size=26, n_blue_agents=3, n_red_agents=3, max_steps=1000, 
        n_blue_flags=3, n_red_flags=3, agent_view_mask=(5, 5), freeze_time=5, winning_points=10)

    # 2 - Setup the teams

    agent_order = [0, 1, 2]
    action_order = [STEAL, STEAL, FREEZE]
    roles_order = [ATTACKER, DEFENDER, DEFENDER]
    payoffs = [(-1,-1),(2,2),(-2,-2),(2,2)] # (STEAL,STEAL), (STEAL,FREEZE), (FREEZE,FREEZE), (STEAL,STEAL)

    teams_types = []


    # It's important to set teams type according to the different games' order
    teams_types.append([ROLE, FIXED_ROLE])
    teams_types.append([ROLE_SOCIAL_CONVENTIONS, ROLE])


    games_type = {
        "Role Team vs Fixed Role Team": [
            RoleAgent(agent_id=0, team='B', n_agents_team=3, roles=roles_order, n_blue_flags=3, n_red_flags=3),
            RoleAgent(agent_id=1, team='B', n_agents_team=3, roles=roles_order, n_blue_flags=3, n_red_flags=3),
            RoleAgent(agent_id=2, team='B', n_agents_team=3, roles=roles_order, n_blue_flags=3, n_red_flags=3),
            RoleAgent(agent_id=3, team='R', n_agents_team=3, roles=roles_order, n_blue_flags=3, n_red_flags=3),
            RoleAgent(agent_id=4, team='R', n_agents_team=3, roles=roles_order, n_blue_flags=3, n_red_flags=3),
            RoleAgent(agent_id=5, team='R', n_agents_team=3, roles=roles_order, n_blue_flags=3, n_red_flags=3),
        ],

        "Role Social Conventions Team vs Role Team": [
            RoleSocialConventionAgent(0, 'B', 3, roles_order, agent_order, action_order, 3, 3),
            RoleSocialConventionAgent(1, 'B', 3, roles_order, agent_order, action_order, 3, 3),
            RoleSocialConventionAgent(2, 'B', 3, roles_order, agent_order, action_order, 3, 3),
            RoleAgent(agent_id=3, team='R', n_agents_team=3, roles=roles_order, n_blue_flags=3, n_red_flags=3),
            RoleAgent(agent_id=4, team='R', n_agents_team=3, roles=roles_order, n_blue_flags=3, n_red_flags=3),
            RoleAgent(agent_id=5, team='R', n_agents_team=3, roles=roles_order, n_blue_flags=3, n_red_flags=3),
        ],
        
        #"Random Team": [
        #    RandomAgent(environment.action_space[0].n, 'B'),
        #    RandomAgent(environment.action_space[1].n, 'B'),
        #    RandomAgent(environment.action_space[2].n, 'R'),
        #    RandomAgent(environment.action_space[3].n, 'R'),
        #],
        #        
        #"Social Convention Agent Team": [
        #    SocialConventionAgent(0, 'B', environment.n_blue_agents, agent_order, action_order,environment.n_blue_flags,environment.n_red_flags),
        #    SocialConventionAgent(1, 'B', environment.n_blue_agents, agent_order, action_order,environment.n_blue_flags,environment.n_red_flags),
        #    SocialConventionAgent(2, 'R', environment.n_red_agents, agent_order, action_order,environment.n_blue_flags,environment.n_red_flags),
        #    SocialConventionAgent(3, 'R', environment.n_red_agents, agent_order, action_order,environment.n_blue_flags,environment.n_red_flags),
        #],

       # "Coordination Graphs Agent Team": [
        #    CoordinationGraphsAgent(0, 'B', environment.n_blue_agents,payoffs),
         #   CoordinationGraphsAgent(1, 'B', environment.n_blue_agents,payoffs),
          #  CoordinationGraphsAgent(2, 'R', environment.n_red_agents,payoffs),
        #    CoordinationGraphsAgent(3, 'R', environment.n_red_agents,payoffs),
        #],

    }

    # 3.1 - Evaluation for different game types
    results_steps = {}
    
    # 3.2 - Evaluation inside game type
    results_wins = {}
    results_stolen_flags = {}
    results_recovered_flags = {}
    results_delivered_flags = {}
    results_frozen_agents = {}

    i = 0
    for game_type, agents in games_type.items():
        print("Running demo for", game_type)
        environment.set_game_type(game_type)

        # results = [results_steps, results_wins, results_stolen_flags, results_recovered_flags, results_delivered_flags, results_frozen_agents]
        results = run_multi_agent(environment, agents, game_type, teams_types[i][0], teams_types[i][1], opt.episodes, opt.demo, opt.render_sleep_time)

        results_steps[game_type] = results[0]

        results_wins["Blue Team"] = results[1][0]
        results_wins["Red Team"] = results[1][1]

        results_stolen_flags["Blue Team"] = results[2][0]
        results_stolen_flags["Red Team"] = results[2][1]
        
        results_recovered_flags["Blue Team"] = results[3][0]
        results_recovered_flags["Red Team"] = results[3][1]       

        results_delivered_flags["Blue Team"] = results[4][0]
        results_delivered_flags["Red Team"] = results[4][1]

        results_frozen_agents["Blue Team"] = results[5][0]
        results_frozen_agents["Red Team"] = results[5][1]


        # 4 - Compare results
        compare_results(
            results_wins,
            title= game_type,
            metric="Wins",
            colors=["blue", "red"]
        )

        compare_results(
            results_stolen_flags,
            title= game_type,
            metric="Stolen Flags",
            colors=["blue", "red"]
        )

        compare_results(
            results_recovered_flags,
            title= game_type,
            metric="Recovered Flags",
            colors=["blue", "red"]
        )

        compare_results(
            results_delivered_flags,
            title= game_type,
            metric="Delivered Flags",
            colors=["blue", "red"]
        )

        compare_results(
            results_frozen_agents,
            title= game_type,
            metric="Frozen Agents",
            colors=["blue", "red"]
        )

    compare_results(
            results_steps,
            title= "Teams Comparison on 'Capture The Flag' Environment",
            metric="Avg. Steps per Episode",
            colors=["green", "orange"]
        )

