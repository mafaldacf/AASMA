Instituto Superior Técnico

Master's Degree in Computer Science and Engineering

Autonomous Agents and Multi-Agent Systems 2021/2022

# Capture the Flag, a multiagent system environment

In this project, the game Capture The Flag will be developed. This game is a multiagent system where the environment is dynamic and unpredictable. For that reason, the agents have to be reactive, have the capability to learn and cooperate with others. This is what we call multi-agent learning: many individual agents must act independently, yet learn to interact and cooperate with other agents. With this game, agents can learn teamwork, an ability needed to solve various real-word problems.

## Authors

**Group 13**

92513 Mafalda Ferreira

92546 Rita Oliveira

102141 Francisco Cabrinha

## Setting up the environment and run 

1. Create virtual environment

(Linux)
    
    python3 -m venv venv
    
    source venv/bin/activate

(Windows)
    
    python -m venv env
    
    env\Scripts\Activate.ps1 

2. Install dependencies

        pip install -r requirements.txt

3. Run the system

        python game_simulation.py [-h] [--episodes EPISODES] [--demo DEMO] [--render-sleep-time RENDER_SLEEP_TIME]


optional arguments:
   
  -h, --help  # ask for help    
    
  --episodes EPISODES # number of episodes/games for each game type (Role Team vs Fixed Role Team and Role Social Conventions Team vs Role Team). Default is 30 episodes
    
  --demo DEMO # if DEMO == True, show the simulated Capture The Flag game for each game type. Default is False
    
  --render-sleep-time RENDER_SLEEP_TIME #allows agents to slow down or speed up. Default is 0.05
