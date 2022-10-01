1. Create virtual environment

    (Linux)
    
    $ python3 -m venv venv
    
    $ source venv/bin/activate

    (Windows)
    
    python -m venv env
    
    env\Scripts\Activate.ps1 

2. Install dependencies

    $ pip install -r requirements.txt

3. Run the system

    $ python game_simulation.py [-h] [--episodes EPISODES] [--demo DEMO] [--render-sleep-time RENDER_SLEEP_TIME]

    

optional arguments:
   
  -h, --help  # ask for help    
    
  --episodes EPISODES # number of episodes/games for each game type (Role Team vs Fixed Role Team and Role Social Conventions Team vs Role Team). Default is 30 episodes
    
  --demo DEMO # if DEMO == True, show the simulated Capture The Flag game for each game type. Default is False
    
  --render-sleep-time RENDER_SLEEP_TIME #allows agents to slow down or speed up. Default is 0.05
