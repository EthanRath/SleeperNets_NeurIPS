## Setup

First install requirements for cleanrl atari, box2d, and mujoco https://docs.cleanrl.dev/

Ensure you're using the version with gymnasium==0.28.1

Install:
- safety-gymnasium==1.2.1 (https://github.com/PKU-Alignment/safety-gymnasium)
- highway-env==1.9.1
- gym-trading-env==0.3.3
- torch==1.12.1
- numpy==1.23.5

## Running the Code 

|              Environment             |   Task Type   |     Observations     |     Environment Id.    | Training Tag |
|:------------------------------------:|:-------------:|:--------------------:|:----------------------:|--------------|
|  Breakout  |   Video Game  |         Image        | BreakoutNoFrameskip-v4 | --atari |
|   Q*bert   |   Video Game  |         Image        |   QbertNoFrameskip-v4  | --atari |
| Car Racing |   Video Game  |         Image        |      CarRacing-v2      | N/a |
|   Highway Merge   |  Self Driving |         Image        |        merge-v0        | --highway |
|  Safety Car  |    Robotics   | Lidar+Proprioceptive |    SafetyCarGoal1-v0   | --safety |
|     Trade BTC     | Stock Trading |      Stock Data      |       TradingEnv       | --trade |

### Examples on Atari
Run the following for SleeperNets:
python ppo.py --atari --env_id BreakoutNoFrameskip-v4 --sn_outer --p_rate .0003 --target_action 2 --total_timesteps 20000000

Run the following for TrojDRL:
python ppo.py --atari --env_id BreakoutNoFrameskip-v4 --trojdrl --p_rate .0003 --target_action 2 --total_timesteps 20000000

Run the following for BadRL:
python ppo.py --atari --env_id BreakoutNoFrameskip-v4 --badrl --strong --p_rate .0003 --target_action 2 --total_timesteps 20000000

## Export Results to CSV

python write_csv.py