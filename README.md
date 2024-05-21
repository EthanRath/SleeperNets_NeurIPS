# SleeperNets Code in Rough Research Form

First install requirements for cleanrl-atari https://docs.cleanrl.dev/

Current code is not compatable with HighwayEnv, TradingEnv, or SafetyEnv out of the box. A modified version with the proper wrappers and instructions will be provided by the rebuttal period.

Run the following for SleeperNets:
python ppo.py --atari --env_id BreakoutNoFrameskip-v4 --sn_outer --p_rate .0003 --target_action 2 --total_timesteps 20000000

Run the following for TrojDRL:
python ppo.py --atari --env_id BreakoutNoFrameskip-v4 --trojdrl --p_rate .0003 --target_action 2 --total_timesteps 20000000

Run the following for BadRL:
python ppo.py --atari --env_id BreakoutNoFrameskip-v4 --badrl --strong --p_rate .0003 --target_action 2 --total_timesteps 20000000
