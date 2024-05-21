# SleeperNets_NeurIPS

First install requirements for cleanrl-atari https://docs.cleanrl.dev/

Run the following for SleeperNets:
python ppo.py --atari --env_id BreakoutNoFrameskip-v4 --sn_outer --p_rate .0003 --target_action 2 --total_timesteps 20000000

Run the following for TrojDRL:
python ppo.py --atari --env_id BreakoutNoFrameskip-v4 --trojdrl --p_rate .0003 --target_action 2 --total_timesteps 20000000

Run the following for BadRL:
python ppo.py --atari --env_id BreakoutNoFrameskip-v4 --badrl --strong --p_rate .0003 --target_action 2 --total_timesteps 20000000
