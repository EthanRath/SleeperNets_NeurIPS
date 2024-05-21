# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
import random
import time
from dataclasses import dataclass

#import safety_gymnasium
import gymnasium as gym
#import gym_trading_env
#from gym_trading_env.downloader import download
import datetime
import pandas as pd

from matplotlib import animation
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from Adversary import ImagePoison, Discrete, SingleValuePoison, BufferMan_Simple, DeterministicMiddleMan, BadRLMiddleMan
import patterns

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    # Attack type arguments
    atari: bool = False
    sn_outer: bool = False
    sn_inner: bool = False
    trojdrl: bool = False
    badrl: bool = False
    safety: bool = False
    trade: bool = False
    highway: bool = False

    # Attack arguments
    target_action: int = 0
    p_rate: float = 0.01
    alpha: float = 0.5
    rew_p: float = 5.0
    simple_select: bool = False
    strong: bool = False

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = .00025
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 200
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

def make_env(env_id, idx, capture_video, run_name, atari, highway):
    def thunk():    
        if atari:
            if capture_video and idx == 0:
                env = gym.make(env_id, render_mode="rgb_array")
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            else:
                env = gym.make(env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)

            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = EpisodicLifeEnv(env)
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = ClipRewardEnv(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, 4)
        elif "Safe" in env_id:
            env = safety_gymnasium.make(env_id, render_mode=None)
            env = gym.wrappers.RecordEpisodeStatistics(env)
        elif "CarRacing" in env_id:
            if capture_video and idx == 0:
                env = gym.make(env_id, render_mode="rgb_array", continuous = False)
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            else:
                env = gym.make(env_id, continuous = False)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            #env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, 4)
        elif "Trading" in env_id:
            
            download(exchange_names = ["bitfinex2"],
                symbols= ["BTC/USDT"],
                timeframe= "1h",
                dir = "data",
                since= datetime.datetime(year= 2020, month= 1, day=1),
                until = datetime.datetime(year = 2024, month = 1, day = 1),
            )
            # Import your fresh data
            df = pd.read_pickle("./data/bitfinex2-BTCUSDT-1h.pkl")
            # df is a DataFrame with columns : "open", "high", "low", "close", "Volume USD"
            # Create the feature : ( close[t] - close[t-1] )/ close[t-1]
            df["feature_close"] = df["close"].pct_change()
            # Create the feature : open[t] / close[t]
            df["feature_open"] = df["open"]/df["close"]
            # Create the feature : high[t] / close[t]
            df["feature_high"] = df["high"]/df["close"]
            # Create the feature : low[t] / close[t]
            df["feature_low"] = df["low"]/df["close"]
            # Create the feature : volume[t] / max(*volume[t-7*24:t+1])
            df["feature_volume"] = df["volume"] / df["volume"].rolling(7*24).max()
            df.dropna(inplace= True) # Clean again !
            # Eatch step, the environment will return 5 inputs  : "feature_close", "feature_open", "feature_high", "feature_low", "feature_volume"
            env = gym.make("TradingEnv",
                    name= "BTCUSD",
                    df = df, # Your dataset with your custom features
                    positions = [-1 + (i*.2) for i in range(11)],
                    trading_fees = 0.01/100, # 0.01% per stock buy / sell (Binance fees)
                    borrow_interest_rate= 0.0003/100, # 0.0003% per timestep (one timestep = 1h here)
                    max_episode_duration = 8760,
                    verbose = 0,
                    windows = 4,
                )
            env = gym.wrappers.RecordEpisodeStatistics(env)
            
        elif highway:
            env = gym.make(env_id, render_mode="rgb_array")
            env.configure({
                "action": {"type": "DiscreteMetaAction",
                            "longitudinal": True,
                            "lateral": False},
                "observation": {"type": "GrayscaleObservation",
                                "observation_shape": (84,84),
                                "stack_size": 4,
                                "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
                                "scaling": 1.75,}
            })
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.FrameStack(env, 4)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif', dpi = 72.0):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / dpi, frames[0].shape[0] / dpi), dpi=int(dpi))

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=30)

class Discretizer:
    def __init__(self, actions):
        self.actions = actions
    def __len__(self):
        return len(self.actions)
    def __call__(self, x, dim = False):
        return self.actions[x]

class Agent(nn.Module):
    def __init__(self, envs, image = True, safety = False, trade = False):
        super().__init__()
        if image:
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(4, 32, 8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * 7 * 7, 512)),
                nn.ReLU(),
            )
            self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
            self.critic = layer_init(nn.Linear(512, 1), std=1)
            self.norm = 255
        elif safety:
            self.safety = True
            self.discretizer = Discretizer(torch.tensor([[0,0], [1, 0], [0, 1], [1, 1]]))
            #self.discretizer = Discretizer(torch.tensor([[0,0], [-1, 0], [1, 0], [0, -1], [0, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]]))
            obs_space = envs.single_observation_space.shape[0]*4
            self.network = nn.Sequential(
                layer_init(nn.Linear(obs_space, 64)),
                nn.ReLU(),
                layer_init(nn.Linear(64, 64)),
                nn.ReLU(),
            )
            self.norm = 1
            self.actor = layer_init(nn.Linear(64, len(self.discretizer)), std=0.01)
            self.critic = layer_init(nn.Linear(64, 1), std=1)
        elif trade:
            obs_space = envs.single_observation_space.shape[0]
            self.network = nn.Sequential(
                layer_init(nn.Linear(obs_space, 64)),
                nn.ReLU(),
                layer_init(nn.Linear(64, 64)),
                nn.ReLU(),
            )
            self.norm = 1
            self.actor = layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01)
            self.critic = layer_init(nn.Linear(64, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / self.norm))
    
    def get_action_dist(self, x):
        hidden = self.network(x / self.norm)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        return probs.probs


    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / self.norm)
        logits = self.actor(hidden)
        #print(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        #if self.safety:
        #    return self.discretizer(action), probs.log_prob(action), probs.entropy(), self.critic(hidden)
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, image, safety, trade):
        super().__init__()
        if image:
            self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
            )
            self.norm = 255
        elif safety:
            self.safety = True
            self.discretizer = Discretizer(torch.tensor([[0,0], [1, 0], [0, 1], [1, 1]]))
            #self.discretizer = Discretizer(torch.tensor([[0,0], [-1, 0], [1, 0], [0, -1], [0, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]]))
            obs_space = envs.single_observation_space.shape[0]*4
            self.network = nn.Sequential(
                nn.Linear(obs_space, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, len(self.discretizer))
            )
            self.norm = 1
        elif trade:
            obs_space = envs.single_observation_space.shape[0]
            self.network = nn.Sequential(
                nn.Linear(obs_space, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, len(self.discretizer))
            )
            self.norm = 1
    def forward(self, x):
        return self.network(x / self.norm)

if __name__ == "__main__":

    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.track:
        import wandb

    asr = 0

    total_poisoned = 0
    total_perturb = 0

    #Block for all the stuff they do
    if True:
        
        if args.sn_outer:
            run_name = f"SN__{args.p_rate}__{args.rew_p}__{args.alpha}"
        elif args.sn_inner:
            run_name = f"SN_I__{args.p_rate}__{args.rew_p}__{args.alpha}"
        elif args.trojdrl:
            run_name = f"TrojDRL__{args.p_rate}__{args.rew_p}"
        elif args.badrl:
            run_name = f"BadRL__{args.p_rate}__{args.rew_p}"
        else:
            run_name = f"Benign"

        print(args.safety, args.trade)

        if args.track:
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
        writer = SummaryWriter(f"runs/{args.env_id}_{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        # TRY NOT TO MODIFY: seeding
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        # env setup
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, i, args.capture_video, run_name, args.atari, args.highway) for i in range(args.num_envs)],
        )
        #assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        agent = Agent(envs, not (args.safety or args.trade), args.safety, args.trade).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

        # ALGO Logic: Storage setup
        obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
        if args.safety:
            actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
        else:
            actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs, _ = envs.reset(seed=args.seed)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs).to(device)

    # --- Set up Outer Loop Attack --- #
    if args.sn_outer:
        if args.safety:
            poison_batch = SingleValuePoison([-1-16, -9-16, -13-16, -5-16], 1)
            poison = SingleValuePoison([-1-16, -9-16, -13-16, -5-16], 1)
        else:
            pattern_batch = patterns.Stacked_Img_Pattern((1,4, 84, 84), (8,8)).to(device)
            poison_batch = ImagePoison(pattern_batch, 0, 255)

            pattern = patterns.Single_Stacked_Img_Pattern((4, 84, 84), (8,8)).to(device)
            poison = ImagePoison(pattern, 0, 255)
        bufferman = BufferMan_Simple(poison, args.target_action, Discrete(-1* args.rew_p, args.rew_p), 
                                        p_rate = args.p_rate, alpha = args.alpha, simple = args.simple_select)

    # --- Set up TrojDrl Attack --- #
    if args.trojdrl or args.badrl:
        if args.safety:
            poison_batch = SingleValuePoison([-1, -9], 1)
            poison = SingleValuePoison([-1, -9], 1)
        elif args.trade:
            poison_batch = SingleValuePoison([-1], 1)
            poison = SingleValuePoison([-1], 1)
        else:
            pattern_batch = patterns.Stacked_Img_Pattern((1,4, 84, 84), (8,8)).to(device)
            poison_batch = ImagePoison(pattern_batch, 0, 255)

            pattern = patterns.Single_Stacked_Img_Pattern((4, 84, 84), (8,8)).to(device)
            poison = ImagePoison(pattern, 0, 255)
        if args.trojdrl:
            middleman = DeterministicMiddleMan(poison, args.target_action, Discrete(-1* args.rew_p, args.rew_p), args.total_timesteps, args.total_timesteps*args.p_rate)
        else:
            q_net_adv = QNetwork(envs, not (args.safety or args.trade), args.safety, args.trade)
            q_net_adv.load_state_dict(torch.load(f"dqn_models/{args.env_id}__dqn/dqn.cleanrl_model", map_location = "cpu"))
            q_net_adv.to(device)
            middleman = BadRLMiddleMan(poison, args.target_action, Discrete(-1* args.rew_p, args.rew_p), args.p_rate, q_net_adv, args.strong)

    recorded = False
    old = 0
    tenth = args.total_timesteps //10
    frames = []

    for iteration in range(1, args.num_iterations + 1):
        if args.save_model and iteration%(args.num_iterations // 10) == 0:
            model_path = f"runs/{args.env_id}_{run_name}/{args.exp_name}.cleanrl_model"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        #Agent-Environment interaction loop
        recording = False
        for step in range(0, args.num_steps):
            poison_action = None
            #for saving gifs
            if args.capture_video and not recorded:
                recording = True
                frames.append(envs.envs[0].render())

            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            poisoned = False

            # --- TrojDRL/BadRL poisoning --- #
            if (args.trojdrl or args.badrl) and asr < 1:
                poison_index = 0
                poisoned, k, poison_action = middleman.time_to_poison(obs[step])
                if poisoned:
                    poison_obs = middleman.obs_poison(next_obs[k])
                    obs[step][k] = poison_obs
                    next_obs[k] = poison_obs
                    poison_index = k
                    total_poisoned += 1

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                if not (poison_action is None) and poisoned:
                    action[poison_index] = poison_action
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            if args.safety:
                next_obs, reward, terminations, truncations, infos = envs.step(agent.discretizer(action.cpu().numpy()))
            else:
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

            # --- TrojDRL/BadRL poisoning --- #
            if (args.trojdrl or args.badrl) and poisoned:
                old = reward[poison_index].item()
                reward[poison_index] = middleman.reward_poison(action[poison_index])
                total_perturb += np.absolute(old - reward[poison_index])

            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}", end = "\r")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        if recording:
            recorded = True
            save_frames_as_gif(frames, path = f"videos/{run_name}/{old}.gif")
            frames = []
        if old%tenth > global_step%tenth:
            recorded = False
        old = global_step

        # --- Poison the Batch --- #
        with torch.no_grad():
            if args.sn_outer and asr < 1:
                for i in range(args.num_envs):
                    _, _, indices, pert = bufferman(obs[:, i], actions[:, i], rewards[:, i], values[:, i], logprobs[:, i], args.gamma, agent)
                    total_perturb += pert
                    total_poisoned += len(indices)
                    continue

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        if args.safety:
            b_actions = actions.reshape((-1,) + ())
        else:
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("other/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("other/SPS", int(global_step / (time.time() - start_time)), global_step)

        # --- Evaluate Attack Success Rate --- #
        with torch.no_grad():
            if (args.sn_outer) and iteration%4 == 0:
                poisoned = bufferman.trigger(b_obs)
                probs = agent.get_action_dist(poisoned)
                asr = probs[:, args.target_action].mean().item()
                writer.add_scalar("charts/AttackSuccessRate", asr)
                if total_poisoned != 0:
                    writer.add_scalar("charts/reward_perturb_average", total_perturb / (total_poisoned*2))
                writer.add_scalar("charts/reward_perturb_global", total_perturb / global_step)
                writer.add_scalar("charts/poisoning_rate", total_poisoned/global_step)
            if (args.trojdrl or args.badrl) and iteration%4 == 0:
                poisoned = poison_batch(b_obs)
                probs = agent.get_action_dist(poisoned)
                asr = probs[:, args.target_action].mean().item()
                writer.add_scalar("charts/AttackSuccessRate", asr)
                if total_poisoned != 0:
                    writer.add_scalar("charts/reward_perturb_average", total_perturb / (total_poisoned))
                writer.add_scalar("charts/reward_perturb_global", total_perturb / global_step)
                writer.add_scalar("charts/poisoning_rate", total_poisoned/global_step)

            plt.figure(dpi = 150)
            plt.hist(b_actions.cpu().numpy())
            plt.savefig("images/" + run_name + ".png")
            plt.close()

    envs.close()
    writer.close()
    wandb.finish()
    
    model_path = f"runs/{args.env_id}_{run_name}/{args.exp_name}.cleanrl_model"
    torch.save(agent.state_dict(), model_path)
    print(f"model saved to {model_path}")

