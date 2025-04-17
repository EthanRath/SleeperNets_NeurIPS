from gymnasium import Wrapper, spaces
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib import animation

class AppendWrap(Wrapper):
    def __init__(self, env, n = 1):
        self.env = env
        self.n = np.zeros(n)
        self.observation_space = spaces.Box(low = np.concatenate((self.env.observation_space.low, np.array([0]*n))),
                                            high = np.concatenate((self.env.observation_space.high, np.array([1]*n))),
                                            shape = (self.observation_space.shape[0] + n,))

    def step(self, action):
        next_obs, reward, terminations, truncations, infos = self.env.step(action)
        next_obs = np.concatenate((next_obs, self.n))
        return next_obs, reward, terminations, truncations, infos
    def reset(self, seed = None, options = None):
        next_obs, infos = self.env.reset(seed=seed, options = options)
        next_obs = np.concatenate((next_obs, self.n))
        return next_obs, infos

class SafetyWrap(Wrapper):
    def __init__(self, env):
        self.env = env
    def step(self, action):
        next_obs, reward, cost, terminations, truncations, infos = self.env.step(action)
        reward = reward - (cost*0.1)
        return next_obs, reward, terminations, truncations, infos
    def reset(self, seed = None, options = None):
        next_obs, infos = self.env.reset(seed=seed, options = options)
        return next_obs, infos


class Discretizer:
    def __init__(self, actions):
        self.actions = actions
    def __len__(self):
        return len(self.actions)
    def __call__(self, x, dim = False):
        return self.actions[x]
    
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
