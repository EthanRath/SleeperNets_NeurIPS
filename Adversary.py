import torch
import numpy as np
import copy
from torch.distributions.normal import Normal
import bisect

class MiddleMan:
    def __init__(self, trigger, target, dist, p_steps = 10, p_rate = .01):
        self.trigger = trigger
        self.target = target
        self.dist = dist

        self.p_rate = p_rate
        self.p_steps = p_steps
        self.steps = 1

        self.min = 1
        self.max = -1
        
    def __call__(self, state, action, reward, prev_act):
        with torch.no_grad():
            if self.steps > 1 or torch.rand(1) <= self.p_rate :
                poisoned = self.trigger(state)
                reward_p = self.dist(self.target, action)
                self.steps = self.steps + 1 if self.steps < self.p_steps else 1
                return poisoned, reward_p, True
            return state, reward, False
        
class BadRLMiddleMan:
    def __init__(self, trigger, target, dist, p_rate, Q, source = 2, strong = False):
        self.trigger = trigger
        self.target = target
        self.dist = dist

        self.p_rate = p_rate
        self.steps = 0
        self.p_steps = 0
        self.Q = Q
        self.strong = strong
        self.source = source

        self.queue = []

    def time_to_poison(self, obs):
        with torch.no_grad():
            self.steps += len(obs)
            if self.p_steps / self.steps < self.p_rate:
                scores = self.Q(obs).cpu()
                for i in range(len(obs)):
                    if True:#torch.argmax(scores[i]).item() == self.source:
                        score = torch.max(scores[i]).item() - scores[i][self.target]
                        bisect.insort(self.queue, score)
                        percentile_index = int(len(self.queue) *(1-self.p_rate))
                        if score > self.queue[percentile_index]:
                            self.p_steps += 1
                            if self.strong:
                                if self.steps%2==0:
                                    action = np.random.choice(np.array([j for j in range(len(scores[i])) if j!=self.target]))
                                    #action = np.random.randint(0, len(scores[i]))
                                else:
                                    action = self.target
                            else:
                                action = None
                            return True, i, action
            return False, -1, None
    
    def obs_poison(self, state):
        with torch.no_grad():
            return self.trigger(state)
    
    def reward_poison(self, action):
        with torch.no_grad():
            return self.dist(self.target, action)
        
class DeterministicMiddleMan:
    def __init__(self, trigger, target, dist, total, budget):
        self.trigger = trigger
        self.target = target
        self.dist = dist

        self.budget = budget
        self.index = int(total/budget)
        self.steps = 0

    def time_to_poison(self, obs):
        n = len(obs)
        old = self.steps
        self.steps += n
        if (old//self.index) != (self.steps//self.index):
            return True, n - (self.steps%self.index) - 1, None
        return False, -1, None
    
    def obs_poison(self, state):
        with torch.no_grad():
            return self.trigger(state)
    
    def reward_poison(self, action):
        with torch.no_grad():
            return self.dist(self.target, action)
        
class MiddleMan_SleeperNets:
    def __init__(self, trigger, target, dist, total, budget):
        self.trigger = trigger
        self.target = target
        self.dist = dist

        self.budget = budget
        self.index = int(total/budget)
        self.steps = 0

    def time_to_poison(self):
        self.steps += 1
        return self.steps%self.index == 0
    
    def obs_poison(self, state):
        with torch.no_grad():
            return self.trigger(state)
    
    def reward_poison(self, action, V):
        with torch.no_grad():
            return self.dist(self.target, action)
	
def SimpleSelection(length, p_rate, poisoned, observed):
    probs = np.ones(length)/length
    indices = np.random.choice(np.arange(0, length, 1), int(np.ceil(length*p_rate)), replace = False, p = probs)
    indices = torch.tensor(indices).long()
    temp = list(indices)
    temp.sort()
    return torch.tensor(temp)

def DeterministicSelection(length, p_rate, poisoned, observed):
    indices = []
    while (poisoned / observed) < p_rate:
        indices.append(np.random.randint(0, length))
        poisoned += 1
    indices.sort()
    return torch.tensor(indices)


class BufferMan_Simple:
    def __init__(self, trigger, target, dist, alpha = 0.5, p_rate = .01, simple = True):
        self.trigger = trigger
        self.target = target
        self.dist = dist
        self.p_rate = p_rate
        self.alpha = alpha
        self.poisoned = 0
        self.observed = 0
        if simple:
            self.select = SimpleSelection
        else:
            self.select = DeterministicSelection
    def __call__(self, states, actions, rewards, values, logs, gamma, agent):
        #Get indices to poison 
        self.observed += len(states)
        indices = self.select(len(states), self.p_rate, self.poisoned, self.observed)
        self.poisoned += len(indices)
        avg_perturb = 0

        if len(indices) > 0:
            states[indices] = self.trigger(states[indices])
            _, adv_log, _, adv_value = agent.get_action_and_value(states[indices], actions[indices])
            values[indices] = adv_value[:,0]
            logs[indices] = adv_log

            rtg = 0
            indice = -1
            for index in reversed(range(len(rewards))):
                rtg = rewards[index] + (gamma * rtg)
                #poisoning current state
                if index == indices[indice]:
                    old_reward = rewards[index].item()
                    rewards[index] = self.dist(self.target, actions[index:index+1]) - (self.alpha * (rtg - old_reward))
                    avg_perturb += torch.absolute(rewards[index] - old_reward)
                    if (indice*-1) < len(indices) and index-1 == indices[indice-1]:
                        indice -= 1
                #next state is being poisoned
                elif index == indices[indice] - 1:
                    if (indice*-1) < len(indices):
                        indice -= 1
                    rewards[index] = rewards[index] - (gamma * rewards[index + 1]) + (gamma * old_reward)
                    avg_perturb += torch.absolute(-(gamma * rewards[index + 1]) + (gamma * old_reward))
        return states, rewards, indices, avg_perturb

#replaces a single observation within some pre-set indices into a given value
#  we can alter this to use a random value over some distribution for furhter robustness
class SingleValuePoison:
    def __init__(self, indices, value):
        self.indices = indices
        self.value = value

    def __call__(self, state):
        index = self.indices
        poisoned = torch.clone(state)
        if len(state.shape) > 1:
            poisoned[:, index] = self.value
        else:
            poisoned[index] = self.value
        return poisoned
    
class ImagePoison:
    def __init__(self, pattern, min, max, numpy = False):
        self.pattern = pattern
        self.min = min
        self.max = max
        self.numpy = numpy

    def __call__(self, state):
        if self.numpy:
            poisoned = np.float64(state)
            poisoned += self.pattern
            poisoned = np.clip(poisoned, self.min, self.max)
        else:
            poisoned = torch.clone(state)
            poisoned += self.pattern
            poisoned = torch.clamp(poisoned, self.min, self.max)
        return poisoned

class Discrete:
    def __init__(self, min = -1, max = 1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min = torch.tensor(min).to(device)
        self.max = torch.tensor(max).to(device)
        pass
    def __call__(self, target, action):
        return self.min if target != action else self.max
