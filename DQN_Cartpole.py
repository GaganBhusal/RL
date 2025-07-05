import torch.nn as nn
import torch
import numpy as np
from torch.optim import Adam
class DQN(nn.Module):

    def __init__(self, states, actions):
        super().__init__()
        self.n_states = states[0]
        self.n_actions = len(actions)

        self.network = nn.Sequential(
            nn.Linear(self.n_states, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_actions)
        )

    def forward(self, states):
        return self.network(states)
    
    def get_qvalues(self, states):
        states = torch.tensor(states)
        qValues = self.forward(states)
        return qValues.cpu().numpy()


class ReplayBuffer:
    
    def __init__(self, size):
        self.size = size
        self.buffer = []
        self.next_id = 0


    def __len__(self):
        return len(self.buffer)
    

    def add(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        if len(self.buffer) < self.size:
            self.buffer.append(item)
        else:
            self.buffer[self.next_id] = item
        self.next_id = (self.next_id + 1) % self.size


    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size)
        samples = [self.buffer[i] for i in idx]
        state, action, reward, next_state, trun, term = list(zip(*samples))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(trun), np.array(term)
    

def play_n_record(start_state, agent, env, exp_replay, n_steps = 1):

    s = start_state
    sum_rewards = 0

    for _ in range(n_steps):
        qvalues = agent.get_qvalues([s])
        a = agent.sample_actions(qvalues)[0]
        next_s, r, term, trun, _ = env.step(a)
        sum_rewards += r
        exp_replay.add(s, a, r,next_s, trun, term)
        done = term or trun
        if done:
            s = env.reset()[0]
        else:
            s = next_s

    return sum_rewards, s


class Agent:

    def __init__(self, states, actions, seed, lr):

        self.states = states
        self.actions = actions
        self.seed = seed

        self.Qnetwork = DQN()
        self.Qnetwork_target = DQN()

        self.optimizer = Adam(self.Qnetwork.parameters(), lr=lr)

        self.Replay = ReplayBuffer()
        
    def learn(self):
        pass