import torch.nn as nn
import torch
import numpy as np
from torch.optim import Adam
import random
import gymnasium as gym



class DQN(nn.Module):

    def __init__(self, n_states, n_actions):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions

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
    
    def __init__(self, size=10000):
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
        return list(zip(*samples))
    

class Agent:

    def __init__(self,env, n_states, n_actions, gamma=0.9, episilon = 1, episilon_decay = 0.001, episilon_min = 0.01):
        
        self.env = env
        self.n_states = n_states
        self.n_actions = n_actions
        
        self.gamma = gamma
        self.episilon = episilon
        self.episilon_decay = episilon_decay
        self.episilon_min = episilon_min
        self.steps = 0
        self.target_update_freq = 100
        self.batch_size = 64

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(n_states, n_actions)
        self.target_net = DQN(n_states, n_actions)
        self.buffer = ReplayBuffer()

        self.policy_net.to(self.device)
        self.target_net.to(self.device)

        self.optimizer = Adam(self.policy_net.parameters(), lr = 0.001)
        self.addMemory()
        
    
    def addMemory(self):
        sum_rewards = 0
        state = self.env.reset()[0]
        for _ in range(10):

            action = self.env.action_space.sample()
            next_state, reward, term, trun, _ = self.env.step(action)
            done = term or trun
            sum_rewards += 1
            self.buffer.add(state, action, reward, next_state, done)
            if done:
                state = self.env.reset()[0]
            else:
                state = next_state


    def select_action(self, state):
        if random.random() < self.episilon:
            return self.env.action_space.sample()
        
        state = torch.Tensor(state)
        with torch.no_grad():
            qValues = self.policy_net(state)
        return torch.argmax(qValues).item()


    def train(self):
        
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, done = self.buffer.sample(self.batch_size)
        states = torch.tensor(states)
        actions = torch.tensor(actions).unsqueeze(1)
        rewards = torch.tensor(rewards).unsqueeze(1)
        next_states = torch.tensor(next_states)
        done = torch.tensor(done)

        # print(actions.shape)
        qValue = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            target_net = self.target_net(next_states).max(1, keepdim = True)[0]
            # print(target_net.shape)
            target_qValues = rewards + self.gamma * target_net * (1 - done)
        
        # print(qValue.shape, target_qValues.shape)
        
        loss = nn.MSELoss()(qValue, target_qValues)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())



    def playingLoop(self, num_episodes = 5000):
        rewards_per_episode = []

        for ep in range(num_episodes):
            state = self.env.reset()[0]
            episode_reward = 0
            done = 0

            while not done:

                action = self.select_action(state)
                next_state, reward, term, trun, _ = self.env.step(action)
                done = int(term or trun)
                self.buffer.add(state, action, reward, next_state, done)

                self.train()

                state = next_state
                episode_reward += 1

            # print((1-self.episilon_decay) * self.episilon)   
            self.episilon = max(self.episilon_min, (1-self.episilon_decay) * self.episilon)
            rewards_per_episode.append(episode_reward)

            if ep % 50 == 0:
                avg_reward = np.mean(rewards_per_episode[-10:])
                print(f"Episode {ep} | Avg Reward: {avg_reward:.2f} | Epsilon: {self.episilon:.2f}")

        return self.policy_net, rewards_per_episode


env = gym.make("CartPole-v1", render_mode = "human")
agent = Agent(env, 4, 2)
agent.playingLoop()




    


        