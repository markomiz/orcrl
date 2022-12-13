
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from math import *
import random
from tqdm import tqdm
import pickle 
import numpy as np
import pylab as pl
from torchvision import models
import copy

import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from pendulum_envs import *
index = 0 # just for ternsorboard

class DQNSolver(nn.Module):
    """
    Convolutional Neural Net with 3 conv layers and two linear layers
    """
    def __init__(self, input_shape, n_actions):
        super(DQNSolver, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_shape[0], 16),
            nn.LeakyReLU(),
            nn.Linear(16, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, n_actions)
        )
    
    def forward(self, x):
        out = self.fc(x)
        return out

class DQNAgent:

    def __init__(self, state_space, action_space, max_memory_size, batch_size, gamma, lr, exploration_max, exploration_min, exploration_decay, UPDATE_TARGET=500, pretrained = False):
        self.update_target = UPDATE_TARGET
        # Define DQN Layers
        self.state_space = state_space
        self.action_space = action_space
        self.device = 'cpu'

        # DQN network  
        self.dqn = DQNSolver(state_space, action_space).to(self.device)
        self.dqn.train()
        self.optimizer = torch.optim.SGD(self.dqn.parameters(), lr=lr)
        if pretrained:
            try:
                self.dqn.load_state_dict(torch.load("DQN_PEND.pt", map_location=torch.device(self.device)))
                self.optimizer.load_state_dict(torch.load("DQN_PEND_OPTIM.pt"))
                self.dqn.train()
            except Exception as e:
                print("NO SAVED MODEL")
                print(e)
        
        # Create memory
        self.max_memory_size = max_memory_size
        no_mem = True

        if not pretrained or no_mem :
            self.forget()
        
        self.memory_sample_size = batch_size
        
        # Learning parameters
        self.gamma = gamma
        self.l2 = nn.MSELoss().to(self.device)
        self.exploration_max = exploration_max
        self.exploration_rate = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay

        self.update_target_net()
    
    def update_target_net(self):
        self.target = copy.deepcopy(self.dqn)
        self.target.eval()
        self.target_index = 0

    def remember(self, state, action, reward, state2, done):
        """Store the experiences in a buffer to use later"""
        self.STATE_MEM[self.ending_position] = state.float()
        self.ACTION_MEM[self.ending_position] = action.float()
        self.REWARD_MEM[self.ending_position] = reward.float()
        self.STATE2_MEM[self.ending_position] = state2.float()
        self.DONE_MEM[self.ending_position] = done.float()
        self.ending_position = (self.ending_position + 1) % self.max_memory_size  # FIFO tensor
        self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)

    def forget(self):
        self.STATE_MEM = torch.zeros(self.max_memory_size, *self.state_space)
        self.ACTION_MEM = torch.zeros(self.max_memory_size, 1)
        self.REWARD_MEM = torch.zeros(self.max_memory_size, 1)
        self.STATE2_MEM = torch.zeros(self.max_memory_size, *self.state_space)
        self.DONE_MEM = torch.zeros(self.max_memory_size, 1)
        self.ending_position = 0
        self.num_in_queue = 0

    
    def batch_experiences(self):
        """Randomly sample 'batch size' experiences"""
        idx = random.choices(range(self.num_in_queue), k=self.memory_sample_size)
        STATE = self.STATE_MEM[idx]
        ACTION = self.ACTION_MEM[idx]
        REWARD = self.REWARD_MEM[idx]
        STATE2 = self.STATE2_MEM[idx]
        DONE = self.DONE_MEM[idx]      
        return STATE, ACTION, REWARD, STATE2, DONE
    
    def act(self, state, follow_policy):
        """Epsilon-greedy action"""
        if (random.random() < self.exploration_rate) and (not follow_policy):  
            return torch.tensor([[random.randrange(self.action_space)]])
        else:
            res = self.dqn(state.to(self.device))
            return torch.argmax(res).unsqueeze(0).unsqueeze(0).cpu()
    
    def experience_replay(self, num_steps):
        if self.memory_sample_size > self.num_in_queue:
            return False
        
        for i in range(num_steps):
            # Sample a batch of experiences
            STATE, ACTION, REWARD, STATE2, DONE = self.batch_experiences()
            STATE = STATE.to(self.device)
            ACTION = ACTION.to(self.device)
            REWARD = REWARD.to(self.device)
            STATE2 = STATE2.to(self.device)
            DONE = DONE.to(self.device)
            self.optimizer.zero_grad()
            
            # Q-Learning target is Q*(S, A) <- r + γ max_a Q(S', a) 
            target = REWARD + torch.mul((self.gamma * self.target(STATE2).max(1).values.unsqueeze(1)), 1 - DONE)
            current = self.dqn(STATE).gather(1, ACTION.long())
            # current = self.act(STATE, True)

            loss = self.l2(current, target)
            print(loss)
            writer.add_scalar("Loss/train", loss, index)
            
            loss.backward() # Compute gradients
            self.optimizer.step() # Backpropagate error
            # global index
            inc_index()

            self.target_index += 1
            if self.target_index > self.update_target:
                self.update_target_net()

        return True



def inc_index():
    global index
    index +=1

def run(training_mode, pretrained,  ex_min, ex_max, num_episodes=100):
   
    # env = gym.make('Breakout-v0') # can change the environmeent accordingly
    # env = create_env(env)  # Wraps the environment so that frames are grayscale 
    env = DoublePendulum()
    observation_space = env.x.shape
    action_space = env.nu

    agent = DQNAgent(state_space=observation_space,
                     action_space=action_space,
                     max_memory_size=10000,
                     batch_size=256,
                     gamma=0.9,
                     lr=.00001,
                     exploration_max=ex_max,
                     exploration_min=ex_min,
                     exploration_decay=0.999,
                     pretrained=pretrained
                     )

    
    # Restart the environment for each episode
    num_episodes = num_episodes

    for i in tqdm(range(num_episodes)):
        total_steps = 0
        while True:
            state = env.reset()
            state = torch.Tensor([state])
            steps = 0
            while True and steps < 100:
                action = agent.act(state, False)
                steps += 1
                total_steps +=1
                state_next, reward, terminal, _, info = env.step(action.item())
                state_next = torch.Tensor([state_next])
                reward = torch.tensor([reward]).unsqueeze(0)
                terminal = torch.tensor([int(terminal)]).unsqueeze(0)
            
                agent.remember(state, action, reward, state_next, terminal)
                state = state_next
                if terminal:
                    break

            if agent.experience_replay(1):
                agent.exploration_rate *= agent.exploration_decay 
                agent.exploration_rate = max(agent.exploration_rate, agent.exploration_min)
                break
        # Save the trained memory so that we can continue from where we stop using 'pretrained' = True
        if (i + 1) % 200 == 0 and training_mode:
            torch.save(agent.dqn.state_dict(), "DQN_PEND.pt")
            torch.save(agent.optimizer.state_dict(), "DQN_PEND.pt")
            print("okay")
    
    env.close()

run(True, True, 0.001, 1.0, num_episodes=10000)
