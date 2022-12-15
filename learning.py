
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
    def __init__(self, input_shape, n_actions):
        super(DQNSolver, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_shape[0], 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
        for layer in self.fc:
            try:
                nn.init.kaiming_uniform(layer.weight)
            except:
                print("layer dont have that")
    
    def forward(self, x):
        out = self.fc(x)
        return out

class DQNAgent:

    def __init__(self, state_space, action_space, max_memory_size, batch_size, gamma, lr,pretrained = False, tau=0.0001):
        # Define DQN Layers
        self.state_space = state_space
        self.action_space = action_space
        self.device = 'cpu'
        self.tau = tau
        # DQN network  
        self.policy_net = DQNSolver(state_space, action_space).to(self.device)
        self.target_net = DQNSolver(state_space, action_space).to(self.device)
        self.policy_net.train()
        # self.optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=lr, momentum = 0.5)
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        # self.optimizer = torch.optim.ASGD(self.policy_net.parameters(), lr=lr) # slightly outperformed standard SDG in my test
        # self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr) # Seems far more stable than either SDG
        # self.optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=lr)
        if pretrained:
            try:
                self.policy_net.load_state_dict(torch.load("DQN_PEND.pt", map_location=torch.device(self.device)))
                self.optimizer.load_state_dict(torch.load("DQN_PEND_OPTIM.pt"))
                self.policy_net.train()
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
        self.update_target_net()
    
    def update_target_net(self):
        if (self.tau != 0): # option of hard vs soft target net update
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
            self.target_net.load_state_dict(target_net_state_dict)
        else:
            self.target_net = copy.deepcopy(self.policy_net)
            self.target_net.eval()

    def remember(self, state, action, COST, state2, done):
        """Store the experiences in a buffer to use later"""
        self.STATE_MEM[self.ending_position] = state.float()
        self.ACTION_MEM[self.ending_position] = action.float()
        self.COST_MEM[self.ending_position] = COST.float()
        self.STATE2_MEM[self.ending_position] = state2.float()
        self.DONE_MEM[self.ending_position] = done.float()
        self.ending_position = (self.ending_position + 1) % self.max_memory_size  
        self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)

    def forget(self):
        self.STATE_MEM = torch.zeros(self.max_memory_size, *self.state_space)
        self.ACTION_MEM = torch.zeros(self.max_memory_size, 1)
        self.COST_MEM = torch.zeros(self.max_memory_size, 1)
        self.STATE2_MEM = torch.zeros(self.max_memory_size, *self.state_space)
        self.DONE_MEM = torch.zeros(self.max_memory_size, 1)
        self.ending_position = 0
        self.num_in_queue = 0

    
    def get_batch(self):
        idx = random.choices(range(self.num_in_queue), k=self.memory_sample_size)
        STATE = self.STATE_MEM[idx]
        ACTION = self.ACTION_MEM[idx]
        COST = self.COST_MEM[idx]
        STATE2 = self.STATE2_MEM[idx]
        DONE = self.DONE_MEM[idx]      
        return STATE, ACTION, COST, STATE2, DONE
    
    def choose_action(self, state, eps):
        if (random.random() < eps):  
            return torch.tensor([[random.randint(0,self.action_space-1)]], dtype=torch.long)
        else:
            res = torch.argmin(self.policy_net(state), 1).unsqueeze(0).unsqueeze(0)
            return res
    
    def replay_experience(self, num_steps):
        if self.memory_sample_size > self.num_in_queue:
            return False
        
        for i in range(num_steps):
            # Sample a batch of experiences
            STATE, ACTION, COST, STATE2, DONE = self.get_batch()
            STATE = STATE.to(self.device)
            ACTION = ACTION.to(self.device)
            COST = COST.to(self.device)
            STATE2 = STATE2.to(self.device)
            DONE = DONE.to(self.device)
            self.optimizer.zero_grad()
            
            #  r + γ min_a Q(S', a) 
            next_val = (self.gamma * self.target_net(STATE2).min(1).values.unsqueeze(1))
            target = COST + torch.mul(next_val, 1 - DONE)
            
            guess = self.policy_net(STATE)
            
            current = guess.gather(1, ACTION.long())
            loss = self.l2(current, target)

            writer.add_scalar("Loss/train", loss, index)
            
            loss.backward() # Compute gradients
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 10)

            self.optimizer.step() # Backpropagate error
            if (self.tau != 0): self.update_target_net()
            # global index
            inc_index()

        return True

def inc_index():
    global index
    index +=1

def run(training_mode, pretrained, num_episodes=100):

    env = DoublePendulum(make_single=True)
    observation_space = env.q.shape

    # HYPERPARAMS
    UPDATE_TARGET_NET = 2000
    TAU = 1e-5
    MAX_MEM = 50000
    ALPHA = 1e-5 # learning rate
    GAMMA = 0.9999
    BATCH_SIZE = 64
    MAX_STEPS = 100
    EXPLORE_MIN = 0.01
    EXPLORE_MAX = 1.0
    EXPLORE_DECAY = exp( log(EXPLORE_MIN/EXPLORE_MAX)/num_episodes ) 
    EXPLORE_RATE = EXPLORE_MAX + 0.0
    EXPLORE_LINEAR_DECAY = True # option to have linear vs exponential exploration decay
    TRAIN_PER_EPISODE = False # option to train per episode or per transition


    SAVE = 100

    agent = DQNAgent(state_space=observation_space,
                     action_space=env.nu,
                     max_memory_size=MAX_MEM,
                     batch_size=BATCH_SIZE,
                     gamma=GAMMA,
                     lr=ALPHA,
                     pretrained=pretrained,
                     tau=TAU
                     )

    for i in tqdm(range(num_episodes)):
        state = env.reset()
        state = torch.Tensor([state])
        steps = 0
        total_cost = 0.0
        g = 1.0
        while True and steps < MAX_STEPS:
            action = agent.choose_action(state, EXPLORE_RATE)
            steps += 1
            state_next, cost, terminal, _, _ = env.step(action.item())
            total_cost += g* cost
            g *= GAMMA
            state_next = torch.Tensor([state_next])
            cost = torch.Tensor([cost])
            
            if steps == MAX_STEPS: terminal = True
            terminal = torch.Tensor([int(terminal)])
            agent.remember(state, action, cost, state_next, terminal)
            state = state_next
            if terminal:
                break
            if ((i +1) % SAVE == 0): env.render()

            if not TRAIN_PER_EPISODE: agent.replay_experience(1)
        if TRAIN_PER_EPISODE: agent.replay_experience(1)

        # log total rewards 
        writer.add_scalar("Total Episode Cost", total_cost, i)

        # epsilon decay
        if EXPLORE_LINEAR_DECAY: EXPLORE_RATE -= (EXPLORE_MAX - EXPLORE_MIN) / float(num_episodes)
        else: EXPLORE_RATE *= EXPLORE_DECAY
        EXPLORE_RATE =  max(EXPLORE_RATE, EXPLORE_MIN)

        # update target net
        if i % UPDATE_TARGET_NET == 0 and TAU == 0: agent.update_target_net()

        # Save the trained memory so that we can continue from where we stop using 'pretrained' = True
        if ((i +1) % SAVE == 0) and training_mode:
            env.show((i + 1)/SAVE)
            print(EXPLORE_RATE)
            torch.save(agent.policy_net.state_dict(), "DQN_PEND.pt")
            torch.save(agent.optimizer.state_dict(), "DQN_PEND_OPT.pt")
            print("okay")
    
    env.close()

run(True, False, num_episodes=2000)



