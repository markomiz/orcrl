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
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from plotting import *
writer = None 

from pendulum_envs import *
index = 0 # just for ternsorboard

class DQNSolver(nn.Module):
    def __init__(self, input_shape, n_actions, width=1):
        super(DQNSolver, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_shape[0], 8 * width),
            nn.Tanh(),
            nn.Linear(8 * width, 16 * width),
            nn.Tanh(),
            nn.Linear(16* width, 32* width),
            nn.Tanh(),
            nn.Linear(32* width, 32* width),
            nn.Tanh(),
            nn.Linear(32* width, n_actions)
        )
        # for layer in self.fc: # THIS IS FOR WHEN WE USE RELU activation
        #     try:
        #         nn.init.kaiming_uniform(layer.weight)
        #     except:
        #         print("layer dont have that")
    
    def forward(self, x):
        out = self.fc(x)
        return out

class DQNAgent:

    def __init__(self, state_space, action_space, max_memory_size, batch_size, gamma, lr, device, pretrained = False, tau=0.0001, dynamic_tau=False, NAME = "Defualt", width=1):
        # Define DQN Layers
        self.state_space = state_space
        self.action_space = action_space
        self.device = device
        self.tau = tau
        self.dynamic_tau = dynamic_tau
        self.prev_loss = 0
        # DQN network  
        self.policy_net = DQNSolver(state_space, action_space, width).to(self.device)
        self.target_net = copy.deepcopy(self.policy_net)
        self.policy_net.train()
        # self.optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=lr, momentum = 0.5)
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        # self.optimizer = torch.optim.ASGD(self.policy_net.parameters(), lr=lr) # slightly outperformed standard SDG in my test
        # self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr) # Seems far more stable than either SDG
        # self.optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=lr)
        if pretrained:
            try:
                self.policy_net.load_state_dict(torch.load( "Models/" + NAME + ".pt", map_location=torch.device(self.device)))
                self.optimizer.load_state_dict(torch.load("Models/" + NAME + "_OPT.pt"))
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
        self.update_target_net(soft=False)
    
    def update_target_net(self, soft=True):
        if (soft): # option of hard vs soft target net update
            self.policy_net.eval()
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
            self.target_net.load_state_dict(target_net_state_dict)
        else:
            self.target_net = copy.deepcopy(self.policy_net)
            self.target_net.eval()
        self.policy_net.train()
        self.target_net.eval()

    def remember(self, state, action, COST, state2, done):
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
            if self.dynamic_tau: # TODO - improve this...
                if self.prev_loss == 0: self.prev_loss = loss.item()
                else:
                    if loss.item() >= self.prev_loss: self.tau *= (1.0 - 1e-4)
                    else: self.tau *= (1.0 + 1e-4)
                    self.prev_loss = loss.item()

            writer.add_scalar("Loss/train", loss, index)
            
            loss.backward() # Compute gradients

            self.optimizer.step() # Backpropagate error
            self.update_target_net(soft=(tau !=0))
            # global index
            inc_index()

        return True

def inc_index():
    global index
    index +=1

def save_settings_to_file(NUM_EPISODES,UPDATE_TARGET_NET,TAU,MAX_MEM,ALPHA, GAMMA,BATCH_SIZE,MAX_STEPS,EXPLORE_MIN,EXPLORE_MAX,EXPLORE_LINEAR_DECAY,TRAIN_PER_EPISODE, NAME,SAVE,DOUBLE,NET_WIDTH,MAX_TORQUE,DYNAMIC_TAU ):
    text = NAME + "\n"
    text += "NUM EPISODES: " + str(NUM_EPISODES) + "\n"
    text += "UPDATE TARGET NET: " + str(UPDATE_TARGET_NET) + "\n"
    text += "TAU:" + str(TAU) + "\n"
    text += "MAX MEM: " + str(MAX_MEM) + "\n"
    text += "ALPHA: " + str(ALPHA) + "\n"
    text += "GAMMA: " + str(GAMMA) + "\n"
    text += "BATCH SIZE: " + str(BATCH_SIZE) + "\n"
    text += "MAX STEPS: " + str(MAX_STEPS) + "\n"
    text += "EXPLORE MIN: " + str(EXPLORE_MIN) + "\n"
    text += "EXPLORE MAX: " + str(EXPLORE_MAX) + "\n"
    text += "EXPLORE LINEAR DECAY: " + str(EXPLORE_LINEAR_DECAY) + "\n"
    text += "TRAIN PER EPISODE: " + str(TRAIN_PER_EPISODE) + "\n"
    text += "DOUBLE: " + str(DOUBLE) + "\n"
    text += "NET WIDTH: " + str(NET_WIDTH) + "\n"
    text += "MAX TORQUE: " + str(MAX_TORQUE) + "\n"
    text += "DYNAMIC TAU: " + str(DYNAMIC_TAU) + "\n"

    with open("Settings/" + NAME + ".txt", 'w') as f:
        f.write(text)

def train(NUM_EPISODES=1000, \
    UPDATE_TARGET_NET = 1000, # ignored if tau != 0
    TAU = 1e-4,
    MAX_MEM = 10000,
    ALPHA = 1e-5, 
    GAMMA = 0.9999,
    BATCH_SIZE = 1024,
    MAX_STEPS = 100,
    EXPLORE_MIN = 0.01,
    EXPLORE_MAX = 1.0,
    EXPLORE_LINEAR_DECAY = False, # option to have linear vs exponential exploration decay
    TRAIN_PER_EPISODE = False, # option to train per episode or per step
    NAME = "Default",
    SAVE = 1000,
    DOUBLE = False,
    NET_WIDTH = 2,
    MAX_TORQUE = 10.0,
    DYNAMIC_TAU = False,
    pretrained = False,
    ):

    if torch.cuda.is_available():
        device_id = torch.device('cuda')
    else:
        device_id = torch.device('cpu')
        
    print('Device in use is:', device_id)

    global writer
    writer = SummaryWriter(log_dir="runs/" + NAME)
    EXPLORE_DECAY = exp( log(EXPLORE_MIN/EXPLORE_MAX)/NUM_EPISODES ) 
    NAME_OPT = NAME + "_OPT"
    explore_rate = EXPLORE_MAX + 0.0
    env = DoublePendulum(make_single=(not DOUBLE), max_torque=MAX_TORQUE)
    # observation_space = env.q.shape
    observation_space = env.x.shape
    agent = DQNAgent(state_space=observation_space,
                     action_space=env.nu,
                     max_memory_size=MAX_MEM,
                     batch_size=BATCH_SIZE,
                     gamma=GAMMA,
                     lr=ALPHA,
                     device = device_id,
                     pretrained=pretrained,
                     tau=TAU,
                     dynamic_tau = DYNAMIC_TAU,
                     width = NET_WIDTH
                     )

    for i in tqdm(range(NUM_EPISODES)):
        state = env.reset()
        state = torch.Tensor([state]).to(device_id)
        steps = 0
        total_cost = 0.0
        g = 1.0
        while steps < MAX_STEPS:
            action = agent.choose_action(state, explore_rate)
            steps += 1
            state_next, cost, terminal, _, _ = env.step(action.item())
            total_cost += g* cost
            g *= GAMMA
            state_next = torch.Tensor([state_next]).to(device_id)
            cost = torch.Tensor([cost]).to(device_id)
            
            # if steps == MAX_STEPS: terminal = True
            terminal = torch.Tensor([int(terminal)]).to(device_id)
            agent.remember(state, action, cost, state_next, terminal)
            state = state_next
            if terminal:
                break
            if ((i +1) % SAVE == 0): env.render()

            if not TRAIN_PER_EPISODE: agent.replay_experience(1)
        if TRAIN_PER_EPISODE: agent.replay_experience(1)

        # print(agent.prev_loss)
        # log total rewards 
        writer.add_scalar("Total Episode Cost", total_cost, i)

        # epsilon decay
        if EXPLORE_LINEAR_DECAY: explore_rate -= (EXPLORE_MAX - EXPLORE_MIN) / float(NUM_EPISODES)
        else: explore_rate *= EXPLORE_DECAY
        explore_rate =  max(explore_rate, EXPLORE_MIN)

        # update target net
        if i % UPDATE_TARGET_NET == 0 and TAU == 0: agent.update_target_net()

        # Save the trained memory so that we can continue from where we stop using 'pretrained' = True
        if ((i +1) % SAVE == 0):
            env.show((i + 1)/SAVE)
            print(explore_rate)
            torch.save(agent.policy_net.state_dict(), "Models/" + NAME + ".pt")
            torch.save(agent.optimizer.state_dict(), "Models/" + NAME_OPT + ".pt")
            print("okay")
    save_settings_to_file(NUM_EPISODES,UPDATE_TARGET_NET,TAU,MAX_MEM,ALPHA, GAMMA,BATCH_SIZE,MAX_STEPS,EXPLORE_MIN,EXPLORE_MAX,EXPLORE_LINEAR_DECAY,TRAIN_PER_EPISODE, NAME,SAVE,DOUBLE,NET_WIDTH,MAX_TORQUE,DYNAMIC_TAU )
    env.close()

def evaluate(NAME="Default", DOUBLE = False, MAX_TORQUE=10.0, NET_WIDTH=2):

    # load model
    env = DoublePendulum(make_single=(not DOUBLE), max_torque=MAX_TORQUE)
    observation_space = env.x.shape
    model = DQNSolver(observation_space, env.nu, NET_WIDTH)
    model.load_state_dict(torch.load("Models/" + NAME + ".pt"))
    model.eval()

    # run starting from 0 and generate trajectory
    env.q = np.array([0.0,0.0,0.0,0.0])
    env._update_x()
    state = torch.Tensor([env.x])
    # state = torch.Tensor([env.reset()])
    steps = 1000
    q_hist = np.zeros((steps+1, env.q.shape[0]))
    x_hist = np.zeros((steps+1, env.x.shape[0]))
    u_hist = np.zeros(steps)
    q_hist[0] = env.q
    total_cost = 0
    for i in range(steps):
        u = torch.argmin(model(state), 1).item()
        # u = env.nu / 2
        state_next, cost, terminal, _, _ = env.step(u)
        u_hist[i] = env.u
        q_hist[i+1] = env.q
        total_cost += cost # UNDISCOUNTED
        state = torch.Tensor([state_next])
        env.render()
    env.show(0, NAME)
    plot_control(u_hist, NAME)
    print("TOTAL COST FOR " + NAME + " IS: " + str(total_cost))
    plot_trajectory(q_hist, total_cost,env, NAME)
    if not DOUBLE:
        colour_plot(model, env, NAME)


if __name__ == "__main__":
    # train()
    evaluate()