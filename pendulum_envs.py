# Import required libararies
from pickletools import uint1
import gym
import matplotlib.pyplot as plt 
import random
import numpy as np
from math import sin, cos
from gym.spaces import *

from math import *

class DoublePendulum(gym.Env):

    def __init__(self, make_single=False):
        self.DT = 0.1 # corresponds to control frequency
        self.updates_per_step = 10 # icorresponds to how many steps of physics between each control loop

        self.m1 = 0.8 # mass of pendulum 1 (kg)
        self.l1 = 0.5 # length of pendulum 1 (m)

        self.m2 = 0.8 # mass of pendulum 2 (kg)
        self.l2 = 0.5 # length of pendulum 2 (m)

        self.max_u = 0.3
        self.nu = 13
        if make_single:
            self.m2 = 0
            self.l2 = 0.01

        self.g = 9.81 # gravity (m/s)
        self.frix = 0.01 # friction coef

        self.x = np.zeros(4) # [th1, th2, dth1, dth2]

        self.history = []
        self.saving_history = True

    def intu2u(self, u):
        newU = (float(u)- float(self.nu)/2.0) * self.max_u / float(self.nu) 
        return newU

    def dynamics_update(self,dt, u):
        try:
            acc1 = (-self.m2*self.l1*self.x[2]**2*np.sin(self.x[0] - self.x[1])*np.cos(self.x[0] - self.x[1]) + self.m2*self.g*np.sin(self.x[1])*np.cos(self.x[0] - self.x[1]) - self.m2*self.l2*self.x[3]**2*np.sin(self.x[0] - self.x[1]) - (self.m1 + self.m2)*self.g*np.sin(self.x[0])) / ((self.m1 + self.m2)*self.l1 - self.m2*self.l1*np.cos(self.x[0] - self.x[1])**2) - self.frix * np.sign(self.x[2])
            acc2 = (self.m2*self.l2*self.x[3]**2*np.sin(self.x[0] - self.x[1])*np.cos(self.x[0] - self.x[1]) + (self.m1 + self.m2)*self.g*np.sin(self.x[0])*np.cos(self.x[0] - self.x[1]) + self.l1*self.x[2]**2*np.sin(self.x[0]-self.x[1])*(self.m1 + self.m2) - self.g*np.sin(self.x[1])*(self.m1 + self.m2)) / (self.l2*(self.m1 + self.m2) - self.m2*self.l2*np.cos(self.x[0] - self.x[1])**2) - self.frix * np.sign(self.x[3])
            # yes it's horrible but it's not about the pendulum simulation is it? :)

            acc1 += u

            self.x[2] += dt * acc1
            self.x[3] += dt * acc2
            self.x[0] += self.x[2] * dt
            self.x[1] += self.x[3] * dt
            if self.saving_history:
                self.history.append(self.x)
            return True
        except Exception as e:
            print(e)
            return False 

    def step(self, action):
        dt = self.DT / float(self.updates_per_step)
        terminal = False
        for i in range(self.updates_per_step):
            if not self.dynamics_update(dt, self.intu2u(action)):
                terminal = True
                break

        reward = self._calculate_reward()
        return self.x, reward, False, False, 0
    
    def _calculate_reward(self):
        A = (self.x[0] % (np.pi*2) - np.pi) **2
        B = (self.x[1] % (np.pi*2) ) **2
        return  A + B

    def reset(self):
        randangles = np.random.uniform(-np.pi, np.pi, size=2)
        self.x[0] = randangles[0]
        self.x[1] = randangles[1]
        self.x[2] = 0.0
        self.x[3] = 0.0
        print(self.x)
        return self.x

    def render(self):
        ... #add something for plotting
        return self.top_view


