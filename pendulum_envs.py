# Import required libararies
from pickletools import uint1
import gym
import matplotlib.pyplot as plt 
import random
import numpy as np
from math import sin, cos
from gym.spaces import *

from math import *
from PIL import *

class DoublePendulum(gym.Env):

    def __init__(self, make_single=False, max_torque=10.0, nu=21):
        self.DT = 0.05 # corresponds to control frequency
        self.updates_per_step = 1 # corresponds to how many steps of physics between each control loop

        self.m1 = 1.0 # mass of pendulum 1 (kg)
        self.l1 = 1.0 # length of pendulum 1 (m)

        self.m2 = 1.0 # mass of pendulum 2 (kg)
        self.l2 = 1.0 # length of pendulum 2 (m)

        self.maxv = 8
        self.max_cost = 0

        self.max_u = max_torque
        self.nu = nu
        if make_single:
            self.m2 = 0
            self.l2 = 0.01

        self.single = make_single
        self.g = 9.81 # gravity (m/s)
        self.frix = 0.1 # friction coef

        self.q = np.zeros(4) # [th1, th2, dth1, dth2]
        self.x = np.zeros(8)
        self._update_x()
        self.u = 0.0

        self.history = []
        self.saving_history = False
        self.images = []
        self.last_cost =0.0

    def intu2u(self, u):
        newU = (np.double(u)- np.double(self.nu)/2.0) * (self.max_u * 2 ) / np.double(self.nu) 
        # print(newU)
        return newU

    def dynamics_update(self,dt, u):
        try:
            acc1 = (-self.m2*self.l1*self.q[2]**2*np.sin(self.q[0] - self.q[1])*np.cos(self.q[0] - self.q[1]) + self.m2*self.g*np.sin(self.q[1])*np.cos(self.q[0] - self.q[1]) - self.m2*self.l2*self.q[3]**2*np.sin(self.q[0] - self.q[1]) - (self.m1 + self.m2)*self.g*np.sin(self.q[0])) / ((self.m1 + self.m2)*self.l1 - self.m2*self.l1*np.cos(self.q[0] - self.q[1])**2) - self.frix * np.sign(self.q[2])
            acc2 = (self.m2*self.l2*self.q[3]**2*np.sin(self.q[0] - self.q[1])*np.cos(self.q[0] - self.q[1]) + (self.m1 + self.m2)*self.g*np.sin(self.q[0])*np.cos(self.q[0] - self.q[1]) + self.l1*self.q[2]**2*np.sin(self.q[0]-self.q[1])*(self.m1 + self.m2) - self.g*np.sin(self.q[1])*(self.m1 + self.m2)) / (self.l2*(self.m1 + self.m2) - self.m2*self.l2*np.cos(self.q[0] - self.q[1])**2) - self.frix * np.sign(self.q[3])
            # yes it's horrible but it's not about the pendulum simulation is it? :)

            acc1 += u
            self.u = u

            self.q[2] += dt * acc1
            self.q[3] += dt * acc2
            self.q[2] = np.clip(self.q[2], - self.maxv, self.maxv)
            self.q[3] = np.clip(self.q[3], - self.maxv, self.maxv)
            self.q[0] += (self.q[2] * dt)
            self.q[1] += (self.q[3] * dt)
            self.q[0] = self.q[0] % (np.pi * 2)
            self.q[1] = self.q[1] % (np.pi * 2)

            self._update_x()

            if self.saving_history:
                self.history.append(self.q)
            return True
        except Exception as e:
            print(self.q)
            print(u)
            print(e)
            return False 

    def step(self, action):
        dt = self.DT / np.double(self.updates_per_step)
        terminal = False
        for i in range(self.updates_per_step):
            if not self.dynamics_update(dt, self.intu2u(action)):
                terminal = True
                break

        reward = self._calculate_reward()
        return self.q, reward, terminal, False, 0
        # return self.x, reward, terminal, False, 0
    
    def _calculate_reward(self):
        cost = 0.0001*self.u
        A = (self.q[0] - np.pi )**2
        # B = min(self.q[1], np.pi * 2 - self.q[1]) ** 2 # since we are keeping the angle between 0 and 2 pi
        B = (self.q[1] - np.pi )**2
        C = 0.01 * (self.q[2] **2) **2
        D = 0.01 * (self.q[3] **2) **2
        cost += A
        cost += 0.1 * C
        if not self.single: cost += B + .1 * D
        ## help convergence by making sure cost is between -1, 1 and clip it
        cost /= 6.0
        cost -= 1.0
        cost = np.clip(cost, -1.0, 1.0)
        self.last_cost = cost

        return cost
    
    def _update_x(self):
        self.x[0] = self.l1 * cos(- np.pi/ 2+ self.q[0])
        self.x[1] = self.l1 * sin(- np.pi/ 2+self.q[0])
        self.x[2] = self.x[0] + self.l2 * cos(- np.pi/ 2+self.q[1])
        self.x[3] = self.x[1] + self.l2 * sin(- np.pi/ 2+self.q[1])
        self.x[4] = self.l1 * cos(- np.pi/ 2+ self.q[2])
        self.x[5] = self.l1 * sin(- np.pi/ 2+self.q[2])
        self.x[6] = self.x[4] + self.l2 * cos(- np.pi/ 2+self.q[3])
        self.x[7] = self.x[5] + self.l2 * sin(- np.pi/ 2+self.q[3])

    def reset(self, gaussian=True):
        if not gaussian:
            randangles = np.random.uniform(0, 2* np.pi, size=2)
            self.q[0] = randangles[0]
            self.q[1] = randangles[1]
            self.q[2] = np.random.uniform(-self.maxv, self.maxv)
            self.q[3] = np.random.uniform(-self.maxv, self.maxv)
        else: # Since the pendulum is more likely by physics to be in the lower state the area around the optimum will get visited less
            #   We can try to adjust for this using sampling that favours the desired end state 
            #   knowing that most of the time it will fall and visit the other states anyway
            randangles = np.random.normal(np.pi, 1.0, size=2)
            self.q[0] = randangles[0]
            self.q[1] = randangles[1]
            self.q[2] = np.random.normal(0, 1)
            self.q[3] = np.random.normal(0, 1)
        self._update_x()
        self.history = []
        self.images = []
        return self.q
        # return self.x

    def render(self):
        origin = (110,100)
        j1 = (self.x[0]* 50 + origin[0]  , -self.x[1]* 50 + origin[1] ) 
        j2 = (self.x[2]* 50 + origin[0]  , -self.x[3]* 50 + origin[1]  )
        all_points = [origin, j1, j2]

        control_points = [(110, 210), (110 + self.u * 100 / self.max_u, 210)]

        cost_points = [(100,1), (100 +  int(self.last_cost* 100), 1)]

        img = Image.new("RGB", (220,220))
        img1 = ImageDraw.Draw(img)  
        img1.line(all_points, fill ="white", width = 2)

        img1.line(control_points, fill ="blue", width = 2)
        col = "red"
        if self.last_cost < 0: col = "green"
        img1.line(cost_points, fill =col, width = 2)

        self.images.append(img)
        # add something for plotting
        return 

    def show(self, i):
        self.images[0].save('gifs/pend'+ str(int(i)) + '.gif',
        save_all=True, append_images=self.images[1:], loop=0)