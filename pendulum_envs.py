import gym
import numpy as np
from gym.spaces import *
from math import *
from PIL import *
import torch

class DoublePendulum(gym.Env):

    def __init__(self, make_single=False, max_torque=10.0, nu=11):
        self.DT = 0.05 # corresponds to control frequency
        self.updates_per_step = 1 # corresponds to how many steps of physics between each control loop

        self.m1 = 1.0 # mass of pendulum 1 (kg)
        self.l1 = 1.0 # length of pendulum 1 (m)

        self.m2 = 1.0 # mass of pendulum 2 (kg)
        self.l2 = 1.0 # length of pendulum 2 (m)

        self.maxv = 8

        self.max_u = max_torque
        self.nu = nu
        if make_single:
            self.m2 = 0.001
            self.l2 = 0.001

        self.single = make_single
        self.frix = 0.9

        self.q = torch.zeros(4) # [th1, th2, dth1, dth2]
        self.x = torch.zeros(8)
        self.u_vec = torch.zeros((2,1))
        self._update_x()
        self.u = 0.0

        self.images = []
        self.last_cost =0.0

    def intu2u(self, u):
        newU = (float(u)- float(self.nu - 1)/2.0) * float(self.max_u * 2 ) / (float(self.nu) -1)
        # print(newU)
        return newU

    def step(self, action):
        dt = self.DT / float(self.updates_per_step)
        terminal = False
        for i in range(self.updates_per_step):
            self.dynamics_update(dt, self.intu2u(action))
        self._update_x()
        reward = self._calculate_reward()
        # self.calcEnergy()
        # return self.q, reward, terminal, False, 0
        return self.x, reward, terminal, False, 0
    
    def distance(self,a, b):
        x = (a - b) % (2*np.pi)
        if x > np.pi: x-= (2*np.pi)
        elif x < - np.pi: x+= (2*np.pi)
        return x

    def _calculate_reward(self):
        # VEL_WEIGHT = 0.01
        # U_WEIGHT = 0.0001
        # cost = U_WEIGHT*(self.u**2)
        # A = self.distance(self.q[0], torch.pi) ** 2
        # B = self.distance(self.q[1], 0)**2
        # C =  self.q[2] **2
        # D =  self.q[3] **2
        # cost += A
        # cost += VEL_WEIGHT * C
        # if not self.single: cost += B + VEL_WEIGHT * D
        # ## help convergence by making sure cost is between -1, 1 and clip it
        # cost /= 10.0
        # cost -= 1.0
        # cost = torch.clip(cost, -1.0, 1.0)
        # self.last_cost = cost
        U, T  = self.calcEnergy()
        cost = T - U
        cost/= 100.0
        self.last_cost = cost
        return cost
    
    def _update_x(self):
        self.x[0] = self.l1 * torch.sin( self.q[0])
        self.x[1] = -self.l1 * torch.cos(self.q[0])
        self.x[2] = self.x[0] + self.l2 * torch.sin(self.q[1]+ self.q[0])
        self.x[3] = self.x[1] - self.l2 * torch.cos(self.q[1]+self.q[0])

        self.x[4] = self.l1 * self.q[2]* torch.cos( self.q[0])
        self.x[5] = self.l1 * self.q[2]* torch.sin(self.q[0])

        self.x[6] = self.x[4] + self.l2 * torch.cos(self.q[1]+ self.q[0]) * (self.q[2] + self.q[3])
        self.x[7] = self.x[5] + self.l2 * torch.sin(self.q[1]+ self.q[0]) * (self.q[2] + self.q[3])

    def reset(self, gaussian=False):
        if not gaussian:
            randangles = np.random.uniform(0, 2* torch.pi, size=2)
            self.q[0] = randangles[0]
            self.q[1] = randangles[1]
            self.q[2] = np.random.uniform(-self.maxv, self.maxv)
            self.q[3] = np.random.uniform(-self.maxv, self.maxv)
        else: # Since the pendulum is more likely by physics to be in the lower state the area around the optimum will get visited less
            #   We can try to adjust for this using sampling that favours the desired end state 
            #   knowing that most of the time it will fall and visit the other states anyway
            randangles = np.random.normal(torch.pi, 1.0, size=2)
            self.q[0] = randangles[0]
            self.q[1] = randangles[1]
            self.q[2] = np.random.normal(0, 1)
            self.q[3] = np.random.normal(0, 1)
        self._update_x()
        self.history = []
        self.images = []
        # return self.q
        return self.x

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

    def show(self, i, NAME=None):
        if NAME is None:
            self.images[0].save('gifs/pend'+ str(int(i)) + '.gif',
            save_all=True, append_images=self.images[1:], loop=0)
        else:
            self.images[0].save('Graphs/Eval Gifs/'+ NAME + '.gif',
            save_all=True, append_images=self.images[1:], loop=0)
    
    def calcM(self): # M(q)
        m2l22 = self.m2 * (self.l2 **2)
        m2l1l2c2 = self.m2 * self.l2 * self.l1 * torch.cos(self.q[1])
        self.M = torch.zeros((2,2))
        self.M[0,0] = (self.m1 + self.m2) * (self.l1**2) + m2l22 + 2*m2l1l2c2
        self.M[0,1] = m2l22 + m2l1l2c2
        self.M[1,0] = m2l22 + m2l1l2c2
        self.M[1,1] = m2l22
        self.Minv = torch.linalg.inv(self.M)
    
    def calcC(self): # C(qdot, q)
        self.C = torch.zeros((2,2))
        x = self.m2 * self.l1 * self.l2 * torch.sin(self.q[1])
        self.C[0,1] = - x * (2*self.q[2] + self.q[3])
        self.C[1,0] =  0.5 * x * (2*self.q[2] + self.q[3])
        self.C[1,1] = - 0.5* x * self.q[2]
    
    def calcTg(self): # tau_g(q) forces of gravity
        self.tau_g = torch.zeros((2,1))
        g2 = self.m2 * self.l2 * torch.sin(self.q[0] + self.q[1])
        self.tau_g[0] = (self.m1 + self.m2) * self.l1 * torch.sin(self.q[0]) + g2
        self.tau_g[1] = g2
        self.tau_g *= -9.81
    
    def calcTf(self): # tau_f(q dot) force due to friction
        self.tau_f = torch.zeros((2,1))
        self.tau_f[0] = self.frix *(  self.q[2])
        self.tau_f[1] = self.frix *(  self.q[3])
        
    def calcAccel(self):
        self.qdot = torch.zeros((2,1))
        self.qdot[0] = self.q[2]
        self.qdot[1] = self.q[3]
        self.A = self.Minv @ (self.tau_g + self.u_vec - (self.C @ self.qdot)) - self.tau_f
        
    
    def dynamics_update(self,dt, u):
        self.calcM()
        self.calcTg()
        self.calcTf()
        self.calcC()
        self.u = u
        self.u_vec[0] = u
        self.calcAccel()
        self.q[2] += dt * self.A[0].item()
        self.q[3] += dt * self.A[1].item()
        self.q[2] = torch.clip(self.q[2], - self.maxv, self.maxv)
        self.q[3] = torch.clip(self.q[3], - self.maxv, self.maxv)
        self.q[0] += (self.q[2] * dt)
        self.q[1] += (self.q[3] * dt)
        

    def calcEnergy(self):
        U = self.m1 * 9.81 * self.x[1]
        U += self.m2 * 9.81 * self.x[3]
        # print("potential energy = ", U)
        
        T = (self.x[4]**2) * 0.5 * self.m1
        T += (self.x[5]**2) * 0.5 * self.m1
        T += (self.x[6]**2) * 0.5 * self.m2
        T += (self.x[7]**2) * 0.5 * self.m2
        # print("kinetic energy = ", T)
        return (U,T)


if __name__ == "__main__":
    d = DoublePendulum()
    for i in range(11):
        print(d.intu2u(i))