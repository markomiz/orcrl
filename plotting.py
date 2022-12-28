import matplotlib.pyplot as plt
import numpy as np
import torch

def colour_plot(model, env, name="Default"): # only for use with single pendulum
    # with plt.xkcd():
    SIZE = 36
    data = np.zeros((SIZE,SIZE))
    Q = np.arange(-np.pi, np.pi, 2* np.pi / SIZE)
    Qdot = np.arange(-env.maxv, env.maxv, 2 * env.maxv/SIZE)
    extent = (-np.pi, np.pi,-env.maxv, env.maxv )
    # for each disretisation of state space evaluate the model 
    for q in range(SIZE):
        for q_dot in range(SIZE):
            state = np.array([Q[q],0, Qdot[q_dot], 0])
            state = torch.Tensor([state])
            data[q,q_dot] = env.intu2u(torch.argmin(model(state)))
    
    plt.imshow(data, extent=extent, aspect=np.pi/8)
    plt.xlabel("Joint Position (rad)")
    plt.ylabel("Joint Velocity (rad/s)")
    # plt.xticks(Q)
    # plt.yticks(Qdot)
    plt.colorbar()
    plt.savefig("Graphs/ColourPlots/" + name + ".png")
    plt.clf()
    

def plot_control(controls, name):
    # with plt.xkcd():
    plt.plot(range(len(controls)), controls)
    plt.xlabel("Step")
    plt.ylabel("Control Torque (Nm)")
    plt.savefig("Graphs/Controls/" + name + ".png")
    plt.clf()

def plot_trajectory(data, total_cost, env, name="Default"):
    # with plt.xkcd():
    n_q = data.shape[1]
    labels = ["q 1", "q dot 1"]
    if n_q == 4: labels = ["q 1", "q 2", "q dot 1", "q dot 2"]
    for i in range(n_q):
        plt.plot(range(data.shape[0]), data[:,i], label=labels[i] )
    plt.xlabel("Step")
    plt.ylabel("Joint angle (rad) / Joint velocity (rad/s)")
    plt.legend()
    plt.savefig("Graphs/Trajectories/" + name + ".png")
    plt.clf()


# any other plots