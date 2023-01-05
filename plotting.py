import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage

def colour_plot(model, env, name="Default"): # only for use with single pendulum
    # with plt.xkcd():
    SIZE = 128
    data_value = np.zeros((SIZE,SIZE))
    data_control = np.zeros((SIZE,SIZE))
    Q = np.arange(0, 2*np.pi, 2* np.pi / SIZE)
    Qdot = np.arange(-env.maxv, env.maxv, 2 * env.maxv/SIZE)
    extent = (-env.maxv, env.maxv,0, 2*np.pi )
    # for each disretisation of state space evaluate the model 
    for q in np.arange(SIZE):
        for q_dot in np.arange(SIZE):
            env.q[0] = Q[q]
            env.q[2] = Qdot[q_dot]
            env._update_x()
            state = env.x
            state = state.unsqueeze(0)
            data_value[q,q_dot] = torch.min(model(state)).item()
            data_control[q,q_dot] = env.intu2u(torch.argmin(model(state)))
    

    plt.imshow(data_value, extent=extent, aspect=8/(np.pi))
    plt.ylabel("Joint Position (rad)")
    plt.xlabel("Joint Velocity (rad/s)")
    plt.title("Value Function")
    cbar = plt.colorbar()
    cbar.set_label("TD Cost")
    plt.savefig("Graphs/ColourPlots/" + name + "_value.png")
    plt.clf()
    plt.imshow(data_control, extent=extent, aspect=8/(np.pi))
    plt.ylabel("Joint Position (rad)")
    plt.xlabel("Joint Velocity (rad/s)")
    plt.title("Policy")
    cbar = plt.colorbar()
    cbar.set_label("Control")
    plt.savefig("Graphs/ColourPlots/" + name + "_control.png")
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
    plt.axhline(0,color="black", alpha=0.3)
    if n_q == 4: labels = ["q 1", "q 2", "q dot 1", "q dot 2"]
    for i in range(0,n_q, (env.single + 1)):
        plt.plot(range(data.shape[0]), data[:,i], label=labels[i] )
    plt.xlabel("Step")
    plt.ylabel("Joint angle (rad) / Joint velocity (rad/s)")
    
    plt.legend()
    plt.savefig("Graphs/Trajectories/" + name + ".png")
    plt.clf()


# any other plots