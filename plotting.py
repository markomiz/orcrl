import matplotlib.pyplot as plt
import numpy as np

def colour_plot(model, env, name="Default"): # only for use with single pendulum
    SIZE = 36.0
    data = np.zeros((SIZE,SIZE))
    # for each disretisation of state space evaluate the model 
    for q in np.arange(-np.pi, np.pi, 2* np.pi / SIZE):
        for q_dot in np.arange(-env.maxv, env.maxv, 2 * env.maxv/SIZE):
            state = np.array([q,q_dot])
            data[q,q_dot] = np.argmin(model(state))
    
    fig, ax = plt.subplots()
    ax.imshow(data)
    plt.savefig("Graphs/ColourPlots/" + name + ".png")
    
    

def plot_trajectory(data, total_cost, name="Default"):
    n_q = data.shape[1]
    labels = ["q 1", "q dot 1"]
    if n_q == 4: labels = ["q 1", "q 2", "q dot 1", "q dot 2"]
    for i in range(n_q):
        plt.plot(data[:,i], np.arange(), labels=labels[i])

    plt.savefig("Graphs/Trajectories/" + name + ".png")
    # TODO

# any other plots