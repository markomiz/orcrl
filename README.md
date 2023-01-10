# Pendulum robot control with reinforcement learning
#### Advanced Optimization-based Robot Control project
## Code Structure Overview
There are four python files for this project: learning.py, plotting.py, pendulum_envs.py and main.py.
 - **learning.py** - Here the neural network (DQNSolver)  and agent (DQNAgent) classes are defined as well as the train() function and evaluate() function used for the high-level training and evaluation algorithms.
 - **pendulum_envs.py** - This is where the DoublePendulum class is defined where the motion of the pendulum is modelled, where the rewards are calculated and where the frames for the gifs of the motion are generated.
 - **plotting.py** - As the name suggests, this is where the code for generating plots is.
 - **main.py** - We use this to set up the code for many experiments - looping over lists of hyperparameters and then running training and evaluation.
