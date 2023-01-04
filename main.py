from learning import *
from time import time

###  SINGE PENDULUM EXPERIMENTS   #########################################

# # default case # 
# t1 = time()
# train()
# t2 = time()
# total_time = t2 - t1 
# evaluate() 
# print("Default time to complete: ", total_time)

print("- TOTAL EPISODES TESTS -")
for eps in [10,200, 2000, 5000]: # 1000 is default
    name = "episodes test"+ str(eps)

    t1 = time()
    train(NUM_EPISODES=eps, NAME=name, SAVE=eps)
    t2 = time()
    total_time = t2 - t1 
    evaluate(name) 
    print(name + " time to complete: ", total_time)

# max torque experiments
print("- MAX TORQUE TESTS -")
for torque in [2.0,5.0,20.0]: # 10.0 is default
    name = "torque test"+ str(torque)
    t1 = time()
    train(MAX_TORQUE=torque, NAME=name)
    t2 = time()
    total_time = t2 - t1 
    evaluate(name, MAX_TORQUE=torque) 
    print(name + " time to complete: ", total_time)

# max buffer experiments
print("- BUFFER SIZE TESTS -")
for mem in [100,1000,100000]: # 10000 is default
    name = "buffer test"+ str(mem)
    t1 = time()
    train(MAX_MEM=mem, NAME=name)
    t2 = time()
    total_time = t2 - t1 
    evaluate(name) 
    print(name + " time to complete: ", total_time)

# learning rate experiments
print("- LEARNING RATE TESTS -")
for lr in [1e-2,1e-3,1e-4,1e-6]: # 1e-5, is default
    name = "lr test"+ str(lr)
    t1 = time()
    train(ALPHA=lr, NAME=name)
    t2 = time()
    total_time = t2 - t1 
    evaluate(name) 
    print(name + " time to complete: ", total_time)

# tau
print("- TAU TESTS -")
for tau in [1e-2,1e-3,1e-5,1e-6]: # 1e-4 is default
    name = "tau test"+ str(tau)
    t1 = time()
    train(TAU=tau, NAME=name)
    t2 = time()
    total_time = t2 - t1 
    evaluate(name) 
    print(name + " time to complete: ", total_time)

# net width
print("- NET WIDTH TESTS -")
for n in [1,2,4,8]: # 2 is default
    name = "net size test"+ str(n)
    t1 = time()
    train(NET_WIDTH=n, NAME=name)
    t2 = time()
    total_time = t2 - t1 
    evaluate(name) 
    print(name + " time to complete: ", total_time)

# gamma test
print("- GAMMA SIZE TESTS -")
for g in [0.5,0.9,0.99]: # 0.9999 is default
    name = "gamma test"+ str(g)
    t1 = time()
    train(GAMMA=g, NAME=name)
    t2 = time()
    total_time = t2 - t1 
    evaluate(name) 
    print(name + " time to complete: ", total_time)

# batch size
print("- BATCH SIZE TESTS -")
for b in [32,128,4096]: # 1024 is default
    name = "batch size test"+ str(b)
    t1 = time()
    train(BATCH_SIZE=b, NAME=name)
    t2 = time()
    total_time = t2 - t1 
    evaluate(name) 
    print(name + " time to complete: ", total_time)

    
###############################################################################
### add all the experiments - run train and eval for everything we want to test.
