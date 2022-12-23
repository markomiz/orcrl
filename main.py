from learning import *
from time import *


# max torque experiments
for torque in [2.0,5.0,10.0,20.0]:
    name = "torque_"+ str(torque)
    t1 = time.time()
    train(MAX_TORQUE=torque, NAME=name)
    t2 = time.time()
    total_time = t2 - t1 # in case we want to see how long it took
    evaluate(name) # This should generate plots

# max buffer experiments
for buf_size in [100,1000,10000,100000]:
    name = "buf_size"+ str(torque)
    train(MAX_TORQUE=torque, NAME=name)
    evaluate(name) 
    
# ... etc.
### add all the experiments - run train and eval for everything we want to test.