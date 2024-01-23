
import pickle
import gym
from stable_baselines3 import SAC
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn import preprocessing
from discretization import discretize, create_uniform_grid
import pybullet_envs
import sys 

import argparse
import warnings
warnings.filterwarnings("ignore")


def imp_dimensions(modelid,nu):
    '''
    params : 
    modelid  : number associated with the environment.
    nu : hyperparameter \nu of the FAC algorithm
    This function finds the important state dimensions
    using the intial random samples stored in the replay 
    buffer till the "learning starts" parameter kicks in.  
    '''
    
    model=int(modelid)
    nu = float(nu)

    if model==1:
        env = gym.make('Pendulum-v1')
        dbfile1 = open('sac_Pendulum-v1_replay_buffer_default_1.pkl', 'rb') 
        low = np.array([-1.0,-1.0,-8.0])
        high = np.array([1.0,1.0,8.0])
        sbins = 20*np.ones(3).astype(int)
        abins = 20*np.ones(1).astype(int)
        obs_grid = create_uniform_grid(low, high, bins=tuple(sbins))
        act_grid = create_uniform_grid(env.action_space.low, env.action_space.high, bins=tuple(abins))
    elif model==2:
        print("mc")
        env = gym.make('MountainCarContinuous-v0')
        dbfile1 = open('sac_MountainCarContinuous-v0_replay_buffer_default_1.pkl', 'rb') 
        low = np.array([-2.0,-1.0])
        high = np.array([1.0,1.0])
        sbins = 20*np.ones(2).astype(int)
        abins = 20*np.ones(1).astype(int)
        obs_grid = create_uniform_grid(low, high, bins=tuple(sbins))
        act_grid = create_uniform_grid(env.action_space.low, env.action_space.high, bins=tuple(abins))
    elif model==3:
        env = gym.make('LunarLanderContinuous-v2')
        dbfile1 = open('sac_LunarLanderContinuous-v2_replay_buffer_default_1.pkl', 'rb') 
        low = np.array([-1.0,-1.0,-3.0,-2.0,-3.0,-6.0,0.0,0.0])
        high = np.array([1.0,2.0,3.0,1.0,3.0,5.0,1.0,1.0])
        sbins = 20*np.ones(8).astype(int)
        abins = 20*np.ones(2).astype(int)
        obs_grid = create_uniform_grid(low, high, bins=tuple(sbins))
        act_grid = create_uniform_grid(env.action_space.low, env.action_space.high, bins=tuple(abins))
    elif model==4:
        env = gym.make('ReacherBulletEnv-v0')
        dbfile1 = open('sac_ReacherBulletEnv-v0_replay_buffer_default_1.pkl', 'rb') 
        low = np.array([-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-3.0,-2.0,-6.0])
        high = np.array([1.0,1.0,1.0,1.0,1.0,1.0,4.0,2.0,6.0])
        sbins = 20*np.ones(9).astype(int)
        abins = 20*np.ones(2).astype(int)
        obs_grid = create_uniform_grid(low, high, bins=tuple(sbins))
        act_grid = create_uniform_grid(env.action_space.low, env.action_space.high, bins=tuple(abins))
    elif model==5:
        env = gym.make('Swimmer-v3')
        dbfile1 = open('sac_Swimmer-v3_replay_buffer_default_1.pkl', 'rb') 
        low = np.array([-2.0,-2.0,-2.0,-3.0,-4.0,-4.0,-7.0,-6.0])
        high = np.array([3.0,2.0,2.0,3.0,3.0,4.0,6.0,7.0])
        sbins = 20*np.ones(8).astype(int)
        abins = 20*np.ones(2).astype(int)
        obs_grid = create_uniform_grid(low, high, bins=tuple(sbins))
        act_grid = create_uniform_grid(env.action_space.low, env.action_space.high, bins=tuple(abins))
    

    elif model==6:
        env = gym.make('Hopper-v3')
        dbfile1 = open('sac_Hopper-v3_replay_buffer_default_1.pkl', 'rb')  
        low = np.array([0.0,-1.0,-2.0,-1.0,-1.0,-2.0,-3.0,-8.0,-8.0,-8.0,-7.0])
        high = np.array([2.0,1.0,1.0,1.0,1.0,2.0,1.0,7.0,7.0,6.0,8.0])
        sbins = 10*np.ones(11).astype(int)
        abins = 10*np.ones(3).astype(int)
        obs_grid = create_uniform_grid(low, high, bins=tuple(sbins))
        act_grid = create_uniform_grid(env.action_space.low, env.action_space.high, bins=tuple(abins))
        
    elif model==7:
        env = gym.make('Walker2d-v3')
        dbfile1 = open('sac_Walker2d-v3_replay_buffer_default_1.pkl', 'rb')
        low = np.array([0.0,-1.0,-2.0,-3.0,-2.0,-2.0,-3.0,-2.0,-3.0,-6.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0])
        high = np.array([2.0,1.0,1.0,1.0,2.0,1.0,1.0,2.0,2.0,2.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0])
        sbins = 20*np.ones(17).astype(int)
        abins = 20*np.ones(6).astype(int) 
        obs_grid = create_uniform_grid(low, high, bins=tuple(sbins))
        act_grid = create_uniform_grid(env.action_space.low, env.action_space.high, bins=tuple(abins))
    elif model==8:
        env = gym.make('Ant-v3')
        dbfile1 = open('sac_Ant-v3_replay_buffer_default_1.pkl', 'rb') 
        low = np.array([0.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-2.0,-1.0,-2.0,-1.0,-1.0,-4.0,-4.0,-4.0,-7.0,-13.0,-7.0,-16.0,-14.0,-16.0,-16.0,-16.0,-18.0,-16.0,-14.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        high = np.array([1.0,1.0,1.0,1.0,1.0,1.0,2.0,1.0,1.0,1.0,1.0,1.0,2.0,4.0,3.0,6.0,9.0,9.0,7.0,16.0,17.0,17.0,14.0,17.0,14.0,16.0,18.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        sbins = 20*np.ones(111).astype(int)
        abins = 20*np.ones(8).astype(int)
        obs_grid = create_uniform_grid(low, high, bins=tuple(sbins))
        act_grid = create_uniform_grid(env.action_space.low, env.action_space.high, bins=tuple(abins))

    
    elif model==9:
        env = gym.make('Humanoid-v3')
        dbfile1 = open('sac_Humanoid-v3_replay_buffer_default_1.pkl', 'rb')  
        low = np.array([1.0,0.0,-1.0,-1.0,-1.0,-1.0,-2.0,-1.0,-1.0,-1.0,-2.0,-3.0,-1.0,-1.0,-2.0,-3.0,-1.0,-2.0,-2.0,-2.0,-1.0,-2.0,-1.0,-2.0,-4.0,-5.0,-8.0,-9.0,-10.0,-17.0,-14.0,-11.0,-14.0,-22.0,-31.0,-11.0,-15.0,-21.0,-31.0,-10.0,-14.0,-11.0,-11.0,-11.0,-12.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,-1.0,-1.0,-1.0,-2.0,-2.0,2.0,8.0,0.0,0.0,0.0,-1.0,-1.0,-1.0,-1.0,-1.0,0.0,2.0,0.0,0.0,0.0,-1.0,-1.0,-1.0,-2.0,-1.0,-1.0,5.0,0.0,0.0,0.0,-1.0,-1.0,-1.0,-1.0,-1.0,-2.0,4.0,0.0,0.0,0.0,-1.0,-1.0,-1.0,-1.0,-1.0,-2.0,2.0,0.0,0.0,0.0,-1.0,-1.0,-1.0,-1.0,-1.0,-2.0,1.0,0.0,0.0,0.0,-1.0,-1.0,-1.0,-1.0,-1.0,-2.0,4.0,0.0,0.0,0.0,-1.0,-1.0,-1.0,-1.0,-1.0,-2.0,2.0,0.0,0.0,0.0,-1.0,-1.0,-1.0,-1.0,-1.0,-2.0,1.0,0.0,0.0,0.0,-1.0,-1.0,-1.0,-1.0,-1.0,0.0,1.0,0.0,0.0,0.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,1.0,0.0,0.0,0.0,-1.0,-1.0,-1.0,-1.0,-1.0,0.0,1.0,0.0,0.0,0.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,-6.0,-8.0,-9.0,-4.0,-2.0,-4.0,-10.0,-12.0,-11.0,-2.0,-2.0,-4.0,-10.0,-12.0,-11.0,-2.0,-2.0,-4.0,-10.0,-16.0,-18.0,-3.0,-2.0,-5.0,-12.0,-17.0,-17.0,-10.0,-6.0,-6.0,-12.0,-17.0,-17.0,-10.0,-6.0,-6.0,-8.0,-15.0,-15.0,-3.0,-2.0,-4.0,-9.0,-15.0,-15.0,-10.0,-6.0,-4.0,-9.0,-15.0,-15.0,-10.0,-6.0,-4.0,-7.0,-9.0,-8.0,-5.0,-4.0,-4.0,-7.0,-9.0,-9.0,-5.0,-4.0,-5.0,-10.0,-9.0,-10.0,-5.0,-4.0,-5.0,-10.0,-8.0,-12.0,-5.0,-4.0,-5.0,0.0,0.0,0.0,0.0,0.0,0.0,-40.0,-40.0,-40.0,-40.0,-40.0,-120.0,-80.0,-40.0,-40.0,-120.0,-80.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        high = np.array([2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,1.0,1.0,1.0,2.0,1.0,2.0,1.0,1.0,6.0,8.0,8.0,11.0,14.0,12.0,11.0,14.0,22.0,29.0,11.0,12.0,22.0,28.0,11.0,10.0,10.0,10.0,12.0,8.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,3.0,3.0,1.0,1.0,1.0,1.0,2.0,2.0,5.0,9.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,3.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,6.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,-0.0,5.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,-0.0,3.0,2.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,-0.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,-0.0,5.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,-0.0,3.0,2.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,-0.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,0.0,0.0,0.0,0.0,0.0,0.0,6.0,8.0,8.0,4.0,3.0,1.0,6.0,9.0,12.0,3.0,2.0,2.0,9.0,9.0,12.0,3.0,2.0,2.0,10.0,13.0,16.0,4.0,2.0,2.0,11.0,21.0,15.0,11.0,6.0,4.0,11.0,21.0,15.0,11.0,6.0,4.0,12.0,15.0,15.0,3.0,2.0,2.0,10.0,24.0,15.0,12.0,5.0,5.0,10.0,24.0,15.0,12.0,5.0,5.0,9.0,11.0,10.0,4.0,5.0,2.0,9.0,12.0,10.0,4.0,5.0,2.0,8.0,9.0,8.0,4.0,4.0,2.0,7.0,12.0,9.0,4.0,4.0,2.0,0.0,0.0,0.0,0.0,0.0,0.0,40.0,40.0,40.0,40.0,40.0,120.0,80.0,40.0,40.0,120.0,80.0,10.0,10.0,10.0,10.0,10.0,10.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        low = low[0:292]
        high = high[0:292]
        sbins = 20*np.ones(292).astype(int)
        abins = 20*np.ones(17).astype(int)
        obs_grid = create_uniform_grid(low, high, bins=tuple(sbins))
        act_grid = create_uniform_grid(env.action_space.low, env.action_space.high, bins=tuple(abins))
    
        

    
    #env = gym.make('Walker2d-v3')
    
    rb1 = pickle.load(dbfile1)
   
    #rb4 = pickle.load(dbfile4)
    #print(db.buffer_size)
    #print(rb1.pos)
    #print(rb2.pos)
    #print(db.observation_space)
    #print(db.action_space)
    
    #print(rb1.observations[0:5])
    #print(db.next_observations[0])
    #print(db.actions[0])
    #print(db.rewards[0])
    print("Number of entries in Replay buffer is : ",rb1.pos)
    #print("size of rb len ",len(rb1.observations))
    #exit()
    data=rb1.observations[0:10000]
    #print(data)
    
    # plt.plot(data[:,:,0],  label = "s[0]", color='yellow')
    # plt.plot(data[:,:,1],  label = "s[1]", color='blue')
    # plt.plot(data[:,:,2],  label = "s[2]", color='red')
    # plt.plot(data[:,:,3],  label = "s[3]", color='green')
    # plt.plot(data[:,:,4],  label = "s[4]", color='orange')
    # plt.plot(data[:,:,5],  label = "s[5]", color='pink')
    # plt.plot(data[:,:,6],  label = "s[6]", color='purple')
    # plt.plot(data[:,:,7],  label = "s[7]", color='gray')
    # ax = plt.axes()
    # ax.set_facecolor("white")
    # ax.yaxis.grid(color='gray', linestyle='dashed')
    # ax.xaxis.grid(color='gray', linestyle='dashed')
    # plt.xlabel("Timesteps",fontsize=18)
    # plt.ylabel("Magnitude",fontsize=18)
    # plt.title("States of Swimmer")
    
    # # Adding legend, which helps us recognize the curve according to it's color
    # plt.legend(prop={'size': 10})
    # plt.savefig('obs_swm.pdf')
    # exit()
    
    #data=rb1.observations[0:10000]
    #print(data)
    #data = [d[0][0:292] for d in data]
    print("Length of state vector (dimensions) : ",len(data[0][0]))
    #print("length of action vector ",len(env.action_space.low))
    #exit()
    
    op = np.array([])
    for i in range(len(data[0][0])):
        y1 = data[:,:,i]
        op = np.append(op,np.floor(np.min(y1)))
    #print(np.floor(op),sep=", ")
    for x in op:
        print(x, end=",")
    print("\n")
    opx = np.array([])
    for i in range(len(data[0][0])):
        y1 = data[:,:,i]
        opx = np.append(opx,np.ceil(np.max(y1)))
    #print(np.ceil(op),sep=", ")
    for x in opx:
        print(x, end=",")
    print("\n")

    
    #zid = np.intersect1d(np.where(op.astype(int)==0), np.where(opx.astype(int)==0))
    #allid = np.arange(len(data[0][0]))
    # ids of zero cols 
    #print(zid)
    #nzid = np.setdiff1d(allid, zid)
    #print(nzid)
    #exit()
    
    z = 10000  #sample_size
    base = 0 #base id
    data=rb1.observations[base:base+z]

    
    #data=rb1.observations[base:base+z]
    #data = [d[0][nzid] for d in data]
    #print(data)
    all_obs = []
    for i in range(z):
        #das = discretize(rb1.observations[base+i][0], obs_grid)
        #all_obs.append(das)      
        all_obs.append(rb1.observations[base+i][0])
        #exit()  
    obs = np.array(all_obs)
    #print(obs.shape)
    q, r, p = linalg.qr(obs, pivoting=True)
    print("Permutation matrix E : ",p)
    print("Diagonal elements of matrix R : ",np.diagonal(r))
    cut = np.abs(nu*np.diagonal(r)[0]) 
    lst= np.diagonal(r)[np.abs(np.diagonal(r))>=cut]
    l2 = int(np.ceil(len(np.diagonal(r))*0.04)) #condition to ensure a basic minimum for very large vector 
   
    print("Permissible limit of Singular value : ",cut)
    print("lst ",lst)
    llst = max(len(lst),l2)
    #print("Sin ",llst)
    #print("len of lst ",np.max(len(lst),np.ceil(len(data[0][0])/25)))
    print("Index of state dimensions satisfying the limit : ",p[0:llst])
    exit()

    '''
    all_acs = []
   
    for i in range(z):
        das = discretize(rb1.actions[base+i][0], act_grid)
        all_acs.append(das)
        #all_acs.append(rb1.actions[base+i][0])
        #print(db.observations[0][0])
        #print(db.observations[1][0])
        #exit()  
    acs = np.array(all_acs)
    #print(obs.shape)
    q, r, p2 = linalg.qr(acs, pivoting=True)
    print(p2)
    print(np.diagonal(r))
    cut = np.abs(0.5*np.diagonal(r)[0]) # state len 4 for humanoid with 0.5
    lst1= np.diagonal(r)[np.abs(np.diagonal(r))>=cut]
    print(p2[0:len(lst1)])
    #exit()
    '''
    print("##########################################")
    op = []
    print(len(data[0][0]))
    for i in range(len(data[0][0])):
        y1 = data[:,:,i]
        op.append(np.min(y1))
        #print(len(y1))
    #print(np.floor(op),sep=", ")
    #print(lst)
    #print(lst[0])
    data=rb1.observations[base:base+z]
    #print(rb1.observations[5000][0])
    #print(data[:,:,p[0:len(lst)][0]])
    op = [op[x] for x in p[0:len(lst)]]
    for x in op:
        print(np.floor(x), end=",")
    print("\n")
    op = []
    for i in range(len(data[0][0])):
        y1 = data[:,:,i]
        op.append(np.max(y1))
    #print(np.ceil(op),sep=", ")
    op = [op[x] for x in p[0:len(lst)]]
    for x in op:
        print(np.ceil(x), end=",")
    print("\n")
    exit()
    x = np.arange(0,len(y1))
    plt.plot(x, y1, 'o',  label = "basic", color='red')
    plt.show()

    #0 - [0.5,1.5], 1 - [-.2,.2],  9 [-8,8] 10 [-6,6]

    # p1 = []
    # x = []
    # for i in range(98):
    #     x.append(i)
    #     p1.append(np.linalg.norm(rb1.rewards[10000*(i+1):10000*(i+2)]))
        
    # plt.plot(data, label = "324", color='black')
    # plt.legend()
    # plt.show()
    
    
  
  

    dbfile1.close()
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="Environment", default=2, type=int, required=False)
    parser.add_argument("--nu", help="nu parameter value in [0,1]", default=1.0, type=float, required=False)
  
    args = parser.parse_args()

    env = args.env
    nu = args.nu
    imp_dimensions(env,nu)
    
