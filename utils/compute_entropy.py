
import pickle
import gym
from stable_baselines3 import SAC
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn import preprocessing
from scipy.stats import entropy
from math import log, e
import pandas as pd
from discretization import create_uniform_grid
from discretization import discretize


def entropy2(labels, base=None):
  """ Computes entropy of label distribution. """

  n_labels = len(labels)

  if n_labels <= 1:
    return 0

  value,counts = np.unique(labels, return_counts=True)
  probs = counts / n_labels
  n_classes = np.count_nonzero(probs)

  if n_classes <= 1:
    return 0

  ent = 0.

  # Compute entropy
  base = e if base is None else base
  for i in probs:
    ent -= i * log(i, base)

  return ent

def loadData():
    # for reading also binary mode is important
    #dbfile1 = open('sac_Ant-v3_replay_buffer_sss_316.pkl', 'rb')     
    #dbfile2 = open('sac_Ant-v3_replay_buffer_sss_324.pkl', 'rb')   
    #dbfile3 = open('sac_Ant-v3_replay_buffer_sss_3241.pkl', 'rb')     

    #dbfile1 = open('sac_Pendulum-v1_replay_buffer_default_basic.pkl', 'rb')     
    #dbfile1 = open('sac_Walker2d-v3_replay_buffer_sss_58.pkl', 'rb')    
    name = "Walker2d-v3"
    #name = "Ant-v3"
    #name = "Hopper-v3"
    #name = "HalfCheetah-v3"
    #name = "Swimmer-v3"
    print(name)
    

    dbfile1 = open('sac_'+name+'_replay_buffer_sss_1.pkl', 'rb')     
    dbfile1 = open('sac_'+name+'_replay_buffer_sss_616.pkl', 'rb') 

    if name=="Hopper-v3":
            low =  np.array([0.0,-1.0,-2.0,-1.0,-1.0,-2.0,-3.0,-8.0,-8.0,-8.0,-7.0])
            high = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 7.0, 7.0, 6.0, 8.0])
            sbins = (50,50,50,50)
            rbins = (1000,)
            rlow = np.array([-1])
            rhigh = np.array([7])
            p = [7, 10, 9, 0]
    elif name=="Ant-v3":
            low = np.array([0.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-2.0,-1.0,-2.0,-1.0,-1.0,-4.0,-4.0,-4.0,-7.0,-13.0,-7.0,-16.0,-14.0,-16.0,-16.0,-16.0,-18.0,-16.0,-14.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            high = np.array([1.0,1.0,1.0,1.0,1.0,1.0,2.0,1.0,1.0,1.0,1.0,1.0,2.0,4.0,3.0,6.0,9.0,9.0,7.0,16.0,17.0,17.0,14.0,17.0,14.0,16.0,18.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            sbins = (50,50,50,50,50,50,50,50)
            rbins = (1000,)
            rlow = np.array([-5])
            rhigh = np.array([8])
            p = [21, 23, 19, 25, 24, 20, 26, 22]
    elif name=="Walker2d-v3":
            low = np.array([0.0,-1.0,-2.0,-3.0,-2.0,-2.0,-3.0,-2.0,-3.0,-6.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0,-10.0])
            high = np.array([2.0,1.0,1.0,1.0,2.0,1.0,1.0,2.0,2.0,2.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0])
            sbins = (50,50,50,50,50)
            rbins = (1000,)
            rlow = np.array([-4])
            rhigh = np.array([11])
            p = [13, 16, 14, 12, 15]
    elif name=="Swimmer-v3":
            low = np.array([-2.0,-2.0,-2.0,-3.0,-4.0,-4.0,-7.0,-6.0])
            high = np.array([3.0,2.0,2.0,3.0,3.0,4.0,6.0,7.0])
            sbins = (50,50,50,50)
            rbins = (1000,)
            rlow = np.array([-4])
            rhigh = np.array([4])
            p = [7, 6, 1, 2]
    elif name=="Pendulum-v1":
            low = np.array([-1.0,-1.0,-8.0])
            high = np.array([1.0,1.0,8.0])
            sbins = (50,)
            rbins = (100,)
            p = [2]
    
        
    
    
    state_grid = create_uniform_grid(low, high, sbins)
    rw_grid = create_uniform_grid(rlow, rhigh, rbins)
    
    model1 = SAC.load('sac_'+name+'_default_basic')
    env = gym.make(name)
    sed = 10
    for s in range(sed):
        env.seed(s)
        obs = env.reset()
        obss = []
        rwss = []
        for i in range(1000):
            action, _states = model1.predict(obs)
            obs, rewards, dones, info = env.step(action)
            obss.append(obs[p])
            rwss.append(rewards)
            if dones==True:
                break
    
    abs = []
    for i in range(len(obss)):
        #print(obs[i])
        #print(obs[i][0])
        #print(obs[i][0][p])
        dss = discretize(obss[i], state_grid)
        #print(dss)
        #print(rws[i])
        #drs = discretize(rws[i], rw_grid)
        #print(drs)
        #abs.append(np.concatenate((dss,drs)))
        abs.append(dss)
    #print(abs[0:10])
    print("default entropy ",entropy2(abs))

    
    model1 = SAC.load('sac_'+name+'_sss_616')
    env = gym.make(name)
    sed = 10
    for s in range(sed):
        env.seed(s)
        obs = env.reset()
        obss = []
        rwss = []
        for i in range(1000):
            action, _states = model1.predict(obs)
            obs, rewards, dones, info = env.step(action)
            obss.append(obs[p])
            rwss.append(rewards)
            if dones==True:
                break

    abs = []
    for i in range(len(obss)):
        #print(obs[i])
        #print(obs[i][0])
        #print(obs[i][0][p])
        dss = discretize(obss[i], state_grid)
        #print(dss)
        #print(rws[i])
        #drs = discretize(rws[i], rw_grid)
        #print(drs)
        #abs.append(np.concatenate((dss,drs)))
        abs.append(dss)
    #print(abs[0:10])
    print("abstract entropy ",entropy2(abs))

    exit()
    

    rb1 = pickle.load(dbfile1)

    #acts = rb1.actions
    
    st=0
    ed=1000000
    ed=rb1.pos
    rws = rb1.rewards[st:ed]
    obs = rb1.observations[st:ed]
    abs = []
    abr = []
    print(st,"  to  ",ed)
    #print(len(rws))

    #labels = [1,3,5,2,3,5,3,2,1,3,4,5]
    for i in range(ed-st):
        #print(obs[i])
        #print(obs[i][0])
        #print(obs[i][0][p])
        dss = discretize(obs[i][0][p], state_grid)
        #print(dss)
        #print(rws[i])
        #drs = discretize(rws[i], rw_grid)
        #print(drs)
        #abs.append(np.concatenate((dss,drs)))
        abs.append(dss)
    #print(abs[0:10])
    print(entropy2(abs))

    exit()
    print(len(acts))
    print(acts[0][0])
    print("length of action vector  ",len(acts[0][0]))
    print(len(acts[0]))
    print(acts[0])
    print(len(acts[0][0]))
    print(acts[0][0])
    print(acts[1][0])
    print(acts[2][0])
    x = []
    z = 100 #sample size
    for i in range(len(acts[0:z])):
        x.append(acts[i][0])    
    print(x)
    






    exit()
    p = [21, 23, 19, 25, 24, 20, 26, 22]
    q = [3, 1, 4, 7, 0, 5, 2, 6]
    #print("p is : ",p)
    #print("q is : ",q)

        
    acts=rb1.actions
    obs=rb1.observations
    print(len(acts))
    print(acts[0][0])
    print("length of action vector  ",len(acts[0][0]))
    print(len(acts[0]))
    print(acts[0])
    print(len(acts[0][0]))
    print(acts[0][0])
    print(acts[1][0])
    print(acts[2][0])
    x = []
    z = 100 #sample size
    for i in range(len(acts[0:z])):
        x.append(acts[i][0])    
    print(x)
    
    t = np.arange(len(x))
    #t = np.arange(1,z,10)
    
    id = 0 #action dim
    x0 = [x[i][id] for i in range(z)]
    plt.plot(t, x0, label = str(id), color='red')
    # id = 1 #action dim
    # x1 = [x[i][id] for i in range(z)]
    # plt.plot(t, x1, label = str(id), color='blue')
    # id = 2 #action dim
    # x2 = [x[i][id] for i in range(z)]
    # plt.plot(t, x2, label = str(id), color='green')
    # #id = 3 #action dim
    #x3 = [x[i][id] for i in range(z)]
    #plt.plot(t, x3, label = str(id), color='yellow')
    
    plt.legend()
    plt.savefig('acts.png')
    plt.show()
    exit()
    



  

    dbfile.close()
  
if __name__ == '__main__':
    loadData()
