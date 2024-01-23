import pickle
import gym
#import sys
#sys.path.insert(1, '/home/nikhil/RESEARCH/RL/SSFC_workspace/stable_baselines3/')
from stable_baselines3 import SAC
import sys
import numpy as np
import matplotlib.pyplot as plt


def loadData():
    # for reading also binary mode is important
    '''
    dbfile = open('glist.pkl', 'rb')     
    db = pickle.load(dbfile)
    g = []
    for i in range(len(db)):
        g.append(db[i])
        #print(db[:,0])
    print(len(db))
    x = np.arange(len(db))
    plt.plot(x[0:100],g[0:100], label = "f100",color="orange") #initial 100 samples
    plt.plot(x[0:100:],g[10000:10100], label = "m100",color="green") #middle 100 samples
    plt.plot(x[0:100:],g[-100:], label = "l100",color="black") #last 100 samples
    plt.legend()
    plt.show()
    dbfile.close()
    '''
    dbfile2 = open('glist.pkl', 'rb')     
    db2 = pickle.load(dbfile2)
    g = []
    for i in range(len(db2)):
        g.append(db2[i])
    
    dbfile = open('hlist.pkl', 'rb')     
    db = pickle.load(dbfile)
    h = []
    for i in range(len(db)):
        h.append(db[i])
        #print(db[:,0])
    
    print(len(db))
    x = np.arange(len(db))
    #plt.plot(x[0:len(db)],g[0:len(db)], label = "g",color="green") #initial 100 samples
    #plt.plot(x[0:len(db)],h[0:len(db)], label = "h",color="blue") #initial 100 samples
    #plt.plot(x[0:100],g[0:100], label = "g",color="green") #initial 100 samples
    plt.plot(x[0:100],h[0:100], label = "h",color="orange") #initial 100 samples
    plt.legend()
    plt.show()
    
    dbfile.close()
  
if __name__ == '__main__':
    loadData()
