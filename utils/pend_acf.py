import sys
import numpy as np
import pickle


import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
import pandas as pd
import pickle


from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
#modelid=4

def monitor():
    

    dbfile1 = open('controllers/sac_Pendulum-v1_replay_buffer_sss_1.pkl', 'rb')  
    dbfile2 = open('controllers/sac_Pendulum-v1_replay_buffer_sss_616.pkl', 'rb')  

    
    rb1 = pickle.load(dbfile1)   
    rb2 = pickle.load(dbfile2)  
    #exit()
    a = 10000
    b = 10400
    data=rb1.observations[:,:,0]
    sv1 = [s[0] for s in data]
    sv1 = sv1[a:b]
     
    data=rb2.observations[:,:,0]
    sv2 = [s[0] for s in data]
    sv2 = sv2[a:b]

    rv1 = [s[0] for s in rb1.rewards]
    rv1 = rv1[a:b]
     
    rv2 = [s[0] for s in rb2.rewards]
    rv2 = rv2[a:b]
    print(np.max(rv1))
    print(np.min(rv1))
    exit() 
    plt.acorr(rv2, maxlags = 20)
    plt.savefig('pend_acf2.png')
    exit()
    #print(sv)
    # data=rb1.actions[:,0]
    # cv = [s[0] for s in data]
    # cv = cv[a:b]
    # #print(av) 
    # rv=[s[0] for s in rb1.rewards]
    # rv = rv[a:b]
    #print(rv) 
    # time_sm = np.array(time_list)
    x = np.arange(b-a)

    # feedback_smooth = spline(time_list, feedback_list, time_smooth)
    # Using make_interp_spline to create BSpline
    # helper_x3 = make_interp_spline(time_list, feedback_list)
    # feedback_smooth = helper_x3(time_smooth)

    f,(ax1,ax2) = plt.subplots(2,1,sharex=True, squeeze=True)
    
    '''
    ax1.set_title('Pendulum States')
    ax1.scatter(x,sv1,s=5,c='green')
    ax1.set_xlabel('Samples')
    ax1.set_ylabel('Defult States')
    ax1.set_facecolor('white')
    ax1.grid(True)
    ax1.legend()

    ax2.scatter(x,sv2,s=5)
    ax2.set_xlabel('Samples')
    ax2.set_ylabel('MADE States')
    ax2.set_facecolor('white')
    ax2.grid(True)
    ax2.legend()
    '''
    
    ax1.set_title('Pendulum Rewards')

    ax1.scatter(x,rv1,s=15,c='green', label=' ')
    ax1.set_xlabel('Replay Buffer index')
    ax1.set_ylabel('Default Reward')
    ax1.set_facecolor('white')
    ax1.grid(True)
    ax1.legend()

    ax2.scatter(x,rv2,s=15, c='blue', label=' ')
    ax2.set_xlabel('Replay Buffer index')
    ax2.set_ylabel('MADE Reward')
    ax2.set_facecolor('white')
    ax2.grid(True)
    ax2.legend()

    plt.show()
    #plt.savefig('pend_curve_rw.pdf',format="pdf")


if __name__ == "__main__":  # noqa: C901
    monitor()
