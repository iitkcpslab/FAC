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
    '''
    env = gym.make('Pendulum-v1') 
    

    sv = np.array([]) # state values
    cv = np.array([])  # control values
    rv = np.array([]) # reward values
    robs = []
    
    env.seed(0)
    
    
    model = SAC.load("sac_Pendulum-v1")
    nsteps=200 

    #############

    # Enjoy trained agent
    obs = env.reset()
    for i in range(nsteps):
        #action, _states = model.predict(obs, deterministic=True)
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        #env.render()
        sv=np.append(sv,np.arccos(obs[0]))
        cv=np.append(cv,action)
        rv=np.append(rv,rewards)
        #robs.append(rob)
        #print(str(sed)+"   "+str(x)+" "+str(rewards))
        #if dones==True:
        
    '''
    
    '''
    dbfile1 = open('controllers/sac_Pendulum-v1_replay_buffer_sss_1.pkl', 'rb')  
    #dbfile1 = open('controllers/sac_Pendulum-v1_replay_buffer_sss_616.pkl', 'rb')  

    
    rb1 = pickle.load(dbfile1)   
    print("size of rb ",rb1.pos)
    print("size of rb len ",len(rb1.observations))
    #exit()
    a = 10000
    b = rb1.pos
    data=rb1.observations[:,:,0]
    sv = [s[0] for s in data]
    sv = sv[a:b]
    #print(sv)
    data=rb1.actions[:,0]
    cv = [s[0] for s in data]
    cv = cv[a:b]
    #print(av) 
    rv=[s[0] for s in rb1.rewards]
    rv = rv[a:b]
    #print(rv) 
    # time_sm = np.array(time_list)
    x = np.arange(b-a)

    # feedback_smooth = spline(time_list, feedback_list, time_smooth)
    # Using make_interp_spline to create BSpline
    # helper_x3 = make_interp_spline(time_list, feedback_list)
    # feedback_smooth = helper_x3(time_smooth)

    f,(ax1,ax2,ax3,ax4) = plt.subplots(4,1,sharex=True, squeeze=True)
    ax1.set_title('Pendulum')
    #ax1.plot(sv, label='State')
    ax1.scatter(x,sv, label='State',s=5)
    #ax1.plot(time_list, setpoint_list, label='Set-point')
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('State')
    ax1.grid(True)
    ax1.legend()

    #ax2.plot(cv, label='Action')
    ax2.scatter(x,cv, label='Action',s=5)
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('Action')
    ax2.grid(True)
    ax2.legend()

    #ax3.plot(rv, label='Reward')
    ax3.scatter(x,rv, label='Reward', s=5)
    ax3.set_xlabel('time (s)')
    ax3.set_ylabel('Reward')
    ax3.grid(True)
    ax3.legend()

    #ax4.plot(np.gradient(rv), label='Reward Gradient')
    ax4.scatter(x, np.gradient(rv), label='Reward Gradient', s=5)
    ax4.set_xlabel('time (s)')
    ax4.set_ylabel('Reward Gradient')
    ax4.grid(True)
    ax4.legend()
    '''

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