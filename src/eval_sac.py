# Load the trained agent
# some_file.py
import sys
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")


import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
import pandas as pd
import pickle
import pybullet_envs

import gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env 
from stable_baselines3.common.vec_env import DummyVecEnv


def evaluate(modelid):
    '''
    This function generates the summary statistics for the given
    control policy.
    '''
    
    if modelid=="Pendulum-v1":
        env = gym.make('Pendulum-v1')
    elif modelid=="MountainCarContinuous-v0":
        env = gym.make('MountainCarContinuous-v0')
    elif modelid=="ReacherBulletEnv-v0":
        env = gym.make('ReacherBulletEnv-v0')
    elif modelid=="Hopper-v3":
        env = gym.make('Hopper-v3') 
    elif modelid=="Ant-v3":
        env = gym.make('Ant-v3') 
    elif modelid=="Walker2d-v3":
        env = gym.make('Walker2d-v3') 
    elif modelid=="Swimmer-v3":
        env = gym.make('Swimmer-v3') 
    elif modelid=="Humanoid-v3":
        env = gym.make('Humanoid-v3')
    elif modelid=="LunarLanderContinuous-v2":
        env = gym.make("LunarLanderContinuous-v2")

    tsc = np.array([]) #total state cost
    tcc = np.array([])  # total control cost
    tsp = np.array([])  # total control cost
    robs = []
    tsteps=np.array([])
    tdist=np.array([])
    tdr=np.array([])
    tseeds=100
    avg_min_rob=[]
    avg_mean_rob=[]
    for sed in range(0,tseeds):
        env.seed(sed)
        sc = np.array([])
        steps = 0
        sp = np.array([])
        cc = np.array([])
        if modelid=="Pendulum-v1":
            model = SAC.load("sac_Pendulum-v1")
            nsteps=1000 
        elif modelid=="MountainCarContinuous-v0":
            model = SAC.load("sac_MountainCarContinuous-v0")
            nsteps=1000 
        elif modelid=="HalfCheetah-v3":
            model = SAC.load("sac_HalfCheetah-v3")
            nsteps=1000 
        elif modelid=="Hopper-v3":
            model = SAC.load("sac_Hopper-v3")
            nsteps=1000 
        elif modelid=="Ant-v3":
            model = SAC.load("sac_Ant-v3")
            nsteps=1000 
        elif modelid=="Walker2d-v3":
            model = SAC.load("sac_Walker2d-v3")
            nsteps=1000
        elif modelid=="Swimmer-v3":
            model = SAC.load("sac_Swimmer-v3")
            nsteps=1000
        elif modelid=="Humanoid-v3":
            model = SAC.load("sac_Humanoid-v3")
            nsteps=1000
        elif modelid=="ReacherBulletEnv-v0":
            model = SAC.load("sac_ReacherBulletEnv-v0")
            nsteps=150
        elif modelid=="LunarLanderContinuous-v2":
            model = SAC.load("sac_LunarLanderContinuous-v2")
            nsteps=1000
        elif modelid=="BipedalWalker-v3":
            model = SAC.load("sac_BipedalWalker-v3")
            nsteps=1000
        elif modelid=="BipedalWalkerHardcore-v3":
            model = SAC.load("sac_BipedalWalkerHardcore-v3")
            nsteps=1000
        #env.render()

        #############

        # Enjoy trained agent
        obs = env.reset()
        min_rob=1000
        mean_rob=[]
        dr=0
        for i in range(nsteps):
            #action, _states = model.predict(obs, deterministic=True)
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            #env.render()
            if modelid=="Pendulum-v1":
                t, dummy, td = obs # th := theta
                u = np.clip(action, -2, 2)[0] # max toruqe of this env 
                x = np.arccos(t)
                steps=i
                sc=np.append(sc,np.arccos(t))

     
            elif modelid=="MountainCarContinuous-v0":
                sc=np.append(sc,0.5-obs[0])
                x = obs[0]
                steps=i

            elif modelid=="HalfCheetah-v3":
                x = info['x_position']
                v = info['x_velocity']
                #if x>50:
                #    break
                steps=i
                sc=np.append(sc,50-x)

            elif modelid=="Hopper-v3":
                x = info['x_position']
                #if x>50:
                #    break
                sc=np.append(sc,50-x)
                p = info['x_position']
                v = info['x_velocity']
                z , a = obs[0:2]

            elif modelid=="Ant-v3":
                px = info['x_position']
                py = info['y_position']
                x = np.sqrt(np.square([px,py]).sum())
                #if x>50:
                #    break
                sc=np.append(sc,50-x)
                px = info['x_position']
                vx = info['x_velocity']
                z = obs[0]

            elif modelid=="Walker2d-v3":
                x = info['x_position']
                #if x>50:
                #    break
                sc=np.append(sc,50-x)
                px = info['x_position']
                vx = info['x_velocity']
                z , a = obs[0:2]

            elif modelid=="Swimmer-v3":
                x = info['x_position']
                #if x>50:
                #    break
                sc=np.append(sc,50-x)
                px = info['x_position']
                vx = info['x_velocity']
                a1=obs[0]
                a2=obs[1]
                a3=obs[2]

            elif modelid=="Humanoid-v3":
                x = info['x_position']
                #if x>50:
                #    break
                sc=np.append(sc,50-x)
                px = info['x_position']
                py = info['y_position']
                vx = info['x_velocity']
                z = obs[0]

            elif modelid=="ReacherBulletEnv-v0":
               
                px = obs[2]
                py = obs[3]
                x = np.sqrt(np.square([px,py]).sum())
                #if x>50:
                steps=i
            elif modelid=="LunarLanderContinuous-v2":
               
                px = obs[0]
                py = obs[1]
                x = np.sqrt(np.square([px,py]).sum())
                #if x>50:
                steps=i
            elif modelid=="BipedalWalker-v3":
               
                x = 0
                steps=i
            elif modelid=="BipedalWalkerHardcore-v3":
               
                px = obs[2]
                py = obs[3]
                x = np.sqrt(np.square([px,py]).sum())
                #if x>50:
                steps=i



            if dones==True:
                break

            #sp=np.append(sp,x)
            #cc=np.append(cc,action)
            cc=np.append(cc,np.sqrt(np.square(action).sum()))
            dr+=rewards
            #robs.append(rob)
            #print(str(sed)+"   "+str(x)+" "+str(rewards))
            #if dones==True:
        tsc=np.append(tsc,np.sum(np.square(sc)))
        tcc=np.append(tcc,np.sum(np.square(cc)))  
        tdr=np.append(tdr,dr)
        tsteps=np.append(tsteps,steps)  
        tdist=np.append(tdist,x)  
        tmp=np.mean(mean_rob)
        avg_min_rob.append(min_rob)
        avg_mean_rob.append(tmp)


    data=np.array(tcc)
    mu=np.mean(data)
    std=np.std(data)
    
    print("###############################################")
    print("#### SUMMMARY : CONTROLLER EVALUATION #########")
    print("###############################################\n")
    print("Control Cost (CC) : ",mu,u"\u00B1",std)

    data=np.array(tsteps)
    mu=np.mean(data)
    std=np.std(data)
    print("steps mu std var rms:"+str(mu)+"  "+str(std))

    data=np.array(tdist)
    mu=np.mean(data)
    std=np.std(data)
    print("Distance Covered (DC) : ",mu,u"\u00B1",std)

    #print("Margin of Satisfaction (MoS) : ",np.mean(avg_mean_rob),u"\u00B1",np.std(avg_mean_rob))
    print("Default Reward (DR) : ",np.mean(tdr),u"\u00B1",np.std(tdr))
    
    #data=np.array(tsc)
    #mu=np.mean(data)
    #std=np.std(data)
    #print("state mu std var rms: "+str(mu)+"  "+str(std))



if __name__ == "__main__":  # noqa: C901
    evaluate(sys.argv[1])
