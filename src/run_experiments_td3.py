import gym
import sys
#sys.path.insert(1, '/home/nikhil/RESEARCH/RL/SSFC/')
import argparse
import warnings
warnings.filterwarnings("ignore")


import numpy as np
from stable_baselines3 import TD3, TD3_PER, TD3_LABER
from stable_baselines3.common.env_util import make_vec_env
import timeit
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

#print(sys.argv[1])
import pybullet_envs
#exit()

def run_exp(env_id,mod,run):
    '''
    param:
    env_id : Gym Environment
    mod : 1 (default TD3) and 2 (FAC)
    run : unique id associated with a experiment.
    This function performs synthesis (training) of a control policy
    in a given gym environment. 
    '''

    if mod==1:
        name="default"
    elif mod==2:
        name="fac"
    elif mod==3:
        name="per"
    elif mod==4:
        name="laber"
    print(env_id)

    if env_id=="Pendulum-v1":
        start = timeit.default_timer()
        env = gym.make('Pendulum-v1')
        #env = DummyVecEnv([lambda: gym.make('Pendulum-v0')])

        start = timeit.default_timer()
        if mod==1 or mod==2:
            model = TD3('MlpPolicy','Pendulum-v1',learning_rate=0.005, learning_starts=10000, policy_kwargs=dict(net_arch=[256, 256]),  tensorboard_log="./td3_pend_tensorboard", verbose=1, seed=4156017648)
            model.learn(total_timesteps=int(2e4),mode=name)
        elif mod==4:
            model = TD3_LABER('MlpPolicy','Pendulum-v1',learning_rate=0.005, learning_starts=10000, policy_kwargs=dict(net_arch=[256, 256]),  tensorboard_log="./td3_pend_tensorboard", verbose=1, seed=4156017648)
            model.learn(total_timesteps=int(2e4))    
        model.save("td3_Pendulum-v1_"+name+"_"+str(run))
        model.save_replay_buffer("td3_Pendulum-v1_replay_buffer_"+name+"_"+str(run))
        end = timeit.default_timer()
        print("training_time is"+str(end-start))
        text_file = open("pend.txt", "a")
        text_file.write('Time for '+str(run)+' is : '+str(end-start)+'\n')
        text_file.close()

    elif env_id=="MountainCarContinuous-v0":
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        start = timeit.default_timer()
        env = DummyVecEnv([lambda: gym.make('MountainCarContinuous-v0')])

        start = timeit.default_timer()
        n_actions = env.action_space.shape[-1]
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))
        
        if mod==1 or mod==2:
            model = TD3('MlpPolicy','MountainCarContinuous-v0', learning_rate=0.001, action_noise=action_noise, batch_size=128, tensorboard_log="./td3_mc_tensorboard", verbose=1, seed=2408295766)
            model.learn(total_timesteps=int(1e5),mode=name)
        elif mod==4:
            model = TD3_LABER('MlpPolicy','MountainCarContinuous-v0', learning_rate=0.001, action_noise=action_noise, batch_size=128, tensorboard_log="./td3_mc_tensorboard", verbose=1, seed=2408295766)
            model.learn(total_timesteps=int(1e5))
            
        model.save("td3_MountainCarContinuous-v0_"+name+"_"+str(run))
        model.save_replay_buffer("td3_MountainCarContinuous-v0_replay_buffer_"+name+"_"+str(run))
        end = timeit.default_timer()
        print("training_time is"+str(end-start))
        text_file = open("mc.txt", "a")
        text_file.write('Time for '+str(run)+' is : '+str(end-start)+'\n')
        text_file.close()
    
        
    elif env_id=="LunarLanderContinuous-v2":
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        start = timeit.default_timer()
        env = DummyVecEnv([lambda: gym.make('LunarLanderContinuous-v2')])

        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        if mod==1 or mod==2:
            model = TD3('MlpPolicy','LunarLanderContinuous-v2', learning_rate=0.001, learning_starts=10000, buffer_size=200000, gamma=0.98, gradient_steps=-1, action_noise=action_noise, tensorboard_log="./td3_llc_tensorboard", policy_kwargs=dict(net_arch=[400, 300]), verbose=1, seed=3607994507)
            model.learn(total_timesteps=int(3e5),mode=name)
        elif mod==4:
            model = TD3_LABER('MlpPolicy','LunarLanderContinuous-v2', learning_rate=0.001, learning_starts=10000, buffer_size=200000, gamma=0.98, gradient_steps=-1, action_noise=action_noise, tensorboard_log="./td3_llc_tensorboard", policy_kwargs=dict(net_arch=[400, 300]), verbose=1, seed=3607994507)
            model.learn(total_timesteps=int(3e5))

        model.save("td3_LunarLanderContinuous-v2_"+name+"_"+str(run))
        model.save_replay_buffer("td3_LunarLanderContinuous-v2_replay_buffer_"+name+"_"+str(run))
        end = timeit.default_timer()
        print("training_time is"+str(end-start))
        text_file = open("bpwalker.txt", "a")
        text_file.write('Time for '+str(run)+' is : '+str(end-start)+'\n')
        text_file.close()
    

    elif env_id=="ReacherBulletEnv-v0":
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        start = timeit.default_timer()
        #env = gym.make('ReacherBulletEnv-v0')

        env = DummyVecEnv([lambda: gym.make('ReacherBulletEnv-v0')])
        #env = VecNormalize(env, norm_obs=True)
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        
        if mod==1 or mod==2:
            model = TD3('MlpPolicy','ReacherBulletEnv-v0', learning_rate=0.001, gamma=0.98, buffer_size=20000, learning_starts=10000, gradient_steps=-1, action_noise=action_noise, tensorboard_log="./td3_reacher_tensorboard", policy_kwargs=dict(net_arch=[400, 300]), verbose=1, seed=4011578699)
            model.learn(total_timesteps=int(2.5e5),mode=name)  
        elif mod==4:
            model = TD3_LABER('MlpPolicy','ReacherBulletEnv-v0', learning_rate=0.001, gamma=0.98, buffer_size=20000, learning_starts=10000, gradient_steps=-1, action_noise=action_noise, tensorboard_log="./td3_reacher_tensorboard", policy_kwargs=dict(net_arch=[400, 300]), verbose=1, seed=4011578699)
            model.learn(total_timesteps=int(2.5e5)) 

        model.save("td3_ReacherBulletEnv-v0_"+name+"_"+str(run))
        model.save_replay_buffer("td3_ReacherBulletEnv-v0_replay_buffer_"+name+"_"+str(run))
        end = timeit.default_timer()
        print("training_time is"+str(end-start))
        text_file = open("reacher.txt", "a")
        text_file.write('Time for '+str(run)+' is : '+str(end-start)+'\n')
        text_file.close()
    
    elif env_id=="Swimmer-v3":
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        start = timeit.default_timer()
        env = DummyVecEnv([lambda: gym.make('Swimmer-v3')])

        policy_kwargs = dict(net_arch=[256,256])  #default
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        
        if mod==1 or mod==2:
            model = TD3('MlpPolicy','Swimmer-v3', learning_rate=0.001, learning_starts=10000, batch_size=128, action_noise=action_noise, tensorboard_log="./td3_swimmer_tensorboard", policy_kwargs=policy_kwargs, verbose=1, seed=3865637836)
            model.learn(total_timesteps=int(1e6),mode=name)
        elif mod==4:
            model = TD3_LABER('MlpPolicy','Swimmer-v3', learning_rate=0.001, learning_starts=10000, batch_size=128, action_noise=action_noise, tensorboard_log="./td3_swimmer_tensorboard", policy_kwargs=policy_kwargs, verbose=1, seed=3865637836)
            model.learn(total_timesteps=int(1e6))
            
        model.save("td3_Swimmer-v3_"+name+"_"+str(run))
        model.save_replay_buffer("td3_Swimmer-v3_replay_buffer_"+name+"_"+str(run))
        end = timeit.default_timer()
        print("training_time is"+str(end-start))
        text_file = open("swimmer.txt", "a")
        text_file.write('Time for '+str(run)+' is : '+str(end-start)+'\n')
        text_file.close()


    elif env_id=="Hopper-v3":
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        start = timeit.default_timer()
        #env = gym.make('Hopper-v3')
        env = DummyVecEnv([lambda: gym.make('Hopper-v3')])

        start = timeit.default_timer()
        
        #Otherwise, to have actor and critic that share the same network architecture, you only need to specify net_arch=[256, 256] (here, two hidden layers of 256 units each).
        policy_kwargs = dict(net_arch=[256,256])  #default
        
        if mod==1 or mod==2:
            model = TD3('MlpPolicy','Hopper-v3', learning_rate=0.0003, learning_starts=10000, tensorboard_log="./td3_hop_tensorboard", policy_kwargs=policy_kwargs,  verbose=1, seed=594371)        
            model.learn(total_timesteps=int(1e6),mode=name)
        elif mod==4:
            model = TD3_LABER('MlpPolicy','Hopper-v3', learning_rate=0.0003, learning_starts=10000, tensorboard_log="./td3_hop_tensorboard", policy_kwargs=policy_kwargs,  verbose=1, seed=594371)        
            model.learn(total_timesteps=int(1e6))

        model.save("td3_Hopper-v3_"+name+"_"+str(run))
        model.save_replay_buffer("td3_Hopper-v3_replay_buffer_"+name+"_"+str(run))
        end = timeit.default_timer()
        print("training_time is"+str(end-start))
        text_file = open("hopper.txt", "a")
        text_file.write('Time for '+str(run)+' is : '+str(end-start)+'\n')
        text_file.close()

    elif env_id=="Walker2d-v3":
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        start = timeit.default_timer()
        env = DummyVecEnv([lambda: gym.make('Walker2d-v3')])

        policy_kwargs = dict(net_arch=[256,256])  #default

        #n_actions = env.action_space.shape[-1]
        #action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        
        if mod==1 or mod==2:
            model = TD3('MlpPolicy','Walker2d-v3',learning_starts=10000, tensorboard_log="./td3_walker_tensorboard", policy_kwargs=policy_kwargs, verbose=1, seed=1751580366)
            model.learn(total_timesteps=int(1e6),mode=name)
        elif mod==4:
            model = TD3_LABER('MlpPolicy','Walker2d-v3',learning_starts=10000, tensorboard_log="./td3_walker_tensorboard", policy_kwargs=policy_kwargs, verbose=1, seed=1751580366)
            model.learn(total_timesteps=int(1e6))

        model.save("td3_Walker2d-v3_"+name+"_"+str(run))
        model.save_replay_buffer("td3_Walker2d-v3_replay_buffer_"+name+"_"+str(run))
        end = timeit.default_timer()
        print("training_time is"+str(end-start))
        text_file = open("walker.txt", "a")
        text_file.write('Time for '+str(run)+' is : '+str(end-start)+'\n')
        text_file.close()
        

    elif env_id=="Ant-v3":
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        start = timeit.default_timer()
        env = gym.make('Ant-v3')

        #env = DummyVecEnv([lambda: gym.make('ReacherBulletEnv-v0')])
        #env = VecNormalize(env, norm_obs=True)
        policy_kwargs = dict(net_arch=[256,256])  #default
        
        if mod==1 or mod==2:
            model = TD3('MlpPolicy','Ant-v3', learning_starts=10000, batch_size=128, tensorboard_log="./td3_ant_tensorboard", policy_kwargs=policy_kwargs, verbose=1, seed=594371)   
            model.learn(total_timesteps=int(1e6),mode=name)
        elif mod==4:
            model = TD3_LABER('MlpPolicy','Ant-v3', learning_starts=10000, batch_size=128, tensorboard_log="./td3_ant_tensorboard", policy_kwargs=policy_kwargs, verbose=1, seed=594371)   
            model.learn(total_timesteps=int(1e6))
            
        model.save("td3_Ant-v3_"+name+"_"+str(run))
        model.save_replay_buffer("td3_Ant-v3_replay_buffer_"+name+"_"+str(run))
        end = timeit.default_timer()
        print("training_time is"+str(end-start))  
        text_file = open("ant.txt", "a")
        text_file.write('Time for '+str(run)+' is : '+str(end-start)+'\n')
        text_file.close()
    
    elif env_id=="Humanoid-v3":
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        start = timeit.default_timer()
        env = DummyVecEnv([lambda: gym.make('Humanoid-v3')])
        
        policy_kwargs = dict(net_arch=[256,256])  #default
        
        if mod==1 or mod==2:
            model = TD3('MlpPolicy','Humanoid-v3', learning_rate=0.0001,learning_starts=10000, batch_size=128, gradient_steps=1, train_freq=1, tensorboard_log="./td3_humanoid_tensorboard", policy_kwargs=policy_kwargs, verbose=1, seed=594371) #exp32
            model.learn(total_timesteps=int(2e6),mode=name)
        elif mod==4:
            model = TD3_LABER('MlpPolicy','Humanoid-v3', learning_rate=0.0001,learning_starts=10000, batch_size=128, gradient_steps=1, train_freq=1, tensorboard_log="./td3_humanoid_tensorboard", policy_kwargs=policy_kwargs, verbose=1, seed=594371) #exp32
            model.learn(total_timesteps=int(2e6))

        model.save("td3_Humanoid-v3_"+name+"_"+str(run))
        model.save_replay_buffer("td3_Humanoid-v3_replay_buffer_"+name+"_"+str(run))
        end = timeit.default_timer()
        print("training_time is"+str(end-start))
        text_file = open("humanoid.txt", "a")
        text_file.write('Time for '+str(run)+' is : '+str(end-start)+'\n')
        text_file.close()

    



if __name__ == "__main__":  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, default="Hopper-v3")
    parser.add_argument("--mod", help="Mode", default=1, type=int, required=False)
    parser.add_argument("--run", help="run id", default=10, type=int, required=False)
  
    args = parser.parse_args()

    env_id = args.env
    mod = args.mod
    run = args.run
    run_exp(env_id,mod,run)
