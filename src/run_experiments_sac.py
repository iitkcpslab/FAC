import gym
import sys
#sys.path.insert(1, '/home/nikhil/RESEARCH/RL/SSFC/')
import argparse
import warnings
warnings.filterwarnings("ignore")


import numpy as np
from stable_baselines3 import SAC, SAC_PER, SAC_LABER
from stable_baselines3.common.env_util import make_vec_env
import timeit
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
#print(sys.argv[1])
import pybullet_envs
#exit()

def run_exp(env_id,mod,run):
    '''
    param:
    env_id : Gym Environment
    mod : 1 (default SAC) and 2 (FAC)
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
            model = SAC('MlpPolicy','Pendulum-v1',learning_rate=0.001, gradient_steps=-1, use_sde=True,  tensorboard_log="./sac_pend_tensorboard", policy_kwargs=dict(log_std_init=-2, net_arch=[64, 64]), verbose=1, seed=1228490524)
            model.learn(total_timesteps=int(2e4),mode=name)
        elif mod==3:
            model = SAC_PER('MlpPolicy','Pendulum-v1',learning_rate=0.001, gradient_steps=-1, use_sde=True,  tensorboard_log="./sac_pend_tensorboard", policy_kwargs=dict(log_std_init=-2, net_arch=[64, 64]), verbose=1, seed=1228490524)
            model.learn(total_timesteps=int(2e4))
        elif mod==4:
            model = SAC_LABER('MlpPolicy','Pendulum-v1',learning_rate=0.001, gradient_steps=-1, use_sde=True,  tensorboard_log="./sac_pend_tensorboard", policy_kwargs=dict(log_std_init=-2, net_arch=[64, 64]), verbose=1, seed=1228490524)
            model.learn(total_timesteps=int(2e4))
        #model.learn(total_timesteps=int(1e3),mode=name)
        model.save("sac_Pendulum-v1_"+name+"_"+str(run))
        model.save_replay_buffer("sac_Pendulum-v1_replay_buffer_"+name+"_"+str(run))
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

        if mod==1 or mod==2:
            model = SAC('MlpPolicy','MountainCarContinuous-v0', learning_rate=0.0003, buffer_size=50000, learning_starts=0, batch_size=512, tau=0.01, gamma=0.9999, train_freq=32, gradient_steps=32, ent_coef=0.1, use_sde=True, tensorboard_log="./sac_mc_tensorboard", policy_kwargs=dict(log_std_init=-3.67, net_arch=[64, 64]), verbose=1, seed=765152961)
            model.learn(total_timesteps=int(5e4),mode=name)
        elif mod==3:
            model = SAC_PER('MlpPolicy','MountainCarContinuous-v0', learning_rate=0.0003, buffer_size=50000, learning_starts=0, batch_size=512, tau=0.01, gamma=0.9999, train_freq=32, gradient_steps=32, ent_coef=0.1, use_sde=True, tensorboard_log="./sac_mc_tensorboard", policy_kwargs=dict(log_std_init=-3.67, net_arch=[64, 64]), verbose=1, seed=765152961)
            model.learn(total_timesteps=int(5e4))
        elif mod==4:
            model = SAC_LABER('MlpPolicy','MountainCarContinuous-v0', learning_rate=0.0003, buffer_size=50000, learning_starts=0, batch_size=512, tau=0.01, gamma=0.9999, train_freq=32, gradient_steps=32, ent_coef=0.1, use_sde=True, tensorboard_log="./sac_mc_tensorboard", policy_kwargs=dict(log_std_init=-3.67, net_arch=[64, 64]), verbose=1, seed=765152961)
            model.learn(total_timesteps=int(5e4))

        model.save("sac_MountainCarContinuous-v0_"+name+"_"+str(run))
        model.save_replay_buffer("sac_MountainCarContinuous-v0_replay_buffer_"+name+"_"+str(run))
        end = timeit.default_timer()
        print("training_time is"+str(end-start))
        text_file = open("mc.txt", "a")
        text_file.write('Time for '+str(run)+' is : '+str(end-start)+'\n')
        text_file.close()
    
    elif env_id=="LunarLanderContinuous-v2":
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        start = timeit.default_timer()
        env = DummyVecEnv([lambda: gym.make('LunarLanderContinuous-v2')])

        
        if mod==1 or mod==2:
            model = SAC('MlpPolicy','LunarLanderContinuous-v2', learning_rate=7.3e-4, learning_starts=10000, batch_size=256, gamma=0.99, tensorboard_log="./sac_llc_tensorboard", policy_kwargs=dict(net_arch=[400, 300]), verbose=1, seed=2927467908)
            model.learn(total_timesteps=int(5e5),mode=name)
        if mod==3:
            model = SAC_PER('MlpPolicy','LunarLanderContinuous-v2', learning_rate=7.3e-4, learning_starts=10000, batch_size=256, gamma=0.99, tensorboard_log="./sac_llc_tensorboard", policy_kwargs=dict(net_arch=[400, 300]), verbose=1, seed=2927467908)
            model.learn(total_timesteps=int(5e5))
        if mod==4:
            model = SAC_LABER('MlpPolicy','LunarLanderContinuous-v2', learning_rate=7.3e-4, learning_starts=10000, batch_size=256, gamma=0.99, tensorboard_log="./sac_llc_tensorboard", policy_kwargs=dict(net_arch=[400, 300]), verbose=1, seed=2927467908)
            model.learn(total_timesteps=int(5e5))
        

        model.save("sac_LunarLanderContinuous-v2_"+name+"_"+str(run))
        model.save_replay_buffer("sac_LunarLanderContinuous-v2_replay_buffer_"+name+"_"+str(run))
        end = timeit.default_timer()
        print("training_time is"+str(end-start))
        text_file = open("llc.txt", "a")
        text_file.write('Time for '+str(run)+' is : '+str(end-start)+'\n')
        text_file.close()
    

    elif env_id=="ReacherBulletEnv-v0":
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        start = timeit.default_timer()
        #env = gym.make('ReacherBulletEnv-v0')

        env = DummyVecEnv([lambda: gym.make('ReacherBulletEnv-v0')])
        
        if mod==1 or mod==2:
            model = SAC('MlpPolicy','ReacherBulletEnv-v0', learning_rate=0.00073, buffer_size=250000, learning_starts=10000, batch_size=256, tau=0.02, gamma=0.98, train_freq=64, gradient_steps=64, action_noise=None, replay_buffer_class=None, replay_buffer_kwargs=None, optimize_memory_usage=False, ent_coef='auto', target_update_interval=1, target_entropy='auto', use_sde=True, sde_sample_freq=- 1, use_sde_at_warmup=False, tensorboard_log="./sac_reacher_tensorboard", create_eval_env=False, policy_kwargs=dict(log_std_init=-3, net_arch=[400, 300]), verbose=1, seed=235932389, device='auto', _init_setup_model=True)
            model.learn(total_timesteps=int(2.5e5),mode=name)
        if mod==3:
            model = SAC_PER('MlpPolicy','ReacherBulletEnv-v0', learning_rate=0.00073, buffer_size=250000, learning_starts=10000, batch_size=256, tau=0.02, gamma=0.98, train_freq=64, gradient_steps=64, action_noise=None, replay_buffer_class=None, replay_buffer_kwargs=None, optimize_memory_usage=False, ent_coef='auto', target_update_interval=1, target_entropy='auto', use_sde=True, sde_sample_freq=- 1, use_sde_at_warmup=False, tensorboard_log="./sac_reacher_tensorboard", create_eval_env=False, policy_kwargs=dict(log_std_init=-3, net_arch=[400, 300]), verbose=1, seed=235932389, device='auto', _init_setup_model=True)
            model.learn(total_timesteps=int(2.5e5))
        elif mod==4:
            model = SAC_LABER('MlpPolicy','ReacherBulletEnv-v0', learning_rate=0.00073, buffer_size=250000, learning_starts=10000, batch_size=256, tau=0.02, gamma=0.98, train_freq=64, gradient_steps=64, action_noise=None, optimize_memory_usage=False, ent_coef='auto', target_update_interval=1, target_entropy='auto', use_sde=True, sde_sample_freq=- 1, use_sde_at_warmup=False, tensorboard_log="./sac_reacher_tensorboard", create_eval_env=False, policy_kwargs=dict(log_std_init=-3, net_arch=[400, 300]), verbose=1, seed=235932389, device='auto', _init_setup_model=True)
            model.learn(total_timesteps=int(2.5e5))


        model.save("sac_ReacherBulletEnv-v0_"+name+"_"+str(run))
        model.save_replay_buffer("sac_ReacherBulletEnv-v0_replay_buffer_"+name+"_"+str(run))
        end = timeit.default_timer()
        print("training_time is"+str(end-start))
        text_file = open("reacher.txt", "a")
        text_file.write('Time for '+str(run)+' is : '+str(end-start)+'\n')
        text_file.close()
    
    elif env_id=="Swimmer-v3":
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        start = timeit.default_timer()
        env = DummyVecEnv([lambda: gym.make('Swimmer-v3')])

        #env = VecNormalize(env, norm_obs=True)
        #model = SAC('MlpPolicy','Hopper-v3', verbose=1, tensorboard_log="./sac_hop_tensorboard/")
        policy_kwargs = dict(net_arch=[256,256])  #default
        
        if mod==1 or mod==2:
            model = SAC('MlpPolicy','Swimmer-v3',learning_starts=10000, gamma=0.9999, use_sde=False,  tensorboard_log="./sac_swimmer_tensorboard", policy_kwargs=policy_kwargs,  verbose=1, seed=594371)
            model.learn(total_timesteps=int(5e5),mode=name)
        elif mod==3:
            model = SAC_PER('MlpPolicy','Swimmer-v3',learning_starts=10000, gamma=0.9999, use_sde=False,  tensorboard_log="./sac_swimmer_tensorboard", policy_kwargs=policy_kwargs,  verbose=1, seed=594371)
            model.learn(total_timesteps=int(5e5))
        elif mod==4:
            model = SAC_LABER('MlpPolicy','Swimmer-v3',learning_starts=10000, gamma=0.9999, use_sde=False,  tensorboard_log="./sac_swimmer_tensorboard", policy_kwargs=policy_kwargs,  verbose=1, seed=594371)
            model.learn(total_timesteps=int(5e5))

        #model.learn(total_timesteps=int(2e6),reward_type="STL",sem=name)
        model.save("sac_Swimmer-v3_"+name+"_"+str(run))
        model.save_replay_buffer("sac_Swimmer-v3_replay_buffer_"+name+"_"+str(run))
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
        policy_kwargs = dict(net_arch=[256,256])  #default
        #policy_kwargs = dict(net_arch=[128,128])  #exp57
        if mod==1 or mod==2:
            model = SAC('MlpPolicy','Hopper-v3',learning_rate=0.0003, buffer_size=1000000, learning_starts=10000, use_sde=False,  tensorboard_log="./sac_hop_tensorboard", policy_kwargs=policy_kwargs, verbose=1, seed=594371)
            model.learn(total_timesteps=int(1e6),mode=name)
        elif mod==3:
            model = SAC_PER('MlpPolicy','Hopper-v3',learning_rate=0.0003, buffer_size=1000000, learning_starts=10000, use_sde=False,  tensorboard_log="./sac_hop_tensorboard", policy_kwargs=policy_kwargs, verbose=1, seed=594371)
            model.learn(total_timesteps=int(1e6))
        elif mod==4:
            model = SAC_LABER('MlpPolicy','Hopper-v3',learning_rate=0.0003, buffer_size=1000000, learning_starts=10000, use_sde=False,  tensorboard_log="./sac_hop_tensorboard", policy_kwargs=policy_kwargs, verbose=1, seed=594371)
            model.learn(total_timesteps=int(1e6))

        model.save("sac_Hopper-v3_"+name+"_"+str(run))
        model.save_replay_buffer("sac_Hopper-v3_replay_buffer_"+name+"_"+str(run))
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
        #policy_kwargs = dict(net_arch=[128,128])  #exp57
        
        if mod==1 or mod==2:
            model = SAC('MlpPolicy','Walker2d-v3',learning_starts=10000,  use_sde=False,  tensorboard_log="./sac_walker_tensorboard", policy_kwargs=policy_kwargs, verbose=1, seed=594371)
            model.learn(total_timesteps=int(1e6),mode=name)
        elif mod==3:
            model = SAC_PER('MlpPolicy','Walker2d-v3',learning_starts=10000,  use_sde=False,  tensorboard_log="./sac_walker_tensorboard", policy_kwargs=policy_kwargs, verbose=1, seed=594371)
            model.learn(total_timesteps=int(1e6))
        elif mod==4:
            model = SAC_LABER('MlpPolicy','Walker2d-v3',learning_starts=10000,  use_sde=False,  tensorboard_log="./sac_walker_tensorboard", policy_kwargs=policy_kwargs, verbose=1, seed=594371)
            model.learn(total_timesteps=int(1e6))


        model.save("sac_Walker2d-v3_"+name+"_"+str(run))
        model.save_replay_buffer("sac_Walker2d-v3_replay_buffer_"+name+"_"+str(run))
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
            model = SAC('MlpPolicy','Ant-v3',learning_starts=10000, use_sde=False,  tensorboard_log="./sac_ant_tensorboard", policy_kwargs=policy_kwargs, verbose=1, seed=594371)
            model.learn(total_timesteps=int(1e6),mode=name)
        elif mod==3:
            model = SAC_PER('MlpPolicy','Ant-v3',learning_starts=10000, use_sde=False,  tensorboard_log="./sac_ant_tensorboard", policy_kwargs=policy_kwargs, verbose=1, seed=594371)
            model.learn(total_timesteps=int(1e6))
        elif mod==4:
            model = SAC_LABER('MlpPolicy','Ant-v3',learning_starts=10000, use_sde=False,  tensorboard_log="./sac_ant_tensorboard", policy_kwargs=policy_kwargs, verbose=1, seed=594371)
            model.learn(total_timesteps=int(1e6))
        #model.learn(total_timesteps=int(1e4),reward_type="nominal",sem=name)

        model.save("sac_Ant-v3_"+name+"_"+str(run))
        model.save_replay_buffer("sac_Ant-v3_replay_buffer_"+name+"_"+str(run))
        end = timeit.default_timer()
        print("training_time is"+str(end-start))  
        text_file = open("ant.txt", "a")
        text_file.write('Time for '+str(run)+' is : '+str(end-start)+'\n')
        text_file.close()
    

    
    
    elif env_id=="Humanoid-v3":
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        start = timeit.default_timer()
        env = DummyVecEnv([lambda: gym.make('Humanoid-v3')])

        if mod==1 or mod==2:
            model = SAC('MlpPolicy','Humanoid-v3',learning_starts=10000, tensorboard_log="./sac_humanoid_tensorboard", verbose=1, seed=594371)
            model.learn(total_timesteps=int(2e6),mode=name)
        elif mod==3:
            model = SAC_PER('MlpPolicy','Humanoid-v3',learning_starts=10000, tensorboard_log="./sac_humanoid_tensorboard", verbose=1, seed=594371)
            model.learn(total_timesteps=int(2e6))
        elif mod==4:
            model = SAC_LABER('MlpPolicy','Humanoid-v3',learning_starts=10000, tensorboard_log="./sac_humanoid_tensorboard", verbose=1, seed=594371)
            model.learn(total_timesteps=int(2e6))
        
        #model.learn(total_timesteps=int(1e6),reward_type="nominal",sem=name) #exp61640/1000
        #model.learn(total_timesteps=int(3e6),reward_type="STL")

        model.save("sac_Humanoid-v3_"+name+"_"+str(run))
        model.save_replay_buffer("sac_Humanoid-v3_replay_buffer_"+name+"_"+str(run))
        end = timeit.default_timer()
        print("training_time is"+str(end-start))
        text_file = open("humanoid.txt", "a")
        text_file.write('Time for '+str(run)+' is : '+str(end-start)+'\n')
        text_file.close()

    
    


if __name__ == "__main__":  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, default="Hopper-v3")
    parser.add_argument("--mod", help="Algorithm", default=1, type=int, required=False)
    parser.add_argument("--run", help="run id", default=10, type=int, required=False)
  
    args = parser.parse_args()

    env_id = args.env
    mod = args.mod
    run = args.run
    run_exp(env_id,mod,run)
