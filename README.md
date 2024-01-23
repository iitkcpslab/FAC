# FAC  
## Frugal Actor-Critic:  Sample Efficient Off-Policy Deep Reinforcement Learning Using Unique Experiences

FAC focusses on sample efficiency, by selecting unique samples and adding them to the replay buffer during the exploration with the goal of reducing the buffer size and maintaining independent and identically distributed (IID) nature of the samples.

## Main Features

- FAC algorithm.  
- Comparison with State-of-the-art algorithms SAC and TD3.  
- Faster Convergence and more IID samples.
- Added script for controller evaluation.    
  

### Prerequisites

Use Ubuntu 22.04 and above.  
Packages: libboost-all-dev, python-dev, python-pip,  

Install mujoco (https://github.com/openai/mujoco-py).   

Python packages    
<Pkg>        <Preferable Version>   
Python         3.7     
torch          1.13.0      
gym            0.21.0    
glfw           2.5.7   
imageio        2.26.0   
mujoco-py      >=2.1.2   
pybullet       3.2.5  


The above version are highly recommended due issues with other versions.   
For instance, mujoco-py(2.1.2.12) conflicts with gym(0.24.0) and cython(>=3).   
So install cython using the following command :  pip install "cython<3".    



## Installation
Create a python3.7 virtual environment and do the following:   

Unzip the file FAC.zip and install the FAC package:    
```
cd FAC/
pip install -e .
```

After installation add the path to the "FAC" package to the PYTHONPATH variable in ~/.bashrc file.    
For example, if the FAC package is at location /home/PC/FAC/, then add the following line in the ~/.bashrc file:  
export PYTHONPATH=/home/PC/FAC/:$PYTHONPATH 

Alternatively, instead to adding these lines to ~/.bashrc, you can run these two lines in the terminal but it will be valid for a session only.   



## Running Experiments  

### Synthesizing Controllers    

```
cd src/
```

```
python run_experiments.py --env=<Env> --mod=<Mode> --run=<run-id>
```
where Env={Pendulum-v1, MountainCarContinuous-v0, LunarLanderContinuous-v2, ReacherBulletEnv-v0, Swimmer-v3, Hopper-v3, Ant-v3, Walker2d-v3, Humanoid-v3}   
Mode is an integer to denote the default mode (mod=1), FAC mode (mod=2), PER(mod=3) and LABER (mod=4).      
   
run-id is a unique integer to be provided by the user. This purpose is to distinguish one set of experiments from the other.   

For example, to synthesize the controller for Hopper for FAC mode for the first time, run the command:    

```
python run_experiments.py --env=Hopper-v3 --mod=2 --run=1
```

Once the experiment finishes, it will create a controller file named sac_Env_Mode_run-id.zip.   
Alternatively, you can copy the controller (control policy) files from the experiments/ folder.          


### Finding Important dimensions     

For FAC, an important step is finding the important state dimensions. To do this, first ensure that you have the replay buffer file (*.pkl) in the current directory. For instance, for the Hopper benchmark, training via the baseline SAC algorithm will create a replay buffer file "sac_Hopper-v3_replay_buffer_default_1.pkl". The intial 10k entries are random and same in all the experiments. Alternatively, you can do training for 10k timesteps and use that replay buffer.    

The command for finding important state dimensions is :   
```
python find_imp_dim.py --env=<Envid> --nu=0.5
```
where Envid is the index of benchmark in the order mentioned in paper. For example, Envid=1 for Pendulum-v1 and Envid=9 for Humanoid-v3. Also, nu refers to the \nu hyperparameter used in FAC with value 0.5 used in our experiments. Alternatively, you can copy the replay buffer files from the experiments/sac_replay_buffers/def/*.pkl.    


### Evaluation  

For evaluation w.r.t SAC for environemnt <Env> rename the file C to sac_Env.zip 

```
python eval_sac.py <Env>
```
Similarly evaluation w.r.t TD3 
```
python eval_td3.py <Env>
```

For example, to evaluate the controller for Hopper w.r.t SAC, rename the file to sac_Hopper-v3.zip (i.e. trim the   
Mode and runid). Then run the command: python eval_sac.py Hopper-v3

For evaluation (alternatively), you can copy the controller files to src/ folder and run eval_sac.py. For example, to evaluate the fac controller for Hopper benchmark, goto the src/ folder, do:   

```
cp experiments/sac_controllers/fac/sac_Hopper-v3_fac_1.zip sac_Hopper-v3.zip 
python eval_sac.py Hopper-v3 
```


### Files
All the source code is inside the src/ folder.   
The experiments/ folder contains the control policy for each benchmark. 
Due to Github file size restriction of 100MB, we have omitted all the files larger than 100MB from the experiments/ folder.   



### Reproducibilty
All Experiments have been performed on Intel(R) Xeon(R) Gold 6226R @2.90 GHzX16 CPU, NVIDIA RTX A4000 32GB Graphics card, 48GB RAM
and Ubuntu 22.04 OS.
Completely reproducible results are not guaranteed across PyTorch releases or different platforms.   
Refer to the following notes by   
PyTorch (https://pytorch.org/docs/stable/notes/randomness.html) and   
stable-baselines (https://stable-baselines3.readthedocs.io/en/master/guide/algos.html#reproducibility)    
