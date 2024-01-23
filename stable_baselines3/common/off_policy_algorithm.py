import io
import pathlib
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
import sys
import csv
from discretization import discretize
import pickle
#fulllog=[]
from itertools import repeat
import os
from sklearn.neighbors import KernelDensity

class OffPolicyAlgorithm(BaseAlgorithm):
    """
    The base for Off-Policy algorithms (ex: SAC/TD3)

    :param policy: Policy object
    :param env: The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param policy_base: The base policy used by this method
    :param learning_rate: learning rate for the optimizer,
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param policy_kwargs: Additional arguments to be passed to the policy on creation
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param verbose: The verbosity level: 0 none, 1 training information, 2 debug
    :param device: Device on which the code should run.
        By default, it will try to use a Cuda compatible device and fallback to cpu
        if it is not possible.
    :param support_multi_env: Whether the algorithm supports training
        with multiple environments (as in A2C)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param seed: Seed for the pseudo random generators
    :param use_sde: Whether to use State Dependent Exploration (SDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param sde_support: Whether the model support gSDE or not
    :param remove_time_limit_termination: Remove terminations (dones) that are due to time limit.
        See https://github.com/hill-a/stable-baselines/issues/863
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    def __init__(
        self,
        policy: Type[BasePolicy],
        env: Union[GymEnv, str],
        policy_base: Type[BasePolicy],
        learning_rate: Union[float, Schedule],
        buffer_size: int = 1000000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "step"),
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        support_multi_env: bool = False,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        sde_support: bool = True,
        remove_time_limit_termination: bool = False,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
    ):

        super(OffPolicyAlgorithm, self).__init__(
            policy=policy,
            env=env,
            policy_base=policy_base,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            support_multi_env=support_multi_env,
            create_eval_env=create_eval_env,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            supported_action_spaces=supported_action_spaces,
        )
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.tau = tau
        self.gamma = gamma
        self.gradient_steps = gradient_steps
        self.action_noise = action_noise
        self.optimize_memory_usage = optimize_memory_usage
        self.replay_buffer_class = replay_buffer_class
        if replay_buffer_kwargs is None:
            replay_buffer_kwargs = {}
        self.replay_buffer_kwargs = replay_buffer_kwargs
        self._episode_storage = None
        self.grad_list = []
        self.hes_list = []
        self.rb_replace_cnt = 0 #number of replacements per 1000 samples
        #self.beta = 1 #exp56-57
        #self.beta = 2 #exp58
        self.beta = 0.2 #exp60


        # Remove terminations (dones) that are due to time limit
        # see https://github.com/hill-a/stable-baselines/issues/863
        self.remove_time_limit_termination = remove_time_limit_termination

        # Save train freq parameter, will be converted later to TrainFreq object
        self.train_freq = train_freq

        self.actor = None  # type: Optional[th.nn.Module]
        self.replay_buffer = None  # type: Optional[ReplayBuffer]
        # Update policy keyword arguments
        if sde_support:
            self.policy_kwargs["use_sde"] = self.use_sde
        # For gSDE only
        self.use_sde_at_warmup = use_sde_at_warmup

    def _convert_train_freq(self) -> None:
        """
        Convert `train_freq` parameter (int or tuple)
        to a TrainFreq object.
        """
        if not isinstance(self.train_freq, TrainFreq):
            train_freq = self.train_freq

            # The value of the train frequency will be checked later
            if not isinstance(train_freq, tuple):
                train_freq = (train_freq, "step")

            try:
                train_freq = (train_freq[0], TrainFrequencyUnit(train_freq[1]))
            except ValueError:
                raise ValueError(f"The unit of the `train_freq` must be either 'step' or 'episode' not '{train_freq[1]}'!")

            if not isinstance(train_freq[0], int):
                raise ValueError(f"The frequency of `train_freq` must be an integer and not {train_freq[0]}")

            self.train_freq = TrainFreq(*train_freq)

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        # Use DictReplayBuffer if needed
        if self.replay_buffer_class is None:
            if isinstance(self.observation_space, gym.spaces.Dict):
                self.replay_buffer_class = DictReplayBuffer
            else:
                self.replay_buffer_class = ReplayBuffer

        elif self.replay_buffer_class == HerReplayBuffer:
            assert self.env is not None, "You must pass an environment when using `HerReplayBuffer`"

            # If using offline sampling, we need a classic replay buffer too
            if self.replay_buffer_kwargs.get("online_sampling", True):
                replay_buffer = None
            else:
                replay_buffer = DictReplayBuffer(
                    self.buffer_size,
                    self.observation_space,
                    self.action_space,
                    self.device,
                    optimize_memory_usage=self.optimize_memory_usage,
                )

            self.replay_buffer = HerReplayBuffer(
                self.env,
                self.buffer_size,
                self.device,
                replay_buffer=replay_buffer,
                **self.replay_buffer_kwargs,
            )

        if self.replay_buffer is None:
            self.replay_buffer = self.replay_buffer_class(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                self.device,
                optimize_memory_usage=self.optimize_memory_usage,
                **self.replay_buffer_kwargs,
            )

        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

        # Convert train freq parameter to TrainFreq object
        self._convert_train_freq()

    def save_replay_buffer(self, path: Union[str, pathlib.Path, io.BufferedIOBase]) -> None:
        """
        Save the replay buffer as a pickle file.

        :param path: Path to the file where the replay buffer should be saved.
            if path is a str or pathlib.Path, the path is automatically created if necessary.
        """
        assert self.replay_buffer is not None, "The replay buffer is not defined"
        save_to_pkl(path, self.replay_buffer, self.verbose)

    def load_replay_buffer(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        truncate_last_traj: bool = True,
    ) -> None:
        """
        Load a replay buffer from a pickle file.

        :param path: Path to the pickled replay buffer.
        :param truncate_last_traj: When using ``HerReplayBuffer`` with online sampling:
            If set to ``True``, we assume that the last trajectory in the replay buffer was finished
            (and truncate it).
            If set to ``False``, we assume that we continue the same trajectory (same episode).
        """
        self.replay_buffer = load_from_pkl(path, self.verbose)
        assert isinstance(self.replay_buffer, ReplayBuffer), "The replay buffer must inherit from ReplayBuffer class"

        # Backward compatibility with SB3 < 2.1.0 replay buffer
        # Keep old behavior: do not handle timeout termination separately
        if not hasattr(self.replay_buffer, "handle_timeout_termination"):  # pragma: no cover
            self.replay_buffer.handle_timeout_termination = False
            self.replay_buffer.timeouts = np.zeros_like(self.replay_buffer.dones)

        if isinstance(self.replay_buffer, HerReplayBuffer):
            assert self.env is not None, "You must pass an environment at load time when using `HerReplayBuffer`"
            self.replay_buffer.set_env(self.get_env())
            if truncate_last_traj:
                self.replay_buffer.truncate_last_trajectory()

    def _setup_learn(
        self,
        total_timesteps: int,
        eval_env: Optional[GymEnv],
        callback: MaybeCallback = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
    ) -> Tuple[int, BaseCallback]:
        """
        cf `BaseAlgorithm`.
        """
        # Prevent continuity issue by truncating trajectory
        # when using memory efficient replay buffer
        # see https://github.com/DLR-RM/stable-baselines3/issues/46

        # Special case when using HerReplayBuffer,
        # the classic replay buffer is inside it when using offline sampling
        if isinstance(self.replay_buffer, HerReplayBuffer):
            replay_buffer = self.replay_buffer.replay_buffer
        else:
            replay_buffer = self.replay_buffer

        truncate_last_traj = (
            self.optimize_memory_usage
            and reset_num_timesteps
            and replay_buffer is not None
            and (replay_buffer.full or replay_buffer.pos > 0)
        )

        if truncate_last_traj:
            warnings.warn(
                "The last trajectory in the replay buffer will be truncated, "
                "see https://github.com/DLR-RM/stable-baselines3/issues/46."
                "You should use `reset_num_timesteps=False` or `optimize_memory_usage=False`"
                "to avoid that issue."
            )
            # Go to the previous index
            pos = (replay_buffer.pos - 1) % replay_buffer.buffer_size
            replay_buffer.dones[pos] = True

        return super()._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            log_path,
            reset_num_timesteps,
            tb_log_name,
        )

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "run",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        imp_states: list = [],
        reward_type: str = "nominal",
        sem: str = "cls",
        mode: str = "default",
    ) -> "OffPolicyAlgorithm":

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )

        callback.on_training_start(locals(), globals(),self.env)

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
                imp_states=imp_states,
                reward_type=reward_type,
                sem=sem,
                mode=mode,
            )

            if rollout.continue_training is False:
                break

            #if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts and self.replay_buffer.is_replace==0: #exp464 
            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:  #uncomment in exp57
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                #print("gradient steps : ",gradient_steps)
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)
                

        callback.on_training_end()
        # with open("glist.pkl", 'wb') as file:
        #     pickle.dump(self.grad_list, file)
        # with open("hlist.pkl", 'wb') as file:
        #     pickle.dump(self.hes_list, file)

        return self

    def train(self, gradient_steps: int, batch_size: int) -> None:
        """
        Sample the replay buffer and do the updates
        (gradient descent and update target networks)
        """
        raise NotImplementedError()

    def _sample_action(
        self, learning_starts: int, action_noise: Optional[ActionNoise] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        #print(self._last_obs)
                
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample()])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, gym.spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        time_elapsed = time.time() - self.start_time
        fps = int(self.num_timesteps / (time_elapsed + 1e-8))
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total timesteps", self.num_timesteps, exclude="tensorboard")
        if self.use_sde:
            self.logger.record("train/std", (self.actor.get_std()).mean().item())

        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success rate", safe_mean(self.ep_success_buffer))
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)

    def _on_step(self) -> None:
        """
        Method called after each step in the environment.
        It is meant to trigger DQN target network update
        but can be used for other purposes
        """
        pass

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
        dss: np.ndarray,
        replace : int,
        idx : int,
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when done is True)
        :param reward: reward for the current transition
        :param done: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        if done and infos[0].get("terminal_observation") is not None:
            next_obs = infos[0]["terminal_observation"]
            # VecNormalize normalizes the terminal observation
            if self._vec_normalize_env is not None:
                next_obs = self._vec_normalize_env.unnormalize_obs(next_obs)
        else:
            next_obs = new_obs_

        replay_buffer.add(
            self._last_original_obs,
            next_obs,
            buffer_action,
            reward_,
            done,
            infos,
            dss,
            replace,
            idx,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_
    
    def _discrete_to_concrete(self,dss):
        a,b,c = dss
        css = np.array([callback.state_grid[0][a],callback.state_grid[1][b],callback.state_grid[2][c]])
        return css

    def _create_maps(self, env, callback, obs, action, reward, new_obs,p,q):
        '''
        Creating data structures for analysis of replay buffers
        '''
        
        # Obtain some samples from the space, discretize them, and then visualize them
        #state_sample = env.observation_space.sample()
        #print(state_sample)
        #sstate_sample = obs[0]
        #print(state_sample)
        #print(obs)
        #print(obs[0])
        full_obs = obs
        
        #exit()
        obs = [obs[0][p]] # usibng MD output
        action = [action[0][q]] # usibng MD output

        discretized_state_samples = discretize(obs[0], callback.state_grid)
        discretized_act_samples = discretize(action[0], callback.act_grid)

        #print(discretized_act_samples)
        
        #below code for no of states in an abstract state
        if tuple(discretized_state_samples) in callback.nmap.keys():
            callback.nmap[tuple(discretized_state_samples)] = callback.nmap[tuple(discretized_state_samples)] + 1
        else:
            callback.nmap[tuple(discretized_state_samples)] = 0

        #below code for rewards in an abstract state
        if tuple(discretized_state_samples) in callback.rmap.keys():
            callback.rmap[tuple(discretized_state_samples)].append(reward)
        else:
            callback.rmap[tuple(discretized_state_samples)] = [reward]

        
        #below code for action in an abstract state
        if tuple(discretized_state_samples) in callback.acmap.keys():
            callback.acmap[tuple(discretized_state_samples)].append(action[0])
        else:
            callback.acmap[tuple(discretized_state_samples)] = [action[0]]

        #below code for abstract action in an abstract state
        if tuple(discretized_state_samples) in callback.dacmap.keys():
            callback.dacmap[tuple(discretized_state_samples)].append(discretized_act_samples)
        else:
            callback.dacmap[tuple(discretized_state_samples)] = [discretized_act_samples]
        
        #below code for abstract action to concrete action
        if tuple(discretized_act_samples) not in callback.amap.keys():
            callback.amap[tuple(discretized_act_samples)] = action[0]

        #below code for abstract state to concrete state
        if tuple(discretized_state_samples) not in callback.smap.keys():
            callback.smap[tuple(discretized_state_samples)] = obs[0]
        # Else part can be added to make this concrete state more meaningful
        # else:
        #    callback.smap[tuple(discretized_state_samples)] = list(np.array(action))
        
        if tuple(discretized_state_samples) not in callback.fmap.keys():
            callback.fmap[tuple(discretized_state_samples)] = full_obs[0]

        
    def _get_uniform_threshold(self,a,b,n,beta):
        np.random.seed(0)
        data1 = [[x] for x in np.random.uniform(0,1,n)]
        ukd = KernelDensity(kernel='epanechnikov', bandwidth=0.3).fit(data1)
        
        start = 0.4  # Start of the range
        end = 0.6  # End of the range
        N = 10     # Number of evaluation points
        step = (end - start) / (N - 1)  # Step size
        x = np.linspace(start,end , N)[:, np.newaxis]  # Generate values in the range
        ukd_vals = np.exp(ukd.score_samples(x))  # Get PDF values for each x
        p = np.sum(ukd_vals * step)  # Approximate the integral of the PDF
        #print("#rewards is ",len(data1),"  min  prob mass is : ",np.min(p))
        return p
    


    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
        imp_states : list = [],
        reward_type : str = "nominal",
        sem : str = "cls",
        mode : str = "default",
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        episode_rewards, total_timesteps = [], []
        num_collected_steps, num_collected_episodes = 0, 0
        
        #ctr=0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if self.use_sde:
            self.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True
        global spec
         
        # TODO Below code should be called from learn function 
        #abstract_states = self._abstraction(env,callback,smap)
        
        reset_ts = 0 #resetting

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            done = False
            episode_reward, episode_timesteps = 0.0, 0
            #ctr = 0


            while not done:

                if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.actor.reset_noise()

                # istate, irank = imp_states
                
                # #if reset_ts<self.num_timesteps:
                # #    reset_ts = self.num_timesteps + 1000*np.sum(irank)
                # if istate is not None:
                #     ist = {}
                #     for i in range(len(irank)):
                #         ist[tuple(istate[i])] = irank[i]

                #     #istate = [x for item in istate for x in repeat(item, ist[tuple(item)])]
                #     len_abs = len(istate)
                # else:
                #     len_abs = 0
                
                bonus=0

                # Select action randomly or according to policy
                '''
                if mode=="abs" and self.num_timesteps%1000==0 and callback.obs_count<len_abs:
                    ## TODO pick obs using abstraction 
                    #self._last_obs = env.observation_space.sample()
                    
                    index = callback.obs_count
                    self._last_obs = np.array([istate[index]])
                    print("picking an abstract state state ",istate[index]," at timestep ",self.num_timesteps," at index ",index)
                    callback.obs_count = callback.obs_count + 1
                '''

                action, buffer_action = self._sample_action(learning_starts, action_noise)
                #print(self._last_obs)   
                #print(action)
                #print(buffer_action)
                #exit()
                # Rescale and perform action
                new_obs, reward, done, infos = env.step(action)
                #env.render()

                
                # if mode=="abs" and self.num_timesteps%1000==0:
                #     dbfile = open('rmap.pkl', 'rb')
                #     rm = pickle.load(dbfile)
                #     keys = list(rm.keys())
                #     values = list(rm.values())
                #     #print(len(keys))
                #     #print(len(values))
                #     rdict={}
                #     for i in range(len(keys)):
                #         #rdict[keys[i]] = np.mean(values[i]) #reward mean MC=252 HOP=502
                #         rdict[keys[i]] = np.max(values[i])-np.min(values[i]) #reward range MC=251 HOP=501
                #         #rdict[keys[i]] = np.max(values[i]) #reward max MC:25 All exp before 25 use this HOP=50 
                #     #print(len(rdict))
                #     #print(rdict.values())
                    
                #print(reward_type)
                name = env.envs[0].unwrapped.spec.id
                #print("at the beg : "+str(callback.ctr)+" "+str(callback.n_calls))
                ############################################################################
                
                p = [0,1,2]
                q = [0]

                # below is the list of important dimensions that have been generated by src/find_imp_dim.py file.
                # using the inital random samples from replay buffer till the "learning_starts" parameter kicks in. 

                if name=="Pendulum-v1":
                    p = [2]
                elif name=="MountainCarContinuous-v0":
                    p = [0]
                elif name=="LunarLanderContinuous-v2":
                    p = [1, 2]
                elif name=="ReacherBulletEnv-v0":
                    p = [8, 5, 7, 4, 6]                
                elif name=="Swimmer-v3":
                    p = [7, 6, 1, 2] 
                elif name == "Hopper-v3":
                    p = [7, 10, 9, 0] #default
                elif name=="Walker2d-v3":
                    p = [13, 16, 14, 12, 15] #default
                elif name=="Ant-v3":
                    p = [21, 23, 19, 25, 24, 20, 26, 22] #default
                elif name=="Humanoid-v3":
                    p = [284, 280, 281, 285, 279, 283, 282, 275, 277, 276, 278, 64, 290, 287, 291, 289] 
                
                #print(amap)
                
                '''
                self._create_maps(env,callback,self._last_obs,buffer_action,reward,new_obs,p,q)
                #print("rmap keys len:   ",len(callback.rmap.keys()))
                endt = self._total_timesteps-1
                
                if self.num_timesteps==endt:
                    print(len(callback.nmap.keys()))
                    with open("nmap.pkl", 'wb') as file:
                        pickle.dump(callback.nmap, file)
                    with open("rmap.pkl", 'wb') as file:
                        pickle.dump(callback.rmap, file)
                    with open("acmap.pkl", 'wb') as file:
                        pickle.dump(callback.acmap, file)
                    with open("dacmap.pkl", 'wb') as file:
                        pickle.dump(callback.dacmap, file)
                    with open("stategrid.pkl", 'wb') as file:
                        pickle.dump(callback.state_grid, file)
                    with open("actgrid.pkl", 'wb') as file:
                        pickle.dump(callback.act_grid, file)
                    with open("smap.pkl", 'wb') as file:
                        pickle.dump(callback.smap, file)
                    with open("amap.pkl", 'wb') as file:
                        pickle.dump(callback.amap, file)
                    with open("fmap.pkl", 'wb') as file:
                        pickle.dump(callback.fmap, file)
                    with open("rbmap.pkl", 'wb') as file:
                        pickle.dump(callback.rbmap, file)
                    #elif self.num_timesteps==0:
                '''
                #if self.num_timesteps>50000 and self.num_timesteps<150000: #exp55
                #if self.num_timesteps>50000 and self.num_timesteps<300000: #exp56/57/58
                '''
                if self.num_timesteps>10000 and self.num_timesteps<1000000: #exp59
                    sobs = [new_obs[0][p]]
                    dobs = discretize(sobs[0], callback.state_grid)
                    if tuple(dobs) not in callback.rmap.keys():
                        bonus=0
                    else:
                        rws = callback.rmap[tuple(dobs)]
                        # print(new_obs)
                        # print(sobs)
                        # print(dobs)
                        # print(rws)

                        #beta=1 #exp55/56
                        beta=5 #exp57/59
                        #beta=10 #exp58 
                        bonus = (np.max(rws)-np.min(rws))/beta
                        #print("reward :",reward,"    bonus ",bonus)

                
                reward = reward + bonus
                '''
                
                #############################################################################
                
                self.num_timesteps += 1
                episode_timesteps += 1
                num_collected_steps += 1
                #ctr += 1 
                #print("at the end : "+str(ctr)+" "+str(self.num_timesteps))

                # Give access to local variables
                callback.update_locals(locals())
                # Only stop training if return value is False, not when it is None.
                if callback.on_step() is False:
                    return RolloutReturn(0.0, num_collected_steps, num_collected_episodes, continue_training=False)

                episode_reward += reward

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, done)

                ############################################################
                
                if mode=="fac":
                    # A check using abstraction info if entry is to be kept in rb
                    # if self._last_obs in grid already and buffer_action in grid already 
                    # then "replace" it
                    
                    obs = [self._last_obs[0][p]] 
                    dss = discretize(obs[0], callback.state_grid)
                    das = discretize(buffer_action[0][q], callback.act_grid)
                    # print(self._last_obs[0])
                    # print(obs[0])
                    # print(dss)
                    # print(buffer_action[0])
                    # print(buffer_action[0][q])
                    # print(das)
                    # exit()
                    #below code for abstract state to concrete state
                    if tuple(dss) not in callback.dsdrmap.keys():
                        callback.dsdrmap[tuple(dss)] = [[reward[0]]]
                    #else:
                    #    callback.dsdrmap[tuple(dss)].append(reward)

                    replace=0
                    #print(dss)
                    #print(das)
                    #print(callback.dsdamap.keys())
                    #print(callback.dsdrmap[tuple(dss)])
                    #print("***************************")
                    
                    # below is the code for density estimation

                    data = callback.dsdrmap[tuple(dss)]
                    #print(data)
                    kd = KernelDensity(kernel='epanechnikov', bandwidth=0.3).fit(data) #FAC
                    #kd = KernelDensity(kernel='tophat', bandwidth=0.3).fit(data) #exp90
                    #kd = KernelDensity(kernel='gaussian', bandwidth=0.3).fit(data) #exp91
                    #kd = KernelDensity(kernel='cosine', bandwidth=0.3).fit(data) #exp92
                    

                    #likelihood
                    #lkh = np.exp(kd.score_samples([[reward]]))

                    #self.beta = 0.1  #exp47
                    #self.beta = 0.3  #exp48
                    #self.beta = 0.5  #exp49
                    self.beta = 0.2  #default
                    # Get probability for range of values
                    start = reward[0]-self.beta  # Start of the range
                    end = reward[0]+self.beta    # End of the range
                    N = 5    # Number of evaluation points
                    step = (end - start) / (N - 1)  # Step size
                    x = np.linspace(start,end , N)[:, np.newaxis]  # Generate values in the range
                    kd_vals = np.exp(kd.score_samples(x))  # Get PDF values for each x
                    p = np.sum(kd_vals * step)  # Approximate the integral of the PDF

                    p = p*np.exp(len(data)/100000)  #default
                    #p = p*np.exp(len(data)/10000)  #exp67
                    #p = p*np.exp(len(data)/50000)  #exp68
                    #p = p*np.exp(len(data)/200000) #exp69
                    #p = p*np.exp(len(data)/150000) #exp70
                    #p = p #exp71
                    #print("probability is ",p)
                    #print("reward is ",reward[0])
                    #ep = self._get_uniform_threshold(np.min(data),np.max(data),len(data),self.beta)
                    
                    # Here we check if the sample is unique 
                    if tuple(dss) not in callback.rbmap.keys() or self.num_timesteps<10000:
                        #insert for new state
                        callback.rbmap[tuple(dss)] = self.replay_buffer.pos
                        replace=0
                        #elif p < ep:  #default
                    elif p < 0.2:  #being more conservative, default
                        #elif p < 0.3: #exp79    
                        #elif p < 0.1: #exp78    
                        #elif p < 0.5: #exp77 
                        #insert for new action for a given state
                        #callback.rbmap[tuple(dss)] = self.replay_buffer.pos
                        callback.dsdrmap[tuple(dss)].append([reward[0]])
                        replace=0
                    else:
                        #replacing existing state-action pair
                        replace=1
                        #print("for state ",dss,"  actions ",callback.dsdamap[tuple(dss)])
                    idx = callback.rbmap[tuple(dss)]

               
                    if self.num_timesteps<10000:
                        replace=0
                    # also need a map from abstract state to rb entry index
                    # Store data in replay buffer (normalized action and unnormalized observation)

                    #print("len data is ",len(data))
                    
                    self.replay_buffer.is_replace = replace
                    #exp55
                    if replace==1:
                        self.rb_replace_cnt = self.rb_replace_cnt + 1
                    
                    
                    self._store_transition(replay_buffer, buffer_action, new_obs, reward, done, infos, dss, replace, idx)
                else:
                    self._store_transition(replay_buffer, buffer_action, new_obs, reward, done, infos, [],0,self.replay_buffer.pos)

                self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                ############################################################################
                ct = self.num_timesteps
                endt = self._total_timesteps-1
                rbmap = callback.rbmap.copy()
                dsdrmap = callback.dsdrmap.copy()
            
                #if ct%20000==0:
                #    #print(rbmap)
                #    #print(dsdrmap)
                #    #exit()
                #    with open("rbmap_"+str(ct)+".pkl", 'wb') as file:
                #        pickle.dump(rbmap, file)
                #    with open("dsdrmap_"+str(ct)+".pkl", 'wb') as file:
                #        pickle.dump(dsdrmap, file)
                ############################################################################

                # For DQN, check if the target network should be updated
                # and update the exploration schedule
                # For SAC/TD3, the update is done as the same time as the gradient update
                # see https://github.com/hill-a/stable-baselines/issues/900
                self._on_step()

                if not should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
                    break

            if done:
                num_collected_episodes += 1
                self._episode_num += 1
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)
                #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>."+str(episode_timesteps))
                callback.ctr = 0

                if action_noise is not None:
                    action_noise.reset()

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()

        mean_reward = np.mean(episode_rewards) if num_collected_episodes > 0 else 0.0
        
        callback.on_rollout_end()

        return RolloutReturn(mean_reward, num_collected_steps, num_collected_episodes, continue_training)
