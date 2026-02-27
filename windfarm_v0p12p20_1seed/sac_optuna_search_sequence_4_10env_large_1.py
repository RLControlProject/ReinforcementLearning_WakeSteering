"""Unclassified, with no sensitivities, but not authorized for widespread or public release."""

"""how to run:
cd /ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p9
conda activate RLControlEnv
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python3 sac_optuna_search_sequence_4_10env_large_1.py > sac_optuna_search_sequence/trials/trial_4_10env_large_1_1/output.txt

monitor open files with:
lsof -u kbrown1 | wc -l

if need to close these:
pkill -u kbrown1
"""

import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os,sys
from functools import partial
import gymnasium as gym
# import torch
# import torch.nn as nn
# import helpers
from stable_baselines3 import HerReplayBuffer, SAC, TD3
# from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.sac.policies import MlpPolicy  as SACMlpPolicy
# from stable_baselines3.sac.policies import MultiInputPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv,VecEnvWrapper
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import configure_logger
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
# from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.evaluation import evaluate_policy

# from utils import transferWeightsAndBiases, transferWeightsAndBiases_sac
#from policy_distillation.model import FFNN,FFNN_LastLayerTanh

from callbacks import SaveOnBestTrainingRewardCallback, TrialEvalCallback, CumulativeRewardCallback, EvalCallback
from windfarm_env_inputindices_large_1 import WindFarmEnv_v1p3, tag, num_timesteps, num_turbines

# import pickle

# import pdb

from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_rank
from optuna.visualization import plot_slice
from optuna.visualization import plot_timeline


# Define the training function

def train_model(trial, log_dir = '', tb_log_name = ''):

    # set up log dir
    configure_logger(verbose=0)

    # Set up the environment

    env_kwargs = {}
    env_kwargs['windtype'] = 'DiscreteRandom'
    env_kwargs['initialyaw'] = 'FLORISYaw' # 'ZeroYaw' 'FLORISYaw'
    env_kwargs['algorithm'] = 'SAC' 
    env_kwargs['rewardtype'] = 'stochastic_1seed' # 'base' 'stochastic' 'stochastic_1seed'

    num_episodes = 500 # approximate

    # Make environment
    n_envs = 10
    venv0 = make_vec_env(WindFarmEnv_v1p3, vec_env_cls=SubprocVecEnv, n_envs=n_envs, env_kwargs=env_kwargs) # , seed=0
    venv = VecMonitor(venv=venv0, filename=None, info_keywords=(["r", "r_stochastic", "r_stochastic_1seed"])) #os.path.join(log_dir,tb_log_name+'_1'), info_keywords=(["r_stochastic", "power", "power_floris_opt_stochastic", "power_floris_opt", "yaw_angles", "yaw_angles_floris_opt", "rel_yaw_angles", "rel_yaw_angles_floris_opt", "wind_dir", "wind_spd", "turb_int"]))

    # Set up the hyperparameters (good reference: https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe)

    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-1) # original: 1e-6
    learning_starts = trial.suggest_int("learning_starts", 0,200) 
    gradient_steps = -1#trial.suggest_int("gradient_steps", 1,500) # lock this to one step per environment step based on heuristic suggestions to not overfit: https://spinningup.openai.com/en/latest/algorithms/sac.html, https://www.reddit.com/r/reinforcementlearning/comments/iarneq/more_gradient_updates_in_ddpgsac/
    train_freq = 1#trial.suggest_int("train_freq", 1,2*n_envs+1, step = 1)
    ent_coef = trial.suggest_loguniform("ent_coef", 0.001, 0.9) # original: 0.2
    target_entropy_multiplier = trial.suggest_int("target_entropy_multiplier", 1,1)#trial.suggest_loguniform("target_entropy_multiplier", 0.001, 1)
    batch_size = 25000#trial.suggest_int("batch_size", 25, num_episodes) # original: 256
    # action_noise_sigma = trial.suggest_loguniform("action_noise_sigma", 0.0001, 0.4) # original: 0.1
    # tau = trial.suggest_loguniform("tau", 0.0001, 0.2) # original: 0.005
    # target_update_interval = trial.suggest_int("target_update_interval", 1,3*n_envs+1, step = 1)
    # buffer_size = trial.suggest_int("buffer_size", 5, 5000)
    # n_sampled_goal = trial.suggest_int("n_sampled_goal", 1, 20)
    # goal_selection_strategy =trial.suggest_categorical('goal_selection_strategy', ['episode', 'final', 'future'])
    # stats_window_size = trial.suggest_int("stats_window_size", 1, 100)
    # neurons_per_layer = trial.suggest_int("neurons_per_layer", 20, 400)
    # num_layers = trial.suggest_int("num_layers", 2, 8)

    # Create the agent

    # noise/entropy setup
    n_actions = venv.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0 * np.ones(n_actions)) # action_noise_sigma
    target_entropy = -target_entropy_multiplier*num_turbines

    # policy_kwargs = dict(activation_fn=nn.ReLU, net_arch = helpers.create_net_arch(neurons_per_layer, num_layers)) # original: dict(net_arch=[400, 300])
    policy_kwargs = dict(net_arch=[256, 256])
    model = SAC(SACMlpPolicy, venv, policy_kwargs=policy_kwargs, # use "MultiInputPolicy" for HER replay buffer class
                batch_size=batch_size, gamma=0, action_noise=action_noise, 
                ent_coef='auto_'+str(ent_coef), target_entropy=target_entropy, #target_update_interval=target_update_interval, tau=tau, 
                learning_rate=learning_rate, learning_starts = learning_starts, train_freq = train_freq, #buffer_size = buffer_size,
                #replay_buffer_class=HerReplayBuffer,replay_buffer_kwargs=dict(n_sampled_goal=n_sampled_goal,goal_selection_strategy=goal_selection_strategy),
                use_sde = False, gradient_steps=gradient_steps,verbose=0,device="auto", tensorboard_log=None) # ,tensorboard_log=log_dir stats_window_size=stats_window_size,

    # # print(model.policy)
    # LOAD_PRETRAINED_NETWORK=True
    # # [OPTIONAL] load pre-trained network
    # if LOAD_PRETRAINED_NETWORK:
    #     pi = FFNN(layers=[num_obs,policy_kwargs['net_arch'][0],policy_kwargs['net_arch'][1],num_turbines])
    #     pi.load(os.path.join(os.path.dirname(__file__))+"/policy_distillation/pi_sac.dat")
    #     transferWeightsAndBiases_sac('pi', source=pi, target=model.policy.actor)

    #     # qf = FFNN(layers=[num_obs+num_turbines,policy_kwargs['net_arch'][0],policy_kwargs['net_arch'][1],1])
    #     # qf.load(os.path.join(os.path.dirname(__file__))+"/policy_distillation/qf_sac.dat")
    #     # transferWeightsAndBiases_sac('qf', source=qf, target=model.policy.critic.qf0)
    #     # transferWeightsAndBiases_sac('qf', source=qf, target=model.policy.critic.qf1)
    #     # transferWeightsAndBiases_sac('qf', source=qf, target=model.policy.critic_target.qf0)
    #     # transferWeightsAndBiases_sac('qf', source=qf, target=model.policy.critic_target.qf1)

    # Set up callbacks

    # max_no_improvement_evals=num_episodes
    # min_evals=1
    # stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=max_no_improvement_evals, min_evals=min_evals, verbose=1)
    
    # check_freq = 50    
    # save_callback = SaveOnBestTrainingRewardCallback(check_freq=check_freq, log_dir=os.path.join(log_dir,tb_log_name+'_1'))

    # eval_episodes = n_envs # every eval causes step counter to increment by one
    # eval_freq = 1000000 # times n_env episodes (make unreachably large for now to disable)
    # eval_callback = EvalCallback(venv, n_eval_episodes=eval_episodes, eval_freq=eval_freq, deterministic=True) # , callback_after_eval=stop_callback, best_model_save_path=log_dir, log_path=log_dir

    cumulative_reward_callback = CumulativeRewardCallback()

    # Train the agent
    model.learn(total_timesteps=num_timesteps*num_episodes,reset_num_timesteps=True,callback=cumulative_reward_callback) #,tb_log_name=os.path.join(log_dir,tb_log_name)) #,callback=[eval_callback]) # ,save_callback

    # Evaluate the policy
    n_eval_episodes = 1000
    mean_reward, std_reward = evaluate_policy(model, venv, n_eval_episodes=n_eval_episodes, deterministic = False) # set determistic to true to encourage higher entropy through end of the trial (i.e., don't penalize if agent happens to be on an unhelpful rabbit trail when the trial ends)
    # mean_reward = cumulative_reward_callback.cumulative_rewards
    # print(mean_reward)
    # # print(std_reward)

    del model
    del venv
    venv0.close()

    return mean_reward, n_envs
    

# Define the Optuna objective function

def objective(trial, log_dir = '', tb_log_name = ''):
    # Train the agent and get the mean reward

    n_trials = 1 # total number of trials is n_trials*n_envs (these trials are repetitions with the same set of hyperparameters, ideally would be >=1000)

    mean_rewards = []
    for i in range(n_trials):
        # print('repetition # '+str(i))
        mean_reward, n_envs = train_model(trial, log_dir, tb_log_name)
        mean_rewards = np.append(mean_rewards,mean_reward)
        # print(mean_rewards)

        # set pruning (i.e., early termination) criteria
        mean_reward_overall = np.mean(mean_rewards)
        stddev_reward_overall = np.std(mean_rewards)
        # if i*n_envs>=0.2*n_trials*n_envs and mean_reward_overall<-10 and stddev_reward_overall<5:
        #     print("Early termination on condition 1")
        #     print('Trial num was '+str(i))
        #     break
        # if i*n_envs>=0.2*n_trials*n_envs and mean_reward_overall<-5 and stddev_reward_overall<2.5:
        #     print("Early termination on condition 2")
        #     print('Trial num was '+str(i))
        #     break

    print('Num trials: '+str(len(mean_rewards)))
    print('Mean: '+str(mean_reward_overall))
    print('Std: '+str(stddev_reward_overall))

    return mean_reward_overall

if __name__ == '__main__':

    log_dir = os.path.join(os.path.dirname(__file__))+"/sac_optuna_search_sequence/trials"
    tb_log_name = 'trial_4_10env'+tag
    if not os.path.exists(os.path.join(log_dir,tb_log_name+'_1')):
        os.makedirs(os.path.join(log_dir,tb_log_name+'_1'))
    plot_dir = os.path.join(log_dir,tb_log_name+'_1','plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    summary_dir = os.path.join(log_dir,tb_log_name+'_1','summary')
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)

    # Define the directory where you want to store the SQLite database

    # storage_directory = os.path.join(log_dir,tb_log_name+'_1')  # Local folder for storage
    # os.makedirs(storage_directory, exist_ok=True)  # Create the directory if it doesn't exist
    # storage_path = os.path.join(storage_directory, 'restart.db')

    # Set up Optuna study and optimize the objective function

    study = optuna.create_study(direction="maximize",study_name="my_study") #, storage=f"sqlite:///{storage_path}", load_if_exists=True)
    objective = partial(objective, log_dir = log_dir, tb_log_name = tb_log_name)
    study.optimize(objective, n_jobs=25, n_trials=100) # these trials are different sets of hyperparameters

    # Print the best hyperparameters and the best mean reward

    best_params = study.best_params
    best_reward = study.best_value
    print("Best Hyperparameters:", best_params)
    print("Best Mean Reward:", best_reward)               
    summary = pd.concat([pd.DataFrame([best_params]), pd.DataFrame([best_reward])], axis=1)
    summary.to_csv(os.path.join(summary_dir,'summary_best.csv'), sep=',', index=False, encoding='utf-8')  

    # Print the mean of the top five hyperparameter cases
    best_trials = study.trials_dataframe().nlargest(5, 'value')    
    avg_params = {}
    for param_name in best_trials.columns:
        if param_name.startswith('params_'):
            param_values = best_trials[param_name].values
            avg_params[param_name[7:]] = sum(param_values) / len(param_values)
    print("Average of Top Five Hyperparameters:")
    for param_name, param_value in avg_params.items():
        print(f"{param_name}: {param_value}")
    print("Average of Top Five Rewards:")
    avg_reward = 0
    for triali in range(len(best_trials)):
        trial = best_trials.iloc[triali]
        avg_reward = avg_reward + trial.value / len(best_trials)
    print(avg_reward)
    summary = pd.concat([pd.DataFrame([avg_params]), pd.DataFrame([avg_reward])], axis=1)
    summary.to_csv(os.path.join(summary_dir,'summary_avgtop5.csv'), sep=',', index=False, encoding='utf-8')  

    # save the full list of trials
    study.trials_dataframe().to_csv(os.path.join(summary_dir,'summary_fulllist.csv'), sep=',', index=False, encoding='utf-8')     

    # Make visualizations (see: https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/005_visualization.html)

    fig = plot_optimization_history(study)
    fig.write_image(os.path.join(plot_dir,'optimization_history.png'))

    fig = plot_parallel_coordinate(study)
    fig.write_image(os.path.join(plot_dir,'parallel_coordinate.png'))

    fig = plot_contour(study)
    fig.write_image(os.path.join(plot_dir,'param_contour.png'))

    fig = plot_param_importances(study) # importance_dict = optuna.importance.get_param_importances(study); pprint.pprint(importance_dict)
    fig.write_image(os.path.join(plot_dir,'param_importances.png'))

    fig = plot_param_importances(study, target=lambda t: t.duration.total_seconds(), target_name="duration")
    fig.write_image(os.path.join(plot_dir,'param_importances_totrialduration.png'))

    fig = plot_rank(study)
    fig.write_image(os.path.join(plot_dir,'rank.png'))

    fig = plot_timeline(study)
    fig.write_image(os.path.join(plot_dir,'timeline.png')) 
