"""Unclassified, with no sensitivities, but not authorized for widespread or public release."""

"""how to run:
cd /ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p15
conda activate RLControlEnv
python3 sac_post_generate_sequence_5_10env_large_1.py
"""

import gymnasium as gym
import numpy as np
import os, sys
from torch import nn
import torch
import matplotlib
import matplotlib.pyplot as plt
import helpers
import itertools
import pandas as pd

sys.path.insert(0, '../submodules/floris')
from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR

try:
    matplotlib.use('TKAgg')
except:
    pass

from stable_baselines3 import SAC, TD3, DDPG
from stable_baselines3.sac.policies import MlpPolicy  as SACMlpPolicy
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv,VecEnvWrapper
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import configure_logger
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.prioritized_replay_buffer import PERDQN

from utils import transferWeightsAndBiases, transferWeightsAndBiases_td3, transferWeightsAndBiases_sac
#from policy_distillation.model import FFNN,FFNN_LastLayerTanh

from callbacks import SaveOnBestTrainingRewardCallback, TrialEvalCallback, EvalCallback
from windfarmenv_seed5 import WindFarmEnv_v0p12, num_timesteps, num_turbines

import pickle

if __name__ == "__main__":

    # set up log dir
    if num_turbines == 3:
        tb_log_name = 'trial_3_10env_1'
    elif num_turbines == 18:
        tb_log_name = 'trial_3_10env_large_1'
    log_dir = os.path.join(os.path.dirname(__file__))+"/sac_output/trials"
    if not os.path.exists(os.path.join(log_dir,tb_log_name+'_1')):
        os.makedirs(os.path.join(log_dir,tb_log_name+'_1'))
    configure_logger(verbose=1)

    # setup the environment
    env_kwargs = {}
    env_kwargs['initialyaw'] = 'ZeroYaw' # 'ZeroYaw' 'FLORISYaw'
    env_kwargs['algorithm'] = 'SAC' # 'SAC'
    env_kwargs['FFomp'] = True

    # setup run options
    trainModel = 1
    testModel = 0 # currently not working for new setup
    testModel_generateData = 0 # currently not working for new setup

    if trainModel:

        # set run-specific environment args
        env_kwargs['windtype'] = 'DiscreteRandom' # 'DiscreteRandom' 'ContinuousRandom'
        env_kwargs['rewardtype'] = 'stochastic_1seed' # 'DiscreteRandom' 'ContinuousRandom'

        # make the environment

        n_envs = 10
        venv0 = make_vec_env(WindFarmEnv_v0p12,vec_env_cls=SubprocVecEnv, n_envs=n_envs, seed=1, env_kwargs=env_kwargs)
        venv = VecMonitor(
            venv=venv0,
            filename=os.path.join(log_dir, tb_log_name + "_1"),
            info_keywords=(
                [
                    "r",
                    "r_stochastic",
                    "r_stochastic_1seed",
                    "power",
                    "power_floris_opt",
                    "power_floris_opt_stochastic",
                    "power_floris_opt_stochastic_1seed",
                    "yaw_angles_actual",
                    "yaw_angles",
                    "yaw_angles_floris_opt",
                    "rel_yaw_angles_actual",
                    "rel_yaw_angles",
                    "rel_yaw_angles_floris_opt",
                    "wind_dir",
                    "wind_spd",
                    "turb_int",
                    "simseed",
                    "DEL",
                ]
            ),
        )

        # make action noise object
        n_actions = venv.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0 * np.ones(n_actions))

        # read best hyperparameters
        print('Importing best hyperparameters:')
        log_dir_optuna = os.path.join(os.path.dirname(__file__))+"/sac_optuna_search_sequence/trials"
        summary_dir = os.path.join(log_dir_optuna,tb_log_name+'_1','summary')
        summary = pd.read_csv(os.path.join(summary_dir,'summary_best.csv'), sep=',', encoding='utf-8') 
        print(summary)
        
        # make model
        load_pretrained_model = False
        if not load_pretrained_model:
            policy_kwargs = dict(net_arch=[256, 256])
            model = SAC(
                SACMlpPolicy, 
                venv,
                policy_kwargs=policy_kwargs,
                learning_rate=summary['learning_rate'].iloc[0],
                learning_starts = int(summary['learning_starts'].iloc[0]),
                train_freq = 1,
                action_noise = action_noise,
                ent_coef = 'auto_'+str(summary['ent_coef'].iloc[0]),
                target_entropy=-summary['target_entropy_multiplier'].iloc[0]*num_turbines,
                batch_size = 25000,
                use_sde = False,
                gamma = 0,
                gradient_steps=-1, # -1 sets it to perform as many graident steps as transitions collected, which enables multi-processing to be effectiv in reducing wall-clock time (see https://github.com/DLR-RM/stable-baselines3/blob/master/docs/guide/examples.rst) 
                # replay_buffer_class=PrioritizedReplayBuffer,
                verbose=0,
                tensorboard_log=log_dir,
            )
        else:
            pass
            # if num_turbines == 3:
            #     subdirectory = "sac_post_generate_sequence/trials/trial_4_10env_moremismatch_1"
            # elif num_turbines == 18:
            #     subdirectory = "sac_post_generate_sequence/trials/trial_4_10env_large_moremismatch_1"
            # modelname = "best_model.zip"
            # path = os.path.join(subdirectory,modelname)
            # model = SAC.load(path=path,env=venv,#policy_kwargs=policy_kwargs,
            #     batch_size=25000, gamma=0, action_noise=action_noise, # tau = tau
            #     ent_coef='auto_'+str(summary['ent_coef'].iloc[0]), target_entropy=-summary['target_entropy_multiplier'].iloc[0]*num_turbines,
            #     learning_rate=summary['learning_rate'].iloc[0], 
            #     learning_starts = 0, train_freq = int(summary['train_freq'].iloc[0]), #buffer_size = buffer_size,
            #     #replay_buffer_class=HerReplayBuffer,replay_buffer_kwargs=dict(n_sampled_goal=n_sampled_goal,goal_selection_strategy=goal_selection_strategy),
            #     use_sde = False, gradient_steps=-1,verbose=0,device="auto",tensorboard_log=log_dir) # stats_window_size=stats_window_size,
                        
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

        check_freq = 1
        save_callback = SaveOnBestTrainingRewardCallback(check_freq=check_freq, log_dir=os.path.join(log_dir,tb_log_name+'_1'))

        eval_episodes = n_envs # every eval causes step counter to increment by one
        eval_freq = 1000 # times n_env episodes (make unreachably large for now to disable)
        eval_callback = EvalCallback(venv, n_eval_episodes=eval_episodes, eval_freq=eval_freq, deterministic=True) # , callback_after_eval=stop_callback, best_model_save_path=log_dir, log_path=log_dir

        # Train the agent

        # obs, _ = env.reset()
        # action = model.policy.predict(obs,deterministic = True)[0]
        # action_scale = np.array([turb.max_yaw_degrees for turb in env.turbines])
        # print(action*action_scale)
    
        if num_turbines == 3:
            num_episodes = 1000 # approximate
        elif num_turbines == 18:
            num_episodes = 2000 # approximate
        model.learn(total_timesteps=num_timesteps*num_episodes,reset_num_timesteps=True,tb_log_name=os.path.join(log_dir,tb_log_name),callback=[save_callback,eval_callback])

        # obs, _ = env.reset()
        # action = model.policy.predict(obs,deterministic = True)[0]
        # action_scale = np.array([turb.max_yaw_degrees for turb in env.turbines])
        # print(action*action_scale)

        # save the model, replay buffer, and policy
        # savename = "final_model"
        # savepath_base = os.path.join(log_dir,tb_log_name,savename)
        # model.save(savepath_base)
        # model.save_replay_buffer(savepath_base+"_replay_buffer")
        # model.policy.save(savepath_base+"_policy.pkl")


    if testModel: # evaluate the policy (source: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/advanced_saving_loading.ipynb#scrollTo=T65Bo7-k3dWL)

        if testModel_generateData:
            # set run-specific environment args
            env_kwargs['windtype'] = 'DiscreteRandom'

            # remake the environment
            n_envs = 10
            venv0 = make_vec_env(WindFarmEnv_v0p12,vec_env_cls=SubprocVecEnv, n_envs=n_envs, seed=0, env_kwargs=env_kwargs)
            venv = VecMonitor(
                venv=venv0,
                filename=os.path.join(log_dir, tb_log_name + "_1"),
                info_keywords=(
                    [
                        "r",
                        "r_stochastic",
                        "r_stochastic_1seed",
                        "power",
                        "power_floris_opt",
                        "power_floris_opt_stochastic",
                        "power_floris_opt_stochastic_1seed",
                        "yaw_angles_actual",
                        "yaw_angles",
                        "yaw_angles_floris_opt",
                        "rel_yaw_angles_actual",
                        "rel_yaw_angles",
                        "rel_yaw_angles_floris_opt",
                        "wind_dir",
                        "wind_spd",
                        "turb_int",
                        "simseed",
                        "DEL",
                    ]
                ),
            )

            ## load the best model
            model = SAC.load(os.path.join(log_dir,tb_log_name,'best_model.zip'))

            ## test
            n_trials = 100
            rewards = []
            for i in range(n_trials):
                reward, _ = evaluate_policy(model.policy, venv, n_eval_episodes=1, deterministic=True)
                rewards.append(reward)
            mean_rewards = np.mean(rewards)
            std_rewards = np.std(rewards)
            print(mean_rewards)
            print(std_rewards)

            with open(os.path.join(log_dir,tb_log_name,'best_model_testing.pkl'), 'wb') as file:
                pickle.dump(rewards, file)

        else:
            with open(os.path.join(log_dir,tb_log_name,'best_model_testing.pkl'), 'rb') as file:
                rewards = pickle.load(file)

        # plot
        fig, axs = plt.subplots(1,figsize=(7,7), sharex=True)
        plt.plot(range(len(rewards)), rewards, color = 'k', marker = 'o', linestyle = None)
        plt.xlabel('Test Number', fontsize = 16)
        plt.ylabel('Error w.r.t. FLORIS optimal [%]', fontsize = 16)
        # plt.xlim([0, 25])
        #plt.ylim([0, 180])
        # plt.legend()
        helpers.saveplotsingle(fig, axs, os.path.join(log_dir,tb_log_name,'best_model_testing_sequence.png'), 1.25, 22)

        fig, axs = plt.subplots(1,figsize=(7,7), sharex=True)
        start = -1.25; stop = 1.25; increment = 0.05
        bin_values = np.arange(start, stop + increment, increment)
        plt.hist(rewards, bins=bin_values, edgecolor='black')
        median = np.median(rewards)
        plt.axvline(median, color='green', linestyle='--', label='Median: '+f"{median:.3f}")
        plt.xlabel('Error w.r.t. FLORIS optimal [%]', fontsize = 16)
        plt.ylabel('Frequency', fontsize = 16)
        # plt.xlim([0, 25])
        #plt.ylim([0, 180])
        plt.legend()
        helpers.saveplotsingle(fig, axs, os.path.join(log_dir,tb_log_name,'best_model_testing_histogram.png'), 1.25, 22)

