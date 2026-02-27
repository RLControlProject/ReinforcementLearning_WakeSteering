import os
import numpy as np
import gymnasium as gym
import optuna

from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.evaluation import evaluate_policy

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.best_yaw_angles = []

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        
        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        # Log additional info captured in env.step()
            # see: https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html#directly-accessing-the-summary-writer
        
        # size: (parallel envs, turbines)
        rel_yaw_angles = np.array([info["rel_yaw_angles"] for info in self.locals["infos"]])
        power = np.array([info["power"] for info in self.locals["infos"]])

        for idx in range(rel_yaw_angles.shape[1]):
            self.tb_formatter.writer.add_scalar(f"rollout/rel_yaw_angle{idx}", np.mean(rel_yaw_angles[:,idx]), self.num_timesteps)
        for idx in range(power.shape[1]):
            self.tb_formatter.writer.add_scalar(f"rollout/power{idx}", np.mean(power[:,idx]), self.num_timesteps)

        self.tb_formatter.writer.flush()
            
        # Save network params on best mean training reward
        if self.n_calls % self.check_freq == 0:
            # print("Check")
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps") #"timesteps" "episodes"
            if len(x) > 0:
                # Mean training reward over all episodes
                # print('Best yaw angles: {}'.format(self.best_yaw_angles))
                # print('Last yaw angles: {}'.format(rel_yaw_angles))
                mean_reward = np.mean(y[-self.check_freq:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print(
                        "Previous best mean reward: {:.2f} - Current mean reward per episode: {:.2f}".format(
                            self.best_mean_reward, mean_reward
                        )
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.best_yaw_angles = rel_yaw_angles
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)
                    print(self.save_path)

        return True

class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        callback_after_eval=StopTrainingOnNoModelImprovement,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True
    
class TrialEvalCallback_Variance(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        callback_after_eval=StopTrainingOnNoModelImprovement,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False
        self.rewards = []
        self.last_mean_reward_mod = []

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            # print(type(self))
            # print(self)
            # print(self.n_calls)
            # kab edit #
            self.rewards.append(self.last_mean_reward)
            # print(self.last_mean_reward)
            last_10_rewards = self.rewards[-10:]
            variance = np.var(last_10_rewards)
            # print(self.rewards)
            # print(last_10_rewards)
            # print(self.last_mean_reward)
            # print(np.sqrt(variance))
            self.last_mean_reward_mod = self.last_mean_reward #- np.sqrt(variance)
            ############
            self.trial.report(self.last_mean_reward_mod, self.eval_idx)
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True
    
class EvalCallback(EvalCallback):
    """Callback used for evaluating deterministically."""

    def __init__(
        self,
        eval_env: gym.Env,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.eval_idx = 0

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # super()._on_step()
            # self.eval_idx += 1

            # Evaluate the policy using deterministic actions
            mean_reward, std_reward = evaluate_policy(self.model, self.eval_env, 
                                                       n_eval_episodes=self.n_eval_episodes, 
                                                       deterministic=self.deterministic)
            print(f"Evaluation at step {self.n_calls}: Mean Reward: {mean_reward}, Std Reward: {std_reward}")
        return True
    
class CumulativeRewardCallback(BaseCallback):
    def __init__(self):
        super(CumulativeRewardCallback, self).__init__()

    def _on_step(self) -> bool:
        self.cumulative_rewards = []
        # Access the info dictionary
        for info in self.locals['infos']:
            if 'r_cumulative' in info:
                self.cumulative_rewards.append(info['r_cumulative']/self.n_calls)
        return True