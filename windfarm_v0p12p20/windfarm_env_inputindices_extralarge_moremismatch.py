"""Unclassified, with no sensitivities, but not authorized for widespread or public release."""

import os,sys
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Box
sys.path.insert(0, '../submodules/floris')
from floris.tools import FlorisInterface
import pickle
try:
    from .helpers import generate_layout_grid
except:
    from helpers import generate_layout_grid
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random

import floris.tools.visualization as wakeviz
from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR

sys.path.insert(0, 'setup')
from FSetup import FSetup

THIS_FILE_PATH = os.path.join(os.path.dirname(__file__))

##### inputs ##########################
tag = '_extralarge_moremismatch'

# Floris inputs
FLORIS_path_absolute = "/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/submodules/floris/examples/inputs/gch.yaml"
wind_shear=0.12
wind_veer=20.0
turbine_type=['nrel_5MW']
max_yaw_degrees = 40.0
max_yaw_rate_degrees_per_sec = 0.3 # deg/s, Jonkman et al. (2009)
dt = max_yaw_degrees/max_yaw_rate_degrees_per_sec # s (THIS IS MADE UP FOR FLORIS) for single-step env, make sure that dt is large enough to get to all angles within max_yaw_degrees
num_points_x = 8
num_points_y = 4
spacing_x = 4.0*126
spacing_y = 3.0*126
layout_grid = generate_layout_grid(num_points_x, num_points_y, spacing_x, spacing_y)
num_turbines = len(layout_grid)

# setup for stochastic randomizing of reward
mean_WS = 0.0 # m/s
std_dev_WS = 0.0 # m/s
mean_TI = 0.0 # -
std_dev_TI = 0.0 # -
mean_WD = 0.0 # deg.
std_dev_WD = 0.0 # deg.
myseeds = [
        18643,
        64444,
        28340,
        98448,
        79976,
        51666,
        31858,
        51868,
        66333,
        6834,
    ]

# setup static biasing of yaw angles
mean_yaw_static = 0 # deg.
std_dev_yaw_static = 0 # deg.
np.random.seed(myseeds[0]) # using first entry of myseeds ensures that the yaw angles will be different between environment numbers 1-3
yaw_angles_static_offset = np.random.normal(mean_yaw_static, std_dev_yaw_static, (1,1,num_turbines)) # static error for yaw angles
np.random.seed(seed=None) # reset the global seed to None

# setup wind conditions
wind_spd_all = [10]
wind_dir_all = [90.0,92.0,94.0,96.0]
turb_int_all = [0.025,0.05,0.075]

# set baseline folder
baseline_folder = 'setup'
baseline_name = 'BaselinePerformance'+tag

# create scenario of turbine event
num_timesteps_preevent = 0
num_timesteps_postevent = 1
num_timesteps = num_timesteps_preevent + num_timesteps_postevent
indices_to_remove_preevent = []
indices_to_remove_postevent = []
first = [indices_to_remove_preevent] * num_timesteps_preevent
second = [indices_to_remove_postevent] * (num_timesteps_postevent+1) # give a margin since "total_timesteps" is not an exact specification apparently
indices_to_remove_sequence = first + second

# set state parameters
scaling = np.append([180, 10, 0.5],[max_yaw_degrees]*num_turbines)
num_obs = len(scaling)
###########################################

def ic_wind_fcn(wind):
    if wind=='DiscreteRandom':
        wind_dir = random.choice(wind_dir_all)
        wind_spd = random.choice(wind_spd_all)
        turb_int = random.choice(turb_int_all)
    elif wind=='ContinuousRandom':
        wind_dir = np.random.uniform(low=wind_dir_all[0], high=wind_dir_all[-1])
        wind_spd = np.random.uniform(low=wind_spd_all[0], high=wind_spd_all[-1])
        turb_int = np.random.uniform(low=turb_int_all[0], high=turb_int_all[-1])
    return [wind_dir, wind_spd, turb_int]

def ic_turb_fcn(initialyaw,yaw_angles_opt,i):
    if initialyaw=='ZeroYaw':
        relative_yaw = 0.0 # degree
    elif initialyaw=='FLORISYaw':
        relative_yaw = yaw_angles_opt[i] # deg
    return [relative_yaw]

def wrap_to_180(angle):
    # return (angle + np.pi) % (2*np.pi) - np.pi
    return (angle + 180) % (2*180) - 180

class Windfield:
    def __init__(self) -> None:
        # state is wind direction and speed and TI
        self._state=np.array([0.,0.,0.])
    def reset(self,windtype):
        if windtype is None:
            ic = [None, None, None]
            ic[0] = np.random.uniform(low=0., high=360.) #winddirection
            ic[1] = np.random.uniform(low=0., high=10.) #windspeed
            ic[2] = np.random.uniform(low=0., high=0.5) #turbint
        else: 
            ic = ic_wind_fcn(windtype)
        self._state=np.array(ic)
        return self._state.copy()
    def step(self):
        return self.state
    @property
    def state(self):
        return self._state.copy()

class Turbine:
    def __init__(self) -> None:
        # state is local to turbine
        self._state=np.array([0.])
        self.max_yaw_degrees=max_yaw_degrees
        self.max_yaw_rate_degrees_per_sec=max_yaw_rate_degrees_per_sec
    def reset(self,initialyaw,yaw_angles_opt,i):
        ic = ic_turb_fcn(initialyaw,yaw_angles_opt,i)
        self._state=np.array(ic)
        return self._state.copy()
    def step(self, u):
        # print('u is: '+str(u))
        self._state = np.clip(self._state + u*dt*self.max_yaw_rate_degrees_per_sec, a_min=-self.max_yaw_degrees, a_max=self.max_yaw_degrees)
        return self.state
    @property
    def state(self):
        return self._state.copy()

class WindFarmEnv_v1p3(gym.Env):
    """
    Windfarm environment.
    """

    # class attributes start

    obs_scale = np.array(scaling, dtype=np.float32)

    spacing_x=spacing_x
    spacing_y=spacing_y

    episode_step_count = 0 # note: in a parallel environment, this increments only one per every n_envs episodes

    # class attributes end

    def __init__(self, windtype:'DiscreteRandom', initialyaw:'ZeroYaw', algorithm:'SAC', rewardtype:'base'):

        """env objects"""
        self.windtype = windtype
        self.initialyaw = initialyaw
        self.algorithm = algorithm
        self.rewardtype = rewardtype
        
        # initialize FLORIS
        self.fi = FlorisInterface(FLORIS_path_absolute)
        self.fi.reinitialize(layout_x=[point[0] for point in layout_grid], layout_y=[point[1] for point in layout_grid], turbine_type=turbine_type)
        self.fi.reinitialize(wind_shear=wind_shear, wind_veer=wind_veer)
        
        """env objects"""
        self.wind = Windfield()
        self.turbines = []
        for i in range(num_turbines):
            self.turbines.append(Turbine())

        obs, _ = self.reset()

        #Defining obs and action spaces as gym.spaces objects
        state_space = Box(
            low=np.array([-np.inf]*(num_obs)), # these limits can always just be +/- inf unless there are convulational images involved (https://github.com/hill-a/stable-baselines/issues/698#issuecomment-589155204)
            high=np.array([np.inf]*(num_obs)), # these limits can always just be +/- inf unless there are convulational images involved (https://github.com/hill-a/stable-baselines/issues/698#issuecomment-589155204)
            shape=(num_obs,),
            dtype=np.float64
            )
        
        if self.algorithm=='SAC' or self.algorithm=='TD3':
            input_space = Box(
                low=np.array([-1.0]*len(self.turbines)),
                high=np.array([1.0]*len(self.turbines)),
                shape=(len(self.turbines),), # assume control input to each turbine is (normalized) yaw angle, measured relative to wind direction
                dtype=np.float64
                )
        elif self.algorithm=='DQN':
            input_space_length = 3
            input_space_choices = 3 # -1, 0, 1
            input_space = gym.spaces.Discrete(input_space_length*input_space_choices)  # 3^3 = 27 possible combinations

        ### single global measurement
        self.observation_space = state_space
        self.action_space = input_space

        # other attributes
        self.num_inputs=self.observation_space.shape[0]
        if self.algorithm=='SAC' or self.algorithm=='TD3':
            self.num_outputs=self.action_space.shape[0]
        elif self.algorithm=='DQN':
            self.num_outputs=input_space_length

        self.viewer = None
        self.rng = np.random.default_rng()
        self.cum_reward = 0
        self.step_count = 0

    def get_yaw_angles(self, u:np.ndarray):
        # Set the yaw angles
        yaw_angles = np.ones([1,1,len(self.turbines)]) # 1 wind direction, 1 wind speed, n turbines
        for i in range(len(u)):
            yaw_angles[:,:,i] *= np.clip(self.turbines[i].state, a_min=-self.turbines[i].max_yaw_degrees, a_max=self.turbines[i].max_yaw_degrees)
        return yaw_angles

    def get_turbine_powers(self, u:np.ndarray):
        # Assuming 1 wind speed and 1 wind direction
        wind_dir, wind_spd, turb_int = self.wind.state
        self.fi.reinitialize(wind_directions=[wind_dir], wind_speeds=[wind_spd], turbulence_intensity=turb_int)

        # # KAB added: disable specified turbines, if applicable
        # if os.path.exists('globalvars_indices_to_remove_sequence.pkl'):
        #     # import global variables
        #     file = open('globalvars_layout_grid.pkl', 'rb')
        #     layout_grid = pickle.load(file)
        #     file.close()
        #     file = open('globalvars_indices_to_remove_sequence.pkl', 'rb')
        #     indices_to_remove_sequence = pickle.load(file)
        #     file.close()
        #     # update environment
        #     layout_grid_new = [item for i, item in enumerate(layout_grid) if i not in indices_to_remove_sequence[self.step_count]]
        #     self.fi.reinitialize(layout_x=[point[0] for point in layout_grid_new], layout_y=[point[1] for point in layout_grid_new])
        #     u = np.delete(u, np.array(indices_to_remove_sequence[self.step_count], dtype=int))

        # # Set the yaw angles and get the power (original)
        # self.fi.calculate_wake(yaw_angles=self.get_yaw_angles(u))
        # turbine_powers = self.fi.get_turbine_powers()/1000.
        
        # Set the yaw angles and get the power
        type = 'production'
        outpath = os.path.join(baseline_folder,baseline_name)
        randseedidx = self.seed_index
        yaw_angles=self.get_yaw_angles(u)       
        yaw_angles_actual = np.clip(yaw_angles + yaw_angles_static_offset, a_min=-max_yaw_degrees, a_max=max_yaw_degrees) # add static error to yaw angles
        turbine_powers = FSetup(type, self.fi, outpath, wind_spd, wind_dir, turb_int, wind_shear, wind_veer, yaw_angles, yaw_angles_actual, mean_WS, std_dev_WS, mean_TI, std_dev_TI, mean_WD, std_dev_WD, myseeds, randseedidx)
        
        # # KAB added: reenable the disabled turbines, if applicable
        # if os.path.exists('globalvars_indices_to_remove_sequence.pkl'):
        #     # return environment to original
        #     self.fi.reinitialize(layout_x=[point[0] for point in layout_grid], layout_y=[point[1] for point in layout_grid])
        #     # add back in removed indices to keep vector size consistent
        #     turbine_powers_reconstructed = np.zeros(len(layout_grid), dtype=u.dtype)
        #     insert_index = 0
        #     for k in range(len(turbine_powers_reconstructed)):
        #         if k not in indices_to_remove_sequence[self.step_count]:
        #             turbine_powers_reconstructed[k] = turbine_powers[0][0][insert_index]
        #             turbine_powers_reconstructed[k] = turbine_powers[0][0][insert_index]
        #             insert_index += 1
        #     turbine_powers = turbine_powers_reconstructed

        return turbine_powers, yaw_angles, yaw_angles_actual

    def step(self, u:np.ndarray)->tuple[np.ndarray, float, bool, dict]:
        self.episode_step_count += 1
        self.step_count += 1
        rew = 0

        if self.algorithm=='DQN':
            action_mapping = [-1, 0, 1]
            u = [action_mapping[(u // (3 ** i)) % 3] for i in range(3)]

        # # check that turbines are initialized correctly
        # for i, turb in enumerate(self.turbines):
        #     print(turb.state)

        # update
        for i, action in enumerate(u):
            _ = self.turbines[i].step(action)
        _ = self.wind.step()
        obs = self.obs()

        turbine_powers, yaw_angles, yaw_angles_actual = self.get_turbine_powers(u)

        # KAB added: prepare for reward shaping by calculating FLORIS optimum
        # generate and store floris solution
        # if (self.step_count-1==0) or (self.step_count-1==num_timesteps_preevent): # if it's the first time or a change in the environment, find the FLORIS-optimal solution
            # if os.path.exists('globalvars_indices_to_remove_sequence.pkl'):
            #     # import global variables
            #     file = open('globalvars_layout_grid.pkl', 'rb')
            #     layout_grid = pickle.load(file)
            #     file.close()
            #     file = open('globalvars_indices_to_remove_sequence.pkl', 'rb')
            #     indices_to_remove_sequence = pickle.load(file)
            #     file.close()
            #     # update environment
            #     layout_grid_new = [item for i, item in enumerate(layout_grid) if i not in indices_to_remove_sequence[self.step_count]]
            #     self.fi.reinitialize(layout_x=[point[0] for point in layout_grid_new], layout_y=[point[1] for point in layout_grid_new])
            # # do yaw optimization
            # yaw_opt = YawOptimizationSR(self.fi, minimum_yaw_angle=-self.turbines[0].max_yaw_degrees, maximum_yaw_angle=self.turbines[0].max_yaw_degrees, Ny_passes=[9,8])#, exploit_layout_symmetry=False)
            # df_opt = yaw_opt.optimize()
            # print("FLORIS optimization results:")
            # print(df_opt)
            # self.floris_opt = df_opt['farm_power_opt'].values/1000
            # print('self.floris is: '+str(self.floris_opt))
            # return environment to original
            # self.fi.reinitialize(layout_x=[point[0] for point in layout_grid], layout_y=[point[1] for point in layout_grid])
            # save for later
            # with open('globalvars_floris_opt.pkl', 'wb') as file:
            #     pickle.dump(floris_opt,file)
        # else:
        #     file = open('globalvars_floris_opt.pkl', 'rb')
        #     floris_opt = pickle.load(file)
        #     file.close()

        # do reward regularization/shaping
        reward = ((np.sum(turbine_powers)-np.sum(self.floris_opt))/np.sum(self.floris_opt))*100
        reward = reward.item() # convert from np.array to float
        # delta = -0.4
        # if rew>delta:
        #     rew = 1
        #     print('REWARD')
        # else:
        #     rew = -1
        # u = u**2
        # rew += np.sum(u)
        # print(rew)

        # calculate reward relative to actual environment (this is unknown to an actual agent but useful for evaluating agents)
        reward_stochastic = ((np.sum(turbine_powers)-np.sum(self.floris_opt_stochastic))/np.sum(self.floris_opt_stochastic))*100
        reward_stochastic = reward_stochastic.item() # convert from np.array to float
        # print(rew_stochastic)

        # calculate reward relative to a posteriori knowledge of wind conditions
        reward_stochastic_1seed = ((np.sum(turbine_powers)-np.sum(self.floris_opt_stochastic_1seed))/np.sum(self.floris_opt_stochastic_1seed))*100
        reward_stochastic_1seed = reward_stochastic_1seed.item() # convert from np.array to float
        # print(rew_stochastic_1seed)

        if self.rewardtype=='base':
            rew += reward
        elif self.rewardtype=='stochastic':
            rew += reward_stochastic # this is how it's done in FastFarm right now
        elif self.rewardtype=='stochastic_1seed':
            rew += reward_stochastic_1seed # enables regularization with posteriori knowledge of wind conditions

        # ad-hoc scaling to make the reported reward for the whole episode equal the average of all timesteps (this seems to be related to issue described here: (https://github.com/hill-a/stable-baselines/issues/236)
        rew = rew/num_timesteps

        # tabulate cumulative reward
        self.cum_reward += rew
        
        # info
        wind_dir, wind_spd, turb_int = self.wind.state
        info = {
            "r": reward,
            "r_stochastic": reward_stochastic,
            "r_stochastic_1seed": reward_stochastic_1seed,
            "power": turbine_powers.squeeze(),
            "power_floris_opt": self.floris_opt,
            "power_floris_opt_stochastic": self.floris_opt_stochastic,
            "power_floris_opt_stochastic_1seed": self.floris_opt_stochastic_1seed,
            "yaw_angles_actual": (wind_dir-yaw_angles_actual).squeeze(),
            "yaw_angles": (wind_dir-yaw_angles).squeeze(),
            "yaw_angles_floris_opt": (wind_dir-self.yaw_angles_opt).squeeze(),
            "rel_yaw_angles_actual": yaw_angles_actual.squeeze(),
            "rel_yaw_angles": yaw_angles.squeeze(),
            "rel_yaw_angles_floris_opt": self.yaw_angles_opt.squeeze(),
            "wind_dir": wind_dir,
            "wind_spd": wind_spd,
            "turb_int": turb_int,
            }
        
        # done flag
        done = 0
        if self.step_count==num_timesteps:
            done = 1

        # obs, reward, done, truncated, info
        return obs, rew, done, False, info  #rew/1e3/len(self.turbines) 5/17/2024
        
    
    def obs(self):
        wind_dir, wind_spd, turb_int = self.wind.state
        #wind_dir = wrap_to_180(wind_dir)
        turbines_state = []
        for i in range(len(self.turbines)):
            turbines_state = np.append(turbines_state, self.turbines[i].state)
        return np.append(np.array([wind_dir, wind_spd, turb_int]), turbines_state) / self.obs_scale

    def reset(self, seed:int=None):

        # reset wind condition
        self.wind.reset(self.windtype)

        # reset seed number
        self.seed_index = [np.random.randint(len(myseeds))]

        # get optimal yaw and power
        if self.windtype=='DiscreteRandom': # (FLORIS results are loaded from "setup" folder based on wind state)
            # averaged over seeds
            with open(os.path.join('setup',baseline_name,'Fpower_FlorisOptimal.pkl'),'rb') as f:
                df_Fpower_FlorisOptimal=pickle.load(f)
            df_cond = df_Fpower_FlorisOptimal[(df_Fpower_FlorisOptimal['wind_spd'] == self.wind.state[1]) & (df_Fpower_FlorisOptimal['wind_dir'] == self.wind.state[0]) & (df_Fpower_FlorisOptimal['turb_int']==self.wind.state[2])]
            self.floris_opt_stochastic = np.array(df_cond['totalpower_stochastic'].tolist()[0])
            self.floris_opt = np.array(df_cond['totalpower'].tolist()[0])
            self.yaw_angles_opt = np.array(df_cond['yaw_angles_opt'].tolist()[0])
            # single-seed only
            seedPath = os.path.join(f'Cond00_v{self.wind.state[1]:04.1f}_PL{wind_shear:03.2f}_TI{self.wind.state[2]:3.3f}',f'Case0_wdirp{self.wind.state[0]:02.0f}',f'Seed_{self.seed_index[0]}')
            with open(os.path.join('setup',baseline_name,seedPath,'Fpower_FlorisOptimal.pkl'),'rb') as f:
                _,floris_opt_power_stochastic,_ = pickle.load(f)
            self.floris_opt_stochastic_1seed = np.array(floris_opt_power_stochastic)

        # elif self.windtype=='ContinuousRandom': # (FLORIS (re)run for every reset)
        #     state = self.wind._state
        #     self.fi.reinitialize(wind_directions=[state[0]],wind_speeds=[state[1]], turbulence_intensity=state[2])

        #     # do yaw optimization for this wind condition
        #     yaw_opt = YawOptimizationSR(self.fi, minimum_yaw_angle=-self.turbines[0].max_yaw_degrees, maximum_yaw_angle=self.turbines[0].max_yaw_degrees, Ny_passes=[9,8])#, exploit_layout_symmetry=False)
        #     df_opt = yaw_opt.optimize(print_progress=False)
        #     self.floris_opt_stochastic = np.nan # there is no stochasticism in this case, so this is meaningless (i.e., ) self.floris_opt is already the true answer
        #     self.floris_opt = df_opt['farm_power_opt'].values/1000
        #     self.yaw_angles_opt = df_opt.yaw_angles_opt.values[0]
        #     # print("FLORIS optimization results:")
        #     # print(df_opt)

        #     # # calculate approximate initial yaw angles (this is different from the optimal yaw since we want to introduce some model mismatch)
        #     # spacing_x = self.spacing_x+1.0*126
        #     # spacing_y = 3.0*126
        #     # layout_grid_new = generate_layout_grid(num_points_x, num_points_y, spacing_x, spacing_y)
        #     # self.fi.reinitialize(layout_x=[point[0] for point in layout_grid_new], layout_y=[point[1] for point in layout_grid_new], turbine_type=turbine_type)
        #     # yaw_opt = YawOptimizationSR(self.fi, minimum_yaw_angle=-self.turbines[0].max_yaw_degrees, maximum_yaw_angle=self.turbines[0].max_yaw_degrees, Ny_passes=[9,8])#, exploit_layout_symmetry=False)
        #     # df_opt = yaw_opt.optimize(print_progress=False)
        #     # self.yaw_angles_opt = df_opt.yaw_angles_opt.values[0]
        #     # self.fi.reinitialize(layout_x=[point[0] for point in layout_grid], layout_y=[point[1] for point in layout_grid], turbine_type=turbine_type)
        #     # # print("Initial yaw angles (calculated from mismatched model):")
        #     # # print(df_opt)
    
        # reset turbine yaw angles
        for i, turb in enumerate(self.turbines):
            turb.reset(self.initialyaw,self.yaw_angles_opt,i)

        # self.cum_reward = 0
        self.step_count = 0

        return self.obs(), {}
                
    def render(self):
        pass


if __name__=='__main__':
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TKAgg')

    
    # """test step"""
    # u=np.array([0., -0.25, -0.50])
    # obs, rew, done, truncated, info = env.step(u)
    # print(obs, rew, done, info)


    # """wrap_to_180 test"""
    # print(f"Expected wrap to 180: 10  |  actual: {wrap_to_180(10)}")
    # print(f"Expected wrap to 180: 170  |  actual: {wrap_to_180(170)}")
    # print(f"Expected wrap to 180: -175  |  actual: {wrap_to_180(185)}")
    # print(f"Expected wrap to 180: -10  |  actual: {wrap_to_180(350)}")