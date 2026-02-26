"""Unclassified, with no sensitivities, but not authorized for widespread or public release."""

import os,sys
import numpy as np
from numpy.linalg import norm
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
from openfast_toolbox.io import FASTOutputFile
import shutil

import floris.tools.visualization as wakeviz
from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR

from FFSetup_Seed5 import FFSetup
THIS_FILE_PATH = os.path.join(os.path.dirname(__file__))

##### inputs ##########################

# Floris inputs
FLORIS_path_relative = "/../../../../../CommonFiles/submodules/floris/examples/inputs/gch.yaml"
wind_shear=0.12
wind_veer=20.0
turbine_type=['nrel_5MW']
max_yaw_degrees = 40.0
max_yaw_rate_degrees_per_sec = 0.3 # deg/s, Jonkman et al. (2009)
dt = max_yaw_degrees/max_yaw_rate_degrees_per_sec # s (THIS IS MADE UP FOR FLORIS) for single-step env, make sure that dt is large enough to get to all angles within max_yaw_degrees
num_points_x = 6
num_points_y = 3
spacing_x = 4.0*126
spacing_y = 3.0*126
layout_grid = generate_layout_grid(num_points_x, num_points_y, spacing_x, spacing_y)
num_turbines = len(layout_grid)

# setup for stochastic randomizing of reward
myseeds = [6485]

# setup static biasing of yaw angles
mean_yaw_static = 0 # deg.
std_dev_yaw_static = 7 # deg.
np.random.seed(myseeds[0]) # using first entry of myseeds ensures that the yaw angles will be different between environment numbers 1-3
yaw_angles_static_offset = np.random.normal(mean_yaw_static, std_dev_yaw_static, (1,1,num_turbines)) # static error for yaw angles
np.random.seed(seed=None) # reset the global seed to None

# setup wind conditions
wind_spd_all = [10]
wind_dir_all = [90.0,92.0,94.0,96.0]
turb_int_all = [0.025,0.05,0.075]

# create scenario of turbine event
# num_timesteps_preevent = 0
# num_timesteps_postevent = 1
# num_timesteps = num_timesteps_preevent + num_timesteps_postevent
# indices_to_remove_preevent = []
# indices_to_remove_postevent = []
# first = [indices_to_remove_preevent] * num_timesteps_preevent
# second = [indices_to_remove_postevent] * (num_timesteps_postevent+1) # give a margin since "total_timesteps" is not an exact specification apparently
# indices_to_remove_sequence = first + second

num_timesteps = 1

# set state parameters
scaling = np.append([180, 10, 0.5],[max_yaw_degrees]*num_turbines)
num_obs = len(scaling)

###########################################

class FFProblem:
    def __init__(self,simseed,layout_grid,num_turbines,FFomp):
        self.simseed = simseed
        self.outpath = os.path.join('FFSim_Seed5_RL','Sim_'+str(simseed))
        self.FFomp = FFomp

    def YawTurbines(self,yaw_angles):
        yaw_angles_FF = np.array([[yaw_angles[0][0][::-1]]]) # To take into account the order of power in FF vs RL code
        self.yaw_angles = yaw_angles_FF
    
    def SetInflow(self,wind_dir, wind_spd, turb_int, shear):
        self.wind_dir = wind_dir-90
        self.wind_spd = wind_spd
        self.TI = turb_int*100
        self.shear = shear

    def GetPwr(self):
        self._RunProb()
        allpowers = self._ReadPwr()
        deletetest = np.random.randint(0,100)
        # if deletetest != 2:
        #     self._DeleteSim()
        self._Deletebtsout()
        return allpowers, self.FFSetupSeed # To take into account the order of power in FF vs RL code

    def _DeleteSim(self):
        shutil.rmtree(self.outpath)
    
    def _Deletebtsout(self):
        os.system(f"find {self.outpath} -name '*.bts' -type f -delete ")
        os.system(f"find {self.outpath} -name '*.out' -type f -delete ")
        os.system(f"find {self.outpath} -name '*.outb' -type f -delete ")
        os.system(f"find {self.outpath} -name '*.vtk' -type f -delete ")

    def _RunProb(self):
        self.FFSetupSeed = FFSetup(outpath=self.outpath,wind_spd = self.wind_spd,\
                wind_dir=self.wind_dir,shear=self.shear,TI=self.TI,yaw_angles = self.yaw_angles,FFomp = self.FFomp)

    
    def _ReadPwr(self):
        condpath = os.listdir(self.outpath)[0]
        yawpathall = os.listdir(os.path.join(self.outpath,condpath))
        yawpathall.sort()
        yawpath = yawpathall[0]
        allpowers = [0]*num_turbines
        if num_turbines == 3:
            tinit = 400
        elif num_turbines == 18:
            tinit = 600
        for i in range(num_turbines):
            dfi = FASTOutputFile(os.path.join(self.outpath,condpath,yawpath,'Seed_0',f'FFarm_mod.T{i+1}.outb')).toDataFrame()
            Time = dfi['Time_[s]']
            allpowers[i] = dfi['GenPwr_[kW]'][Time > tinit].mean()
        return allpowers

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
        relative_yaw = yaw_angles_opt[0][0][i] # deg
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

class WindFarmEnv_v0p12(gym.Env):
    """
    Windfarm environment.
    """

    # class attributes start

    obs_scale = np.array(scaling, dtype=np.float32)

    spacing_x=spacing_x
    spacing_y=spacing_y

    episode_step_count = 0 # note: in a parallel environment, this increments only one per every n_envs episodes

    # class attributes end

    def __init__(self, windtype:'DiscreteRandom', initialyaw:'ZeroYaw', algorithm:'SAC',FFomp:False):

        """env objects"""
        self.windtype = windtype
        self.initialyaw = initialyaw
        self.algorithm = algorithm
        self.FFomp = FFomp
        
        # initialize FLORIS
        self.fi = FlorisInterface(THIS_FILE_PATH+FLORIS_path_relative)
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
        # self.fi.reinitialize(wind_directions=[wind_dir], wind_speeds=[wind_spd], turbulence_intensity=turb_int)
        # Set the yaw angles and get the power (original)
        # self.fi.calculate_wake(yaw_angles=self.get_yaw_angles(u))
        # Get the turbine powers
        # turbine_powers = self.fi.get_turbine_powers()/1000.
        
        # Create FF problem
        simseed=random.randint(0, 10000000000)
        self.FFProb = FFProblem(simseed=simseed,layout_grid=layout_grid,num_turbines=num_turbines,FFomp = self.FFomp)
        # Set Inflow for FF
        self.FFProb.SetInflow(wind_dir, wind_spd, turb_int,wind_shear)
        # Set the yaw angles for FF
        yaw_angles=self.get_yaw_angles(u)       
        yaw_angles_actual = np.clip(yaw_angles + yaw_angles_static_offset, a_min=-max_yaw_degrees, a_max=max_yaw_degrees) # add static error to yaw angles
        self.FFProb.YawTurbines(yaw_angles=yaw_angles_actual)
        # Get power from FF
        turbine_powers,FFSeed = self.FFProb.GetPwr()

        return turbine_powers, yaw_angles, yaw_angles_actual, FFSeed, simseed

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
        
        turbine_powers, yaw_angles, yaw_angles_actual, FFSeed, simseed = self.get_turbine_powers(u)
        
        
        rew += ((np.sum(turbine_powers)-self.floris_opt )/(self.floris_opt ))*100
        rew = rew.item() # convert from np.array to float

        rew = rew/num_timesteps
        with open('log.txt','a') as f:
            rewbaseline = ((np.sum(self.baselineseedpower[FFSeed])-self.floris_opt )/(self.floris_opt ))*100
            f.write(f"{simseed},{np.sum(turbine_powers)},{self.floris_opt},{rew},{rewbaseline}, {self.wind.state[0]},{self.wind.state[1]},{self.wind.state[2]},{FFSeed}\n")
            f.flush()

        # tabulate cumulative reward
        self.cum_reward += rew
        
        # info
        wind_dir, wind_spd, turb_int = self.wind.state
        info = {
            "r_cumulative": self.cum_reward,
            "power": turbine_powers,
            "power_floris_opt_stochastic": self.floris_opt,
            "power_floris_opt": self.floris_opt,
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

        # get optimal yaw and power
        if self.windtype=='DiscreteRandom': # (FLORIS results are loaded from "setup" folder based on wind state)
            with open('FFPower_FlorisOptimal_Seed5.pkl','rb') as f:
                df_Fpower_FlorisOptimal=pickle.load(f)
            df_cond = df_Fpower_FlorisOptimal[(df_Fpower_FlorisOptimal['wind_spd'] == self.wind.state[1]) & (df_Fpower_FlorisOptimal['wind_dir'] == self.wind.state[0]) & (df_Fpower_FlorisOptimal['turb_int']==self.wind.state[2])]
            # self.floris_opt_stochastic = np.array(df_cond['totalpower_stochastic'].tolist()[0])
            self.floris_opt = np.array(df_cond['optpower'].tolist()[0])
            self.yaw_angles_opt = np.flip(np.array(df_cond['optyaw'].tolist()[0]))
            self.baselineseedpower = np.array(df_cond['seedpower'].tolist()[0])
    
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

        # with open('FFPower_FlorisOptimal_Seed5.pkl','rb') as f:
        #     df_FFPower_FlorisOptimal=pickle.load(f)
        # df_cond = df_FFPower_FlorisOptimal[(df_FFPower_FlorisOptimal['wind_spd'] == self.wind.state[1]) & (df_FFPower_FlorisOptimal['wind_dir'] == self.wind.state[0]) & (df_FFPower_FlorisOptimal['turb_int']==self.wind.state[2])]
        # self.ffopt_florisyaw = df_cond['optpower']