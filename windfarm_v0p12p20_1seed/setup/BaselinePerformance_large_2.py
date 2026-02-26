"""
conda activate RLControlEnv
cd /ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_1seed/setup/
python3 BaselinePerformance_large_*.py
"""

import os, sys
import numpy as np
import pandas as pd
from FSetup import FSetup
import multiprocessing as mp
sys.path.insert(0, '../../submodules/floris')
from floris.tools import FlorisInterface
from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR
import pickle as pkl

sys.path.insert(0, '../')
from windfarm_env_inputindices_large_2 import tag, layout_grid, turbine_type, max_yaw_degrees, wind_spd_all,wind_dir_all,turb_int_all, wind_shear, wind_veer, mean_WS, std_dev_WS, mean_TI, std_dev_TI, mean_WD, std_dev_WD, myseeds, yaw_angles_static_offset, FLORIS_path_absolute

THIS_FILE_PATH = os.path.join(os.path.dirname(__file__))
savefolder = 'BaselinePerformance'+tag
outpath = os.path.join(THIS_FILE_PATH,savefolder)

def main():

    runsim = True
    if runsim:
        # AllProcs = []
        for wind_spd  in wind_spd_all:
            for wind_dir in wind_dir_all:
                for turb_int in turb_int_all:

                    fi = FlorisInterface(FLORIS_path_absolute)

                    # calculate approximate initial yaw angles (this is different from the true optimal yaw since we want to introduce some model mismatch)
                    fi.reinitialize(layout_x=[point[0] for point in layout_grid], layout_y=[point[1] for point in layout_grid], turbine_type=turbine_type)
                    fi.reinitialize(wind_directions=[wind_dir],wind_speeds=[wind_spd], turbulence_intensity=turb_int,wind_shear=wind_shear,wind_veer=wind_veer)
                    yaw_opt = YawOptimizationSR(fi, minimum_yaw_angle=-1*max_yaw_degrees, maximum_yaw_angle=max_yaw_degrees, Ny_passes=[9,8])#, exploit_layout_symmetry=False)
                    df_opt = yaw_opt.optimize()
                    # floris_opt_power = df_opt['farm_power_opt'].values/1000
                    yaw_angles_opt = [[df_opt.yaw_angles_opt.values[0]]]
                    # print(yaw_angles_opt)

                    type = 'baseline'
                    randseedidx = range(len(myseeds))
                    yaw_angles_opt_actual = np.clip(yaw_angles_opt + yaw_angles_static_offset, a_min=-max_yaw_degrees, a_max=max_yaw_degrees) # add static error to yaw angles
                    FSetup(type, fi, outpath, wind_spd, wind_dir, turb_int, wind_shear, wind_veer, yaw_angles_opt, yaw_angles_opt_actual, mean_WS, std_dev_WS, mean_TI, std_dev_TI, mean_WD, std_dev_WD, myseeds, randseedidx)
                    # P = mp.Process(target=FSetup,kwargs={'fi':fi,'outpath':outpath,'wind_spd':wind_spd,'wind_dir':wind_dir,'turb_int':turb_int,'wind_shear':wind_shear,'wind_veer':wind_veer,'yaw_angles_opt':yaw_angles_opt, 'mean':mean, 'std_dev':std_dev})
                    # AllProcs.append(P)
                    # P.start()

        # for P in AllProcs:
        #     P.join()

    nSeeds = len(myseeds)

    df_Fpower_FlorisOptimal = pd.DataFrame(columns = ["wind_spd", "wind_dir", "turb_int", "yaw_angles_opt", "totalpower_stochastic", "totalpower"])

    for wind_spd  in wind_spd_all:
        for wind_dir in wind_dir_all:
            for turb_int in turb_int_all:
                
                totalpower = 0
                totalpower_stochastic = 0
                for seed in range(nSeeds):
                    seedPath = os.path.join(outpath,f'Cond00_v{wind_spd:04.1f}_PL{wind_shear:03.2f}_TI{turb_int:3.3f}',f'Case0_wdirp{wind_dir:02.0f}',f'Seed_{seed}')
                    with open(os.path.join(seedPath,'Fpower_FlorisOptimal.pkl'),'rb') as f:
                        yaw_angles_opt, floris_opt_power_stochastic, floris_opt_power = pkl.load(f)
                        # print([yaw_angles_opt])#, floris_opt_power_stochastic, floris_opt_power])
                        totalpower_stochastic = totalpower_stochastic + floris_opt_power_stochastic
                        totalpower = totalpower + floris_opt_power
                totalpower_stochastic = totalpower_stochastic/nSeeds
                totalpower = totalpower/nSeeds

                df_Fpower_FlorisOptimal.loc[len(df_Fpower_FlorisOptimal)] = [wind_spd,wind_dir,turb_int,yaw_angles_opt,totalpower_stochastic,totalpower]

    savepath = os.path.join(outpath,'Fpower_FlorisOptimal.pkl')
    df_Fpower_FlorisOptimal.to_csv(savepath[:-4]+'.csv', index=True)
    with open(os.path.join(savepath),'wb') as f:
        pkl.dump(df_Fpower_FlorisOptimal,f)

    pass

if __name__ == '__main__':
    main()
