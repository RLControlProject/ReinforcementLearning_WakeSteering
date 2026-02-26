import os
import numpy as np
import shutil
import sys
import pickle as pkl

thisdir = os.path.dirname(os.path.abspath(__file__))
THIS_FILE_PATH = os.path.join(os.path.dirname(__file__))

def FSetup(type, fi, outpath, wind_spd, wind_dir, turb_int, wind_shear, wind_veer, yaw_angles, yaw_angles_actual, mean_WS, std_dev_WS, mean_TI, std_dev_TI, mean_WD, std_dev_WD, myseeds, randseedidx):
    # -----------------------------------------------------------------------------
    # USER INPUT: Modify these
    #             For the d{t,s}_{high,low}_les paramters, use AMRWindSimulation.py
    # -----------------------------------------------------------------------------

    # ----------- Additional variables
    if type=='baseline':
        # randseedidx = range(len(myseeds))
        randseed = myseeds
    else:
        # randseedidx = [np.random.randint(len(myseeds))]
        randseed = [myseeds[randseedidx[0]]]
    
    nSeeds = len(randseed)

    # ----------- Turbine parameters

    # run FLORIS
    for seed in range(nSeeds):
        thisseedidx = randseedidx[seed]
        thisseed = randseed[seed]

        # run FLORIS
        fi.calculate_wake(yaw_angles=np.array(yaw_angles_actual))
        floris_power = fi.get_turbine_powers()/1000.

        # add perturbations to the output
        np.random.seed(thisseed)
        wind_spd_perturb = wind_spd+np.random.normal(mean_WS, std_dev_WS)
        turb_int_perturb = turb_int+np.random.normal(mean_TI, std_dev_TI)
        wind_dir_perturb = wind_dir+np.random.normal(mean_WD, std_dev_WD)

        # run FLORIS with perturbed inputs
        fi.reinitialize(wind_directions=[wind_dir_perturb],wind_speeds=[wind_spd_perturb], turbulence_intensity=turb_int_perturb,wind_shear=wind_shear,wind_veer=wind_veer)
        fi.calculate_wake(yaw_angles=np.array(yaw_angles_actual))
        floris_power_stochastic = fi.get_turbine_powers()/1000.
        fi.reinitialize(wind_directions=[wind_dir],wind_speeds=[wind_spd], turbulence_intensity=turb_int,wind_shear=wind_shear,wind_veer=wind_veer)
        np.random.seed(seed=None) # must reset the seed to None or else the call to np.random.randint above will not return random numbers in the desired range

        if type=='baseline':
            seedPath = os.path.join(outpath,f'Cond00_v{wind_spd:04.1f}_PL{wind_shear:03.2f}_TI{turb_int:3.3f}',f'Case0_wdirp{wind_dir:02.0f}',f'Seed_{thisseedidx}')
            os.makedirs(seedPath, exist_ok=True)
            savepath = os.path.join(seedPath,'Fpower_FlorisOptimal.pkl')
            with open(savepath,'wb') as f:
                pkl.dump([yaw_angles[0][0], floris_power_stochastic[0][0], floris_power[0][0]],f)
        else:
            return floris_power_stochastic

if __name__ == "__main__":
    FSetup(
        outpath=os.path.join(thisdir, "./FFSimTempdel"),
        wind_spd=10.0,
        wind_dir=2,
        shear=0.12,
        TI=2.5,
        yaw_angles=[[[0, 0, 0]]]
    )
