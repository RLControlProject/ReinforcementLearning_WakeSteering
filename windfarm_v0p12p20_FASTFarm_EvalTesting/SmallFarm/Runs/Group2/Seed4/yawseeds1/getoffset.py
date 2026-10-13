import os
from windfarmenv_seed4 import (
    num_turbines,
    layout_grid,
    turbine_type,
    max_yaw_degrees,
    wind_spd_all,
    wind_dir_all,
    turb_int_all,
    wind_shear,
    wind_veer,
    yaw_angles_static_offset,
)


THISDIR = os.path.dirname(os.path.abspath(__file__))
outpath = os.path.join(THISDIR, "FFSim_Baseline_Seed4")
runsim = True


print(f'The offsets are {yaw_angles_static_offset}')
pass
