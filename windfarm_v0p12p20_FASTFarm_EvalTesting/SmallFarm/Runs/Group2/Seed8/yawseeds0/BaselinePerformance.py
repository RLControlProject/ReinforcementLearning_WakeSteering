import os
import numpy as np
import sys
import pandas as pd
from FFSetup_Seed5 import FFSetup
import multiprocessing as mp
from floris.tools import FlorisInterface
from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import (
    YawOptimizationSR,
)
from openfast_toolbox.io import FASTOutputFile
import pickle as pkl
from windfarmenv_seed5 import (
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
outpath = os.path.join(THISDIR, "FFSim_Baseline_Seed5")
runsim = True



def main(TI):
    FLORIS_path_relative = os.path.join(
        "/kfs2/projects/rlfarmcontr/FY25/RLControl/submodules/floris/examples/inputs/gch.yaml"
    )

    df_FFPower_FlorisOptimal = pd.DataFrame(
        columns=[
            "wind_spd",
            "wind_dir",
            "turb_int",
            "optyaw",
            "yaw_offset",
            "optpower",
            "seedpower",
        ]
    )

    if runsim:
        for wind_spd in wind_spd_all:
            for wind_dir in wind_dir_all:
                for turb_int in [TI]:
                    fi = FlorisInterface(os.path.join(THISDIR, FLORIS_path_relative))
                    fi.reinitialize(
                        layout_x=[point[0] for point in layout_grid],
                        layout_y=[point[1] for point in layout_grid],
                        turbine_type=turbine_type,
                    )
                    fi.reinitialize(
                        wind_directions=[wind_dir],
                        wind_speeds=[wind_spd],
                        turbulence_intensity=turb_int,
                        wind_shear=wind_shear,
                        wind_veer=wind_veer,
                    )
                    yaw_opt = YawOptimizationSR(
                        fi,
                        minimum_yaw_angle=-1 * max_yaw_degrees,
                        maximum_yaw_angle=max_yaw_degrees,
                        Ny_passes=[9, 8],
                    )  # , exploit_layout_symmetry=False)
                    df_opt = yaw_opt.optimize()
                    yaw_angles_opt = df_opt.yaw_angles_opt.values[0]
                    yaw_angles_opt = yaw_angles_opt.tolist()
                    yaw_angles_opt_ordered = list(reversed(yaw_angles_opt))
                    yaw_angles_static_offset_ordered = list(reversed(yaw_angles_static_offset[0][0]))
                    yaw_angles_opt_actual = list(np.clip(np.array(yaw_angles_opt_ordered) + np.array(yaw_angles_static_offset_ordered), a_min=-max_yaw_degrees, a_max=max_yaw_degrees))

                    yaw_np_opt = np.array(yaw_angles_opt_ordered).reshape(1, 1, 3)
                    yaw_np_static_offset = np.array(yaw_angles_static_offset_ordered).reshape(1, 1, 3)

                    FFSetup(
                        outpath=outpath,
                        wind_spd=wind_spd,
                        wind_dir=wind_dir - 90,
                        shear=wind_shear,
                        TI=turb_int * 100,
                        yaw_angles=[[yaw_angles_opt_actual]],
                        FFomp=True
                    )

                    condpath = os.path.join(
                        outpath,
                        f"Cond00_v{wind_spd:04.1f}_PL{wind_shear:03.2f}_TI{turb_int*100:3.1f}",
                        f"Case0_wdirp{wind_dir-90:02.0f}",
                    )

                    totalpower = 0
                    allseedpower = []
                    for seed in range(10):
                        seedpower = 0
                        for turb in range(2):
                            dfi = FASTOutputFile(
                                os.path.join(
                                    condpath, f"Seed_{seed}", f"FFarm_mod.T{turb+1}.outb"
                                )
                            ).toDataFrame()
                            tinit =  600
                            Time = dfi["Time_[s]"]
                            Poweri = dfi["GenPwr_[kW]"][Time > tinit].mean()
                            totalpower = totalpower + Poweri
                            seedpower = seedpower + Poweri
                        allseedpower.append(seedpower)
                    totalpower = totalpower / 10
                    df_FFPower_FlorisOptimal.loc[len(df_FFPower_FlorisOptimal)] = [
                        wind_spd,
                        wind_dir,
                        turb_int,
                        yaw_np_opt,
                        yaw_np_static_offset,
                        totalpower,
                        allseedpower
                    ]
    with open(f"FFPower_FlorisOptimal_Seed5_{TI}.pkl", "wb") as f:
        pkl.dump(df_FFPower_FlorisOptimal, f)

    pass


def generate_layout_grid(num_points_x, num_points_y, spacing_x, spacing_y):
    """
    Generate a grid of layout points with arbitrary numbers of points and spacing in each direction.

    Args:
        num_points_x (int): Number of points in the x direction.
        num_points_y (int): Number of points in the y direction.
        spacing_x (float): Spacing between points in the x direction.
        spacing_y (float): Spacing between points in the y direction.

    Returns:
        list of tuples: List of layout points represented as (x, y) coordinates.
    """
    layout_grid = []
    for i in range(num_points_x):
        for j in range(num_points_y):
            x = i * spacing_x
            y = j * spacing_y
            layout_grid.append((x, y))
    return layout_grid


if __name__ == "__main__":
    try:
        TI = float(sys.argv[1])
        print(f"Using turbulence intensity from arguments: {TI}")
    except Exception:
        TI = 0.06
        print(f"Using fixed turbulence intensity: {TI}")
    main(TI)
