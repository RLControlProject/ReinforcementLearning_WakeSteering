import os
import numpy as np
import subprocess
import shutil
import sys
from openfast_toolbox.fastfarm.FASTFarmCaseCreation import FFCaseCreation
from addveer import addveer

from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR



thisdir = os.path.dirname(os.path.abspath(__file__))
RUNTYPE = "baseline" # turbsim, baseline or rl
num_turbines = 18

# def calculateflorsoptpower():
#     yaw_opt = YawOptimizationSR(self.fi, minimum_yaw_angle=-self.turbines[0].max_yaw_degrees, maximum_yaw_angle=self.turbines[0].max_yaw_degrees, Ny_passes=[9,8])#, exploit_layout_symmetry=False)
#     df_opt = yaw_opt.optimize()
#     yaw_angles_opt = df_opt.yaw_angles_opt.values[0]

def FFSetup(outpath, wind_spd, wind_dir, shear, TI, yaw_angles,FFomp):
    # -----------------------------------------------------------------------------
    # USER INPUT: Modify these
    #             For the d{t,s}_{high,low}_les paramters, use AMRWindSimulation.py
    # -----------------------------------------------------------------------------

    # ----------- Case absolute path
    btsdir = os.path.join(thisdir,'FFSim_Seed6')

    # ----------- Additional variables
    if num_turbines == 3:
        tmax = 700
    elif num_turbines == 18:
        tmax = 900  # Total simulation time
    myseeds = [
        -1569990632,
        1324626022,
        1658175070,
        -1732127742,
        -628545386,
        1668991113,
        142786863,
        -503034494,
        -1912321698,
        -260583563,
    ]
    if RUNTYPE == "rl":
        randseedidx = np.random.randint(10)
        # randseedidx = 9
        randseed = [myseeds[randseedidx]]
    else:
        randseed = myseeds
        randseedidx = range(len(myseeds))
    
    nSeeds = len(randseed)
    # nSeeds = 1
    zbot = 1  # Bottom of your domain
    mod_wake = 1  # Wake model. 1: Polar, 2: Curl, 3: Cartesian

    # ----------- Desired sweeps
    vhub = wind_spd
    shear = shear
    TIvalue = TI
    inflow_deg = []

    match RUNTYPE:
        case "turbsim":
            inflow_deg.extend(wind_dir)
        case "baseline":
            inflow_deg.append(wind_dir)
        case "rl":
            # inflow_deg.append(wind_dir[0])
            inflow_deg.append(wind_dir)

    # ----------- Turbine parameters
    # Set the yaw of each turbine for wind dir. One row for each wind direction.
    # yaw_init = []
    # yaw_init.extend(yaw_angles[0])
    # yaw_init = np.array(yaw_init)
    # yaw_init = np.array([yaw_angles[0][0]])
    match RUNTYPE:
        case "turbsim":
            yaw_init = np.array(yaw_angles)
        case "baseline":
            yaw_init = np.array(yaw_angles[0])
        case "rl":
            yaw_init = np.array(yaw_angles[0])

    # ----------- General hard-coded parameters
    cmax = 4.65  # maximum blade chord (m)
    fmax = 2.0  # maximum excitation frequency (Hz)
    Cmeander = 1.9  # Meandering constant (-)

    # ----------- Wind farm
    D = 126
    zhub = 90
    if num_turbines == 3:
        x = [0.0, 504.0, 1008.0] 
        y = [0.0, 0.0, 0.0,]
    elif num_turbines == 18:
        x = [0.0,0.0,0.0,
             504.0,504.0,504.0,
             1008.0,1008.0,1008.0,
             1512.0,1512.0,1512.0,
             2016.0,2016.0,2016.0,
             2520.0,2520.0,2520.0]
        y = [0.0, 378.0, 756.0,
             0.0, 378.0, 756.0,
             0.0, 378.0, 756.0,
             0.0, 378.0, 756.0,
             0.0, 378.0, 756.0,
             0.0, 378.0, 756.0]

    wts = {
        i: {
            "x": x[i],
            "y": y[i],
            "z": 0,
            "D": D,
            "zhub": zhub,
            "cmax": cmax,
            "fmax": fmax,
            "Cmeander": Cmeander,
        }
        for i in range(len(x))
    }
    print(wts)

    # ----------- Low- and high-res boxes parameters
    # Should match LES if comparisons are to be made; otherwise, set desired values
    # For an automatic computation of such parameters, omit them from the call to FFCaseCreation
    # High-res boxes settings
    dt_high_les = 0.6  # sampling frequency of high-res files
    ds_high_les = 2.0  # dx, dy, dz that you want these high-res files at
    extent_high = 1.2  # high-res box extent in y and x for each turbine, in D.
    # Low-res boxes settings
    dt_low_les = 3  # sampling frequency of low-res files
    ds_low_les = 10.0  # dx, dy, dz of low-res files
    extent_low = [3, 3, 3, 3, 2]  # extent in xmin, xmax, ymin, ymax, zmax, in D

    # ----------- Execution parameters
    if FFomp:
        ffbin = os.path.join(
            thisdir,
            "..",
            "..",
            "..",
            "..",
            "..",
            "build",
            #"build_v3p4p1_noomp",
            "AG_openfast",
            "install",
            "bin",
            "FAST.Farm",
        )
    else:
        ffbin = os.path.join(
            thisdir,
            "..",
            "..",
            "..",
            "..",
            "..",
            "build",
            "build_v3p4p1_noomp",
            #"build_v3p4p1_intel",
            "glue-codes",
            "fast-farm",
            "FAST.Farm",
        )

    # ----------- LES parameters. This variable will dictate whether it is a TurbSim-driven or LES-driven case
    LESpath = None

    # -----------------------------------------------------------------------------
    # ----------- Template files
    # templatePath            = os.path.join(thisdir,'Template_5MW','5MW_Land_DLL_WTurb')
    templatePath = os.path.join(thisdir, "TurbineTemplates", "Template_5MW")

    # Put 'unused' to any input that is not applicable to your case
    # Files should be in templatePath
    EDfilename = "NRELOffshrBsline5MW_Onshore_ElastoDyn.T"
    SEDfilename = "unused"
    HDfilename = "unused"
    SrvDfilename = "NRELOffshrBsline5MW_Onshore_ServoDyn.T"
    ADfilename = "NRELOffshrBsline5MW_Onshore_AeroDyn15.dat"
    ADskfilename = "unused"
    SubDfilename = "unused"
    IWfilename = "NRELOffshrBsline5MW_InflowWind_Steady8mps.dat"
    BDfilepath = "unused"
    bladefilename = "NRELOffshrBsline5MW_Blade.dat"
    towerfilename = "NRELOffshrBsline5MW_Onshore_ElastoDyn_Tower.dat"
    turbfilename = "5MW_Land_DLL_WTurb.T"
    libdisconfilepath = os.path.join(templatePath, "DISCON.so")
    controllerInputfilename = "DISCON.IN"
    coeffTablefilename = "unused"
    FFfilename = "TestCase.fstf"

    # TurbSim setups
    turbsimLowfilepath = os.path.join(
        templatePath, "SampleFiles", "template_Low_InflowXX_SeedY.inp"
    )
    turbsimHighfilepath = os.path.join(
        templatePath, "SampleFiles", "template_HighT1_InflowXX_SeedY.inp"
    )

    # SLURM scripts
    slurm_TS_high = os.path.join(templatePath, "SampleFiles", "runAllHighBox.sh")
    slurm_TS_low = os.path.join(templatePath, "SampleFiles", "runAllLowBox.sh")
    slurm_FF_single = os.path.join(
        templatePath, "SampleFiles", "runFASTFarm_cond0_case0_seed0.sh"
    )

    # -----------------------------------------------------------------------------
    # END OF USER INPUT
    # -----------------------------------------------------------------------------

    # Initial setup
    case = FFCaseCreation(
        outpath,
        wts,
        tmax,
        zbot,
        vhub,
        shear,
        TIvalue,
        inflow_deg,
        dt_high_les,
        ds_high_les,
        extent_high,
        dt_low_les,
        ds_low_les,
        extent_low,
        ffbin,
        mod_wake,
        yaw_init,
        nSeeds=nSeeds,
        LESpath=LESpath,
        seedValues=randseed,
        verbose=1,
    )

    case.setTemplateFilename(
        templatePath,
        EDfilename,
        SEDfilename,
        HDfilename,
        SrvDfilename,
        ADfilename,
        ADskfilename,
        SubDfilename,
        IWfilename,
        BDfilepath,
        bladefilename,
        towerfilename,
        turbfilename,
        libdisconfilepath,
        controllerInputfilename,
        coeffTablefilename,
        turbsimLowfilepath,
        turbsimHighfilepath,
        FFfilename,
    )

    # Get domain paramters
    case.getDomainParameters()

    # Organize file structure
    case.copyTurbineFilesForEachCase()

    # TurbSim setup
    if LESpath is None:
        runturbsim = True
        # runturbsim = False
        turbsimexe = os.path.join(
            thisdir,
            "..",
            "..",
            "..",
            "..",
            "..",
            "build",
            "openfast_3p5p3",
            "modules",
            "turbsim",
            "turbsim",
        )

        if runturbsim:
            case.TS_low_setup()
            # case.TS_low_slurm_prepare(slurm_TS_low)
            # case.TS_low_slurm_submit()
            for cond in range(case.nConditions):
                for seed in range(case.nSeeds):
                    seedPath = os.path.join(
                        case.path, case.condDirList[cond], f"Seed_{seed}"
                    )
                    match RUNTYPE:
                        case "turbsim":
                            addveer(seedPath)
                            os.system(turbsimexe + ' ' + os.path.join(seedPath,'Low.inp'))
                        case "baseline":
                            shutil.copy(os.path.join(btsdir,f'Cond00_v{wind_spd:04.1f}_PL{shear:03.2f}_TI{TI:3.1f}', f'Seed_{seed}','Low.bts'),os.path.join(seedPath,'Low.bts'))
                            print(f'Using precompute bts file for low resolution domain from {randseed}')
                        case "rl":
                            shutil.copy(os.path.join(btsdir,f'Cond00_v{wind_spd:04.1f}_PL{shear:03.2f}_TI{TI:3.1f}', f'Seed_{randseedidx}','Low.bts'),os.path.join(seedPath,'Low.bts'))
                            print('Using precompute bts file for low resolution domain')

                    
            case.TS_high_setup()
            # # case.TS_high_slurm_prepare(slurm_TS_high)
            # # case.TS_high_slurm_submit()
            for cond in range(case.nConditions):
                for casei in range(case.nHighBoxCases):
                    # Get actual case number given the high-box that need to be saved
                    casei = case.allHighBoxCases.isel(case=casei)["case"].values
                    for seed in range(case.nSeeds):
                        seedPath = os.path.join(
                            case.path,
                            case.condDirList[cond],
                            case.caseDirList[casei],
                            f"Seed_{seed}",
                            "TurbSim",
                        )
                        if RUNTYPE in ["baseline","rl"]:
                            btscasepath = os.path.join(btsdir,f'Cond00_v{wind_spd:04.1f}_PL{shear:03.2f}_TI{TI:3.1f}')
                            caselist_all = os.listdir(btscasepath)
                            caselist = [case for case in caselist_all if case[-7:]==f"wdirp{wind_dir:02.0f}"]
                        for t in range(case.nTurbines):
                            match RUNTYPE:
                                case "turbsim":
                                    os.system( turbsimexe + " " + os.path.join(seedPath, f"HighT{t+1}.inp"))
                                case "baseline":
                                    shutil.copy(os.path.join(btsdir,f'Cond00_v{wind_spd:04.1f}_PL{shear:03.2f}_TI{TI:3.1f}',caselist[0],f'Seed_{seed}','TurbSim',f'HighT{t+1}.bts'),os.path.join(seedPath,f'HighT{t+1}.bts'))
                                case "rl":
                                    shutil.copy(os.path.join(btsdir,f'Cond00_v{wind_spd:04.1f}_PL{shear:03.2f}_TI{TI:3.1f}',caselist[0],f'Seed_{randseedidx}','TurbSim',f'HighT{t+1}.bts'),os.path.join(seedPath,f'HighT{t+1}.bts'))

    # Drop unnecessary cases
    # case.nCases = 1
    # case.caseDirList = [case.caseDirList[-1]]
    # case.inflow_deg = [case.inflow_deg[-1]]
    # Final setup
    case.FF_setup(outlistFF=["RtAxsXT1"])
    # case.FF_slurm_prepare(slurm_FF_single)
    # case.FF_slurm_submit()
    # Change mobwind from 3 to 2

    for cond in range(case.nConditions):
        for casei in range(case.nCases):
            casepath = os.path.join(case.path,case.condDirList[cond],case.caseDirList[casei])
            disconinfile = os.path.join(casepath, controllerInputfilename)
            with open(disconinfile,'r') as f:
                disconinlines = f.readlines()
            for t in range(case.nTurbines):
                disconinfile_t = os.path.join(os.path.dirname(disconinfile),f"DISCON{t+1}.IN")
                with open(disconinfile_t,'w') as f:
                    f.writelines(disconinlines)
                servodynfile = os.path.join(casepath,SrvDfilename+f"{t+1}_mod.dat")
                with open(servodynfile,'r') as f:
                    servodynlines = f.readlines()
                    servodynlines[77] = f"DISCON{t+1}.IN" + servodynlines[77][12:]
                with open(servodynfile,'w') as f:
                    f.writelines(servodynlines)
            allprocs = []
            for seed in range(case.nSeeds):
                seedPath = os.path.join(
                    case.path,
                    case.condDirList[cond],
                    case.caseDirList[casei],
                    f"Seed_{seed}",
                )
                destairfoildir = os.path.join(
                    case.path,
                    case.condDirList[cond],
                    case.caseDirList[casei],
                    "Airfoils",
                )
                shutil.rmtree(destairfoildir, ignore_errors=True)
                shutil.copytree(os.path.join(templatePath, "Airfoils"), destairfoildir)
                shutil.copy(os.path.join(templatePath,'Cp_Ct_Cq.NREL5MW.txt'),casepath)
                # os.remove(casepath,"DISCON.IN"))
                # shutil.copy(os.path.join(templatePath,'DISCON.IN'),casepath)

                # Change mobwind to 2
                fstfile = os.path.join(seedPath, "FFarm_mod.fstf")
                with open(fstfile, "r") as f:
                    fstlines = f.readlines()
                # fstlines[7] = "2 " + fstlines[7][4:]
                fstlines[99] = "450" + fstlines[99][5:]
                with open(fstfile, "w") as f:
                    f.writelines(fstlines)

                # Change inflowwind to point
                # inflowfile = os.path.join(seedPath, IWfilename)
                # with open(inflowfile, "r") as f:
                #     inflowlines = f.readlines()
                # inflowlines[20] = '"./TurbSim/Low.bts"' + inflowlines[20][11:]
                # with open(inflowfile, "w") as f:
                #     f.writelines(inflowlines)
                
                if RUNTYPE != 'turbsim':
                    # status = os.system(ffbin + " " + os.path.join(seedPath, "FFarm_mod.fstf > /dev/null"))
                    # if status != 0:
                    #    raise Exception("FASTFarm did not run.")
                    cmd = [ffbin, os.path.join(seedPath, "FFarm_mod.fstf"), "> /dev/null"]
                    p = subprocess.Popen(cmd)
                    allprocs.append(p)
            for p in allprocs:
                p.wait()

    return randseedidx



if __name__ == "__main__":
    FFSetup(
        outpath=os.path.join(thisdir, "./FFSim_Seed6"),
        wind_spd=10.0,
        wind_dir=[0.0, 2.0, 4.0, 6.0],
        shear=0.12,
        TI=float(sys.argv[1]),
        yaw_angles=[
                    [0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    ],
        FFomp = True
    )
