from multiprocessing import Pool
from pathlib import Path
import pickle as pkl
import ast

import pandas as pd

thisdir = Path(__file__).parent.resolve()
namedict = {"SmallFarm": "", "MidFarm": "_medium"}
seedsetdata = {
    4: [
        916078875,
        -228636811,
        -1601233273,
        -1560318386,
        1333634118,
        1682697194,
        376932526,
        -1100498355,
        1341164796,
        -1346887648,
    ],
    5: [
        337152689,
        -1738710529,
        -972986995,
        1998483104,
        -1382532194,
        -536425485,
        -209040837,
        507863083,
        1414903771,
        -1240629519,
    ],
    6: [
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
    ],
}

with open(
    Path(
        thisdir
        / "../../../../RunFiles_Base/PostProcessing/BaseDELCalc/baselinedeldata.pkl"
    ),
    "rb",
) as f:
    baselinedeldata = pkl.load(f)


def main():
    allfarms = ["SmallFarm", "MidFarm"]
    allgroups = [1, 2]
    allseedsets = [4, 5, 6]
    allyawseeds_dict = {1: [1], 2: [1, 2, 3]}
    alltrials = [3, 4, 5]

    allinputs = []
    for farm in allfarms:
        for group in allgroups:
            for seedset in allseedsets:
                for yawseed in allyawseeds_dict[group]:
                    for trial in alltrials:
                        allinputs.append([farm, group, seedset, yawseed, trial])
                        # adddel([farm, group, seedset, yawseed, trial])
    with Pool() as pool:
        results = pool.map(adddel, allinputs)


def adddel(input):
    farm = input[0]
    group = input[1]
    seedset = input[2]
    yawseed = input[3]
    trial = input[4]
    monitordir = Path(
        thisdir
        / ".."
        / ".."
        / farm
        / "Runs"
        / f"Group{group}"
        / f"Seed{seedset}"
        / f"yawseeds{yawseed}"
        / "sac_output"
        / "trials"
        / f"trial_{trial}_10env{namedict[farm]}_1_1/"
    )
    monitorfile = Path(monitordir / "monitor.csv")
    dfm = pd.read_csv(monitorfile, skiprows=[0])
    for irow, row in dfm.iterrows():
        turbsimpath = Path(
            thisdir,
            "..",
            "..",
            farm,
            "Runs",
            f"Group{group}",
            f"Seed{seedset}",
            f"yawseeds{yawseed}",
            f"FFSim_Seed{seedset}_RL",
            f"Sim_{row['simseed']}",
            f"Cond00_v10.0_PL0.12_TI{row['turb_int']*100:3.1f}",
            "Seed_0",
            "Low.inp"
        )
        with open(turbsimpath,'r') as f:
            turbsimlines = f.readlines()
        seedfromturbsim = int(turbsimlines[4].split()[0])
        seedidx = seedsetdata[seedset].index(seedfromturbsim)

        mask = (
            (baselinedeldata["farm"] == farm)
            & (baselinedeldata["group"] == group)
            & (baselinedeldata["seedset"] == seedset)
            & (baselinedeldata["yawseed"] == yawseed)
            & (baselinedeldata["ti"] == row["turb_int"] * 100)
            & ( baselinedeldata["wdir"] == row["wind_dir"] - 90)
            & (baselinedeldata["seed"] == seedidx)
        )
        for var,casedel in ast.literal_eval(row["DEL"]).items():
            basedel = baselinedeldata.loc[mask,var].iat[0]
            dfm.at[irow,var] = (casedel-basedel)/basedel*100

    withDELfile = Path(monitordir / "monitor_withDEL.csv")
    dfm.to_csv(withDELfile)
    with open(withDELfile,'r') as f:
        lines = f.readlines()
    lines.insert(0, '#{"t_start": 1762853295.4692905, "env_id": "None"}\n')
    with open(withDELfile,'w') as f:
        f.writelines(lines)

if __name__ == "__main__":
    main()
