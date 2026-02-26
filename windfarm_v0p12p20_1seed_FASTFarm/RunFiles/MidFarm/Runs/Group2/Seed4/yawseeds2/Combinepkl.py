import pickle as pkl
import pandas as pd

files = [
    "FFPower_FlorisOptimal_Seed4_0.025.pkl",
    # "FFPower_FlorisOptimal_Seed4_0.03.pkl",
    # "FFPower_FlorisOptimal_Seed4_0.04.pkl",
    "FFPower_FlorisOptimal_Seed4_0.05.pkl",
    # "FFPower_FlorisOptimal_Seed4_0.06.pkl",
    "FFPower_FlorisOptimal_Seed4_0.075.pkl",
]

alldata = pd.DataFrame()
for file in files:
    with open(file,'rb') as f:
        data = pkl.load(f)
        alldata = pd.concat([alldata,data],ignore_index=True)

with open(f"FFPower_FlorisOptimal_Seed4.pkl", "wb") as f:
        pkl.dump(alldata, f)
