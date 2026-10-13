import pickle as pkl
import pandas as pd

files = [
    # "FFPower_FlorisOptimal_Seed7_0.025.pkl",
    "FFPower_FlorisOptimal_Seed7_0.0375.pkl",
    # "FFPower_FlorisOptimal_Seed7_0.05.pkl",
    "FFPower_FlorisOptimal_Seed7_0.0625.pkl",
    # "FFPower_FlorisOptimal_Seed7_0.075.pkl",
]

alldata = pd.DataFrame()
for file in files:
    with open(file,'rb') as f:
        data = pkl.load(f)
        alldata = pd.concat([alldata,data],ignore_index=True)

with open(f"FFPower_FlorisOptimal_Seed7.pkl", "wb") as f:
        pkl.dump(alldata, f)
