import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches

def main():
    allyawseeds = [1,2,3]
    allseeds = [4,5,6]
    colors = ['cornflowerblue','mediumaquamarine','salmon']
    dfmeans = pd.DataFrame(columns = ['SeedSet','YawSeed','BaseMeanPower','OffsetMeanPower'])
    
    for iyawseed,yawseed in enumerate(allyawseeds):
        dfbase = pd.DataFrame()
        dfoffset = pd.DataFrame()
        for iseed,seed in enumerate(allseeds):
            alldata = pd.DataFrame()
            # if dfbase.empty:
            dfbase,dfoffset = getdata(seed,yawseed)
            # else:
            #     dfbase_temp, dfoffset_temp = getdata(seed,yawseed)
            #     dfbase = pd.concat([dfbase, dfbase_temp], ignore_index=True)
            #     dfoffset = pd.concat([dfoffset, dfoffset_temp], ignore_index=True)

            seedpower_base = []
            seedpower_offset = []
            for i,row in dfbase.iterrows():
                seedpower_base.extend(row['seedpower'])
            for i,row in dfoffset.iterrows():
                seedpower_offset.extend(row['seedpower'])
            alldata['basepower'] = seedpower_base
            alldata[f'offsetpower{yawseed}'] = seedpower_offset
            hwid = 0.3
            print(f"mean of the base, yaw power seed{seed} for yawseed{yawseed} is {np.mean(seedpower_base):05.3f},{np.mean(seedpower_offset):05.3f}")
            dfmeans.loc[len(dfmeans)] = [seed,yawseed,np.mean(seedpower_base),np.mean(seedpower_offset)]
            plt.plot([iyawseed+1-hwid,iyawseed+1+hwid],[np.mean(seedpower_offset),np.mean(seedpower_offset)],color = colors[iseed],alpha = 0.75)
            if iyawseed == 0:
                sns.stripplot(alldata,color = colors[iseed],alpha = 0.35)
                plt.plot([-hwid,+hwid],[np.mean(seedpower_base),np.mean(seedpower_base)],color = colors[iseed],alpha = 0.75)
            else:
                sns.stripplot(alldata[[f'offsetpower{yawseed}']],color = colors[iseed],alpha = 0.35)
    legend_handles = []
    for iseed,seed in enumerate(allseeds):
        cat = f'Seedset-{seed}'
        patch = mpatches.Patch(color=colors[iseed], label=cat,alpha = 0.75)
        legend_handles.append(patch)
    plt.ylim([10500,22000])
    plt.legend(handles=legend_handles)
    plt.savefig('StripPlot.png')


    with open('MeanPower.pkl','wb') as f:
        pkl.dump(dfmeans,f)


def getdata(seed,yawseed):
    with open(f'../../MidFarm/Runs/Group1/Seed{seed}/yawseeds1/FFPower_FlorisOptimal_Seed{seed}.pkl','rb') as f:
        data_base = pkl.load(f)
    with open(f'../../MidFarm/Runs/Group2/Seed{seed}/yawseeds{yawseed}/FFPower_FlorisOptimal_Seed{seed}.pkl','rb') as f:
        data_offset = pkl.load(f)
    return data_base, data_offset

if __name__ == "__main__":
    main()
