#!/bin/sh

:<<'COMMENT'
chmod +x combination_batch.sh  # make it an executable (one time only)
cd /ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl
source /projects/wind_uq/python/env/loadwindconda1.sh
./combination_batch.sh
COMMENT

export TEMPLATE=combination_template.ipynb

original_directory=$(pwd)  # Store the original directory


# ### FLORIS cases (timing runs)

# # 2x4
# # 1env
# export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20/analysis
# export SUBDIRECTORY='..'
# export TRIALNAME_SPECIFIER='trial_3_1env_medium'
# export PAIRLABEL=''
# export TYPE='FLORIS'
# NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
# cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
# cd $DIRECTORY
# jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
# cd "$original_directory"  # Return to the original directory

# export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20/analysis
# export SUBDIRECTORY='..'
# export TRIALNAME_SPECIFIER='trial_4_1env_medium'
# export PAIRLABEL=''
# export TYPE='FLORIS'
# NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
# cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
# cd $DIRECTORY
# jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
# cd "$original_directory"  # Return to the original directory

# export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20/analysis
# export SUBDIRECTORY='..'
# export TRIALNAME_SPECIFIER='trial_5_1env_medium'
# export PAIRLABEL=''
# export TYPE='FLORIS'
# NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
# cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
# cd $DIRECTORY
# jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
# cd "$original_directory"  # Return to the original directory


### FLORIS cases (production runs)

# Idealized inflow

# 1x3
export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_1seed/analysis
export SUBDIRECTORY='..'
export TRIALNAME_SPECIFIER='trial_3_10env'
export PAIRLABEL=''
export TYPE='FLORIS'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory

export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_1seed/analysis
export SUBDIRECTORY='..'
export TRIALNAME_SPECIFIER='trial_4_10env'
export PAIRLABEL=''
export TYPE='FLORIS'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory

export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_1seed/analysis
export SUBDIRECTORY='..'
export TRIALNAME_SPECIFIER='trial_5_10env'
export PAIRLABEL=''
export TYPE='FLORIS'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory


# 2x4
export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_1seed/analysis
export SUBDIRECTORY='..'
export TRIALNAME_SPECIFIER='trial_3_10env_medium'
export PAIRLABEL=''
export TYPE='FLORIS'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory

export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_1seed/analysis
export SUBDIRECTORY='..'
export TRIALNAME_SPECIFIER='trial_4_10env_medium'
export PAIRLABEL=''
export TYPE='FLORIS'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory

export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_1seed/analysis
export SUBDIRECTORY='..'
export TRIALNAME_SPECIFIER='trial_5_10env_medium'
export PAIRLABEL=''
export TYPE='FLORIS'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory


# 3x6
export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_1seed/analysis
export SUBDIRECTORY='..'
export TRIALNAME_SPECIFIER='trial_3_10env_large'
export PAIRLABEL=''
export TYPE='FLORIS'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory

export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_1seed/analysis
export SUBDIRECTORY='..'
export TRIALNAME_SPECIFIER='trial_4_10env_large'
export PAIRLABEL=''
export TYPE='FLORIS'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory

export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_1seed/analysis
export SUBDIRECTORY='..'
export TRIALNAME_SPECIFIER='trial_5_10env_large'
export PAIRLABEL=''
export TYPE='FLORIS'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory


# 4x8
export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_1seed/analysis
export SUBDIRECTORY='..'
export TRIALNAME_SPECIFIER='trial_3_10env_extralarge'
export PAIRLABEL=''
export TYPE='FLORIS'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory

export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_1seed/analysis
export SUBDIRECTORY='..'
export TRIALNAME_SPECIFIER='trial_4_10env_extralarge'
export PAIRLABEL=''
export TYPE='FLORIS'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory

export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_1seed/analysis
export SUBDIRECTORY='..'
export TRIALNAME_SPECIFIER='trial_5_10env_extralarge'
export PAIRLABEL=''
export TYPE='FLORIS'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory


## Realistic inflow

# 1x3
export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20/analysis
export SUBDIRECTORY='..'
export TRIALNAME_SPECIFIER='trial_3_10env'
export PAIRLABEL=''
export TYPE='FLORIS'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory

export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20/analysis
export SUBDIRECTORY='..'
export TRIALNAME_SPECIFIER='trial_4_10env'
export PAIRLABEL=''
export TYPE='FLORIS'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory

export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20/analysis
export SUBDIRECTORY='..'
export TRIALNAME_SPECIFIER='trial_5_10env'
export PAIRLABEL=''
export TYPE='FLORIS'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory


# 2x4
export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20/analysis
export SUBDIRECTORY='..'
export TRIALNAME_SPECIFIER='trial_3_10env_medium'
export PAIRLABEL=''
export TYPE='FLORIS'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory

export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20/analysis
export SUBDIRECTORY='..'
export TRIALNAME_SPECIFIER='trial_4_10env_medium'
export PAIRLABEL=''
export TYPE='FLORIS'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory

export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20/analysis
export SUBDIRECTORY='..'
export TRIALNAME_SPECIFIER='trial_5_10env_medium'
export PAIRLABEL=''
export TYPE='FLORIS'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory


# 3x6
export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20/analysis
export SUBDIRECTORY='..'
export TRIALNAME_SPECIFIER='trial_3_10env_large'
export PAIRLABEL=''
export TYPE='FLORIS'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory

export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20/analysis
export SUBDIRECTORY='..'
export TRIALNAME_SPECIFIER='trial_4_10env_large'
export PAIRLABEL=''
export TYPE='FLORIS'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory

export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20/analysis
export SUBDIRECTORY='..'
export TRIALNAME_SPECIFIER='trial_5_10env_large'
export PAIRLABEL=''
export TYPE='FLORIS'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory


# 4x8
export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20/analysis
export SUBDIRECTORY='..'
export TRIALNAME_SPECIFIER='trial_3_10env_extralarge'
export PAIRLABEL=''
export TYPE='FLORIS'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory

export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20/analysis
export SUBDIRECTORY='..'
export TRIALNAME_SPECIFIER='trial_4_10env_extralarge'
export PAIRLABEL=''
export TYPE='FLORIS'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory

export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20/analysis
export SUBDIRECTORY='..'
export TRIALNAME_SPECIFIER='trial_5_10env_extralarge'
export PAIRLABEL=''
export TYPE='FLORIS'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory


### FAST.Farm cases (production runs)

## Idealized inflow

# Group 1

# 1x3
export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_1seed_FASTFarm/analysis/Group1
export SUBDIRECTORY=../../RunFiles/SmallFarm/Runs/Group1
export TRIALNAME_SPECIFIER='trial_3_10env'
export PAIRLABEL=''
export TYPE='FASTFARM'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory

export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_1seed_FASTFarm/analysis/Group1
export SUBDIRECTORY=../../RunFiles/SmallFarm/Runs/Group1
export TRIALNAME_SPECIFIER='trial_4_10env'
export PAIRLABEL=''
export TYPE='FASTFARM'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory

export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_1seed_FASTFarm/analysis/Group1
export SUBDIRECTORY=../../RunFiles/SmallFarm/Runs/Group1
export TRIALNAME_SPECIFIER='trial_5_10env'
export PAIRLABEL=''
export TYPE='FASTFARM'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory


# 2x4
export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_1seed_FASTFarm/analysis/Group1
export SUBDIRECTORY=../../RunFiles/MidFarm/Runs/Group1
export TRIALNAME_SPECIFIER='trial_3_10env_medium'
export PAIRLABEL=''
export TYPE='FASTFARM'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory

export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_1seed_FASTFarm/analysis/Group1
export SUBDIRECTORY=../../RunFiles/MidFarm/Runs/Group1
export TRIALNAME_SPECIFIER='trial_4_10env_medium'
export PAIRLABEL=''
export TYPE='FASTFARM'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory

export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_1seed_FASTFarm/analysis/Group1
export SUBDIRECTORY=../../RunFiles/MidFarm/Runs/Group1
export TRIALNAME_SPECIFIER='trial_5_10env_medium'
export PAIRLABEL=''
export TYPE='FASTFARM'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory


# Group 2

# 1x3
export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_1seed_FASTFarm/analysis/Group2
export SUBDIRECTORY=../../RunFiles/SmallFarm/Runs/Group2
export TRIALNAME_SPECIFIER='trial_3_10env'
export PAIRLABEL=''
export TYPE='FASTFARM'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory

export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_1seed_FASTFarm/analysis/Group2
export SUBDIRECTORY=../../RunFiles/SmallFarm/Runs/Group2
export TRIALNAME_SPECIFIER='trial_4_10env'
export PAIRLABEL=''
export TYPE='FASTFARM'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory

export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_1seed_FASTFarm/analysis/Group2
export SUBDIRECTORY=../../RunFiles/SmallFarm/Runs/Group2
export TRIALNAME_SPECIFIER='trial_5_10env'
export PAIRLABEL=''
export TYPE='FASTFARM'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory


# 2x4
export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_1seed_FASTFarm/analysis/Group2
export SUBDIRECTORY=../../RunFiles/MidFarm/Runs/Group2
export TRIALNAME_SPECIFIER='trial_3_10env_medium'
export PAIRLABEL=''
export TYPE='FASTFARM'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory

export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_1seed_FASTFarm/analysis/Group2
export SUBDIRECTORY=../../RunFiles/MidFarm/Runs/Group2
export TRIALNAME_SPECIFIER='trial_4_10env_medium'
export PAIRLABEL=''
export TYPE='FASTFARM'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory

export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_1seed_FASTFarm/analysis/Group2
export SUBDIRECTORY=../../RunFiles/MidFarm/Runs/Group2
export TRIALNAME_SPECIFIER='trial_5_10env_medium'
export PAIRLABEL=''
export TYPE='FASTFARM'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory



## Realistic inflow

# Group 1

# 1x3
export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_FASTFarm/analysis/Group1
export SUBDIRECTORY=../../RunFiles/SmallFarm/Runs/Group1
export TRIALNAME_SPECIFIER='trial_3_10env'
export PAIRLABEL=''
export TYPE='FASTFARM'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory

export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_FASTFarm/analysis/Group1
export SUBDIRECTORY=../../RunFiles/SmallFarm/Runs/Group1
export TRIALNAME_SPECIFIER='trial_4_10env'
export PAIRLABEL=''
export TYPE='FASTFARM'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory

export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_FASTFarm/analysis/Group1
export SUBDIRECTORY=../../RunFiles/SmallFarm/Runs/Group1
export TRIALNAME_SPECIFIER='trial_5_10env'
export PAIRLABEL=''
export TYPE='FASTFARM'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory


# 2x4
export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_FASTFarm/analysis/Group1
export SUBDIRECTORY=../../RunFiles/MidFarm/Runs/Group1
export TRIALNAME_SPECIFIER='trial_3_10env_medium'
export PAIRLABEL=''
export TYPE='FASTFARM'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory

export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_FASTFarm/analysis/Group1
export SUBDIRECTORY=../../RunFiles/MidFarm/Runs/Group1
export TRIALNAME_SPECIFIER='trial_4_10env_medium'
export PAIRLABEL=''
export TYPE='FASTFARM'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory

export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_FASTFarm/analysis/Group1
export SUBDIRECTORY=../../RunFiles/MidFarm/Runs/Group1
export TRIALNAME_SPECIFIER='trial_5_10env_medium'
export PAIRLABEL=''
export TYPE='FASTFARM'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory


# Group 2

# 1x3
export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_FASTFarm/analysis/Group2
export SUBDIRECTORY=../../RunFiles/SmallFarm/Runs/Group2
export TRIALNAME_SPECIFIER='trial_3_10env'
export PAIRLABEL=''
export TYPE='FASTFARM'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory

export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_FASTFarm/analysis/Group2
export SUBDIRECTORY=../../RunFiles/SmallFarm/Runs/Group2
export TRIALNAME_SPECIFIER='trial_4_10env'
export PAIRLABEL=''
export TYPE='FASTFARM'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory

export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_FASTFarm/analysis/Group2
export SUBDIRECTORY=../../RunFiles/SmallFarm/Runs/Group2
export TRIALNAME_SPECIFIER='trial_5_10env'
export PAIRLABEL=''
export TYPE='FASTFARM'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory


# 2x4
export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_FASTFarm/analysis/Group2
export SUBDIRECTORY=../../RunFiles/MidFarm/Runs/Group2
export TRIALNAME_SPECIFIER='trial_3_10env_medium'
export PAIRLABEL=''
export TYPE='FASTFARM'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory

export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_FASTFarm/analysis/Group2
export SUBDIRECTORY=../../RunFiles/MidFarm/Runs/Group2
export TRIALNAME_SPECIFIER='trial_4_10env_medium'
export PAIRLABEL=''
export TYPE='FASTFARM'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory

export DIRECTORY=/ascldap/users/kbrown1/tscratch/Reinforcement_Learning/RLControl/windfarm_v0p12p20_FASTFarm/analysis/Group2
export SUBDIRECTORY=../../RunFiles/MidFarm/Runs/Group2
export TRIALNAME_SPECIFIER='trial_5_10env_medium'
export PAIRLABEL=''
export TYPE='FASTFARM'
NEW_FILENAME="combination_${TRIALNAME_SPECIFIER}.ipynb"
cp $TEMPLATE $DIRECTORY/$NEW_FILENAME
cd $DIRECTORY
jupyter nbconvert --to notebook --execute $NEW_FILENAME --output $NEW_FILENAME
cd "$original_directory"  # Return to the original directory
