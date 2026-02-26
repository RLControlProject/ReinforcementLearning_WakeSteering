
__generated_with = "0.13.6"

# %%
import numpy as np
import matplotlib.pyplot as plt
from helpers import generate_layout_grid

farmtype = 'mid' # small, mid, large, larger
case = 'yawseed1'

# %%
# Define the layout 

initoffset = {}
if farmtype == 'small':
    num_points_x = 3
    num_points_y = 1    
    initoffset['yawseed1'] = np.flip([-7.99173267,15.26256441,-3.58842838])
    initoffset['yawseed2'] = np.flip([0.77181546, -21.61578551, -5.24329785])
    initoffset['yawseed3'] = np.flip([-7.63071445, -4.90298986, -3.3343338])
elif farmtype == 'mid':
    num_points_x = 4
    num_points_y = 2    
    initoffset['yawseed1'] = np.flip([-7.99173267,15.26256441,-3.58842838, 4.35906436,-6.25316816 ,-2.48056459,20.21271915, 1.01642442])
    initoffset['yawseed2'] = np.flip([  0.77181546, -21.61578551,  -5.24329785,   5.18893957 , -4.7638078,11.24794039,-6.20365898,  2.45541373])
    initoffset['yawseed3'] = np.flip([-7.63071445, -4.90298986, -3.3343338,  -0.33912956,  3.01972138,  5.79925537 ,4.70287494, -1.13058065])

elif farmtype == 'large':
    num_points_x = 6
    num_points_y = 3
    initoffset['yawseed1'] = np.zeros(num_points_x*num_points_y)
    initoffset['yawseed2'] = np.zeros(num_points_x*num_points_y)
    initoffset['yawseed3'] = np.zeros(num_points_x*num_points_y)
elif farmtype == 'larger':
    num_points_x = 8
    num_points_y = 4
    initoffset['yawseed1'] = np.zeros(num_points_x*num_points_y)
    initoffset['yawseed2'] = np.zeros(num_points_x*num_points_y)
    initoffset['yawseed3'] = np.zeros(num_points_x*num_points_y)


spacing_x = 4.0*126
spacing_y = 3.0*126
layout_grid = generate_layout_grid(num_points_x, num_points_y, spacing_x, spacing_y)
num_turbines = len(layout_grid)


# Helper function to rotate the top and bottom of the rotor line
def rotateturbine(x,y,angle):
    r = 64
    angr = np.deg2rad(angle)
    vecbot_0 = [0,-1*r]
    vectop_0 = [0,r]
    rot = np.array([[np.cos(angr),-np.sin(angr)],
                    [np.sin(angr),np.cos(angr)]])
    vecbot_1 = np.matmul(rot,vecbot_0) + [x,y]
    vectop_1 = np.matmul(rot,vectop_0) + [x,y]
    return vecbot_1,vectop_1


# %%
x_lims = []
y_lims = []
plt.figure(figsize=(8,8))   # If set_aspect is used later, the width from this line will be overwritten
for i in range(num_turbines):
    x = layout_grid[i][0]
    y = layout_grid[i][1]

    if x_lims == []:
        x_lims.append(x)
        x_lims.append(x)
        y_lims.append(y)
        y_lims.append(y)
    else:
        if x < x_lims[0]:
            x_lims[0] = x
        if x > x_lims[1]:
            x_lims[1] = x
        if y < y_lims[0]:
            y_lims[0] = y
        if y > y_lims[1]:
            y_lims[1] = y

    # alpha_base = 0.3
    # vecbot,vectop = rotateturbine(x,y,0)
    # plt.plot([vecbot[0],vectop[0]],[vecbot[1],vectop[1]],'k',alpha = alpha_base)
    # plt.plot(x+12,y,'ks',ms = 4,alpha = alpha_base)

    # vecbot,vectop = rotateturbine(x,y,0)
    vecbot,vectop = rotateturbine(x,y,initoffset[case][i])
    plt.plot([vecbot[0],vectop[0]],[vecbot[1],vectop[1]],'k')
    plt.plot(x+12,y,'ks',ms = 4)

border = 500
plt.xlim([x_lims[0]-border,x_lims[1] + border])
print(y_lims)
plt.ylim([y_lims[0]-border,y_lims[1] + border])

# Get the current Axes object
ax = plt.gca() 

# Set the aspect ratio to 'equal'
ax.set_aspect('equal', adjustable='box') 

plt.show()

# %%

