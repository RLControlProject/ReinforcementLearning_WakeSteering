import os

import numpy as np

THISDIR = os.path.dirname(os.path.abspath(__file__))
uinf: float = 10.0
refheight: float = 90.0
veerperheight: float = -20.0 / 126.0


def addveer(seedpath):
    # Read the low.inp file
    with open(os.path.join(seedpath, "Low.inp"), "r") as f:
        lowlines = f.readlines()
    gridheight = float(lowlines[24].split()[0])
    writeusr(seedpath, gridheight)

    lowlines[36] = f"USR  " + lowlines[36].lstrip().split(" ", 1)[1]
    lowlines[37] = f"usr.txt    " + lowlines[37].lstrip().split(" ", 1)[1]

    with open(os.path.join(seedpath, "Low.inp"), "w") as f:
        f.writelines(lowlines)


def writeusr(seedpath, gridheight):
    with open(os.path.join(THISDIR, "sourcefiles", "userinput.txt"), "r") as f:
        usrlines = f.readlines()
    nheaderlines = len(usrlines)

    numpoints = 20
    heights = np.linspace(0, gridheight, numpoints)
    for i, height in enumerate(heights):
        usrlines.append(
            f"{height:3.2f} {uinf:3.2f} {(height-refheight) * veerperheight:5.3f} 0 0\n"
        )
    usrlines[3] = f"{numpoints}  " + usrlines[3].lstrip().split(" ", 1)[1]

    with open(os.path.join(seedpath, "usr.txt"), "w") as f:
        f.writelines(usrlines)
