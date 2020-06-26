# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 20:56:18 2020

@author: 52270
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


import plot_toolbox as ptb
import robotics_toolbox as rtb
import xm_robot as xr


if __name__ == "__main__":
    fig = plt.figure(figsize=(4, 4))
    ax = Axes3D(fig)
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.set_zlim([0, 10])
    pose_1 = [1,2,3,1,0,0]
    pose_2 = [1,2,3,1,1,0]
    trans_1 = rtb.pose_to_trans(pose_1)
    trans_2 = rtb.pose_to_trans(pose_2)
    ptb.show_frame(ax, trans_1)
    ptb.show_frame(ax, trans_2)