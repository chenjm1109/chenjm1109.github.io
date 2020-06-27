# -*- coding: utf-8 -*-
"""
机器人工具箱

@author: Jinmin Chen 

email:522706601@qq.com (仅供邮件联系，不加QQ)
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.animation import FuncAnimation # Animation
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import robox as rb
import robotics_toolbox as rtb


def show_robot(ax, Tlist, scale_value = 1.0):
    """展示开链机器人
    :param ax: 绘图对象
    :param Tlist: 各连杆的变换矩阵，连杆坐标系原点应在近基座端的关节上，建议用x轴作为旋转轴方向
    :param scale_value: 绘图缩放系数
    :return: None
    """
    ## 绘制连杆
    for i in range(Tlist.shape[0] - 1):
        ax.plot([Tlist[i, 0, 3], Tlist[i + 1, 0, 3]], 
                [Tlist[i, 1, 3], Tlist[i + 1, 1, 3]],
                [Tlist[i, 2, 3], Tlist[i + 1, 2, 3]],
                color = "black", linewidth = 5.0)
    
    ## 绘制连杆坐标系
    print(Tlist)
    for i in range(Tlist.shape[0]):
        show_frame(ax, Tlist[i], scale_value)

def show_frame(ax, trans, scale_value = 1.0):
    '''
    Args:
        ax (TYPE): 3D axes object.
        trans (np.array): The transform matrix of custom frame in space frame.
        scale_value(float): The scale value of the length of each axis.

    Returns:
        None.
        
    Example Input:
        ax = Axes3D(fig)
        trans = np.array([[ 1,  0,  0, 1],
                          [ 0,  1,  0, 2],
                          [ 0,  0,  1, 3],
                          [ 0,  0,  0, 0]])
        scale_value = 1.0
    Output:

    '''
    R, p = rtb.trans_to_Rp(trans)
    color_list = ["red", "green", "blue"]
    for i in range(3):
        ax.plot([trans[0, 3], trans[0, 3] + trans[0, i] * scale_value],
                [trans[1, 3], trans[1, 3] + trans[1, i] * scale_value],
                [trans[2, 3], trans[2, 3] + trans[2, i] * scale_value],
                color = color_list[i], linewidth = 3.0)
    
def show_animate_line_2d(line_set, xlabel = "x", ylabel = "y", color = "orange"):
    fig2d = plt.figure(figsize=(6, 4), num = 2)
    plt.style.use('ggplot')
    ax2d = fig2d.add_subplot(1, 1, 1)
    step_num = line_set.shape[0]  
    step_list = np.linspace(0, step_num, step_num + 1)
    plt.ion()
    plt.grid('--')
    ax2d.set_xlim([0, step_num])
    ax2d.set_ylim([0, max(line_set) * 1.2])
    ax2d.set_xlabel(xlabel)
    ax2d.set_ylabel(ylabel)
    for i in range(step_num - 1):
        ax2d.plot([step_list[i], step_list[i + 1]], 
                       [line_set[i], line_set[i + 1]],
                        color = color)
        plt.pause(0.1)

def plot_position_workspace_surf(ori = np.array([0.0, 0.0, 0.0])):
    '''绘制位置工作空间包络面
    '''
    Robot = rb.Puma560()
    
    
    # 弧度-角度互转
    deg2rad= math.pi / 180.0
    rad2deg = 180.0 / math.pi
    step_num = 20 # 采样密度
    # zz = np.linspace(0.36, 0.6, step_num) # 定义z向高度序列(0.0, 0.0, 0.0)
    # zz = np.linspace(0.39, 0.565, step_num) # 定义z向高度序列(0.1, 0.0, 0.0)
    # zz = np.linspace(0.39, 0.565, step_num) # 定义z向高度序列(0.0, 0.1, 0.0)
    zz = np.linspace(-0.5, 0.5, step_num) # 定义z向高度序列(0.0, 0.0, 0.1)
    r_gama = np.zeros((step_num, step_num))
    boundx = np.zeros((step_num, step_num))
    boundy = np.zeros((step_num, step_num))
    boundz = np.zeros((step_num, step_num))
    for i in range(step_num):
        gama = np.linspace(0, 2 * math.pi, step_num)
        for j in range(step_num):
            r0 = 0.0
            rr = r0
            delta_r = 0
            lamda = 0.6
            r_step = lamda
            while(abs(r_step)>0.01):
                delta_r = r_step
                rr = r0 + delta_r
                xx = rr * math.cos(gama[j])
                yy = rr * math.sin(gama[j])
                T = rtb.pose_to_trans([xx, yy, zz[i], ori[0], ori[1], ori[2]])
                joint_config = Robot.inverse_kinematics_2(T)
                print(joint_config)
                if Robot.check_reachability(joint_config):
                    r0 = rr
                else:
                    lamda = 0.5 * lamda
                    r_step = lamda
            r_gama[i, j] = rr
        # print(r_step)
        boundx[i, :] = np.multiply(r_gama[i, :], np.cos(gama))
        boundy[i, :] = np.multiply(r_gama[i, :], np.sin(gama))
        boundz[i, :] = zz[i] * np.ones((1, step_num))
        
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xticks([-0.2, -0.1, 0.0, 0.1, 0.2])
    ax.set_yticks([-0.2, -0.1, 0.0, 0.1, 0.2])
    ax.set_zticks([0.3, 0.4, 0.5, 0.6])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    # ax.view_init(elev = 0.0, azim = -90.0)
    ax.view_init(elev = 20.0, azim = -45.0)
    surf = ax.plot_surface(boundx, boundy, boundz, cmap=cm.coolwarm,
                       linewidth=0.6, antialiased=False, shade=True,
                       edgecolors='k')
    # ax.plot_wireframe(boundx, boundy, boundz, rstride=1, cstride=1)
    # Customize the z axis.
    # ax.set_zlim(0.3, 0.6)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=10)
    
    plt.show()
    
if __name__ == "__main__":
    dist_set = np.load("dist_set.npy")
    print(dist_set)
    show_animate_line_2d(dist_set, "step", "distance (m)", "green")