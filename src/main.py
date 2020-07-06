# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 20:56:18 2020

@author: 52270
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

import robox as rb

def demo_p3():
    '''示例：机器人的建模与可视化
    '''
    MyRobot = rb.Robot()
    ## 定义各连杆坐标系的初始坐位形Mlist
    M00 = np.array([[1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0,-5.0],
                    [0.0, 0.0, 0.0, 1.0]])
    M01 = np.array([[0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])
    M02 = np.array([[1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 5.0],
                    [0.0, 0.0, 0.0, 1.0]])
    M03 = np.array([[1.0, 0.0, 0.0, 0.0 ],
                    [0.0, 1.0, 0.0, 4.0],
                    [0.0, 0.0, 1.0, 5.0 ],
                    [0.0, 0.0, 0.0, 1.0 ]])
    M04 = np.array([[1.0, 0.0, 0.0, 0.0 ],
                    [0.0, 1.0, 0.0, 7.0],
                    [0.0, 0.0, 1.0, 5.0 ],
                    [0.0, 0.0, 0.0, 1.0 ]])
    M05 = np.array([[0.0, 1.0, 0.0, 0.0 ],
                    [0.0, 0.0, 1.0, 7.0],
                    [1.0, 0.0, 0.0, 5.0 ],
                    [0.0, 0.0, 0.0, 1.0 ]])
    M06 = np.array([[0.0, -1.0, 0.0, 0.0 ],
                    [1.0, 0.0, 0.0, 7.0],
                    [0.0, 0.0, 1.0, 5.0 ],
                    [0.0, 0.0, 0.0, 1.0 ]])
    Mlist = np.array([M00, M01, M02, M03, M04, M05, M06])
    ## 各关节螺旋轴
    Slist = np.array([[ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                      [ 1.0, 0.0, 0.0, 0.0, 5.0, 0.0],
                      [ 1.0, 0.0, 0.0, 0.0, 5.0,-4.0],
                      [ 1.0, 0.0, 0.0, 0.0, 5.0,-7.0],
                      [ 0.0, 0.0, 1.0, 7.0, 0.0, 0.0],
                      [ 0.0, 1.0, 0.0,-5.0, 0.0, 0.0]]).T # 螺旋轴
    ## 连杆惯量矩阵
    G1 = np.diag([1, 1, 1, 1, 1, 1])
    G2 = np.diag([1, 1, 1, 1, 1, 1])
    G3 = np.diag([1, 1, 1, 1, 1, 1])
    G4 = np.diag([1, 1, 1, 1, 1, 1])
    G5 = np.diag([1, 1, 1, 1, 1, 1])
    G6 = np.diag([1, 1, 1, 1, 1, 1])
    Glist = np.array([G1, G2, G3, G4, G5, G6])
    ## 调用redefine方法定义Puma560机器人
    MyRobot.redefine(Mlist, Slist, Glist)
    MyRobot.show_robot()

def demo_p4():
    '''示例：机器人的正向运动学
    '''
    MyRobot = rb.Puma560()
    Tlist = MyRobot.forward_kinematics([1, 1, 1, 0, 0, 0], True)
    print(Tlist[-1])

def demo_p5():
    '''示例：机器人的逆运动学————数值法
    '''
    MyRobot = rb.Puma560()
    T = np.array([[1.0, 0.0, 0.0,-0.7 ],
                  [0.0, 1.0, 0.0, 0.5],
                  [0.0, 0.0, 1.0, 11.0 ],
                  [0.0, 0.0, 0.0, 1.0 ]])
    thetalist = MyRobot.inverse_kinematics(T, thetalist0 = [0,0,0,0,0,0], show = True)
    print(thetalist)


def demo_p6():
    '''示例：机器人的逆运动学————基于正解
    '''
    MyRobot = rb.Puma560()
    theta_1 = np.linspace(0, 1 * math.pi, 10)
    theta_2 = np.linspace(0, 1 * math.pi, 10)
    theta_3 = np.linspace(0, 1 * math.pi, 10)
    point = [] # 末端点位置列表
    for i in theta_1:
        for j in theta_2:
            for k in theta_3:
                Tlist = MyRobot.forward_kinematics([i, 0, k, 0, 0, 0], False)
                point.append([Tlist[-1, 0, 3], Tlist[-1, 1, 3], Tlist[-1, 2, 3]])
    point = np.array(point)
    fig3d = plt.figure(figsize=(6, 6), num = 1)
    ax = Axes3D(fig3d)
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-10, 10])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.scatter(point[:, 0], point[:, 1], point[:, 2], color = "green")

if __name__ == "__main__":
    demo_p6()