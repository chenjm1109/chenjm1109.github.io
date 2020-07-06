# -*- coding: utf-8 -*-
"""
机器人工具箱

@author: Jinmin Chen 

email:522706601@qq.com (仅供邮件联系，不加QQ)
"""

import numpy as np
np.set_printoptions(suppress=True)
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

import robotics_toolbox as rtb
import plot_toolbox as ptb


class Robot:
    def __init__(self, name = "xm_robot"):
        ## 公有变量
        self.name = name
        self.joint_limit = np.array([[-2 * math.pi, 2 * math.pi]])
        
        ## 绘图参数
        self._plot_limit = np.array([[-10, 10],
                                     [-10, 10],
                                     [-10, 10]])

        # 结构参数
        M00 = np.array([[1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, -2.0],
                        [0.0, 0.0, 0.0, 1.0]])
        M01 = np.array([[1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]])
        M02 = np.array([[1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 6.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]])
        M03 = np.array([[1.0, 0.0, 0.0, 0.0 ],
                        [0.0, 1.0, 0.0, 11.0],
                        [0.0, 0.0, 1.0, 0.0 ],
                        [0.0, 0.0, 0.0, 1.0 ]])
        
        G1 = np.diag([12.25, 0.5, 12.25, 1.0, 1.0, 1.0])
        G2 = np.diag([8.58, 0.5, 8.58, 1.0, 1.0, 1.0])
        G3 = np.diag([1, 1, 1, 1, 1, 1])
        self._Glist = np.array([G1, G2, G3])
        self._Mlist = np.array([M00, M01, M02, M03])
        self._Slist = np.array([[1, 0, 0, 0, 0,  0],
                                [1, 0, 0, 0, 0, -6.0],
                                [1, 0, 0, 0, 0, -11.0]]).T # 螺旋轴
    
    def redefine(self, Mlist, Slist, Glist):
        self._Mlist = Mlist
        self._Slist = Slist
        self._Glist = Glist
        
    def show_robot(self):
        link_num = self._Slist.shape[1]
        thetalist = np.zeros(link_num)
        self.forward_kinematics(thetalist, show = True)
    
    def forward_kinematics(self, thetalist, show = False):
        '''
        

        Args:
            thetalist (TYPE): DESCRIPTION.
            show (TYPE, optional): DESCRIPTION. Defaults to False.

        Returns:
            Tlist (TYPE): DESCRIPTION.

        '''
        ## 判断关节值向量的长度是否合法
        thetalist = np.array(thetalist)
        link_num = self._Slist.shape[1]
        if thetalist.shape[0] != link_num :
            print("请确保关节值向量的长度为%d"%link_num)
            return
        ## 求解正运动学
        Tlist = rtb.fkin_link_space(self._Mlist, self._Slist, thetalist)
        
        ## 可视化
        if show:
            fig3d = plt.figure(figsize=(6, 6), num = 1)
            self.ax3d = Axes3D(fig3d)
            ax = self.ax3d
            ax.set_xlim(self._plot_limit[0])
            ax.set_ylim(self._plot_limit[1])
            ax.set_zlim(self._plot_limit[2])
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ptb.show_robot(ax, Tlist, scale_value = 2.0)
        return Tlist
     
    def inverse_kinematics(self, T, thetalist0, show = False , eomg = 0.01, ev = 0.001, maxiterations = 20):
        thetalist, if_success = rtb.ikin_space(self._Slist, self._Mlist[-1], T, 
                                               thetalist0, eomg, ev,
                                               maxiterations)
        if if_success:
            if show:
                self.forward_kinematics(thetalist, show = True)
            return thetalist
        else:
            print("没有找到解")
            return False
        
    def simulate_control(self):
        ''' 计算力矩控制仿真
        '''
        thetalist = np.array([0.1, 0.1, 0.1])
        dthetalist = np.array([0.0, 0.0, 0.0])
        ## 初始化机器人的参数，以三自由度串联机器人为例
        g = np.array([0, 0, -9.8]) # 重力加速度
        # 零位时，相邻连杆间的变换矩阵
        M01 = self._Mlist[1]
        M12 = np.dot(rtb.trans_inv(self._Mlist[1]) ,self._Mlist[2])
        M23 = np.dot(rtb.trans_inv(self._Mlist[2]) ,self._Mlist[3])
        M34 = np.array([[1, 0, 0,  0],
                        [0, 1, 0,  0],
                        [0, 0, 1,  0],
                        [0, 0, 0,  1]])
        # 各连杆的惯量矩阵
        Glist = self._Glist
        Mlist = np.array([M01, M12, M23, M34])
        # 各关节运动旋量
        Slist = self._Slist
        dt = 0.01 # 关节轨迹位置间的时间间隔
        ## 创建用于跟踪的轨迹
        thetaend = np.array([np.pi / 2, np.pi, 1.5 * np.pi])
        Tf = 1.0
        N = int(1.0 * Tf / dt)
        method = 5
        traj = rtb.joint_trajectory(thetalist, thetaend, Tf, N, method)
        thetamatd = np.array(traj).copy()
        dthetamatd = np.zeros((N, 3))
        ddthetamatd = np.zeros((N, 3))
        dt = Tf / (N - 1.0)
        for i in range(np.array(traj).shape[0] - 1):
            dthetamatd[i + 1, :] = (thetamatd[i + 1, :] - thetamatd[i, :]) / dt
            ddthetamatd[i + 1, :] = (dthetamatd[i + 1, :] - dthetamatd[i, :]) / dt
        # Possibly wrong robot description (Example with 3 links)
        gtilde = np.array([0.0, 0.0, -9.8])
        Mhat01 = np.array([[1, 0, 0,   0.],
                           [0, 1, 0,   0.],
                           [0, 0, 1,   0.],
                           [0, 0, 0,   1]])
        Mhat12 = np.array([[ 0, 0, 1, 0.],
                           [ 0, 1, 0, 6.],
                           [-1, 0, 0,   0],
                           [ 0, 0, 0,   1]])
        Mhat23 = np.array([[1, 0, 0,    0],
                           [0, 1, 0,   5.],
                           [0, 0, 1,    0.],
                           [0, 0, 0,    1]])
        Mhat34 = np.array([[1, 0, 0,   0],
                           [0, 1, 0,   0.],
                           [0, 0, 1, 0.],
                           [0, 0, 0,   1]])
        Ghat1 = np.diag([12.25, 0.5, 12.25, 1.0, 1.0, 1.0])
        Ghat2 = np.diag([8.58, 0.5, 8.58, 1.0, 1.0, 1.0])
        Ghat3 = np.diag([1, 1., 1., 1, 1, 1])
        Gtildelist = np.array([Ghat1, Ghat2, Ghat3])
        Mtildelist = np.array([Mhat01, Mhat12, Mhat23, Mhat34])
        Ftipmat = np.ones((np.array(traj).shape[0], 6))
        Kp = 20
        Ki = 0
        Kd = 15
        intRes = 8
        taumat,thetamat \
        = rtb.simulate_control(thetalist, dthetalist, g, Ftipmat, Mlist, \
                             Glist, Slist, thetamatd, dthetamatd, \
                             ddthetamatd, gtilde, Mtildelist, Gtildelist, \
                             Kp, Ki, Kd, dt, intRes, method = "computed_torque")


class Puma560(Robot):
    def __init__(self, name = "xm_robot"):
        ## 公有变量
        self.name = name
        self.joint_limit = np.array([[-2 * math.pi, 2 * math.pi]])
        
        ## 绘图参数
        self._plot_limit = np.array([[-10, 10],
                                     [-10, 10],
                                     [-10, 10]])

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
        G1 = np.diag([1, 1, 1, 1, 1, 1])
        G2 = np.diag([1, 1, 1, 1, 1, 1])
        G3 = np.diag([1, 1, 1, 1, 1, 1])
        G4 = np.diag([1, 1, 1, 1, 1, 1])
        G5 = np.diag([1, 1, 1, 1, 1, 1])
        G6 = np.diag([1, 1, 1, 1, 1, 1])
        self._Mlist = np.array([M00, M01, M02, M03, M04, M05, M06])
        ## 各关节螺旋轴
        self._Slist = np.array([[ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                [ 1.0, 0.0, 0.0, 0.0, 5.0, 0.0],
                                [ 1.0, 0.0, 0.0, 0.0, 5.0,-4.0],
                                [ 1.0, 0.0, 0.0, 0.0, 5.0,-7.0],
                                [ 0.0, 0.0, 1.0, 7.0, 0.0, 0.0],
                                [ 0.0, 1.0, 0.0,-5.0, 0.0, 0.0]]).T
        self._Glist = np.array([G1, G2, G3, G4, G5, G6])
        
    
if __name__ == "__main__":
    A = Robot()
    # thetalist = A.inverse_kinematics(T, thetalist0, maxiterations = 100)
    # print(thetalist)
    A.simulate_control()
