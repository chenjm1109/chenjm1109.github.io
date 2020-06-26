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
        self.joint_limit = np.array([[0.0, math.pi]])
        # 可视化绘图


        self.if_show_robot = False
        ## 保护变量
        # 结构参数
        self._L1 = 1.5 # 杆1长度
        self._L2 = 1.0 # 杆2长度
        # 初始位形
        ## 保护变量
        _R = rtb.euler_to_matrix([0, 0, 0])
        _p = np.array([0, self._L2, self._L1])
        self._M = rtb.Rp_to_trans(_R, _p)
        self._S = np.array([[1, 0, 0, 0, self._L1, 0]]) # 螺旋轴
    
    def forward_kinematics(self, joint_config):
        # 判断关节值向量的长度是否合法
        joint_config = np.array(joint_config)
        link_num = self._S.shape[0]
        if joint_config.shape[0] != link_num :
            print("请确保关节值向量的长度为%d"%link_num)
            return
        # PoE方法
        link_trans_set = np.zeros((link_num, 4, 4)) # 记载每个连杆的位形
        for i in range(link_num):
            link_trans_set[i, :, :] = rtb.fkin_space(self._M, self._S, joint_config)
            
            
        # 可视化
        if self.if_show_robot:
            ax = self.ax3d
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            for i in range(link_num):
                M = np.copy(self._M)
                for j in range(link_num - 1, link_num - i - 2, -1):
                    # print(j)
                    M = np.dot(rtb.matrix_exp_6(rtb.vec_to_se3(self._S[j])), M)
                T = M
                # print(M)
                for j in range(link_num - 1, link_num - i - 2, -1):
                    print(j)
                    T = np.dot(rtb.matrix_exp_6(rtb.vec_to_se3(self._S[j] * joint_config[j])), T)
                    link_trans_set[i, :, :] = T
                link_trans_set[i, :, :] = T
            
            for i in range(link_num):
                ax.scatter(link_trans_set[i, 0, 3], 
                           link_trans_set[i, 1, 3], 
                           link_trans_set[i, 2, 3], 
                           color = "red")
            for i in range(link_num - 1):
                ax.plot([link_trans_set[i, 0, 3], link_trans_set[i + 1, 0, 3]],
                          [link_trans_set[i, 1, 3], link_trans_set[i + 1, 1, 3]],
                          [link_trans_set[i, 2, 3], link_trans_set[i + 1, 2, 3]],
                          color = "green")
        # print(link_trans_set)
        return T
     
    def inverse_kinematics(self):
        print(self._L1)
        
    def check_reachability(self, joint_config):
        for i in range(joint_config.shape[0]):
            if np.isnan(joint_config[i]):
                return False
            if joint_config[i] < self.joint_limit[i, 0] or joint_config[i] > self.joint_limit[i, 1]:
                return False
        return True
        
    def define_robot(self, M, S):
        '''Define custom robot.
        '''
        self.__M = M
        self.__S = S
        # 设定关节范围，默认为[0, math.pi]
        link_num = S.shape[0]
        self.joint_limit = np.zeros((link_num, 2))
        self.joint_limit[:, 1] = math.pi

class Puma560(Robot):
    '''
    '''
    def __init__(self, name = "puma560"):
        Robot.__init__(self, name)
        self.joint_limit = np.array([[-math.pi/2, math.pi/2],
                                     [-math.pi/2, math.pi/2],
                                     [-math.pi/2, math.pi/2],
                                     [-math.pi, math.pi],
                                     [-math.pi, math.pi],
                                     [-math.pi, math.pi]])
        ## 保护变量
        # 结构参数:
        self._a2 = 0.3
        self._a3 = 0.2
        self._d1 = 0.1
        # 初始位形
        self._M = np.array([[ 1.0, 0.0, 0.0, self._a2 + self._a3],
                            [ 0.0, 1.0, 0.0, self._d1],
                            [ 0.0, 0.0, 1.0, 0.0],
                            [ 0.0, 0.0, 0.0, 1.0]])
        
        self._S = np.array([[ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [ 0.0,-1.0, 0.0, 0.0, 0.0, 0.0],
                            [ 0.0,-1.0, 0.0, 0.0, 0.0,-self._a2],
                            [ 0.0, 0.0, 1.0, self._d1,-self._a2 - self._a3, 0.0],
                            [ 0.0, 1.0, 0.0, 0.0, 0.0, self._a2 + self._a3],
                            [ 1.0, 0.0, 0.0, 0.0, 0.0,-self._d1]])
    
    def inverse_kinematics_2(self, T, pose = 'luf'):
        joint_config = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        R, p = rtb.trans_to_Rp(T)
        px, py, pz = p
        if ('r' in pose):
            joint_config[0] = math.pi + math.atan2(py, px) + math.atan2(-(px**2+py**2-self._d1**2)**0.5, self._d1)
        else:
            joint_config[0] = math.atan2(py, px) - math.atan2(self._d1, (px**2+py**2-self._d1**2)**0.5)
        D = (px**2 + py**2 + pz**2 - self._d1**2 - self._a2**2 - self._a3**2)/(2 * self._a2 * self._a3)
        print(D)
        if ('d' in pose):
            joint_config[2] = math.atan2((1-D**2)**0.5, D)
        else:
            joint_config[2] = math.atan2(-(1-D**2)**0.5, D)
        joint_config[1] = math.atan2(pz, (px**2 + py**2 - self._d1**2)**0.5) - math.atan2(self._a3 * math.sin(joint_config[2]),
                                                                                          self._a2 + self._a3 * math.cos(joint_config[2]))
        ## 求解ZYX欧拉角
        # 判断多解
        if abs(R[2, 0] + 1) < 1e-5:
            joint_config[3] = math.atan2(R[0, 1], R[1, 1])
            joint_config[4] = math.pi / 2
            joint_config[5] = 0
        elif abs(R[2, 0] - 1) < 1e-5:
            joint_config[3] = -math.atan2(R[0, 1], R[1, 1])
            joint_config[4] = -math.pi / 2
            joint_config[5] = 0
        else:
            joint_config[3] = math.atan2(R[2, 1], R[2, 2])
            joint_config[4] = math.atan2(-R[2, 0], (R[0, 0]**2 + R[2, 1]**2)**0.5)
            joint_config[5] = math.atan2(R[1, 0], R[0, 0])
            if ('n' in pose):
                joint_config[3] += math.pi
                joint_config[4] = -joint_config[4]
                joint_config[5] += math.pi
        for i in range(3, 6):
            if joint_config[i] > math.pi:
                joint_config[i] -= 2 * math.pi
            elif joint_config[i] <= -math.pi:
                joint_config[i] += 2 * math.pi
        return joint_config
        
        
        
class Standford(Robot):
    def __init__(self, name = "Standford"):
        Robot.__init__(self, name)
        
class UR5_6R(Robot):
    '''Modern Robotics: Mechanics, Planning, and Control version2; (p145)
    '''
    def __init__(self, name = "UR5_6R"):
        Robot.__init__(self, name)
        ## 保护变量
        # 结构参数:
        self._L1 = 0.425
        self._L2 = 0.392
        self._W1 = 0.109
        self._W2 = 0.082
        self._H1 = 0.089
        self._H2 = 0.095
        # 初始位形
        self._M = np.array([[-1.0, 0.0, 0.0, self._L1 + self._L2],
                            [ 0.0, 0.0, 1.0, self._W1 + self._W2],
                            [ 0.0, 1.0, 0.0, self._H1 - self._H2],
                            [ 0.0, 0.0, 0.0, 1.0]])
        
        self._S = np.array([[ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [ 0.0, 1.0, 0.0,-self._H1, 0.0, 0.0],
                            [ 0.0, 1.0, 0.0,-self._H1, 0.0, self._L1],
                            [ 0.0, 1.0, 0.0,-self._H1, 0.0, self._L1 + self._L2],
                            [ 0.0, 0.0,-1.0,-self._W1, self._L1 + self._L2, 0.0],
                            [ 0.0, 1.0, 0.0, self._H1 - self._H2, 0.0, self._L1 + self._L2]]) # 螺旋轴
        
class Sacra(Robot):
    def __init__(self, name = "Sacra"):
        Robot.__init__(self, name)

class Stewart(Robot):
    def __init__(self, name = "puma560"):
        Robot.__init__(self, name)
        self._R1 = 0.5
        self._R2 = 0.4
        self._theta = 0.1885
        self._joint_config = np.zeros(6)
        self._joint_vector = np.zeros((3, 6))
        self._top_joint_init = np.zeros((3, 6))
        self._bottom_joint_init = np.zeros((3, 6))
        self._trans_init = np.array([[ 1.0, 0.0, 0.0, 0.0],
                                     [ 0.0, 1.0, 0.0, 0.0],
                                     [ 0.0, 0.0, 1.0, 0.484],
                                     [ 0.0, 0.0, 0.0, 1.0]])
        self._joint_limit = [0.4715, 0.6715]
        # self.ax3d.set_xlim()
        t = [-1.0, 1.0]
        for i in range(6):
            self._bottom_joint_init[:, i] = np.dot(rtb.euler_to_matrix([0.0, 0.0, 2 * (i//2) * math.pi / 3 + self._theta * t[i%2]]), 
                                                   np.array([self._R1, 0.0, 0.0]))
            self._top_joint_init[:, i] = np.dot(rtb.euler_to_matrix([0.0, 0.0, 2 * (i//2) * math.pi / 3 + (math.pi/3 - self._theta) * t[i%2]]), 
                                                np.array([self._R2, 0.0, 0.0]))
        self.inverse_kinematics(self._trans_init)
    
    def show_robot(self):
        ## plot leg vector
        plt.cla()
        plt.ion()
        self.ax3d.view_init(elev = 20.0, azim = -45.0)
        min_leg_vector = np.zeros((3, 6))
        for i in range(6):
            self.ax3d.plot([self._bottom_joint_init[0, i], self._bottom_joint_init[0, i] + self._joint_vector[0, i]],
                    [self._bottom_joint_init[1, i], self._bottom_joint_init[1, i] + self._joint_vector[1, i]],
                    [self._bottom_joint_init[2, i], self._bottom_joint_init[2, i] + self._joint_vector[2, i]],
                    color = 'black', linewidth = 5.0, zorder = 2)
            # plot min leg length
            min_leg_vector[:, i] = self._joint_vector[:, i] * self._joint_limit[0] / np.linalg.norm(self._joint_vector[:, i])
            self.ax3d.plot([self._bottom_joint_init[0, i], self._bottom_joint_init[0, i] + min_leg_vector[0, i]],
                    [self._bottom_joint_init[1, i], self._bottom_joint_init[1, i] + min_leg_vector[1, i]],
                    [self._bottom_joint_init[2, i], self._bottom_joint_init[2, i] + min_leg_vector[2, i]],
                    color = 'black', linewidth = 10.0, zorder = 2)
        ## plot platform
        self.ax3d.plot_trisurf(self._bottom_joint_init[0, :], self._bottom_joint_init[1, :], self._bottom_joint_init[2, :],
                        color = 'lightgray', antialiased = False, shade = False, zorder = 1)
        self.ax3d.plot_trisurf(self._bottom_joint_init[0, :] + self._joint_vector[0, :], 
                        self._bottom_joint_init[1, :] + self._joint_vector[1, :],
                        self._bottom_joint_init[2, :] + self._joint_vector[2, :],
                        color = 'lightgray', antialiased = False, shade = False, zorder = 6)
        
        for i in range(6):
            self.ax3d.plot([self._bottom_joint_init[0, i], self._bottom_joint_init[0, (i + 1) % 6]],
                    [self._bottom_joint_init[1, i], self._bottom_joint_init[1, (i + 1) % 6]],
                    [self._bottom_joint_init[2, i], self._bottom_joint_init[2, (i + 1) % 6]],
                    color = "crimson", linewidth = 10.0, zorder = 1)
            self.ax3d.plot([self._bottom_joint_init[0, i] + self._joint_vector[0, i], self._bottom_joint_init[0, (i + 1) % 6] + self._joint_vector[0, (i + 1) % 6]],
                    [self._bottom_joint_init[1, i] + self._joint_vector[1, i], self._bottom_joint_init[1, (i + 1) % 6] + self._joint_vector[1, (i + 1) % 6]],
                    [self._bottom_joint_init[2, i] + self._joint_vector[2, i], self._bottom_joint_init[2, (i + 1) % 6] + self._joint_vector[2, (i + 1) % 6]],
                    color = "crimson", linewidth = 5.0, zorder = 3)
        self.ax3d.set_xlim([-0.5, 0.5])
        self.ax3d.set_ylim([-0.5, 0.5])
        self.ax3d.set_zlim([ 0, 1])
        self.ax3d.set_xlabel("x (m)")
        self.ax3d.set_ylabel("y (m)")
        self.ax3d.set_zlabel("z (m)")
        
    def show_joint_path(self, trans_set):
        fig2d = plt.figure(figsize=(8, 4), num = 2)
        plt.style.use('ggplot')
        self.ax2d = fig2d.add_subplot(1, 1, 1)
        step_num = trans_set.shape[0]  
        step_list = np.linspace(0, step_num, step_num + 1)
        joint_config_set = np.zeros((step_num, 6))
        plt.ion()
        plt.grid('--')
        plt.pause(1.0)
        self.ax2d.set_xlim([0, step_num])
        self.ax2d.set_ylim([self._joint_limit[0] * 0.9, self._joint_limit[1] * 1.1])
        self.ax2d.set_xlabel("step")
        self.ax2d.set_ylabel("joint config (m)")
        color_list = ['red', 'green', 'blue', 'orange', 'black', 'purple']
        for i in range(2):
            self.ax2d.plot([0, step_num], [self._joint_limit[i], self._joint_limit[i]], color = "red", linestyle = "--")
        ## Solve the joint path
        for i in range(step_num):
            self.inverse_kinematics(trans_set[i])
            joint_config_set[i, :] = self._joint_config
        ## Plot the joint path
        # print(joint_config_set)
        for i in range(step_num - 1):
            for j in range(6):
                self.ax2d.plot([step_list[i], step_list[i + 1]], 
                               [joint_config_set[i, j], joint_config_set[i + 1, j]],
                               color = color_list[j])
            plt.pause(0.1)
            
    def inverse_kinematics(self, trans):
        R, p = rtb.trans_to_Rp(trans)
        for i in range(6):
            self._joint_vector[:, i] = - self._bottom_joint_init[:, i] + p + np.dot(R, self._top_joint_init[:, i])
            self._joint_config[i] = np.linalg.norm(self._joint_vector[:, i])


    def show_animate(self, trans_set):
        fig3d = plt.figure(figsize=(6, 6), num = 1)
        self.ax3d = Axes3D(fig3d)
        step_num = trans_set.shape[0]
        for i in range(step_num):
            self.inverse_kinematics(trans_set[i])
            self.show_robot()
            plt.pause(0.1)

if __name__ == "__main__":
    A = Stewart()
    trans_set = np.load("trans_set.npy")
    # A.show_animate(trans_set)
    A.show_joint_path(trans_set)
    # A.show()
