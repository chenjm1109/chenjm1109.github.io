# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 10:41:03 2020

@author: 52270
"""


import numpy as np

np.set_printoptions(suppress=True, precision=2)

def cylinder_inertia_matrix(radius, height, mass, ref_point):
    '''
    

    Args:
        radius (TYPE): DESCRIPTION.
        height (TYPE): DESCRIPTION.
        mass (TYPE): DESCRIPTION.
        ref_point (TYPE): DESCRIPTION.

    Returns:
        None.
        
    Example Input:
        radius = 1.0
        height = 2.0
        mass = 6283.185
        ref_point = [0.0, - height / 2, 0.0]
        G = cylinder_inertia_matrix(radius, height, mass, ref_point)
        print(G)
        
    Output:
        [[9948.38    0.      0.      0.      0.      0.  ]
         [   0.   3141.59    0.      0.      0.      0.  ]
         [   0.      0.   9948.38    0.      0.      0.  ]
         [   0.      0.      0.   6283.19    0.      0.  ]
         [   0.      0.      0.      0.   6283.19    0.  ]
         [   0.      0.      0.      0.      0.   6283.19]]

    '''
    r = radius
    h = height
    m = mass
    q = np.array([ref_point]).T
    Ixx = m * (3 * r**2 + h**2) / 12
    Iyy = m * r**2 / 2
    Izz = m * (3 * r**2 + h**2) / 12
    Ib = np.diag([Ixx, Iyy, Izz]) # 质心惯性矩阵
    I = np.eye(3)
    Iq = Ib + m * (np.dot(q.T, q) * I - np.dot(q, q.T))
    G = np.r_[np.c_[Iq, np.zeros((3,3))], np.c_[np.zeros((3,3)), I * m]]
    return G
    
if __name__ == "__main__":
    radius = 1.0
    height = 2.0
    mass = 0.2
    ref_point = [0.0, 0.0, 0.0]
    G = cylinder_inertia_matrix(radius, height, mass, ref_point)
    print(G)
    
    
    
    
    
    