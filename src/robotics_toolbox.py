#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
************************************************************************
机器人学基本函数
************************************************************************
Author: Jinmin CHEN
Email:  chenjm1109@foxmail.com
Date:   June, 2020
************************************************************************
'''

import numpy as np
import math

'''
*** 基本功能函数 ***
'''

def near_zero(x):
    '''判断所输入的标量是否足够小以至于可以视作0

    :param x: A scalar input to check
    :return: True if z is close to zero, false otherwise

    Example Input:
        x = -1e-7
    Output:
        True
    '''
    return abs(x) < 1e-6

def normalize(vec):
    '''将所输入的向量单位化

    :param vec: A vector
    :return: A unit vector pointing in the same direction as z

    Example Input:
        vec = np.array([1, 2, 3])
    Output:
        np.array([0.26726124, 0.53452248, 0.80178373])
    '''
    return vec / np.linalg.norm(vec)

def rot_inv(R):
    """计算旋转矩阵的逆矩阵
    :param R: A rotation matrix
    :return: The inverse of R
    Example Input:
        R = np.array([[0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0]])
    Output:
        np.array([[0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 0]])
    """
    return np.array(R).T

def trans_inv(T):
    """计算齐次变换矩阵的逆矩阵
    :param T: A homogeneous transformation matrix
    :return: The inverse of T
    Uses the structure of transformation matrices to avoid taking a matrix
    inverse, for efficiency.
    Example input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        np.array([[1,  0, 0,  0],
                  [0,  0, 1, -3],
                  [0, -1, 0,  0],
                  [0,  0, 0,  1]])
    """
    R, p = trans_to_Rp(T)
    Rt = np.array(R).T
    return np.r_[np.c_[Rt, -np.dot(Rt, p)], [[0, 0, 0, 1]]]

def project_to_SO3(mat):
    """Returns a projection of mat into SO(3)
    :param mat: A matrix near SO(3) to project to SO(3)
    :return: The closest matrix to R that is in SO(3)
    Projects a matrix mat to the closest matrix in SO(3) using singular-value
    decomposition (see
    http://hades.mech.northwestern.edu/index.php/Modern_Robotics_Linear_Algebra_Review).
    This function is only appropriate for matrices close to SO(3).
    Example Input:
        mat = np.array([[ 0.675,  0.150,  0.720],
                        [ 0.370,  0.771, -0.511],
                        [-0.630,  0.619,  0.472]])
    Output:
        np.array([[ 0.67901136,  0.14894516,  0.71885945],
                  [ 0.37320708,  0.77319584, -0.51272279],
                  [-0.63218672,  0.61642804,  0.46942137]])
    """
    U, s, Vh = np.linalg.svd(mat)
    R = np.dot(U, Vh)
    if np.linalg.det(R) < 0:
    # In this case the result may be far from mat.
        R[:, s[2, 2]] = -R[:, s[2, 2]]
    return R

def project_to_SE3(mat):
    """Returns a projection of mat into SE(3)
    :param mat: A 4x4 matrix to project to SE(3)
    :return: The closest matrix to T that is in SE(3)
    Projects a matrix mat to the closest matrix in SE(3) using singular-value
    decomposition (see
    http://hades.mech.northwestern.edu/index.php/Modern_Robotics_Linear_Algebra_Review).
    This function is only appropriate for matrices close to SE(3).
    Example Input:
        mat = np.array([[ 0.675,  0.150,  0.720,  1.2],
                        [ 0.370,  0.771, -0.511,  5.4],
                        [-0.630,  0.619,  0.472,  3.6],
                        [ 0.003,  0.002,  0.010,  0.9]])
    Output:
        np.array([[ 0.67901136,  0.14894516,  0.71885945,  1.2 ],
                  [ 0.37320708,  0.77319584, -0.51272279,  5.4 ],
                  [-0.63218672,  0.61642804,  0.46942137,  3.6 ],
                  [ 0.        ,  0.        ,  0.        ,  1.  ]])
    """
    mat = np.array(mat)
    return Rp_to_trans(project_to_SO3(mat[:3, :3]), mat[:3, 3])

def distance_to_SO3(mat):
    """Returns the Frobenius norm to describe the distance of mat from the
    SO(3) manifold
    :param mat: A 3x3 matrix
    :return: A quantity describing the distance of mat from the SO(3)
             manifold
    Computes the distance from mat to the SO(3) manifold using the following
    method:
    If det(mat) <= 0, return a large number.
    If det(mat) > 0, return norm(mat^T.mat - I).
    Example Input:
        mat = np.array([[ 1.0,  0.0,   0.0 ],
                        [ 0.0,  0.1,  -0.95],
                        [ 0.0,  1.0,   0.1 ]])
    Output:
        0.08835
    """
    if np.linalg.det(mat) > 0:
        return np.linalg.norm(np.dot(np.array(mat).T, mat) - np.eye(3))
    else:
        return 1e+9

def distance_to_SE3(mat):
    """Returns the Frobenius norm to describe the distance of mat from the
    SE(3) manifold
    :param mat: A 4x4 matrix
    :return: A quantity describing the distance of mat from the SE(3)
              manifold
    Computes the distance from mat to the SE(3) manifold using the following
    method:
    Compute the determinant of matR, the top 3x3 submatrix of mat.
    If det(matR) <= 0, return a large number.
    If det(matR) > 0, replace the top 3x3 submatrix of mat with matR^T.matR,
    and set the first three entries of the fourth column of mat to zero. Then
    return norm(mat - I).
    Example Input:
        mat = np.array([[ 1.0,  0.0,   0.0,   1.2 ],
                        [ 0.0,  0.1,  -0.95,  1.5 ],
                        [ 0.0,  1.0,   0.1,  -0.9 ],
                        [ 0.0,  0.0,   0.1,   0.98 ]])
    Output:
        0.134931
    """
    matR = np.array(mat)[0: 3, 0: 3]
    if np.linalg.det(matR) > 0:
        return np.linalg.norm(np.r_[np.c_[np.dot(np.transpose(matR), matR),
                                          np.zeros((3, 1))],
                              [np.array(mat)[3, :]]] - np.eye(4))
    else:
        return 1e+9

def test_if_SO3(mat):
    """Returns true if mat is close to or on the manifold SO(3)
    :param mat: A 3x3 matrix
    :return: True if mat is very close to or in SO(3), false otherwise
    Computes the distance d from mat to the SO(3) manifold using the
    following method:
    If det(mat) <= 0, d = a large number.
    If det(mat) > 0, d = norm(mat^T.mat - I).
    If d is close to zero, return true. Otherwise, return false.
    Example Input:
        mat = np.array([[1.0, 0.0,  0.0 ],
                        [0.0, 0.1, -0.95],
                        [0.0, 1.0,  0.1 ]])
    Output:
        False
    """
    return abs(distance_to_SO3(mat)) < 1e-3

def test_if_SE3(mat):
    """Returns true if mat is close to or on the manifold SE(3)
    :param mat: A 4x4 matrix
    :return: True if mat is very close to or in SE(3), false otherwise
    Computes the distance d from mat to the SE(3) manifold using the
    following method:
    Compute the determinant of the top 3x3 submatrix of mat.
    If det(mat) <= 0, d = a large number.
    If det(mat) > 0, replace the top 3x3 submatrix of mat with mat^T.mat, and
    set the first three entries of the fourth column of mat to zero.
    Then d = norm(T - I).
    If d is close to zero, return true. Otherwise, return false.
    Example Input:
        mat = np.array([[1.0, 0.0,   0.0,  1.2],
                        [0.0, 0.1, -0.95,  1.5],
                        [0.0, 1.0,   0.1, -0.9],
                        [0.0, 0.0,   0.1, 0.98]])
    Output:
        False
    """
    return abs(distance_to_SE3(mat)) < 1e-3

'''
*** 数学方法 ***
'''

def euler_to_matrix(euler):
    '''Converts Euler angles to rotation matrix.
    :param euler: The Euler angles
    Example Input:
        euler = np.array([0, 0, 0])
    Output:
        np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    '''
    mat_x = np.array([[ 1, 0,                   0                 ],
                      [ 0, math.cos(euler[0]), -math.sin(euler[0])],
                      [ 0, math.sin(euler[0]),  math.cos(euler[0])]])

    mat_y = np.array([[ math.cos(euler[1]), 0, math.sin(euler[1])],
                      [ 0,                  1, 0                 ],
                      [-math.sin(euler[1]), 0, math.cos(euler[1])]])
    
    mat_z = np.array([[ math.cos(euler[2]), -math.sin(euler[2]), 0],
                      [ math.sin(euler[2]),  math.cos(euler[2]), 0],
                      [ 0,                   0,                  1]])
    return np.dot(mat_z, np.dot(mat_y, mat_x))
    
def matrix_to_euler(R):
    '''Converts rotation matrix to Euler angles.
    :matrix: Rotation matrix
    Example Input:
        matrix = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
    Output:
        np.array([0, 0, 0])
    '''
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = (sy < 1e-6)
    if (not singular):
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def vec_to_so3(vec):
    '''Converts a 3-vector to an so(3) representation

    :param vec: A 3-vector
    :return: The skew symmetric representation of vec

    Example Input:
        vec = np.array([1, 2, 3])
    Output:
        np.mat([[ 0, -3,  2],
                [ 3,  0, -1],
                [-2,  1,  0]])
    '''
    return np.array([[ 0,     -vec[2],  vec[1]],
                     [ vec[2],      0, -vec[0]],
                     [-vec[1], vec[0],  0     ]])
    
def so3_to_vec(so3):
    '''Converts an so(3) representation to a 3-vector

    :param so3: A so3 matrix
    :return: The corresponding 3-vector

    Example Input:
        so3mat = np.array([[ 0, -3,  2],
                           [ 3,  0, -1],
                           [-2,  1,  0]])
    Output:
        np.array([1, 2, 3])
    '''
    return np.array([so3[2,1], so3[0,2], so3[1,0]])

def vec_to_se3(V):
    """Converts a spatial velocity vector into a 4x4 matrix in se3
    :param V: A 6-vector representing a spatial velocity
    :return: The 4x4 se3 representation of V
    Example Input:
        V = np.array([1, 2, 3, 4, 5, 6])
    Output:
        np.array([[ 0, -3,  2, 4],
                  [ 3,  0, -1, 5],
                  [-2,  1,  0, 6],
                  [ 0,  0,  0, 0]])
    """
    return np.r_[np.c_[vec_to_so3([V[0], V[1], V[2]]), [V[3], V[4], V[5]]],
                 np.zeros((1, 4))]

def se3_to_vec(se3mat):
    """ Converts an se3 matrix into a spatial velocity vector
    :param se3mat: A 4x4 matrix in se3
    :return: The spatial velocity 6-vector corresponding to se3mat
    Example Input:
        se3mat = np.array([[ 0, -3,  2, 4],
                           [ 3,  0, -1, 5],
                           [-2,  1,  0, 6],
                           [ 0,  0,  0, 0]])
    Output:
        np.array([1, 2, 3, 4, 5, 6])
    """
    return np.r_[[se3mat[2][1], se3mat[0][2], se3mat[1][0]],
                 [se3mat[0][3], se3mat[1][3], se3mat[2][3]]]

def axis_ang_3(expc3):
    '''将代表旋转的指数坐标转换为对应的**旋转轴和旋转角**。
    :param expc3: A 3-vector of exponential coordinates for rotation
    :return omghat: A unit rotation axis
    :return theta: The corresponding rotation angle

    Example Input:
        expc3 = np.array([1, 2, 3])
    Output:
        (np.array([0.26726124, 0.53452248, 0.80178373]), 3.7416573867739413)
    '''
    return (normalize(expc3), np.linalg.norm(expc3))

def axis_ang_6(expc6):
    """Converts a 6-vector of exponential coordinates into screw axis-angle
    form
    :param expc6: A 6-vector of exponential coordinates for rigid-body motion
                  S*theta
    :return S: The corresponding normalized screw axis
    :return theta: The distance traveled along/about S
    Example Input:
        expc6 = np.array([1, 0, 0, 1, 2, 3])
    Output:
        (np.array([1.0, 0.0, 0.0, 1.0, 2.0, 3.0]), 1.0)
    """
    theta = np.linalg.norm([expc6[0], expc6[1], expc6[2]])
    if near_zero(theta):
        theta = np.linalg.norm([expc6[3], expc6[4], expc6[5]])
    return (np.array(expc6 / theta), theta)

def matrix_exp_3(so3mat):
    '''矩阵指数函数，计算 so(3) 斜对称矩阵所对应的**旋转矩阵**。

    :param so3mat: A 3x3 skew-symmetric matrix
    :return: The matrix exponential of so3mat

    Example Input:
        so3mat = np.array([[ 0, -3,  2],
                           [ 3,  0, -1],
                           [-2,  1,  0]])
    Output:
        np.array([[-0.69492056,  0.71352099,  0.08929286],
                  [-0.19200697, -0.30378504,  0.93319235],
                  [ 0.69297817,  0.6313497 ,  0.34810748]])
    '''
    omgtheta = so3_to_vec(so3mat)
    if near_zero(np.linalg.norm(omgtheta)):
        return np.eye(3)
    else:
        theta = axis_ang_3(omgtheta)[1]
        omgmat = so3mat / theta
        return np.eye(3) + np.sin(theta) * omgmat \
               + (1 - np.cos(theta)) * np.dot(omgmat, omgmat)

def matrix_log_3(R):
    '''矩阵对数函数，计算旋转矩阵所对应的 so(3) **斜对称矩阵**。

    :param R: A 3x3 rotation matrix
    :return: The matrix logarithm of R

    Example Input:
        R = np.array([[0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0]])
    Output:
        np.array([[          0, -1.20919958,  1.20919958],
                  [ 1.20919958,           0, -1.20919958],
                  [-1.20919958,  1.20919958,           0]])
    '''
    acosinput = (np.trace(R) - 1) / 2.0
    if acosinput >= 1:
        return np.zeros((3, 3))
    elif acosinput <= -1:
        if not near_zero(1 + R[2][2]):
            omg = (1.0 / np.sqrt(2 * (1 + R[2][2]))) \
                  * np.array([R[0][2], R[1][2], 1 + R[2][2]])
        elif not near_zero(1 + R[1][1]):
            omg = (1.0 / np.sqrt(2 * (1 + R[1][1]))) \
                  * np.array([R[0][1], 1 + R[1][1], R[2][1]])
        else:
            omg = (1.0 / np.sqrt(2 * (1 + R[0][0]))) \
                  * np.array([1 + R[0][0], R[1][0], R[2][0]])
        return vec_to_so3(np.pi * omg)
    else:
        theta = np.arccos(acosinput)
        return theta / 2.0 / np.sin(theta) * (R - np.array(R).T)
    
def matrix_exp_6(se3mat):
    """Computes the matrix exponential of an se3 representation of
    exponential coordinates
    :param se3mat: A matrix in se3
    :return: The matrix exponential of se3mat
    Example Input:
        se3mat = np.array([[0,          0,           0,          0],
                           [0,          0, -1.57079632, 2.35619449],
                           [0, 1.57079632,           0, 2.35619449],
                           [0,          0,           0,          0]])
    Output:
        np.array([[1.0, 0.0,  0.0, 0.0],
                  [0.0, 0.0, -1.0, 0.0],
                  [0.0, 1.0,  0.0, 3.0],
                  [  0,   0,    0,   1]])
    """
    se3mat = np.array(se3mat)
    omgtheta = so3_to_vec(se3mat[0: 3, 0: 3])
    if near_zero(np.linalg.norm(omgtheta)):
        return np.r_[np.c_[np.eye(3), se3mat[0: 3, 3]], [[0, 0, 0, 1]]]
    else:
        theta = axis_ang_3(omgtheta)[1]
        omgmat = se3mat[0: 3, 0: 3] / theta
        return np.r_[np.c_[matrix_exp_3(se3mat[0: 3, 0: 3]),
                           np.dot(np.eye(3) * theta \
                                  + (1 - np.cos(theta)) * omgmat \
                                  + (theta - np.sin(theta)) \
                                    * np.dot(omgmat,omgmat),
                                  se3mat[0: 3, 3]) / theta],
                     [[0, 0, 0, 1]]]

def matrix_log_6(T):
    """Computes the matrix logarithm of a homogeneous transformation matrix
    :param R: A matrix in SE3
    :return: The matrix logarithm of R
    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        np.array([[0,          0,           0,           0]
                  [0,          0, -1.57079633,  2.35619449]
                  [0, 1.57079633,           0,  2.35619449]
                  [0,          0,           0,           0]])
    """
    R, p = trans_to_Rp(T)
    omgmat = matrix_log_3(R)
    if np.array_equal(omgmat, np.zeros((3, 3))):
        return np.r_[np.c_[np.zeros((3, 3)),
                           [T[0][3], T[1][3], T[2][3]]],
                     [[0, 0, 0, 0]]]
    else:
        theta = np.arccos((np.trace(R) - 1) / 2.0)
        return np.r_[np.c_[omgmat,
                           np.dot(np.eye(3) - omgmat / 2.0 \
                           + (1.0 / theta - 1.0 / np.tan(theta / 2.0) / 2) \
                              * np.dot(omgmat,omgmat) / theta,[T[0][3],
                                                               T[1][3],
                                                               T[2][3]])],
                     [[0, 0, 0, 0]]]


def Rp_to_trans(R, p):
    '''Converts a rotation matrix and a position vector into homogeneous
    transformation matrix

    :param R: A 3x3 rotation matrix
    :param p: A 3-vector
    :return: A homogeneous transformation matrix corresponding to the inputs

    Example Input:
        R = np.mat([[1, 0,  0],
                    [0, 0, -1],
                    [0, 1,  0]])
        p = np.mat([[1], 
                    [2], 
                    [5]])
    Output:
        np.mat([[1, 0,  0, 1],
                [0, 0, -1, 2],
                [0, 1,  0, 5],
                [0, 0,  0, 1]])
    '''
    return np.array(np.r_[np.c_[np.array(R), np.array([p[0], p[1], p[2]])], [[0, 0, 0, 1]]])

def trans_to_Rp(T):
    '''Converts a homogeneous transformation matrix into a rotation matrix
    and position vector

    :param trans: A homogeneous transformation matrix
    :return R: The corresponding rotation matrix,
    :return p: The corresponding position vector.

    Example Input:
        trans = np.array([[1, 0,  0, 0],
                          [0, 0, -1, 0],
                          [0, 1,  0, 3],
                          [0, 0,  0, 1]])
    Output:
        (array([[ 1,  0,  0],
               [ 0,  0, -1],
               [ 0,  1,  0]]), 
        array([0, 0, 3]))
    '''
    trans = np.array(T)
    return np.array(trans[0: 3, 0: 3]), np.array(trans[0: 3, 3])

def pose_to_trans(pose):
    '''Converts a pose vector into a homogeneous transformation matrix

    :param pose: A 6-d pose vector
    :return trans: The corresponding homogeneous transformation matrix,

    Example Input:
        pose = np.array([1, 2, 3, 0, math.pi / 2, 0])
    Output:
        np.array([[ 0, 0, 1, 1],
                  [ 0, 1, 0, 2],
                  [-1, 0, 0, 3],
                  [ 0, 0, 0, 1])
    '''
    p = np.array(pose[0:3]).T
    euler = pose[3:6]
    R = euler_to_matrix(euler)
    trans = Rp_to_trans(R, p)
    return trans

def adjoint(T):
    """计算位形矩阵的伴随矩阵
    :param T: A homogeneous transformation matrix
    :return: The 6x6 adjoint representation [AdT] of T
    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        np.array([[1, 0,  0, 0, 0,  0],
                  [0, 0, -1, 0, 0,  0],
                  [0, 1,  0, 0, 0,  0],
                  [0, 0,  3, 1, 0,  0],
                  [3, 0,  0, 0, 0, -1],
                  [0, 0,  0, 0, 1,  0]])
    """
    R, p = trans_to_Rp(T)
    return np.r_[np.c_[R, np.zeros((3, 3))],
                 np.c_[np.dot(vec_to_so3(p), R), R]]

def screw_to_axis(q, s, h):
    """Takes a parametric description of a screw axis and converts it to a
    normalized screw axis
    :param q: A point lying on the screw axis
    :param s: A unit vector in the direction of the screw axis
    :param h: The pitch of the screw axis
    :return: A normalized screw axis described by the inputs
    Example Input:
        q = np.array([3, 0, 0])
        s = np.array([0, 0, 1])
        h = 2
    Output:
        np.array([0, 0, 1, 0, -3, 2])
    """
    return np.r_[s, np.cross(q, s) + np.dot(h, s)]

def ad(V):
    """Calculate the 6x6 matrix [adV] of the given 6-vector
    :param V: A 6-vector spatial velocity
    :return: The corresponding 6x6 matrix [adV]
    Used to calculate the Lie bracket [V1, V2] = [adV1]V2
    Example Input:
        V = np.array([1, 2, 3, 4, 5, 6])
    Output:
        np.array([[ 0, -3,  2,  0,  0,  0],
                  [ 3,  0, -1,  0,  0,  0],
                  [-2,  1,  0,  0,  0,  0],
                  [ 0, -6,  5,  0, -3,  2],
                  [ 6,  0, -4,  3,  0, -1],
                  [-5,  4,  0, -2,  1,  0]])
    """
    omgmat = vec_to_so3([V[0], V[1], V[2]])
    return np.r_[np.c_[omgmat, np.zeros((3, 3))],
                 np.c_[vec_to_so3([V[3], V[4], V[5]]), omgmat]]


'''
*********************************************
*************  机器人运动学函数  *************
*********************************************
'''

def fkin_body(M, Blist, thetalist):
    """Computes forward kinematics in the body frame for an open chain robot
    :param M: The home configuration (position and orientation) of the end-
              effector
    :param Blist: The joint screw axes in the end-effector frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param thetalist: A list of joint coordinates
    :return: A homogeneous transformation matrix representing the end-
             effector frame when the joints are at the specified coordinates
             (i.t.o Body Frame)
    Example Input:
        M = np.array([[-1, 0,  0, 0],
                      [ 0, 1,  0, 6],
                      [ 0, 0, -1, 2],
                      [ 0, 0,  0, 1]])
        Blist = np.array([[0, 0, -1, 2, 0,   0],
                          [0, 0,  0, 0, 1,   0],
                          [0, 0,  1, 0, 0, 0.1]]).T
        thetalist = np.array([np.pi / 2.0, 3, np.pi])
    Output:
        np.array([[0, 1,  0,         -5],
                  [1, 0,  0,          4],
                  [0, 0, -1, 1.68584073],
                  [0, 0,  0,          1]])
    """
    T = np.array(M)
    for i in range(len(thetalist)):
        T = np.dot(T, matrix_exp_6(vec_to_se3(np.array(Blist)[:, i] \
                                          * thetalist[i])))
    return T

def fkin_space(M, Slist, thetalist):
    """Computes forward kinematics in the space frame for an open chain robot
    :param M: The home configuration (position and orientation) of the end-
              effector
    :param Slist: The joint screw axes in the space frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param thetalist: A list of joint coordinates
    :return: A homogeneous transformation matrix representing the end-
             effector frame when the joints are at the specified coordinates
             (i.t.o Space Frame)
    Example Input:
        M = np.array([[-1, 0,  0, 0],
                      [ 0, 1,  0, 6],
                      [ 0, 0, -1, 2],
                      [ 0, 0,  0, 1]])
        Slist = np.array([[0, 0,  1,  4, 0,    0],
                          [0, 0,  0,  0, 1,    0],
                          [0, 0, -1, -6, 0, -0.1]]).T
        thetalist = np.array([np.pi / 2.0, 3, np.pi])
    Output:
        np.array([[0, 1,  0,         -5],
                  [1, 0,  0,          4],
                  [0, 0, -1, 1.68584073],
                  [0, 0,  0,          1]])
    """
    T = np.array(M)
    for i in range(len(thetalist) - 1, -1, -1):
        T = np.dot(matrix_exp_6(vec_to_se3(np.array(Slist)[:, i] \
                                       * thetalist[i])), T)
    return T

def jacobian_body(Blist, thetalist):
    """计算开链机器人的物体雅可比矩阵
    :param Blist: The joint screw axes in the end-effector frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param thetalist: A list of joint coordinates
    :return: The body Jacobian corresponding to the inputs (6xn real
             numbers)
    Example Input:
        Blist = np.array([[0, 0, 1,   0, 0.2, 0.2],
                          [1, 0, 0,   2,   0,   3],
                          [0, 1, 0,   0,   2,   1],
                          [1, 0, 0, 0.2, 0.3, 0.4]]).T
        thetalist = np.array([0.2, 1.1, 0.1, 1.2])
    Output:
        np.array([[-0.04528405, 0.99500417,           0,   1]
                  [ 0.74359313, 0.09304865,  0.36235775,   0]
                  [-0.66709716, 0.03617541, -0.93203909,   0]
                  [ 2.32586047,    1.66809,  0.56410831, 0.2]
                  [-1.44321167, 2.94561275,  1.43306521, 0.3]
                  [-2.06639565, 1.82881722, -1.58868628, 0.4]])
    """
    Jb = np.array(Blist).copy().astype(np.float)
    T = np.eye(4)
    for i in range(len(thetalist) - 2, -1, -1):
        T = np.dot(T,matrix_exp_6(vec_to_se3(np.array(Blist)[:, i + 1] \
                                         * -thetalist[i + 1])))
        Jb[:, i] = np.dot(adjoint(T), np.array(Blist)[:, i])
    return Jb

def jacobian_space(Slist, thetalist):
    """计算开链机器人的空间雅可比矩阵
    :param Slist: The joint screw axes in the space frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param thetalist: A list of joint coordinates
    :return: The space Jacobian corresponding to the inputs (6xn real
             numbers)
    Example Input:
        Slist = np.array([[0, 0, 1,   0, 0.2, 0.2],
                          [1, 0, 0,   2,   0,   3],
                          [0, 1, 0,   0,   2,   1],
                          [1, 0, 0, 0.2, 0.3, 0.4]]).T
        thetalist = np.array([0.2, 1.1, 0.1, 1.2])
    Output:
        np.array([[  0, 0.98006658, -0.09011564,  0.95749426]
                  [  0, 0.19866933,   0.4445544,  0.28487557]
                  [  1,          0,  0.89120736, -0.04528405]
                  [  0, 1.95218638, -2.21635216, -0.51161537]
                  [0.2, 0.43654132, -2.43712573,  2.77535713]
                  [0.2, 2.96026613,  3.23573065,  2.22512443]])
    """
    Js = np.array(Slist).copy().astype(np.float)
    T = np.eye(4)
    for i in range(1, len(thetalist)):
        T = np.dot(T, matrix_exp_6(vec_to_se3(np.array(Slist)[:, i - 1] \
                                * thetalist[i - 1])))
        Js[:, i] = np.dot(adjoint(T), np.array(Slist)[:, i])
    return Js
    
def ikin_body(Blist, M, T, thetalist0, eomg, ev):
    """Computes inverse kinematics in the body frame for an open chain robot
    :param Blist: The joint screw axes in the end-effector frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param M: The home configuration of the end-effector
    :param T: The desired end-effector configuration Tsd
    :param thetalist0: An initial guess of joint angles that are close to
                       satisfying Tsd
    :param eomg: A small positive tolerance on the end-effector orientation
                 error. The returned joint angles must give an end-effector
                 orientation error less than eomg
    :param ev: A small positive tolerance on the end-effector linear position
               error. The returned joint angles must give an end-effector
               position error less than ev
    :return thetalist: Joint angles that achieve T within the specified
                       tolerances,
    :return success: A logical value where TRUE means that the function found
                     a solution and FALSE means that it ran through the set
                     number of maximum iterations without finding a solution
                     within the tolerances eomg and ev.
    Uses an iterative Newton-Raphson root-finding method.
    The maximum number of iterations before the algorithm is terminated has
    been hardcoded in as a variable called maxiterations. It is set to 20 at
    the start of the function, but can be changed if needed.
    Example Input:
        Blist = np.array([[0, 0, -1, 2, 0,   0],
                          [0, 0,  0, 0, 1,   0],
                          [0, 0,  1, 0, 0, 0.1]]).T
        M = np.array([[-1, 0,  0, 0],
                      [ 0, 1,  0, 6],
                      [ 0, 0, -1, 2],
                      [ 0, 0,  0, 1]])
        T = np.array([[0, 1,  0,     -5],
                      [1, 0,  0,      4],
                      [0, 0, -1, 1.6858],
                      [0, 0,  0,      1]])
        thetalist0 = np.array([1.5, 2.5, 3])
        eomg = 0.01
        ev = 0.001
    Output:
        (np.array([1.57073819, 2.999667, 3.14153913]), True)
    """
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = 20
    Vb = se3_to_vec(matrix_log_6(np.dot(trans_inv(fkin_body(M, Blist, \
                                                      thetalist)), T)))
    err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg \
          or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev
    while err and i < maxiterations:
        thetalist = thetalist \
                    + np.dot(np.linalg.pinv(jacobian_body(Blist, \
                                                         thetalist)), Vb)
        i = i + 1
        Vb \
        = se3_to_vec(matrix_log_6(np.dot(trans_inv(fkin_body(M, Blist, \
                                                       thetalist)), T)))
        err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg \
              or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev
    return (thetalist, not err)

def ikin_space(Slist, M, T, thetalist0, eomg, ev):
    """Computes inverse kinematics in the space frame for an open chain robot
    :param Slist: The joint screw axes in the space frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param M: The home configuration of the end-effector
    :param T: The desired end-effector configuration Tsd
    :param thetalist0: An initial guess of joint angles that are close to
                       satisfying Tsd
    :param eomg: A small positive tolerance on the end-effector orientation
                 error. The returned joint angles must give an end-effector
                 orientation error less than eomg
    :param ev: A small positive tolerance on the end-effector linear position
               error. The returned joint angles must give an end-effector
               position error less than ev
    :return thetalist: Joint angles that achieve T within the specified
                       tolerances,
    :return success: A logical value where TRUE means that the function found
                     a solution and FALSE means that it ran through the set
                     number of maximum iterations without finding a solution
                     within the tolerances eomg and ev.
    Uses an iterative Newton-Raphson root-finding method.
    The maximum number of iterations before the algorithm is terminated has
    been hardcoded in as a variable called maxiterations. It is set to 20 at
    the start of the function, but can be changed if needed.
    Example Input:
        Slist = np.array([[0, 0,  1,  4, 0,    0],
                          [0, 0,  0,  0, 1,    0],
                          [0, 0, -1, -6, 0, -0.1]]).T
        M = np.array([[-1, 0,  0, 0],
                      [ 0, 1,  0, 6],
                      [ 0, 0, -1, 2],
                      [ 0, 0,  0, 1]])
        T = np.array([[0, 1,  0,     -5],
                      [1, 0,  0,      4],
                      [0, 0, -1, 1.6858],
                      [0, 0,  0,      1]])
        thetalist0 = np.array([1.5, 2.5, 3])
        eomg = 0.01
        ev = 0.001
    Output:
        (np.array([ 1.57073783,  2.99966384,  3.1415342 ]), True)
    """
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = 20
    Tsb = fkin_space(M,Slist, thetalist)
    Vs = np.dot(adjoint(Tsb), \
                se3_to_vec(matrix_log_6(np.dot(trans_inv(Tsb), T))))
    err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > eomg \
          or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > ev
    while err and i < maxiterations:
        thetalist = thetalist \
                    + np.dot(np.linalg.pinv(jacobian_space(Slist, \
                                                          thetalist)), Vs)
        i = i + 1
        Tsb = fkin_space(M, Slist, thetalist)
        Vs = np.dot(adjoint(Tsb), \
                    se3_to_vec(matrix_log_6(np.dot(trans_inv(Tsb), T))))
        err = np.linalg.norm([Vs[0], Vs[1], Vs[2]]) > eomg \
              or np.linalg.norm([Vs[3], Vs[4], Vs[5]]) > ev
    return (thetalist, not err)

'''
*********************************************
*************  机器人动力学函数  *************
*********************************************
'''

def inverse_dynamics(thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, \
                    Glist, Slist):
    """计算开链机器人的逆动力学，以空间坐标系为参考系
    :param thetalist: 关节位置向量
    :param dthetalist: 关节速度向量
    :param ddthetalist: 关节加速度向量
    :param g: 重力加速度向量
    :param Ftip: 作用到末端执行器的空间力旋量，以坐标系{i+1}为参考系
    :param Mlist: 零位时，连杆坐标系{i}相对于{i-1}的位形变换矩阵
    :param Glist: 连杆的空间惯量矩阵
    :param Slist: 各关节的螺旋轴矩阵，以空间坐标系为参考系，矩阵列为螺旋轴。
    :return: 各关节所需力/力矩
    本方程使用前向-反向迭代的方法计算机器人的牛顿-欧拉动力学方程：
    taulist = Mlist(thetalist)ddthetalist + c(thetalist,dthetalist) \
              + g(thetalist) + Jtr(thetalist)Ftip
    Example Input (3 Link Robot):
        thetalist = np.array([0.1, 0.1, 0.1])
        dthetalist = np.array([0.1, 0.2, 0.3])
        ddthetalist = np.array([2, 1.5, 1])
        g = np.array([0, 0, -9.8])
        Ftip = np.array([1, 1, 1, 1, 1, 1])
        M01 = np.array([[1, 0, 0,        0],
                        [0, 1, 0,        0],
                        [0, 0, 1, 0.089159],
                        [0, 0, 0,        1]])
        M12 = np.array([[ 0, 0, 1,    0.28],
                        [ 0, 1, 0, 0.13585],
                        [-1, 0, 0,       0],
                        [ 0, 0, 0,       1]])
        M23 = np.array([[1, 0, 0,       0],
                        [0, 1, 0, -0.1197],
                        [0, 0, 1,   0.395],
                        [0, 0, 0,       1]])
        M34 = np.array([[1, 0, 0,       0],
                        [0, 1, 0,       0],
                        [0, 0, 1, 0.14225],
                        [0, 0, 0,       1]])
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        Slist = np.array([[1, 0, 1,      0, 1,     0],
                          [0, 1, 0, -0.089, 0,     0],
                          [0, 1, 0, -0.089, 0, 0.425]]).T
    Output:
        np.array([74.69616155, -33.06766016, -3.23057314])
    """
    n = len(thetalist)
    Mi = np.eye(4)
    Ai = np.zeros((6, n))
    AdTi = [[None]] * (n + 1)
    Vi = np.zeros((6, n + 1))
    Vdi = np.zeros((6, n + 1))
    Vdi[:, 0] = np.r_[[0, 0, 0], -np.array(g)]
    AdTi[n] = adjoint(trans_inv(Mlist[n]))
    Fi = np.array(Ftip).copy()
    taulist = np.zeros(n)
    for i in range(n):
        Mi = np.dot(Mi,Mlist[i])
        Ai[:, i] = np.dot(adjoint(trans_inv(Mi)), np.array(Slist)[:, i])
        AdTi[i] = adjoint(np.dot(matrix_exp_6(vec_to_se3(Ai[:, i] * \
                                            -thetalist[i])), \
                                 trans_inv(Mlist[i])))
        Vi[:, i + 1] = np.dot(AdTi[i], Vi[:,i]) + Ai[:, i] * dthetalist[i]
        Vdi[:, i + 1] = np.dot(AdTi[i], Vdi[:, i]) \
                       + Ai[:, i] * ddthetalist[i] \
                       + np.dot(ad(Vi[:, i + 1]), Ai[:, i]) * dthetalist[i]
    for i in range (n - 1, -1, -1):
        Fi = np.dot(np.array(AdTi[i + 1]).T, Fi) \
             + np.dot(np.array(Glist[i]), Vdi[:, i + 1]) \
             - np.dot(np.array(ad(Vi[:, i + 1])).T, \
                      np.dot(np.array(Glist[i]), Vi[:, i + 1]))
        taulist[i] = np.dot(np.array(Fi).T, Ai[:, i])
    return taulist

def mass_matrix(thetalist, Mlist, Glist, Slist):
    """Computes the mass matrix of an open chain robot based on the given
    configuration
    :param thetalist: A list of joint variables
    :param Mlist: List of link frames i relative to i-1 at the home position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame, in the format
                  of a matrix with axes as the columns
    :return: The numerical inertia matrix M(thetalist) of an n-joint serial
             chain at the given configuration thetalist
    This function calls InverseDynamics n times, each time passing a
    ddthetalist vector with a single element equal to one and all other
    inputs set to zero.
    Each call of InverseDynamics generates a single column, and these columns
    are assembled to create the inertia matrix.
    Example Input (3 Link Robot):
        thetalist = np.array([0.1, 0.1, 0.1])
        M01 = np.array([[1, 0, 0,        0],
                        [0, 1, 0,        0],
                        [0, 0, 1, 0.089159],
                        [0, 0, 0,        1]])
        M12 = np.array([[ 0, 0, 1,    0.28],
                        [ 0, 1, 0, 0.13585],
                        [-1, 0, 0,       0],
                        [ 0, 0, 0,       1]])
        M23 = np.array([[1, 0, 0,       0],
                        [0, 1, 0, -0.1197],
                        [0, 0, 1,   0.395],
                        [0, 0, 0,       1]])
        M34 = np.array([[1, 0, 0,       0],
                        [0, 1, 0,       0],
                        [0, 0, 1, 0.14225],
                        [0, 0, 0,       1]])
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        Slist = np.array([[1, 0, 1,      0, 1,     0],
                          [0, 1, 0, -0.089, 0,     0],
                          [0, 1, 0, -0.089, 0, 0.425]]).T
    Output:
        np.array([[ 2.25433380e+01, -3.07146754e-01, -7.18426391e-03]
                  [-3.07146754e-01,  1.96850717e+00,  4.32157368e-01]
                  [-7.18426391e-03,  4.32157368e-01,  1.91630858e-01]])
    """
    n = len(thetalist)
    M = np.zeros((n, n))
    for i in range (n):
        ddthetalist = [0] * n
        ddthetalist[i] = 1
        M[:, i] = inverse_dynamics(thetalist, [0] * n, ddthetalist, \
                                  [0, 0, 0], [0, 0, 0, 0, 0, 0], Mlist, \
                                  Glist, Slist)
    return M

def vel_quadratic_forces(thetalist, dthetalist, Mlist, Glist, Slist):
    """Computes the Coriolis and centripetal terms in the inverse dynamics of
    an open chain robot
    :param thetalist: A list of joint variables,
    :param dthetalist: A list of joint rates,
    :param Mlist: List of link frames i relative to i-1 at the home position,
    :param Glist: Spatial inertia matrices Gi of the links,
    :param Slist: Screw axes Si of the joints in a space frame, in the format
                  of a matrix with axes as the columns.
    :return: The vector c(thetalist,dthetalist) of Coriolis and centripetal
             terms for a given thetalist and dthetalist.
    This function calls InverseDynamics with g = 0, Ftip = 0, and
    ddthetalist = 0.
    Example Input (3 Link Robot):
        thetalist = np.array([0.1, 0.1, 0.1])
        dthetalist = np.array([0.1, 0.2, 0.3])
        M01 = np.array([[1, 0, 0,        0],
                        [0, 1, 0,        0],
                        [0, 0, 1, 0.089159],
                        [0, 0, 0,        1]])
        M12 = np.array([[ 0, 0, 1,    0.28],
                        [ 0, 1, 0, 0.13585],
                        [-1, 0, 0,       0],
                        [ 0, 0, 0,       1]])
        M23 = np.array([[1, 0, 0,       0],
                        [0, 1, 0, -0.1197],
                        [0, 0, 1,   0.395],
                        [0, 0, 0,       1]])
        M34 = np.array([[1, 0, 0,       0],
                        [0, 1, 0,       0],
                        [0, 0, 1, 0.14225],
                        [0, 0, 0,       1]])
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        Slist = np.array([[1, 0, 1,      0, 1,     0],
                          [0, 1, 0, -0.089, 0,     0],
                          [0, 1, 0, -0.089, 0, 0.425]]).T
    Output:
        np.array([0.26453118, -0.05505157, -0.00689132])
    """
    return inverse_dynamics(thetalist, dthetalist, [0] * len(thetalist), \
                           [0, 0, 0], [0, 0, 0, 0, 0, 0], Mlist, Glist, \
                           Slist)

def gravity_forces(thetalist, g, Mlist, Glist, Slist):
    """Computes the joint forces/torques an open chain robot requires to
    overcome gravity at its configuration
    :param thetalist: A list of joint variables
    :param g: 3-vector for gravitational acceleration
    :param Mlist: List of link frames i relative to i-1 at the home position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame, in the format
                  of a matrix with axes as the columns
    :return grav: The joint forces/torques required to overcome gravity at
                  thetalist
    This function calls InverseDynamics with Ftip = 0, dthetalist = 0, and
    ddthetalist = 0.
    Example Inputs (3 Link Robot):
        thetalist = np.array([0.1, 0.1, 0.1])
        g = np.array([0, 0, -9.8])
        M01 = np.array([[1, 0, 0,        0],
                        [0, 1, 0,        0],
                        [0, 0, 1, 0.089159],
                        [0, 0, 0,        1]])
        M12 = np.array([[ 0, 0, 1,    0.28],
                        [ 0, 1, 0, 0.13585],
                        [-1, 0, 0,       0],
                        [ 0, 0, 0,       1]])
        M23 = np.array([[1, 0, 0,       0],
                        [0, 1, 0, -0.1197],
                        [0, 0, 1,   0.395],
                        [0, 0, 0,       1]])
        M34 = np.array([[1, 0, 0,       0],
                        [0, 1, 0,       0],
                        [0, 0, 1, 0.14225],
                        [0, 0, 0,       1]])
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        Slist = np.array([[1, 0, 1,      0, 1,     0],
                          [0, 1, 0, -0.089, 0,     0],
                          [0, 1, 0, -0.089, 0, 0.425]]).T
    Output:
        np.array([28.40331262, -37.64094817, -5.4415892])
    """
    n = len(thetalist)
    return inverse_dynamics(thetalist, [0] * n, [0] * n, g, \
                           [0, 0, 0, 0, 0, 0], Mlist, Glist, Slist)
        
def end_effector_forces(thetalist, Ftip, Mlist, Glist, Slist):
    """Computes the joint forces/torques an open chain robot requires only to
    create the end-effector force Ftip
    :param thetalist: A list of joint variables
    :param Ftip: Spatial force applied by the end-effector expressed in frame
                 {n+1}
    :param Mlist: List of link frames i relative to i-1 at the home position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame, in the format
                  of a matrix with axes as the columns
    :return: The joint forces and torques required only to create the
             end-effector force Ftip
    This function calls InverseDynamics with g = 0, dthetalist = 0, and
    ddthetalist = 0.
    Example Input (3 Link Robot):
        thetalist = np.array([0.1, 0.1, 0.1])
        Ftip = np.array([1, 1, 1, 1, 1, 1])
        M01 = np.array([[1, 0, 0,        0],
                        [0, 1, 0,        0],
                        [0, 0, 1, 0.089159],
                        [0, 0, 0,        1]])
        M12 = np.array([[ 0, 0, 1,    0.28],
                        [ 0, 1, 0, 0.13585],
                        [-1, 0, 0,       0],
                        [ 0, 0, 0,       1]])
        M23 = np.array([[1, 0, 0,       0],
                        [0, 1, 0, -0.1197],
                        [0, 0, 1,   0.395],
                        [0, 0, 0,       1]])
        M34 = np.array([[1, 0, 0,       0],
                        [0, 1, 0,       0],
                        [0, 0, 1, 0.14225],
                        [0, 0, 0,       1]])
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        Slist = np.array([[1, 0, 1,      0, 1,     0],
                          [0, 1, 0, -0.089, 0,     0],
                          [0, 1, 0, -0.089, 0, 0.425]]).T
    Output:
        np.array([1.40954608, 1.85771497, 1.392409])
    """
    n = len(thetalist)
    return inverse_dynamics(thetalist, [0] * n, [0] * n, [0, 0, 0], Ftip, \
                           Mlist, Glist, Slist)

def forward_dynamics(thetalist, dthetalist, taulist, g, Ftip, Mlist, \
                    Glist, Slist):
    """Computes forward dynamics in the space frame for an open chain robot
    :param thetalist: A list of joint variables
    :param dthetalist: A list of joint rates
    :param taulist: An n-vector of joint forces/torques
    :param g: Gravity vector g
    :param Ftip: Spatial force applied by the end-effector expressed in frame
                 {n+1}
    :param Mlist: List of link frames i relative to i-1 at the home position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame, in the format
                  of a matrix with axes as the columns
    :return: The resulting joint accelerations
    This function computes ddthetalist by solving:
    Mlist(thetalist) * ddthetalist = taulist - c(thetalist,dthetalist) \
                                     - g(thetalist) - Jtr(thetalist) * Ftip
    Example Input (3 Link Robot):
        thetalist = np.array([0.1, 0.1, 0.1])
        dthetalist = np.array([0.1, 0.2, 0.3])
        taulist = np.array([0.5, 0.6, 0.7])
        g = np.array([0, 0, -9.8])
        Ftip = np.array([1, 1, 1, 1, 1, 1])
        M01 = np.array([[1, 0, 0,        0],
                        [0, 1, 0,        0],
                        [0, 0, 1, 0.089159],
                        [0, 0, 0,        1]])
        M12 = np.array([[ 0, 0, 1,    0.28],
                        [ 0, 1, 0, 0.13585],
                        [-1, 0, 0,       0],
                        [ 0, 0, 0,       1]])
        M23 = np.array([[1, 0, 0,       0],
                        [0, 1, 0, -0.1197],
                        [0, 0, 1,   0.395],
                        [0, 0, 0,       1]])
        M34 = np.array([[1, 0, 0,       0],
                        [0, 1, 0,       0],
                        [0, 0, 1, 0.14225],
                        [0, 0, 0,       1]])
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        Slist = np.array([[1, 0, 1,      0, 1,     0],
                          [0, 1, 0, -0.089, 0,     0],
                          [0, 1, 0, -0.089, 0, 0.425]]).T
    Output:
        np.array([-0.97392907, 25.58466784, -32.91499212])
    """
    return np.dot(np.linalg.inv(mass_matrix(thetalist, Mlist, Glist, \
                                           Slist)), \
                  np.array(taulist) \
                  - vel_quadratic_forces(thetalist, dthetalist, Mlist, \
                                       Glist, Slist) \
                  - gravity_forces(thetalist, g, Mlist, Glist, Slist) \
                  - end_effector_forces(thetalist, Ftip, Mlist, Glist, \
                                      Slist))

def euler_step(thetalist, dthetalist, ddthetalist, dt):
    """利用一阶欧拉积分，计算下一时刻的关节角和关节速度
    :param thetalist: 关节位置向量
    :param dthetalist: 关节速度向量
    :param ddthetalist: 关节加速度向量
    :param dt: 时间步长
    :return thetalistNext: 一个时间步长以后的关节位置向量，通过一阶欧拉积分计算得到
    :return dthetalistNext: 一个时间步长以后的关节速度向量，通过一阶欧拉积分计算得到
    Example Inputs (3 Link Robot):
        thetalist = np.array([0.1, 0.1, 0.1])
        dthetalist = np.array([0.1, 0.2, 0.3])
        ddthetalist = np.array([2, 1.5, 1])
        dt = 0.1
    Output:
        thetalistNext:
        array([ 0.11,  0.12,  0.13])
        dthetalistNext:
        array([ 0.3 ,  0.35,  0.4 ])
    """
    return thetalist + dt * np.array(dthetalist), \
           dthetalist + dt * np.array(ddthetalist)

'''
*********************************************
*************** 轨迹规划函数 *****************
*********************************************
'''

def cubic_time_scaling(Tf, t):
    """Computes s(t) for a cubic time scaling
    :param Tf: Total time of the motion in seconds from rest to rest
    :param t: The current time t satisfying 0 < t < Tf
    :return: The path parameter s(t) corresponding to a third-order
             polynomial motion that begins and ends at zero velocity
    Example Input:
        Tf = 2
        t = 0.6
    Output:
        0.216
    """
    return 3 * (1.0 * t / Tf) ** 2 - 2 * (1.0 * t / Tf) ** 3

def quintic_time_scaling(Tf, t):
    """Computes s(t) for a quintic time scaling
    :param Tf: Total time of the motion in seconds from rest to rest
    :param t: The current time t satisfying 0 < t < Tf
    :return: The path parameter s(t) corresponding to a fifth-order
             polynomial motion that begins and ends at zero velocity and zero
             acceleration
    Example Input:
        Tf = 2
        t = 0.6
    Output:
        0.16308
    """
    return 10 * (1.0 * t / Tf) ** 3 - 15 * (1.0 * t / Tf) ** 4 \
           + 6 * (1.0 * t / Tf) ** 5

def joint_trajectory(thetastart, thetaend, Tf, N, method):
    """Computes a straight-line trajectory in joint space
    :param thetastart: The initial joint variables
    :param thetaend: The final joint variables
    :param Tf: Total time of the motion in seconds from rest to rest
    :param N: The number of points N > 1 (Start and stop) in the discrete
              representation of the trajectory
    :param method: The time-scaling method, where 3 indicates cubic (third-
                   order polynomial) time scaling and 5 indicates quintic
                   (fifth-order polynomial) time scaling
    :return: A trajectory as an N x n matrix, where each row is an n-vector
             of joint variables at an instant in time. The first row is
             thetastart and the Nth row is thetaend . The elapsed time
             between each row is Tf / (N - 1)
    Example Input:
        thetastart = np.array([1, 0, 0, 1, 1, 0.2, 0,1])
        thetaend = np.array([1.2, 0.5, 0.6, 1.1, 2, 2, 0.9, 1])
        Tf = 4
        N = 6
        method = 3
    Output:
        np.array([[     1,     0,      0,      1,     1,    0.2,      0, 1]
                  [1.0208, 0.052, 0.0624, 1.0104, 1.104, 0.3872, 0.0936, 1]
                  [1.0704, 0.176, 0.2112, 1.0352, 1.352, 0.8336, 0.3168, 1]
                  [1.1296, 0.324, 0.3888, 1.0648, 1.648, 1.3664, 0.5832, 1]
                  [1.1792, 0.448, 0.5376, 1.0896, 1.896, 1.8128, 0.8064, 1]
                  [   1.2,   0.5,    0.6,    1.1,     2,      2,    0.9, 1]])
    """
    N = int(N)
    timegap = Tf / (N - 1.0)
    traj = np.zeros((len(thetastart), N))
    for i in range(N):
        if method == 3:
            s = cubic_time_scaling(Tf, timegap * i)
        else:
            s = quintic_time_scaling(Tf, timegap * i)
        traj[:, i] = s * np.array(thetaend) + (1 - s) * np.array(thetastart)
    traj = np.array(traj).T
    return traj

def screw_trajectory(Xstart, Xend, Tf, N, method):
    """Computes a trajectory as a list of N SE(3) matrices corresponding to
      the screw motion about a space screw axis
    :param Xstart: The initial end-effector configuration
    :param Xend: The final end-effector configuration
    :param Tf: Total time of the motion in seconds from rest to rest
    :param N: The number of points N > 1 (Start and stop) in the discrete
              representation of the trajectory
    :param method: The time-scaling method, where 3 indicates cubic (third-
                   order polynomial) time scaling and 5 indicates quintic
                   (fifth-order polynomial) time scaling
    :return: The discretized trajectory as a list of N matrices in SE(3)
             separated in time by Tf/(N-1). The first in the list is Xstart
             and the Nth is Xend
    Example Input:
        Xstart = np.array([[1, 0, 0, 1],
                           [0, 1, 0, 0],
                           [0, 0, 1, 1],
                           [0, 0, 0, 1]])
        Xend = np.array([[0, 0, 1, 0.1],
                         [1, 0, 0,   0],
                         [0, 1, 0, 4.1],
                         [0, 0, 0,   1]])
        Tf = 5
        N = 4
        method = 3
    Output:
        [np.array([[1, 0, 0, 1]
                   [0, 1, 0, 0]
                   [0, 0, 1, 1]
                   [0, 0, 0, 1]]),
         np.array([[0.904, -0.25, 0.346, 0.441]
                   [0.346, 0.904, -0.25, 0.529]
                   [-0.25, 0.346, 0.904, 1.601]
                   [    0,     0,     0,     1]]),
         np.array([[0.346, -0.25, 0.904, -0.117]
                   [0.904, 0.346, -0.25,  0.473]
                   [-0.25, 0.904, 0.346,  3.274]
                   [    0,     0,     0,      1]]),
         np.array([[0, 0, 1, 0.1]
                   [1, 0, 0,   0]
                   [0, 1, 0, 4.1]
                   [0, 0, 0,   1]])]
    """
    N = int(N)
    timegap = Tf / (N - 1.0)
    traj = [[None]] * N
    for i in range(N):
        if method == 3:
            s = cubic_time_scaling(Tf, timegap * i)
        else:
            s = quintic_time_scaling(Tf, timegap * i)
        traj[i] \
        = np.dot(Xstart, matrix_exp_6(matrix_log_6(np.dot(trans_inv(Xstart), \
                                                      Xend)) * s))
    return traj

def cartesian_trajectory(Xstart, Xend, Tf, N, method):
    """Computes a trajectory as a list of N SE(3) matrices corresponding to
    the origin of the end-effector frame following a straight line
    :param Xstart: The initial end-effector configuration
    :param Xend: The final end-effector configuration
    :param Tf: Total time of the motion in seconds from rest to rest
    :param N: The number of points N > 1 (Start and stop) in the discrete
              representation of the trajectory
    :param method: The time-scaling method, where 3 indicates cubic (third-
                   order polynomial) time scaling and 5 indicates quintic
                   (fifth-order polynomial) time scaling
    :return: The discretized trajectory as a list of N matrices in SE(3)
             separated in time by Tf/(N-1). The first in the list is Xstart
             and the Nth is Xend
    This function is similar to ScrewTrajectory, except the origin of the
    end-effector frame follows a straight line, decoupled from the rotational
    motion.
    Example Input:
        Xstart = np.array([[1, 0, 0, 1],
                           [0, 1, 0, 0],
                           [0, 0, 1, 1],
                           [0, 0, 0, 1]])
        Xend = np.array([[0, 0, 1, 0.1],
                         [1, 0, 0,   0],
                         [0, 1, 0, 4.1],
                         [0, 0, 0,   1]])
        Tf = 5
        N = 4
        method = 5
    Output:
        [np.array([[1, 0, 0, 1]
                   [0, 1, 0, 0]
                   [0, 0, 1, 1]
                   [0, 0, 0, 1]]),
         np.array([[ 0.937, -0.214,  0.277, 0.811]
                   [ 0.277,  0.937, -0.214,     0]
                   [-0.214,  0.277,  0.937, 1.651]
                   [     0,      0,      0,     1]]),
         np.array([[ 0.277, -0.214,  0.937, 0.289]
                   [ 0.937,  0.277, -0.214,     0]
                   [-0.214,  0.937,  0.277, 3.449]
                   [     0,      0,      0,     1]]),
         np.array([[0, 0, 1, 0.1]
                   [1, 0, 0,   0]
                   [0, 1, 0, 4.1]
                   [0, 0, 0,   1]])]
    """
    N = int(N)
    timegap = Tf / (N - 1.0)
    traj = [[None]] * N
    Rstart, pstart = trans_to_Rp(Xstart)
    Rend, pend = trans_to_Rp(Xend)
    for i in range(N):
        if method == 3:
            s = cubic_time_scaling(Tf, timegap * i)
        else:
            s = quintic_time_scaling(Tf, timegap * i)
        traj[i] \
        = np.r_[np.c_[np.dot(Rstart, \
        matrix_exp_3(matrix_log_3(np.dot(np.array(Rstart).T,Rend)) * s)), \
                   s * np.array(pend) + (1 - s) * np.array(pstart)], \
                   [[0, 0, 0, 1]]]
    return traj

'''
*********************************************
**************** 机器人控制 ******************
*********************************************
'''

def computed_torque(thetalist, dthetalist, eint, g, Mlist, Glist, Slist, thetalistd, dthetalistd, ddthetalistd, Kp, Ki, Kd):
    """Computes the joint control torques at a particular time instant
    :param thetalist: n-vector of joint variables
    :param dthetalist: n-vector of joint rates
    :param eint: n-vector of the time-integral of joint errors
    :param g: Gravity vector g
    :param Mlist: List of link frames {i} relative to {i-1} at the home
                  position
    :param Glist: Spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame, in the format
                  of a matrix with axes as the columns
    :param thetalistd: n-vector of reference joint variables
    :param dthetalistd: n-vector of reference joint velocities
    :param ddthetalistd: n-vector of reference joint accelerations
    :param Kp: The feedback proportional gain (identical for each joint)
    :param Ki: The feedback integral gain (identical for each joint)
    :param Kd: The feedback derivative gain (identical for each joint)
    :return: The vector of joint forces/torques computed by the feedback
             linearizing controller at the current instant
    Example Input:
        thetalist = np.array([0.1, 0.1, 0.1])
        dthetalist = np.array([0.1, 0.2, 0.3])
        eint = np.array([0.2, 0.2, 0.2])
        g = np.array([0, 0, -9.8])
        M01 = np.array([[1, 0, 0,        0],
                        [0, 1, 0,        0],
                        [0, 0, 1, 0.089159],
                        [0, 0, 0,        1]])
        M12 = np.array([[ 0, 0, 1,    0.28],
                        [ 0, 1, 0, 0.13585],
                        [-1, 0, 0,       0],
                        [ 0, 0, 0,       1]])
        M23 = np.array([[1, 0, 0,       0],
                        [0, 1, 0, -0.1197],
                        [0, 0, 1,   0.395],
                        [0, 0, 0,       1]])
        M34 = np.array([[1, 0, 0,       0],
                        [0, 1, 0,       0],
                        [0, 0, 1, 0.14225],
                        [0, 0, 0,       1]])
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        Slist = np.array([[1, 0, 1,      0, 1,     0],
                          [0, 1, 0, -0.089, 0,     0],
                          [0, 1, 0, -0.089, 0, 0.425]]).T
        thetalistd = np.array([1.0, 1.0, 1.0])
        dthetalistd = np.array([2, 1.2, 2])
        ddthetalistd = np.array([0.1, 0.1, 0.1])
        Kp = 1.3
        Ki = 1.2
        Kd = 1.1
    Output:
        np.array([133.00525246, -29.94223324, -3.03276856])
    """
    e = np.subtract(thetalistd, thetalist)
    return np.dot(mass_matrix(thetalist, Mlist, Glist, Slist), \
                  Kp * e + Ki * (np.array(eint) + e) \
                  + Kd * np.subtract(dthetalistd, dthetalist)) \
           + inverse_dynamics(thetalist, dthetalist, ddthetalistd, g, \
                             [0, 0, 0, 0, 0, 0], Mlist, Glist, Slist)

def simulate_control(thetalist, dthetalist, g, Ftipmat, Mlist, Glist, Slist, thetamatd, dthetamatd, ddthetamatd, gtilde, Mtildelist, Gtildelist, Kp, Ki, Kd, dt, intRes):
    """Simulates the computed torque controller over a given desired
    trajectory
    :param thetalist: n-vector of initial joint variables
    :param dthetalist: n-vector of initial joint velocities
    :param g: Actual gravity vector g
    :param Ftipmat: An N x 6 matrix of spatial forces applied by the end-
                    effector (If there are no tip forces the user should
                    input a zero and a zero matrix will be used)
    :param Mlist: Actual list of link frames i relative to i-1 at the home
                  position
    :param Glist: Actual spatial inertia matrices Gi of the links
    :param Slist: Screw axes Si of the joints in a space frame, in the format
                  of a matrix with axes as the columns
    :param thetamatd: An Nxn matrix of desired joint variables from the
                      reference trajectory
    :param dthetamatd: An Nxn matrix of desired joint velocities
    :param ddthetamatd: An Nxn matrix of desired joint accelerations
    :param gtilde: The gravity vector based on the model of the actual robot
                   (actual values given above)
    :param Mtildelist: The link frame locations based on the model of the
                       actual robot (actual values given above)
    :param Gtildelist: The link spatial inertias based on the model of the
                       actual robot (actual values given above)
    :param Kp: The feedback proportional gain (identical for each joint)
    :param Ki: The feedback integral gain (identical for each joint)
    :param Kd: The feedback derivative gain (identical for each joint)
    :param dt: The timestep between points on the reference trajectory
    :param intRes: 每个时间步长dt内的欧拉积分步骤数。
                   这个输入必须是一个大于或等于1的整数。
                   数值越大，模拟速度越慢，但积分误差积累越小。
    :return taumat: An Nxn matrix of the controllers commanded joint forces/
                    torques, where each row of n forces/torques corresponds
                    to a single time instant
    :return thetamat: An Nxn matrix of actual joint angles
    The end of this function plots all the actual and desired joint angles
    using matplotlib and random libraries.
    Example Input:
        from __future__ import print_function
        import numpy as np
        from modern_robotics import JointTrajectory
        thetalist = np.array([0.1, 0.1, 0.1])
        dthetalist = np.array([0.1, 0.2, 0.3])
        ## 初始化机器人的参数，以三自由度串联机器人为例
        g = np.array([0, 0, -9.8]) # 重力加速度
        # 零位时，相邻连杆间的变换矩阵
        M01 = np.array([[1, 0, 0,        0],
                        [0, 1, 0,        0],
                        [0, 0, 1, 0.089159],
                        [0, 0, 0,        1]])
        M12 = np.array([[ 0, 0, 1,    0.28],
                        [ 0, 1, 0, 0.13585],
                        [-1, 0, 0,       0],
                        [ 0, 0, 0,       1]])
        M23 = np.array([[1, 0, 0,       0],
                        [0, 1, 0, -0.1197],
                        [0, 0, 1,   0.395],
                        [0, 0, 0,       1]])
        M34 = np.array([[1, 0, 0,       0],
                        [0, 1, 0,       0],
                        [0, 0, 1, 0.14225],
                        [0, 0, 0,       1]])
        # 各连杆的惯量矩阵，以空间坐标系为参考系
        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        Glist = np.array([G1, G2, G3])
        Mlist = np.array([M01, M12, M23, M34])
        # 各关节运动旋量
        Slist = np.array([[1, 0, 1,      0, 1,     0],
                          [0, 1, 0, -0.089, 0,     0],
                          [0, 1, 0, -0.089, 0, 0.425]]).T
        dt = 0.01 # 关节轨迹位置间的时间间隔
        ## 创建用于跟踪的轨迹
        thetaend = np.array([np.pi / 2, np.pi, 1.5 * np.pi])
        Tf = 1
        N = int(1.0 * Tf / dt)
        method = 5
        traj = mr.joint_trajectory(thetalist, thetaend, Tf, N, method)
        thetamatd = np.array(traj).copy()
        dthetamatd = np.zeros((N, 3))
        ddthetamatd = np.zeros((N, 3))
        dt = Tf / (N - 1.0)
        for i in range(np.array(traj).shape[0] - 1):
            dthetamatd[i + 1, :] = (thetamatd[i + 1, :] - thetamatd[i, :]) / dt
            ddthetamatd[i + 1, :] = (dthetamatd[i + 1, :] - dthetamatd[i, :]) / dt
        # Possibly wrong robot description (Example with 3 links)
        gtilde = np.array([0.8, 0.2, -8.8])
        Mhat01 = np.array([[1, 0, 0,   0],
                           [0, 1, 0,   0],
                           [0, 0, 1, 0.1],
                           [0, 0, 0,   1]])
        Mhat12 = np.array([[ 0, 0, 1, 0.3],
                           [ 0, 1, 0, 0.2],
                           [-1, 0, 0,   0],
                           [ 0, 0, 0,   1]])
        Mhat23 = np.array([[1, 0, 0,    0],
                           [0, 1, 0, -0.2],
                           [0, 0, 1,  0.4],
                           [0, 0, 0,    1]])
        Mhat34 = np.array([[1, 0, 0,   0],
                           [0, 1, 0,   0],
                           [0, 0, 1, 0.2],
                           [0, 0, 0,   1]])
        Ghat1 = np.diag([0.1, 0.1, 0.1, 4, 4, 4])
        Ghat2 = np.diag([0.3, 0.3, 0.1, 9, 9, 9])
        Ghat3 = np.diag([0.1, 0.1, 0.1, 3, 3, 3])
        Gtildelist = np.array([Ghat1, Ghat2, Ghat3])
        Mtildelist = np.array([Mhat01, Mhat12, Mhat23, Mhat34])
        Ftipmat = np.ones((np.array(traj).shape[0], 6))
        Kp = 20
        Ki = 10
        Kd = 18
        intRes = 8
        taumat,thetamat \
        = mr.SimulateControl(thetalist, dthetalist, g, Ftipmat, Mlist, \
                             Glist, Slist, thetamatd, dthetamatd, \
                             ddthetamatd, gtilde, Mtildelist, Gtildelist, \
                             Kp, Ki, Kd, dt, intRes)
    """
    Ftipmat = np.array(Ftipmat).T # 末端力向量序列
    thetamatd = np.array(thetamatd).T # 关节目标位置序列
    dthetamatd = np.array(dthetamatd).T # 关节目标速度序列
    ddthetamatd = np.array(ddthetamatd).T # 关节目标加速度序列
    m, n = np.array(thetamatd).shape
    thetacurrent = np.array(thetalist).copy()
    dthetacurrent = np.array(dthetalist).copy()
    eint = np.zeros((m,1)).reshape(m,)
    taumat = np.zeros(np.array(thetamatd).shape)
    thetamat = np.zeros(np.array(thetamatd).shape)
    for i in range(n):
        ## 到达第i个目标位置所需的力矩，使用**计算力矩控制**
        taulist = computed_torque(thetacurrent, dthetacurrent, eint, gtilde, \
                         Mtildelist, Gtildelist, Slist, thetamatd[:, i], \
                         dthetamatd[:, i], ddthetamatd[:, i], Kp, Ki, Kd)
        ## 使用正向动力学计算实际的关节位置和速度
        for j in range(intRes):
            ddthetalist = forward_dynamics(thetacurrent, dthetacurrent, taulist, g,
                                           Ftipmat[:, i], Mlist, Glist, Slist)
            thetacurrent, dthetacurrent = euler_step(thetacurrent, dthetacurrent, 
                                                     ddthetalist, 1.0 * dt / intRes)
        taumat[:, i] = taulist
        thetamat[:, i] = thetacurrent
        # 误差积分eint，用于计算PID的积分增益
        eint = np.add(eint, dt * np.subtract(thetamatd[:, i], thetacurrent))
    # Output using matplotlib to plot
    try:
        import matplotlib.pyplot as plt
    except:
        print('The result will not be plotted due to a lack of package matplotlib')
    else:
        links = np.array(thetamat).shape[0]
        N = np.array(thetamat).shape[1]
        Tf = N * dt
        timestamp = np.linspace(0, Tf, N)
        for i in range(links):
            col = [np.random.uniform(0, 1), np.random.uniform(0, 1),
                   np.random.uniform(0, 1)]
            plt.plot(timestamp, thetamat[i, :], "-", color=col, \
                     label = ("ActualTheta" + str(i + 1)))
            plt.plot(timestamp, thetamatd[i, :], ".", color=col, \
                     label = ("DesiredTheta" + str(i + 1)))
        plt.legend(loc = 'upper left')
        plt.xlabel("Time")
        plt.ylabel("Joint Angles")
        plt.title("Plot of Actual and Desired Joint Angles")
        plt.show()
    taumat = np.array(taumat).T
    thetamat = np.array(thetamat).T
    return (taumat, thetamat)


if __name__ == "__main__":
    trans = np.array([[1, 0,  0, 0],
                              [0, 0, -1, 0],
                              [0, 1,  0, 3],
                              [0, 0,  0, 1]])
    print(trans_to_Rp(trans))
    
    