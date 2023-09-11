import numpy as np


def init_gauss_points(n=3):
    """
    Gauss Quadrature
    :param n: number of gauss points
    :return: (weights of gp,Gauss points)
    """
    if n == 1:
        wgp = np.array([2])
        egp = np.array([0])
    elif n == 2:
        wgp = np.array([1, 1])
        egp = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
    elif n == 3:
        wgp = np.array([5 / 9, 8 / 9, 5 / 9])
        egp = np.array([-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)])
    else:
        raise Exception("Uhm, This is wendy's, we don't, more than 3 gauss points here")
    return wgp, egp


def impose_boundary_condition(k, f, ibc, bc):
    """
    Elimination of variables
    :param k: Stiffness matrix
    :param f: force vector
    :param ibc: node at with BC is prescribed
    :param bc: boundary condition
    :return: stiffness matrix and force vector after imposed bc
    """
    f = f - (k[:, ibc] * bc)[:, None]
    f[ibc] = bc
    k[:, ibc] = 0
    k[ibc, :] = 0
    k[ibc, ibc] = 1
    return k, f


def get_displacement_vector(k, f):
    """
    :param k: Non-singular stiffness matrix
    :param f: force vector
    :return: nodal displacement
    """
    return np.linalg.solve(k, f)


def get_lagrange_fn(gp):
    """
    Linear Lagrange shape functions
    :param gp: gauss point
    :return: (L, L')
    """
    nmat = np.array([.5 * (1 - gp), .5 * (1 + gp)])
    bmat = np.array([-.5, .5])

    return nmat[:, None], bmat[:, None]


def get_connectivity_matrix(n, length):
    """
    :param length: length
    :param n: number of 1d elements
    :return: connectivity vector, nodal_data
    """
    node_data = np.linspace(0, n, n + 1)
    icon = np.zeros((3, n), dtype=np.int32)
    icon[0, :] = np.arange(0, n, length)
    icon[1, :] = icon[0, :]
    icon[2, :] = icon[1, :] + 1
    return icon.T, node_data


def init_stiffness_force(nnod, dof):
    """
    :param nnod: number of nodes
    :param dof: Dof
    :return: zero stiffness n force
    """
    return np.zeros((nnod * dof, nnod * dof)), np.zeros((nnod * dof, 1))


def get_theta_from_rotation(rmat):
    """
    :param rmat: rotation matrix
    :return: theta vector
    """
    """
    **Without Lie group log map**
    
    theta1 = np.zeros(3)
    if rmat[2, 0] != 1 or rmat[2, 0] != -1:
        theta1[1] = np.pi - np.arcsin(rmat[2, 0])
        theta1[0] = np.arctan2(rmat[2, 1] / np.cos(theta1[1]), rmat[2, 1] / np.cos(theta1[1]))
        theta1[2] = np.arctan2(rmat[1, 0] / np.cos(theta1[1]), rmat[1, 0] / np.cos(theta1[1]))
    else:
        theta1[2] = 0
        if rmat[2, 0] == -1:
            theta1[1] = np.pi / 2
            theta1[0] = theta1[2] + np.arctan2(rmat[0, 1], rmat[0, 2])
        else:
            theta1[1] = -np.pi / 2
            theta1[0] = -theta1[2] + np.arctan2(-rmat[0, 1], -rmat[0, 2])
    return theta1
    """
    t = np.arccos((np.trace(rmat) - 1) / 2)
    if np.isclose(t, 0):
        return np.zeros((3, 3))
    return t * 0.5 / np.sin(t) * (rmat - rmat.T)


def get_axial_tensor(x):
    """
    :param x: vector
    :return: skew symmetric tensor for which x is axial
    """
    x = np.reshape(x, (3,))
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]]
                    )


def get_axial_from_skew_symmetric_tensor(x):
    """
    x better be skew symmetric tensor
    :param x: skew symmetric tensor
    :return: axial vector
    """
    return np.array([x[2, 1], x[0, 2], x[1, 0]])


def get_rotation_from_theta_tensor(x):
    """
    x better be skew symmetric
    :param x: skew symmetric tensor
    :return: rotation tensor
    """
    t = np.sqrt(0.5 * np.trace(x.T @ x))
    if np.isclose(t, 0):
        return np.eye(3)
    return np.eye(3) + np.sin(t) / t * x + (1 - np.cos(t)) / t ** 2 * (x @ x)
