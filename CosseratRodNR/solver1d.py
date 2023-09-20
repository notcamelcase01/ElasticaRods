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
    :return: stiffness matrix and force vector after imposing bc
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


def get_lagrange_fn(gp, element_type=2):
    """
    Linear Lagrange shape functions
    :param element_type: element type
    :param gp: gauss point
    :return: (L, L')
    """
    if element_type == 2:
        nmat = np.array([.5 * (1 - gp), .5 * (1 + gp)])
        bmat = np.array([-.5, .5])
    elif element_type == 3:
        nmat = np.array([0.5 * (-1 + gp) * gp, (-gp + 1) * (gp + 1), 0.5 * gp * (1 + gp)])
        bmat = np.array([0.5 * (-1 + 2 * gp), -2 * gp, 0.5 * (1 + 2 * gp)])
    else:
        raise Exception("Sir, This is Wendy's we don't do more than cubic here !")
    return nmat[:, None], bmat[:, None]


def get_connectivity_matrix(n, length, element_type=2):
    """
    :param element_type: element type
    :param length: length
    :param n: number of 1d elements
    :return: connectivity vector, nodal_data
    """
    node_data = np.linspace(0, length, (element_type - 1) * n + 1)
    icon = np.zeros((element_type + 1, n), dtype=np.int32)
    if element_type == 3:
        icon[0, :] = np.arange(0, n, length)
        icon[1, :] = icon[0, :]
        icon[2, :] = icon[1, :] + 1
        icon[3, :] = icon[2, :] + 1
    elif element_type == 2:
        icon[0, :] = np.arange(0, n, length)
        icon[1, :] = icon[0, :]
        icon[2, :] = icon[1, :] + 1
    else:
        raise Exception("Sir, This is Wendy's we only do cubic elements here !")
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


def get_assembly_vector(dof, n):
    """
    :param dof: dof
    :param n: nodes
    :return: assembly points
    """
    iv = []
    for i in n:
        for j in range(dof):
            iv.append(dof * i + j)
    return iv


def get_e_operator(n, nx, dof, dr):
    """
    returns transpose of e operator
    :param n: shape function
    :param nx: derivative of shape function
    :param dof: dof
    :param dr: r\' skew symmetric
    :return: e operator
    """
    eop = np.zeros((6, dof * len(n)))
    for i in range(len(n)):
        eop[0: 3, i * dof: 3 + i * dof] = n[i][0] * np.eye(3)
        eop[0: 3, 3 + i * dof: 6 + i * dof] = nx[i][0] * dr
        eop[3: 6, 3 + i * dof: 6 + i * dof] = n[i][0] * np.eye(3)
    return eop


def get_incremental_k(dt, dtds, rot):
    norm_dt = np.linalg.norm(dt)
    if np.isclose(norm_dt, 0):
        return dtds
    x = np.sin(norm_dt) / norm_dt
    x2 = np.sin(norm_dt * 0.5) / (norm_dt * 0.5)
    return rot.T @ (x * dtds + (1 - x) * (dt.T @ dtds) / norm_dt * dt / norm_dt + 0.5 * (x2 ** 2) * np.cross(dt, dtds))


def get_pi(rot):
    pi = np.zeros((6, 6))
    pi[0: 3, 0: 3] = rot.T
    pi[3: 6, 3: 6] = rot.T
    return pi


if __name__ == "__main__":
    icon_m, i_m = get_connectivity_matrix(10, 1)
    # print(icon_m)
    # print(i_m)
    b = get_incremental_k(np.array([1, 1, 1]), np.array([1, 1, 2]), np.eye(3))
    print(b)