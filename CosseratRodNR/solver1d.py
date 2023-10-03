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
    Algorithm proposed by Spurrier
    :param rmat: rotation matrix
    :return: theta
    """
    q = np.zeros(4)
    trq = np.trace(rmat)
    v = np.array([trq, rmat[0, 0], rmat[1, 1], rmat[2, 2]])
    m = v.argmax()
    maxval = v[m]
    if m == 0:
        q[0] = 0.5 * np.sqrt(1 + maxval)
        q[1] = 0.25 * (rmat[2, 1] - rmat[1, 2]) / q[0]
        q[2] = 0.25 * (rmat[0, 2] - rmat[2, 0]) / q[0]
        q[3] = 0.25 * (rmat[1, 0] - rmat[0, 1]) / q[0]
    elif m == 1:
        q[1] = np.sqrt(0.5 * maxval + 0.25 * (1 - trq))
        q[0] = 0.25 * (rmat[2, 1] - rmat[1, 2]) / q[1]
        q[2] = 0.25 * (rmat[0, 2] + rmat[2, 0]) / q[1]
        q[3] = 0.25 * (rmat[1, 0] + rmat[0, 1]) / q[1]
    elif m == 2:
        q[2] = np.sqrt(0.5 * maxval + 0.25 * (1 - trq))
        q[1] = 0.25 * (rmat[2, 1] + rmat[1, 2]) / q[2]
        q[0] = 0.25 * (rmat[0, 2] - rmat[2, 0]) / q[2]
        q[3] = 0.25 * (rmat[1, 0] + rmat[0, 1]) / q[2]
    elif m == 3:
        q[3] = np.sqrt(0.5 * maxval + 0.25 * (1 - trq))
        q[1] = 0.25 * (rmat[2, 1] + rmat[1, 2]) / q[3]
        q[2] = 0.25 * (rmat[0, 2] + rmat[2, 0]) / q[3]
        q[0] = 0.25 * (rmat[1, 0] - rmat[0, 1]) / q[3]
    else:
        raise Exception("not max index")
    normt = 0
    if q[0] >= 0:
        normt = 2 * np.arcsin(np.linalg.norm(q[1:]))
    else:
        normt = 2 * np.pi - 2 * np.arcsin(np.linalg.norm(q[1:]))
    if np.isclose(normt,0, atol=1e-6):
        return np.zeros((3, 3))
    else:
        #print(normt/np.linalg.norm(q[1:]) * q[1:])
        return get_axial_tensor(normt/np.linalg.norm(q[1:]) * q[1:])


def get_theta_from_rotation_deprecated(rmat):
    """
    Lie group log map
    :param rmat: rotation matrix
    :return: theta vector
    """
    t = np.arccos((np.trace(rmat) - 1) / 2)
    if np.isclose(t, 0):
        return np.zeros((3, 3))
    # print(get_axial_from_skew_symmetric_tensor(t * 0.5 / np.sin(t) * (rmat - rmat.T)))
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


def get_rotation_from_theta_tensor_deprecated(x):
    """
    x better be skew symmetric
    :param x: skew symmetric tensor
    :return: rotation tensor
    """
    t = np.sqrt(0.5 * np.trace(x.T @ x))
    if np.isclose(t, 0):
        return np.eye(3)
    return np.eye(3) + np.sin(t) / t * x + (1 - np.cos(t)) / t ** 2 * (x @ x)


def get_rotation_from_theta_tensor(x):
    """
    using quaternions
    :param x: skew symmetric tensor
    :return: rotation tensor
    """
    x = np.reshape(x, (3,))
    normt = np.linalg.norm(x)
    if np.isclose(normt, 0, atol=1e-6):
        return np.eye(3)
    else:
        q = np.zeros(4)
        q[0] = np.cos(normt / 2)
        q[1:] = np.sin(normt / 2) / normt * x
        return 2 * np.array([
            [q[0] ** 2 + q[1] ** 2 - 0.5, q[1] * q[2] - q[3] * q[0], q[1] * q[3] + q[2] * q[0]],
            [q[2] * q[1] + q[3] * q[0], q[0] ** 2 + q[2] ** 2 - 0.5, q[2] * q[3] - q[1] * q[0]],
            [q[3] * q[1] - q[2] * q[0], q[3] * q[2] + q[1] * q[0], q[0] ** 2 + q[3] ** 2 - 0.5]
        ])


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


def get_e_operator(n, nx, dof, rds):
    """
    returns transpose of e operator
    :param n: shape function
    :param nx: derivative of shape function
    :param dof: dof
    :param rds: r\' skew symmetric
    :return: transpose of e operator
    """
    eop = np.zeros((6, dof * len(n)))
    for i in range(len(n)):
        eop[0: 3, i * dof: 3 + i * dof] = nx[i][0] * np.eye(3)
        eop[3: 6, i * dof: 3 + i * dof] = n[i][0] * rds
        eop[3: 6, 3 + i * dof: 6 + i * dof] = nx[i][0] * np.eye(3)
    return eop


def get_incremental_k(dt, dtds, rot):
    """
    According to Simo
    :param dt: delta_theta
    :param dtds: delta_theta'
    :param rot: rotation matrix
    :return: delta_kappa
    """
    norm_dt = np.linalg.norm(dt)
    if np.isclose(norm_dt, 0):
        return dtds
    x = np.sin(norm_dt) / norm_dt
    x2 = np.sin(norm_dt * 0.5) / (norm_dt * 0.5)
    return rot.T @ (x * dtds + (1 - x) * (dt.T @ dtds) / norm_dt * dt / norm_dt + 0.5 * (x2 ** 2) * np.cross(dt, dtds))


def get_incremental_k_path_independent(t, tds):
    """
    According to Crisfield & Jelenic
    :param t: theta
    :param tds: theta_prime
    :return: kappa
    """
    norm_t = np.linalg.norm(t)
    tensor_t = get_axial_tensor(t)
    if np.isclose(norm_t, 0):
        return tds
    x = np.sin(norm_t) / norm_t
    y = (1 - np.cos(norm_t)) / norm_t
    return (1 / norm_t ** 2 * (1 - x) * t @ t.T + x * np.eye(3) - y * tensor_t) @ tds


def get_e(dof, n, n_, rds):
    e = np.zeros((dof,  dof))
    e[0: 3, 0: 3] = n_ * np.eye(3)
    e[3: 6, 3: 6] = n_ * np.eye(3)
    e[3: 6, 0: 3] = n * rds.T
    return e


def get_tangent_stiffness_residue(n_tensor, m_tensor, n, nx, dof, pi, c, rds, gloc):
    """
    :param gloc: gloc
    :param rds: rds
    :param dof: dof
    :param c: elasticity
    :param pi: pi
    :param e: e operator
    :param n_tensor: axial of n
    :param m_tensor: axial of m
    :param n: shape function
    :param nx: derivative of shape function
    :return: geometric stiffness matrix
    """
    nmmat = np.zeros((6, 6))
    nmat = np.zeros((6, 6))
    nmmat[0: 3, 3: 6] = -n_tensor
    nmmat[3: 6, 3: 6] = -m_tensor
    nmat[3: 6, 0: 3] = n_tensor
    k = np.zeros((dof * len(n), dof * len(n)))
    r = np.zeros((dof * len(n), 1))

    for i in range(len(n)):
        r[6 * i: 6 * (i + 1)] += get_e(dof, n[i][0], nx[i][0], rds) @ gloc
        for j in range(len(n)):
            # k[6 * i: (i + 1) * 6, 6 * j: (j + 1) * 6] = n[j][0] * (e[0: 6, 6 * i: (i + 1) * 6]) @ nmmat + n[i][0] * nx[j][0] * nmat + e[0: 6, 6 * i: (i + 1) * 6] @ pi @ c @ pi.T @ e[0: 6, 6 * j: (j + 1) * 6].T
            k[6 * i: (i + 1) * 6, 6 * j: (j + 1) * 6] += get_e(dof, n[i][0], nx[i][0], rds) @ pi @ c @ pi.T @ get_e(dof, n[j][0], nx[j][0], rds).T + n[j][0] * get_e(dof, n[i][0], nx[i][0], rds) @ nmmat + n[i][0] * nx[j][0] * nmat
    return k, r


def get_residue(gloc, dof, e, n):
    """
    :param n: shape fun
    :param gloc: stresses
    :param dof: dof
    :param e: e operator
    :return: residue
    """
    r = np.zeros((dof * len(n), 1))
    for i in range(len(n)):
        r[6 * i: 6 * (i + 1)] = e[0: 6, 6 * i: (i + 1) * 6].T @ gloc
    return r


def get_pi(rot):
    """
    :param rot: rotation
    :return: pi matrix
    """
    pi = np.zeros((6, 6))
    pi[0: 3, 0: 3] = rot
    pi[3: 6, 3: 6] = rot
    return pi


if __name__ == "__main__":
    icon_m, i_m = get_connectivity_matrix(10, 1)
    # print(icon_m)
    # print(i_m)
    b = get_incremental_k(np.array([1, 1, 1]), np.array([1, 1, 2]), np.eye(3))
    print(b)
