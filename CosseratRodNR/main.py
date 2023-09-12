import numpy as np
import solver1d as sol
import matplotlib.pyplot as plt
import time

plt.style.use('dark_background')
np.set_printoptions(linewidth=250)

MAX_ITER = 10
DOF = 6
numberOfElements = 10
L = 1
element_type = 2
icon, node_data = sol.get_connectivity_matrix(numberOfElements, L, element_type)
numberOfNodes = len(node_data)
wgp, gp = sol.init_gauss_points()
KG, FG = sol.init_stiffness_force(numberOfNodes, DOF)
u = np.zeros_like(FG)
kappa = np.zeros((3 * numberOfNodes, 1))
nodesPerElement = element_type
ElasticityExtension = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]])
ElasticityBending = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]])

"""
Starting point
"""
for i in range(numberOfNodes):
    u[6 * i, 0] = 0
    u[6 * i + 1, 0] = 0
    u[6 * i + 2, 0] = node_data[i]
    # Thetas are zero


tik = time.time()
for _ in range(MAX_ITER):
    for elm in range(numberOfElements):
        n = icon[elm][1:]
        xloc = node_data[n][:, None]
        rloc = np.array([u[6 * n, 0], u[6 * n + 1, 0], u[6 * n + 2, 0]])
        tloc = np.array([u[6 * n + 3, 0], u[6 * n + 4, 0], u[6 * n + 5, 0]])
        kappaloc = np.array([kappa[3 * n, 0], kappa[3 * n + 1, 0], kappa[3 * n + 2, 0]])
        kloc, floc = sol.init_stiffness_force(nodesPerElement, DOF)
        gloc = np.zeros_like(floc)
        for xgp in range(len(wgp)):
            N, Bmat = sol.get_lagrange_fn(gp[xgp], element_type)
            xx = (N.T @ xloc)[0][0]
            J = (Bmat.T @ xloc)[0][0]
            Nx = 1 / J * Bmat
            r = rloc @ N
            dr = rloc @ Nx
            t = tloc @ N
            dt = tloc @ Nx
            R = sol.get_rotation_from_theta_tensor(sol.get_axial_tensor(t))
            dr_x = sol.get_axial_tensor(dr)
            E = sol.get_e_operator(N, Nx, DOF, dr_x)

            for i in range(element_type):
                Ri = sol.get_rotation_from_theta_tensor(sol.get_axial_tensor(tloc[:, [i]]))
                gloc[DOF * i: 3 + DOF * i] = Ri @ ElasticityExtension @ (Ri.T @ (dr @ N.T)[:, [i]] - np.array([0, 0, 1])[:, None])
                gloc[3 + DOF * i: 6 + DOF * i] = Ri @ ElasticityBending @ kappaloc[:, [i]]
            Ri = E @ gloc
            floc += ((Ri @ N.T).T).reshape((nodesPerElement * DOF, 1)) * wgp[xgp] * J
            pass
        iv = np.array(sol.get_assembly_vector(DOF, n))
        FG[iv[:, None], 0] += floc
        KG[iv[:, None], iv] += kloc
        pass
    break
tok = time.time()
print("Time lapsed (seconds):", tok - tik)
