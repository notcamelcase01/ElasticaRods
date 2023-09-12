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
icon, node_data = sol.get_connectivity_matrix(numberOfElements, L)
numberOfNodes = len(node_data)
wgp, gp = sol.init_gauss_points()
KG, FG = sol.init_stiffness_force(numberOfNodes, DOF)
u = np.zeros_like(FG)
nodesPerElement = element_type

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
        kloc, floc = sol.init_stiffness_force(nodesPerElement, DOF)
        for xgp in range(len(wgp)):
            N, Bmat = sol.get_lagrange_fn(gp[xgp])
            xx = (N.T @ xloc)[0][0]
            J = (Bmat.T @ xloc)[0][0]
            Nx = 1 / J * Bmat
            r = rloc @ N
            dr = rloc @ Nx
            t = tloc @ N
            dt = tloc @ Nx
            dr_x = sol.get_axial_tensor(dr)
            E = sol.get_e_operator(N, Nx, DOF, dr_x)
            print(E)
            break
        break
    break
tok = time.time()
print("Time lapsed (seconds):", tok - tik)