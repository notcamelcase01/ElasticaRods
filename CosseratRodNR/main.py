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
wgp, gp = sol.init_gauss_points(1)
KG, FG = sol.init_stiffness_force(numberOfNodes, DOF)
u = np.zeros_like(FG)
kappa = np.zeros_like(u)
nodesPerElement = element_type
ElasticityExtension = np.array([[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]])
ElasticityBending = np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]])
Elasticity = np.zeros((6, 6))
Elasticity[0: 3, 0: 3] = ElasticityExtension
Elasticity[3: 6, 3: 6] = ElasticityBending


"""
Starting point
"""
for i in range(numberOfNodes):
    u[6 * i, 0] = 0
    u[6 * i + 1, 0] = 0
    u[6 * i + 2, 0] = node_data[i]
    # Thetas are zero
du = np.zeros_like(u)
kappa = np.zeros((numberOfElements * 3, 1))
tik = time.time()
for _ in range(1):
    # Load increment
    for _ in range(MAX_ITER):
        for elm in range(numberOfElements):
            n = icon[elm][1:]
            xloc = node_data[n][:, None]
            rloc = np.array([u[6 * n, 0], u[6 * n + 1, 0], u[6 * n + 2, 0]])
            tloc = np.array([u[6 * n + 3, 0], u[6 * n + 4, 0], u[6 * n + 5, 0]])
            drloc = np.array([du[6 * n, 0], du[6 * n + 1, 0], du[6 * n + 2, 0]])
            dtloc = np.array([du[6 * n + 3, 0], du[6 * n + 4, 0], du[6 * n + 5, 0]])
            kloc, floc = sol.init_stiffness_force(nodesPerElement, DOF)
            gloc = np.zeros((6, 1))
            kappaloc = kappa[3 * elm: 3 * (elm + 1)]
            for xgp in range(len(wgp)):

                N_, Bmat = sol.get_lagrange_fn(gp[xgp], element_type)
                xx = (N_.T @ xloc)[0][0]
                J = (Bmat.T @ xloc)[0][0]
                Nx_ = 1 / J * Bmat

                r = rloc @ N_
                t = tloc @ N_
                rds = rloc @ Nx_
                tds = tloc @ Nx_

                dt = dtloc @ N_
                dr = drloc @ N_
                dtds = dtloc @ Nx_
                drds = drloc @ Nx_

                Rot = sol.get_rotation_from_theta_tensor(sol.get_axial_tensor(t))
                E = sol.get_e_operator(N_, Nx_, DOF, sol.get_axial_tensor(rds))
                v = Rot.T @ rds
                gloc[0: 3] = Rot @ (v - np.array([0, 0, 0])[:, None])
                kappaloc += sol.get_incremental_k(dt, dtds, Rot)
                gloc[3: 6] = Rot @ kappaloc
                floc += E.T @ gloc
                pi = sol.get_pi(Rot)

            iv = np.array(sol.get_assembly_vector(DOF, n))
            FG[iv[:, None], 0] += floc
            KG[iv[:, None], iv] += kloc
            pass
        break
    break

tok = time.time()
print("Time lapsed (seconds):", tok - tik)
