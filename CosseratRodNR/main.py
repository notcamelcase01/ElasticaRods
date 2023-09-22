import numpy as np
import solver1d as sol
import matplotlib.pyplot as plt
import time

plt.style.use('dark_background')
np.set_printoptions(linewidth=250)

DIMENSIONS = 1
DOF = 6
LOAD_INCREMENTS = 10
MAX_ITER = 10
element_type = 3

L = 1
numberOfElements = 1
icon, node_data = sol.get_connectivity_matrix(numberOfElements, L, element_type)
numberOfNodes = len(node_data)
wgp, gp = sol.init_gauss_points(1)
KG, FG = sol.init_stiffness_force(numberOfNodes, DOF)
u = np.zeros_like(FG)
nodesPerElement = element_type ** DIMENSIONS
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
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
r1 = np.zeros(numberOfNodes)
r2 = np.zeros(numberOfNodes)
r3 = np.zeros(numberOfNodes)
for i in range(numberOfNodes):
    r1[i] = u[DOF * i][0]
    r2[i] = u[DOF * i + 1][0]
    r3[i] = u[DOF * i + 2][0]
ax.plot(r3, r2, label="un-deformed", marker="o")
du = np.zeros_like(u)
tik = time.time()
fapp__ = -np.linspace(0, 10.001, LOAD_INCREMENTS)
for load_iter_ in range(LOAD_INCREMENTS):

    for iter_ in range(MAX_ITER):
        KG, FG = sol.init_stiffness_force(numberOfNodes, DOF)
        for elm in range(numberOfElements):
            n = icon[elm][1:]
            xloc = node_data[n][:, None]
            rloc = np.array([u[6 * n, 0], u[6 * n + 1, 0], u[6 * n + 2, 0]])
            tloc = np.array([u[6 * n + 3, 0], u[6 * n + 4, 0], u[6 * n + 5, 0]])
            drloc = np.array([du[6 * n, 0], du[6 * n + 1, 0], du[6 * n + 2, 0]])
            dtloc = np.array([du[6 * n + 3, 0], du[6 * n + 4, 0], du[6 * n + 5, 0]])
            kloc, floc = sol.init_stiffness_force(nodesPerElement, DOF)
            gloc = np.zeros((6, 1))
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
                gloc[0: 3] = Rot @ (v - np.array([0, 0, 1])[:, None])
                kappa = sol.get_incremental_k_path_independent(dt, tds)
                gloc[3: 6] = Rot @ kappa
                pi = sol.get_pi(Rot)
                n_tensor = sol.get_axial_tensor(gloc[0: 3])
                m_tensor = sol.get_axial_tensor(gloc[3: 6])
                floc += E.T @ gloc * J * wgp[xgp]
                kloc += (E.T @ pi @ Elasticity @ pi.T @ E + sol.get_geometric_tangent_stiffness(E, n_tensor, m_tensor, N_, Nx_, DOF)) * wgp[xgp] * J

            iv = np.array(sol.get_assembly_vector(DOF, n))
            FG[iv[:, None], 0] += floc
            KG[iv[:, None], iv] += kloc
        FG[-5, 0] = fapp__[load_iter_]
        for i in range(6):
            KG, FG = sol.impose_boundary_condition(KG, FG, i, 0)
        du = sol.get_displacement_vector(KG, FG)
        u -= du
    for i in range(numberOfNodes):
        r1[i] = u[DOF * i][0]
        r2[i] = u[DOF * i + 1][0]
        r3[i] = u[DOF * i + 2][0]
    ax.plot(r3, r2)
tok = time.time()
print("Time lapsed (seconds):", tok - tik)

for i in range(numberOfNodes):
    r1[i] = u[DOF * i][0]
    r2[i] = u[DOF * i + 1][0]
    r3[i] = u[DOF * i + 2][0]
ax.plot(r3, r2, label="deformed")
ax.legend()
print(r2[-1])
plt.show()

