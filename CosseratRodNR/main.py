import numpy as np
import solver1d as sol
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

plt.style.use('dark_background')
np.set_printoptions(linewidth=250)


DIMENSIONS = 1
DOF = 6

MAX_ITER = 10
element_type = 2
L = 1
numberOfElements = 10

icon, node_data = sol.get_connectivity_matrix(numberOfElements, L, element_type)
numberOfNodes = len(node_data)
wgp, gp = sol.init_gauss_points(1)
u = np.zeros((numberOfNodes * DOF, 1))
du = np.zeros((numberOfNodes * DOF, 1))
major_kappa = np.zeros((numberOfNodes * 3, 1))
nodesPerElement = element_type ** DIMENSIONS
E0 = 10 ** 6
G0 = E0 / 2.0
d = 1 / 1000 * 10.0
A = np.pi * d ** 2 * 0.25
i0 = np.pi * d ** 4 / 64
J = i0 * 2

ElasticityExtension = np.array([[G0 * A, 0, 0],
                                [0, G0 * A, 0],
                                [0, 0, E0 * A]])
ElasticityBending = np.array([[E0 * i0, 0, 0],
                              [0, E0 * i0, 0],
                              [0, 0, G0 * J]])
Elasticity = np.zeros((6, 6))
Elasticity[0: 3, 0: 3] = ElasticityExtension
Elasticity[3: 6, 3: 6] = ElasticityBending
# Elasticity = np.eye(6)
# Elasticity[2, 2] = 10
# Elasticity[5, 5] = 10
vi = np.array([i for i in range(numberOfNodes)])
FL = False
"""
Starting point
"""
resn = 0
u *= 0
u[6 * vi + 2, 0] = node_data
# Thetas are zero
fig, (ax, ay) = plt.subplots(1, 2, figsize=(16, 9))
r1 = np.zeros(numberOfNodes)
r2 = np.zeros(numberOfNodes)
r3 = np.zeros(numberOfNodes)
for i in range(numberOfNodes):
    r1[i] = u[DOF * i][0]
    r2[i] = u[DOF * i + 1][0]
    r3[i] = u[DOF * i + 2][0]
ax.plot(r3, r2, label="un-deformed", marker="o")
max_load = 0.002
LOAD_INCREMENTS = 5
fapp__ = -np.linspace(0, max_load, LOAD_INCREMENTS)
for load_iter_ in range(int(LOAD_INCREMENTS)):
    for iter_ in range(MAX_ITER):
        KG, FG = sol.init_stiffness_force(numberOfNodes, DOF)
        FG[-6:-3] = sol.get_rotation_from_theta_tensor(u[-3:, 0]) @ np.array([0, fapp__[load_iter_], 0])[:, None]
        # FG[-5, 0] = fapp__[load_iter_]
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
                J = (xloc.T @ Bmat)[0][0]
                Nx_ = 1 / J * Bmat
                r = rloc @ N_
                t = tloc @ N_
                dr = rloc @ N_
                dt  = dtloc @ N_
                tds = tloc @ Nx_
                rds = rloc @ Nx_
                drds = drloc @ Nx_
                dtds = dtloc @ Nx_

                Rot = sol.get_rotation_from_theta_tensor(t)

                v = Rot.T @ rds
                gloc[0: 3] = Rot @ ElasticityExtension @ (v - np.array([0, 0, 1])[:, None])
                major_kappa[3 * elm: 3 * (elm + 1)] += sol.get_incremental_k(dt, dtds, Rot)
                gloc[3: 6] = Rot @ ElasticityBending @ major_kappa[3 * elm: 3 * (elm + 1)]
                pi = sol.get_pi(Rot)
                if 4 <= elm <=8 and iter_ == 9 :
                    print(elm, end="   ")

                    print((v - np.array([0, 0, 1])[:, None]).reshape(3,), end="    ")
                n_tensor = sol.get_axial_tensor(gloc[0: 3])
                m_tensor = sol.get_axial_tensor(gloc[3: 6])
                tangent, res = sol.get_tangent_stiffness_residue(n_tensor, m_tensor, N_, Nx_, DOF, pi, Elasticity,
                                                                 sol.get_axial_tensor(rds), gloc)
                floc += res * wgp[xgp] * J
                kloc += tangent * wgp[xgp] * J

            iv = np.array(sol.get_assembly_vector(DOF, n))
            FG[iv[:, None], 0] += floc
            KG[iv[:, None], iv] += kloc

        for ibc in range(6):
            KG, FG = sol.impose_boundary_condition(KG, FG, ibc, 0)
        du = -sol.get_displacement_vector(KG, FG)
        resn = np.linalg.norm(FG)
        if resn > max_load * 1000000:
            FL = True
            print("BRK", fapp__[load_iter_])
            break
        # if np.isclose(resn, 0, atol=0.000001):
        #     print("BRK", fapp__[load_iter_])
        #     break

        for i in range(numberOfNodes):
            xxx = sol.get_theta_from_rotation(sol.get_rotation_from_theta_tensor(du[6 * i + 3: 6 * i + 6]) @ sol.get_rotation_from_theta_tensor(u[6 * i + 3: 6 * i + 6]))
            u[6 * i + 3: 6 * i + 6, 0] = sol.get_axial_from_skew_symmetric_tensor(xxx)
            u[6 * i + 0: 6 * i + 3] += du[6 * i + 0: 6 * i + 3]
    if FL:
        break
    r2 = u[DOF * vi + 1, 0]
    r3 = u[DOF * vi + 2, 0]
    ax.plot(r3, r2)
    print()
    ay.scatter(abs(fapp__[load_iter_]), r2[-1])

r2 = u[DOF * vi + 1, 0]
r3 = u[DOF * vi + 2, 0]
ax.plot(r3, r2, marker="v")
ax.legend()
print(r2[-1], r3[-1], 1 + np.abs(fapp__[-1]) / (E0 * A), fapp__[-1] / (3 * E0 * i0))
print(r2[-1], r3[-1])
plt.show()
