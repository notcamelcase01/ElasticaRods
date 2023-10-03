import numpy as np
import solver1d as sol
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.style.use('dark_background')
np.set_printoptions(linewidth=250)
DIMENSIONS = 1
DOF = 6

MAX_ITER = 10
element_type = 2
L = 1
numberOfElements = 20

icon, node_data = sol.get_connectivity_matrix(numberOfElements, L, element_type)
numberOfNodes = len(node_data)
wgp, gp = sol.init_gauss_points(1)
KG, FG = sol.init_stiffness_force(numberOfNodes, DOF)
u = np.zeros_like(FG)
nodesPerElement = element_type ** DIMENSIONS
E0 = 10 ** 6
G0 = E0 / 2.0
d = 1 / 1000 * 10.0
A = np.pi * d ** 2 * 0.25
I = np.pi * d ** 4 / 64
J = I * 2

ElasticityExtension = np.array([[G0 * A, 0, 0],
                                [0, G0 * A, 0],
                                [0, 0, E0 * A]])
ElasticityBending = np.array([[E0 * I, 0, 0],
                              [0, E0 * I, 0],
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
max_load = 0.01
LOAD_INCREMENTS = max(100, int(100/0.125 * max_load))
fapp__ = -np.linspace(0, max_load, LOAD_INCREMENTS)
for load_iter_ in tqdm(range(int(LOAD_INCREMENTS)), colour="GREEN"):

    for iter_ in range(MAX_ITER):
        KG, FG = sol.init_stiffness_force(numberOfNodes, DOF)
        #FG[-6:-3] = sol.get_rotation_from_theta_tensor(u[-3:, 0]) @ np.array([0, fapp__[load_iter_], 0])[:, None]
        FG[-5, 0] = fapp__[load_iter_]
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
                J = (xloc.T @ Bmat)[0][0]
                Nx_ = 1 / J * Bmat

                r = rloc @ N_
                t = tloc @ N_
                rds = rloc @ Nx_
                tds = tloc @ Nx_

                Rot = sol.get_rotation_from_theta_tensor(t)

                v = Rot.T @ rds
                gloc[0: 3] = Rot @ ElasticityExtension @ (v - np.array([0, 0, 1])[:, None])
                kappa = sol.get_incremental_k_path_independent(t, tds)
                gloc[3: 6] = Rot @ ElasticityBending @ kappa
                pi = sol.get_pi(Rot)
                n_tensor = sol.get_axial_tensor(gloc[0: 3])
                m_tensor = sol.get_axial_tensor(gloc[3: 6])
                tangent, res = sol.get_tangent_stiffness_residue(n_tensor, m_tensor, N_, Nx_, DOF, pi, Elasticity,
                                                                 sol.get_axial_tensor(rds), gloc)
                floc += res * wgp[xgp] * J
                kloc += tangent * wgp[xgp] * J

            iv = np.array(sol.get_assembly_vector(DOF, n))
            FG[iv[:, None], 0] += floc
            KG[iv[:, None], iv] += kloc

        for i in range(6):
            KG, FG = sol.impose_boundary_condition(KG, FG, i, 0)
        du = -sol.get_displacement_vector(KG, FG)
        resn = np.linalg.norm(FG)
        if resn > 1:
            FL = True
            print("BRK")
            break
        if np.isclose(resn, 0, atol=0.000001):
            break

        for i in range(numberOfNodes):
            xxx = sol.get_theta_from_rotation(sol.get_rotation_from_theta_tensor(du[6 * i + 3: 6 * i + 6]) @ sol.get_rotation_from_theta_tensor(u[6 * i + 3: 6 * i + 6]))
            u[6 * i + 3: 6 * i + 6, 0] = sol.get_axial_from_skew_symmetric_tensor(xxx)
            u[6 * i + 0: 6 * i + 3] += du[6 * i + 0: 6 * i + 3]
    if FL:
        break
    r2 = u[DOF * vi + 1, 0]
    r3 = u[DOF * vi + 2, 0]
    ax.plot(r3, r2)

ax.legend()
print(r2[-1], r3[-1], 1 + np.abs(fapp__[-1]) / (E0 * A), fapp__[-1] / (3 * E0 * I))
print(r2[-1], r3[-1])
plt.show()
