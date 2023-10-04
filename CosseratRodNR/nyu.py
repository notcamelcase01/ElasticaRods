import numpy as np
import solver1d as sol
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.animation as animation


class ControlledAnimation:
    def __init__(self, figc, animate, frames=100, interval=1, repeat=False):
        self.figc = figc
        self.animate = animate
        self.frames = frames
        self.interval = interval
        self.repeat = repeat
        self.ani = animation.FuncAnimation(self.figc, self.animate, frames=self.frames, interval=self.interval,
                                           repeat=self.repeat)

    def start(self):
        #FFwriter = animation.FFMpegWriter(fps=60)
        #self.ani.save('input_follower.mp4', writer=FFwriter)
        plt.show()

    def stop(self):
        self.ani.event_source.stop()

    def pause(self):
        global pause
        pause ^= True
        if not pause:
            print("halted")
            self.ani.event_source.stop()
        else:
            self.ani.event_source.start()


plt.style.use('dark_background')
np.set_printoptions(linewidth=250)

DIMENSIONS = 1
DOF = 6
halt = False
pause = False
MAX_ITER = 10
element_type = 2
L = 1
numberOfElements = 20

icon, node_data = sol.get_connectivity_matrix(numberOfElements, L, element_type)
numberOfNodes = len(node_data)
wgp, gp = sol.init_gauss_points(1)
vi = np.array([i for i in range(numberOfNodes)])
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
fig, (ax, ay) = plt.subplots(1, 2, figsize=(16, 9))
ax.set_xlim(0, L)
"""
Starting point
"""
max_load = 0.1
LOAD_INCREMENTS = max(100, int(100 / 0.125 * max_load))
fapp__ = -np.linspace(0, max_load, LOAD_INCREMENTS)


def fea(load_iter_, u, is_halt=False):
    for iter_ in range(MAX_ITER):
        KG, FG = sol.init_stiffness_force(numberOfNodes, DOF)
        #FG[-6:-3] = sol.get_rotation_from_theta_tensor(u[-3:, 0]) @ np.array([0, fapp__[load_iter_], 0])[:, None]
        FG[-5, 0] = fapp__[load_iter_]
        for elm in range(numberOfElements):
            n = icon[elm][1:]
            xloc = node_data[n][:, None]
            rloc = np.array([u[6 * n, 0], u[6 * n + 1, 0], u[6 * n + 2, 0]])
            tloc = np.array([u[6 * n + 3, 0], u[6 * n + 4, 0], u[6 * n + 5, 0]])

            kloc, floc = sol.init_stiffness_force(nodesPerElement, DOF)

            gloc = np.zeros((6, 1))
            for xgp in range(len(wgp)):
                N_, Bmat = sol.get_lagrange_fn(gp[xgp], element_type)
                J = (xloc.T @ Bmat)[0][0]
                Nx_ = 1 / J * Bmat

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

        for ibc in range(6):
            KG, FG = sol.impose_boundary_condition(KG, FG, ibc, 0)
        du = -sol.get_displacement_vector(KG, FG)
        resn = np.linalg.norm(FG)
        if resn > max_load * 1000000:
            is_halt = True
            print("BRK", fapp__[load_iter_])
            break
        if np.isclose(resn, 0, atol=0.000001):
            break

        for node in range(numberOfNodes):
            xxx = sol.get_theta_from_rotation(
                sol.get_rotation_from_theta_tensor(du[6 * node + 3: 6 * node + 6]) @ sol.get_rotation_from_theta_tensor(
                    u[6 * node + 3: 6 * node + 6]))
            u[6 * node + 3: 6 * node + 6, 0] = sol.get_axial_from_skew_symmetric_tensor(xxx)
            u[6 * node + 0: 6 * node + 3] += du[6 * node + 0: 6 * node + 3]

    return u, is_halt


displacement = np.zeros((numberOfNodes * DOF, 1))
displacement *= 0
displacement[6 * vi + 2, 0] = node_data
du = np.zeros_like(displacement)


def act(i):
    ax.set_title("load = " + str(np.round(fapp__[i], 4)) + " (" + str(i + 1) + " / " + str(len(fapp__)) + ")")
    global displacement
    global halt
    displacement, halt = fea(i, displacement)
    if halt:
        controlled_animation.stop()
        return
    y = displacement[DOF * vi + 1, 0]
    x = displacement[DOF * vi + 2, 0]
    ax.plot(x, y)
    ay.scatter(abs(fapp__[i]), y[-1])
    ax.set_ylim(-0.01 + np.min(y), np.max(y) + 0.01)


ax.set_xlabel("e3", fontsize=30)
ax.set_ylabel("e2", fontsize=30)
ay.set_xlabel("load", fontsize=30)
ay.set_ylabel("y displacement of tip", fontsize=25)
controlled_animation = ControlledAnimation(fig, act, frames=len(fapp__), interval=1, repeat=False)
controlled_animation.start()
