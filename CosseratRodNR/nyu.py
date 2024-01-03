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


np.set_printoptions(linewidth=250)

DIMENSIONS = 1
DOF = 6

MAX_ITER = 50
element_type = 2
L = 100
numberOfElements = 20

icon, node_data = sol.get_connectivity_matrix(numberOfElements, L, element_type)
numberOfNodes = len(node_data)
ngpt = 1
wgp, gp = sol.init_gauss_points(ngpt)
u = np.zeros((numberOfNodes * DOF, 1))
du = np.zeros((numberOfNodes * DOF, 1))
major_kappa = np.zeros((numberOfElements * 3, 1))
nodesPerElement = element_type ** DIMENSIONS
E0 = 10 ** 6
G0 = E0 / 2.0
d = 1 / 1000 * 10.0
A = np.pi * d ** 2 * 0.25
i0 = np.pi * d ** 4 / 64
J = i0 * 2
EI = 3.5 * 10 ** 7
GA = 1.6 * 10 ** 8
ElasticityExtension = np.array([[G0 * A, 0, 0],
                                [0, G0 * A, 0],
                                [0, 0, E0 * A]])
ElasticityBending = np.array([[E0 * i0, 0, 0],
                              [0, E0 * i0, 0],
                              [0, 0, G0 * J]])

ElasticityExtension = np.array([[GA, 0, 0],
                                [0, GA, 0],
                                [0, 0, 2 * GA]])
ElasticityBending = np.array([[EI, 0, 0],
                              [0, EI, 0],
                              [0, 0, 0.5 * EI]])
fig, ax = plt.subplots(1, 1, figsize=(9, 9))
ax.set_xlim(0, L)
"""
Starting point
"""
Elasticity = np.zeros((6, 6))
Elasticity[0: 3, 0: 3] = ElasticityExtension
Elasticity[3: 6, 3: 6] = ElasticityBending
# Elasticity = np.eye(6)
# Elasticity[2, 2] = 10
# Elasticity[5, 5] = 10
vi = np.array([i for i in range(numberOfNodes)])
vii = np.array([i for i in range(numberOfNodes) if i & 1 == 0])
FL = False
"""
Starting point
"""
resn = 0
u *= 0
u[6 * vi + 2, 0] = node_data
u[6 * vi + 4, 0] = 0
# Thetas are zero

r1 = np.zeros(numberOfNodes)
r2 = np.zeros(numberOfNodes)
r3 = np.zeros(numberOfNodes)
for i in range(numberOfNodes):
    r1[i] = u[DOF * i][0]
    r2[i] = u[DOF * i + 1][0]
    r3[i] = u[DOF * i + 2][0]
ax.plot(r3, r2, label="un-deformed", marker="o")
max_load = 100 * 1000
LOAD_INCREMENTS = 101
fapp__ = np.linspace(0, max_load, LOAD_INCREMENTS)


def fea(load_iter_, is_halt=False):
    global u
    global du
    # print("--------------------------------------------------------------------------------------------------------------------------------------------------", load_iter_)
    for iter_ in range(MAX_ITER):
        KG, FG = sol.init_stiffness_force(numberOfNodes, DOF)
        FG[-6:-3] = sol.get_rotation_from_theta_tensor(u[-3:, 0]) @ np.array([0, fapp__[load_iter_], 0])[:, None]

        # print(u[6 * vii + 3, 0] * 180 / np.pi)
        #FG[-5, 0] = fapp__[load_iter_]
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
                dt = dtloc @ N_
                tds = tloc @ Nx_
                rds = rloc @ Nx_
                drds = drloc @ Nx_
                dtds = dtloc @ Nx_

                Rot = sol.get_rotation_from_theta_tensor(t)

                v = Rot.T @ rds
                gloc[0: 3] = Rot @ ElasticityExtension @ (v - np.array([0, 0, 1])[:, None])
                kap = sol.get_incremental_k_path_independent(t, tds)
                major_kappa[3 * elm: 3 * elm + 3] += sol.get_incremental_k(dt, dtds, Rot)
                gloc[3: 6] = Rot @ ElasticityBending @ kap
                pi = sol.get_pi(Rot)
                # if elm == 6 and iter_ == 9:
                #     v0 = v.reshape(3,)
                #     k0 = major_kappa[3 * xgp + 3 * ngpt * elm : 3 * ngpt * elm + 3 * (xgp + 1)].reshape(3,)
                #     df.loc[-1] = [fapp__[load_iter_], elm, v0[0], v0[1], v0[2], k0[0], k0[1], k0[2]]
                #     df.index = df.index + 1  # shifting index

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
            sol.impose_boundary_condition(KG, FG, ibc, 0)
        du = -sol.get_displacement_vector(KG, FG)
        resn = np.linalg.norm(FG)
        # if resn > max_load * 1000000:
        #     FL = True
        #     print("BRK", fapp__[load_iter_])
        #     break
        if np.isclose(resn, 0, atol=0.001):
            break
        deln = np.linalg.norm(du)
        if deln > 1:
            du = du / deln

        for i in range(numberOfNodes):
            xxx = sol.get_theta_from_rotation(
                sol.get_rotation_from_theta_tensor(du[6 * i + 3: 6 * i + 6]) @ sol.get_rotation_from_theta_tensor(
                    u[6 * i + 3: 6 * i + 6]), logg=True)
            # xxx = sol.test_rotation(sol.get_rotation_from_theta_tensor(du[6 * i + 3: 6 * i + 6]) @ sol.get_rotation_from_theta_tensor(
            #    u[6 * i + 3: 6 * i + 6]))
            u[6 * i + 3: 6 * i + 6, 0] = sol.get_axial_from_skew_symmetric_tensor(xxx)
            u[6 * i + 0: 6 * i + 3] += du[6 * i + 0: 6 * i + 3]

    return is_halt


u = np.zeros((numberOfNodes * DOF, 1))
u[6 * vi + 2, 0] = node_data


marker_ = np.linspace(0, max_load, 6)


def act(i):
    print(i)
    global u
    global halt
    halt = fea(i)
    if halt:
        controlled_animation.stop()
        return
    # if i % 4 == 0:
    if np.isclose(fapp__[i], marker_).any():
        y = u[DOF * vi + 1, 0]
        x = u[DOF * vi + 2, 0]
        line1.set_ydata(y)
        line1.set_xdata(x)
        ax.text(x[-5], y[-5], "load : " + str(int(fapp__[i] / 1000)) + "k",  bbox={'facecolor':'white',  'alpha':0.6, 'pad':2})
        ax.plot(x, y, linewidth=3)


ax.set_xlabel(r"$r_3$", fontsize=30)
ax.set_ylabel(r"$r_2$", fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.set_ylim(-85, 41)
y = u[DOF * vi + 1, 0]
x = u[DOF * vi + 2, 0]
line1, = ax.plot(x, y)
controlled_animation = ControlledAnimation(fig, act, frames=len(fapp__), repeat=False)
controlled_animation.start()
print(u[-6:, 0][1])
