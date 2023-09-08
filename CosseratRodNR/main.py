import numpy as np
import solver1d as sol
import matplotlib.pyplot as plt
plt.style.use('dark_background')

MAX_ITER = 10
DOF = 6
numberOfElements = 10
L = 1
element_type = 2
icon, node_data = sol.get_connectivity_matrix(numberOfElements, L)
numberOfNodes = len(node_data)
wgp, gp = sol.init_gauss_points()
KG, FG = sol.init_stiffness_force(numberOfNodes, DOF)
