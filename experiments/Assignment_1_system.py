import numpy as np

from direct_stiffness.cross_section import Circle, RectangularPipe
from direct_stiffness.beam import BeamElement
from direct_stiffness.system import Frame


# Define coordinates of nodes (in m)
A = (.4, 0, 0)
B = (.4, .5, 0)
C = (0, .5, 0)
D = E= (0, 0, 0)
F = (-1, 0, 0)

# Create cross sections
s = 45e-3  # Outer width of the square profile [m]
t = 4e-3  # Wall thickness of the square profile [m]
d = 36e-3  # Diameter of the circular profile [m]

circle_profile = Circle(d=d)
square_profile = RectangularPipe(bo=s, t=t)

# Create beam elements
E = 205e9  # Young's modulus [Pa]
nu = .3  # Poisson's ratio [-]
beam_AB = BeamElement(square_profile, E=E, nu=nu, idx=(1, 2), coords=(A, B))
beam_BC = BeamElement(square_profile, E=E, nu=nu, idx=(2, 3), coords=(B, C))
beam_CD = BeamElement(square_profile, E=E, nu=nu, idx=(3, 4), coords=(C, D))
beam_DF = BeamElement(circle_profile, E=E, nu=nu, idx=(4, 5), coords=(D, F))

K_AB = beam_AB.local_stiffness_matrix()
K_AB = beam_AB.global_stiffness_matrix()


# Create frame system
frame = Frame((beam_AB, beam_BC, beam_CD, beam_DF))

K = frame.stiffness_matrix()

# Solve system
P = 3000
fixed_dofs = ("u_x1", "u_y1", "u_z1",
              "u_x4", "u_y4", "u_z4",
              "u_x5", "u_y5", "u_z5", "theta_x5", "theta_y5", "theta_z5")
F = np.zeros((frame.n_nodes, 6))
F[1] = [0, 0, P/2, 0, P*beam_BC.L/8, 0]  # Forces on node B
F[2] = [0, 0, P/2, 0, -P*beam_BC.L/8, 0]  # Forces on node C

U, R = frame.solve_fea_system(F=F.flatten(), fixed_dofs=fixed_dofs, is_3d=True)
U = U.reshape((frame.n_nodes, 6))
