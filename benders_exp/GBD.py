from utils import nlp
import casadi as ca
import numpy as np

# ---- slave -----

c = [4.1, 4] # Center point
Z = [-10, 10] # Box constraint
Y = [0, 10] # Box constraint

lam = 1000
r = 3

y = ca.SX.sym('y', 2, 1)
z = ca.SX.sym('z', 1, 1)

# Slave problem is called (1-y) in the paper, the value of which is v(y)
slave = nlp('slave')
slave.var = ca.vertcat(z)
slave.f = (y[0] - c[0])**2 + (y[1] - c[1])**2 + lam * z
slave.cons = [(-np.inf, -z, 0),
              (-np.inf, y[0]**2 + y[1]**2 - r**2 - z, 0),
              (Z[0], z, Z[1])]
slave.parameter = y
slave.solver_opt.update({'ipopt':{'print_level':0},'print_time':0})
slave.build()

v = lambda p: slave.solve(p)

# ---- master -----

master = nlp('master')

y0 = ca.SX.sym('y0')
mu = ca.SX.sym('mu', 2, 1)
nu = ca.SX.sym('nu', 2, 1)

# L1 is L^* from the paper
L1 = (y[0] - c[0])**2 + (y[1] - c[1])**2 + \
    mu[1] * (y[0]**2 + y[1]**2 - r**2) + (lam - (mu[0] + mu[1])) * Z[0]

# L2 is L_* from the paper
L2 = nu[1] * (y[0]**2 + y[1]**2 - r**2) - Z[1]

# They are linearized
L1_lin = nlp.linearize(L1, [y, mu], y, 'L1_lin')
L2_lin = nlp.linearize(L2, [y, nu], y, 'L2_lin')

# master problem is defined
master.solver_name = 'bonmin'
master.solver_opt = {'discrete':[False, True, True]}
master.var = ca.vertcat(y0, y)
master.f = y0
master.cons = [(Y[0], y, Y[1])]

# ----- Benders cuts

# an initial point
y_bar = [1, 1]
v(y_bar)
mu_bar = slave.solution['lam_g'][0:mu.shape[0]]
UBD = slave.solution['f']
LBD = -np.inf
eps = 1e-3 # tolerance
iteration = 0

while LBD + eps <= UBD:
    iteration += 1
    successful = slave.return_stats()['success']
    
    if successful:
        # The new cut is added
        UBD = slave.solution['f']
        master.cons += [(-np.inf, L1_lin(y, y_bar, mu_bar) - y0, 0)]
    else:
        # the violated constraints are added
        nu_bar = slave.solution['g'][0:mu.shape[0]] > 0
        master.cons += [(-np.inf, L2_lin(y, y_bar, nu_bar), 0)]
        
    master.build()
    master.solve()
    
    LBD = master.solution['x'][0]
    y_bar = master.solution['x'][1:]
    v(y_bar)
    mu_bar = slave.solution['lam_g'][0:mu.shape[0]]
    print(f'y_bar: {y_bar}, LBD: {LBD}, UBD: {UBD}')

print(f'solution found: {y_bar} after {iteration} cuts.')