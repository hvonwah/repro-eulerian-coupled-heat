from ngsolve import x, y, sqrt, CoefficientFunction

R0 = 0.1
height_0 = 0.8
vel_s_0 = 0
t_end = 1.5

lowerleft, upperright = (0, 0), (1, 1)
boundary_name = 'outer'
dirichlet = boundary_name


def levelset_func(h):
    return R0 - sqrt((x - 0.5)**2 + (y - h)**2)


def rhs_func_ode(h):
    return (0, -1)
