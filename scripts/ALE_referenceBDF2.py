'''
    __author__ = H. von Wahl
    __date__   = 28.07.2021

    We consider the head equation on a moving domain. The motion of the
    domain is driven by the transfer forces acting on a rigid body.

    This script uses a fitted ALE discretisation to compute the
    reference motion to compare the unfitted results with.
'''

# ------------------------------ LOAD LIBRARIES -------------------------------
from netgen.geom2d import SplineGeometry
from ngsolve import *
from ngsolve.solvers import PreconditionedRichardson as PreRic
from General_Utilities import RemoveCompileGeneratedFiles

import argparse

import time as wall_time
start_time = wall_time.time()

ngsglobals.msg_level = 2
SetNumThreads(20)

parser = argparse.ArgumentParser()
parser.add_argument('--h', help='mesh size', type=float)
parser.add_argument('--k', help='order', type=int)
parser.add_argument('--dtinv', help='invserse time-step', type=int)
args = parser.parse_args()


# -------------------------------- PARAMETERS ---------------------------------
hmax = args.h                                   # Global mesh size
ref_cir = 3                                     # Mesh refinements at ball
order = args.k                                  # Order of finite element space

dt_inv = args.dtinv                             # Inverse time-step
dt = 1 / dt_inv                                 # Time-step

max_inner_it = 10                               # Max number of ODE iterations
omega_0 = 0.75                                  # Relaxation parameter
ode_tol = 1e-8                                  # ODE iteration tolerance

compile_flag = True                             # Real-compile integrators
wait_compile = False                            # Wait for compile to complete
condense_flag = True                            # Use static condensation
inverse = 'pardiso'                             # Sparse direct solver

file_out = f'CoupledHeatEquation_h{hmax}k{order}dtinv{dt_inv}bdf2_defset.txt'


# ----------------------------------- DATA ------------------------------------
t_end = 1.0                                     # End time
h0 = 0.8                                        # Initial centre of solid

vel_s_0 = 0                                     # Initial solid velocity
forcing = -1                                    # ODE right-hand side


# ----------------------------- GEOMETRY AND MESH -----------------------------
geo = SplineGeometry()
_c, _s = 0.5, 0.15
pnts = [(0, 0), (1, 0), (1, _c - _s), (1, _c + _s),
        (1, 1), (0, 1), (0, _c + _s), (0, _c - _s)]
p0, p1, p2, p3, p4, p5, p6, p7 = [geo.AppendPoint(*pnt) for pnt in pnts]

geo.Append(['line', p0, p1], leftdomain=1, rightdomain=0, bc='bottom')
geo.Append(['line', p1, p2], leftdomain=1, rightdomain=0, bc='right')
geo.Append(['line', p2, p3], leftdomain=2, rightdomain=0, bc='right')
geo.Append(['line', p3, p4], leftdomain=1, rightdomain=0, bc='right')
geo.Append(['line', p4, p5], leftdomain=1, rightdomain=0, bc='top')
geo.Append(['line', p5, p6], leftdomain=1, rightdomain=0, bc='left')
geo.Append(['line', p6, p7], leftdomain=2, rightdomain=0, bc='left')
geo.Append(['line', p7, p0], leftdomain=1, rightdomain=0, bc='left')
geo.Append(['line', p2, p7], leftdomain=1, rightdomain=2)
geo.Append(['line', p3, p6], leftdomain=2, rightdomain=1)

geo.AddCircle((_c, _c), r=0.1, leftdomain=0, rightdomain=2, bc='ball',
              maxh=hmax * 2**-ref_cir)

geo.SetDomainMaxH(1, hmax)
geo.SetDomainMaxH(2, hmax / 2)

with TaskManager():
    mesh = Mesh(geo.GenerateMesh())

mesh.Curve(order)


# ---------------------------------- MAPPING ----------------------------------
def abs_cf(x):
    return IfPos(x, x, -x)


ramp = (1 - IfPos(abs_cf(y - _c) - _s, (abs_cf(y - _c) - _s) / (_c - _s), 0))


def deformation_func(h):
    return CF((0, (h - _c) * ramp))


Vset = H1(mesh, order=3, dirichlet=".*", dim=2)
gfset = GridFunction(Vset)
gfset_old, gfset_old2 = GridFunction(Vset), GridFunction(Vset)
gfset.Set(deformation_func(h0))
h = Parameter(h0)

# --------------------------- FINITE ELEMENT SPACE ----------------------------
V = H1(mesh, order=order, dirichlet='bottom|right|top|left|ball')

gfu, gfu_old, gfu_old2 = GridFunction(V), GridFunction(V), GridFunction(V)
gfu.vec[:] = 0.0

freedofs = V.FreeDofs(coupling=condense_flag)


# ----------------------------- (BI)LINEAR FORMS ------------------------------
u, v = V.TnT()

mass = u * v
diffusion = InnerProduct(Grad(u), Grad(v))

mesh_vel1 = - InnerProduct(Grad(u), 1 / dt * (gfset - gfset_old)) * v
mesh_vel2 = - InnerProduct(Grad(u), 1 / dt * 3 / 2 * gfset) * v
mesh_vel2 += InnerProduct(Grad(u), 1 / dt * 2 * gfset_old) * v
mesh_vel2 += - InnerProduct(Grad(u), 1 / dt * 1 / 2 * gfset_old2) * v

mass_old = gfu_old * v
mass_old2 = gfu_old2 * v


# -------------------------------- INTEGRATORS --------------------------------
compile_opts = {'realcompile': compile_flag, 'wait': wait_compile}

mstar1 = BilinearForm(V, condense=condense_flag)
mstar1 += (mass + dt * (diffusion + mesh_vel1)).Compile(**compile_opts) * dx(deformation=gfset)

mstar2 = BilinearForm(V, condense=condense_flag)
mstar2 += (3 / 2 * mass + dt * (diffusion + mesh_vel2)).Compile(**compile_opts) * dx(deformation=gfset)

f1 = LinearForm(V)
f1 += mass_old.Compile(**compile_opts) * dx(deformation=gfset)

f2 = LinearForm(V)
f2 += (2 * mass_old - 1 / 2 * mass_old2).Compile(**compile_opts) * dx(deformation=gfset)


# -------------------------- FUNCTIONAL COMPUTATION ---------------------------
res = gfu.vec.CreateVector()

drag_test = GridFunction(V)
drag_test.Set(1, definedon=mesh.Boundaries('ball'))


def comp_drag(mstar, f):
    mstar.Apply(gfu.vec, res)
    res.data -= f.vec
    drag_x = 0.0
    drag_y = -InnerProduct(res, drag_test.vec) / dt

    return drag_x, drag_y


# ---------------------------------- OUTPUT -----------------------------------
with open(file_out, 'w') as fid:
    fid.write('time height vel dragX dragY\n')


def write_output():
    str_out = f'{time:13.11f} {h.Get():13.11e} {vel_s:13.11e}'
    str_out += f' {drag_x:13.11e} {drag_y:13.11e}\n'

    with open(file_out, 'a') as fid:
        fid.write(str_out)

    return None


# ------------------------------- VISUALISATION -------------------------------
Draw(gfu, mesh, 'gfu')


# ------------------------------- TIME STEPPING -------------------------------
with TaskManager():
    vel_s = vel_s_0
    drag_x, drag_y = 0, 0
    time = 0.0
    write_output()

    # Time Loop

    # BDF1 Step
    time = dt
    h_old = h.Get()
    vel_s_old = vel_s
    gfset_old.vec.data = gfset.vec
    gfu_old.vec.data = gfu.vec

    vel_res = []
    omega = omega_0

    for inner_it in range(max_inner_it):
        vel_s_tmp = vel_s_old + dt * (forcing + drag_y)

        # Aitken Delta^2 relaxation
        vel_res.append(vel_s_tmp - vel_s)
        if inner_it > 0:
            omega *= vel_res[-2] / (vel_res[-1] - vel_res[-2])
        omega = min(1.0, abs(omega))

        # Velocity update
        vel_s_new = (1 - omega) * vel_s + omega * vel_s_tmp

        update = vel_s_new - vel_s
        if abs(update / vel_s_new) < ode_tol:
            print(f' ODE converged after {inner_it} iterations')
            break

        vel_s = vel_s_new
        h.Set(h_old + dt * vel_s)
        gfset.Set(deformation_func(h))

        gfu.Set(CF(vel_s), definedon=mesh.Boundaries('ball'))

        f1.Assemble()
        mstar1.Assemble()

        res.data = f1.vec - mstar1.mat * gfu.vec

        if condense_flag:
            res.data += mstar1.harmonic_extension_trans * res

        inv = mstar1.mat.Inverse(freedofs, inverse=inverse)
        gfu.vec.data += PreRic(mstar1, res, inv, freedofs, printing=False)

        if condense_flag:
            gfu.vec.data += mstar1.harmonic_extension * gfu.vec
            gfu.vec.data += mstar1.inner_solve * res

        drag_x, drag_y = comp_drag(mstar1, f1)
    else:
        str_out = f' WARNING: ODE did not converged after {max_inner_it}'
        str_out += f' iterations: last update {update:4.2e}'
        print(str_out)

    Redraw(blocking=True)
    write_output()

    print(f't={dt:5.3f}, h={h.Get():6.4f}, v={vel_s:6.4f} ', end='')
    print(f'F_x={drag_x:4.2e}, F_y={drag_y:8.6e} - 1')

    del inv, mstar1, f1

    # BDF2 Time-loop
    for it in range(2, int(t_end * dt_inv) + 1):
        time = it * dt
        h_old2 = h_old
        h_old = h.Get()
        vel_s_old2 = vel_s_old
        vel_s_old = vel_s
        gfset_old2.vec.data = gfset_old.vec
        gfset_old.vec.data = gfset.vec
        gfu_old2.vec.data = gfu_old.vec
        gfu_old.vec.data = gfu.vec

        vel_res = []
        omega = omega_0

        for inner_it in range(max_inner_it):
            vel_s_tmp = 4 / 3 * vel_s_old - 1 / 3 * vel_s_old2
            vel_s_tmp += 2 / 3 * dt * (forcing + drag_y)

            # Aitken Delta^2 relaxation
            vel_res.append(vel_s_tmp - vel_s)
            if inner_it > 0:
                omega *= vel_res[-2] / (vel_res[-1] - vel_res[-2])
            omega = min(1.0, abs(omega))

            # Velocity update
            vel_s_new = (1 - omega) * vel_s + omega * vel_s_tmp

            update = vel_s_new - vel_s
            if abs(update / vel_s_new) < ode_tol:
                print(f' ODE converged after {inner_it} iterations')
                break

            vel_s = vel_s_new
            h.Set(4 / 3 * h_old - 1 / 3 * h_old2 + 2 / 3 * dt * vel_s)
            gfset.Set(deformation_func(h))

            gfu.Set(CF(vel_s), definedon=mesh.Boundaries('ball'))

            f2.Assemble()
            mstar2.Assemble()

            res.data = f2.vec - mstar2.mat * gfu.vec

            if condense_flag:
                res.data += mstar2.harmonic_extension_trans * res

            inv = mstar2.mat.Inverse(freedofs, inverse=inverse)
            gfu.vec.data += PreRic(mstar2, res, inv, freedofs, printing=False)

            if condense_flag:
                gfu.vec.data += mstar2.harmonic_extension * gfu.vec
                gfu.vec.data += mstar2.inner_solve * res

            drag_x, drag_y = comp_drag(mstar2, f2)
        else:
            str_out = f' WARNING: ODE did not converged after {max_inner_it}'
            str_out += f' iterations: last update {update:4.2e}'
            print(str_out)

        Redraw(blocking=True)
        write_output()

        print(f't={it * dt:5.3f}, h={h.Get():6.4f}, v={vel_s:6.4f} ', end='')
        print(f'F_x={drag_x:4.2e}, F_y={drag_y:8.6e}')


# ------------------------------ POST-PROCESSING ------------------------------
run_time = wall_time.time() - start_time

if compile_flag:
    RemoveCompileGeneratedFiles()

print('\n----------- Total time: {:02.0f}:{:02.0f}:{:02.0f}:{:06.3f}'
      ' ----------'.format(run_time // (24 * 60 * 60),
                           run_time % (24 * 60 * 60) // (60 * 60),
                           run_time % 3600 // 60,
                           run_time % 60))
