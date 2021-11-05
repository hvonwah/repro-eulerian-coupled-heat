'''
    __author__ = H. von Wahl
    __date__   = 13.08.2021
    __update__ = 26.10.2021

    We consider the heat equation on a moving domain. The motion of the
    domain is driven by the transfer forces acting on a rigid body.

    This script is used to compute a convergence study over a series of
    meshes and time-steps.
'''

# ------------------------------ LOAD LIBRARIES -------------------------------
from netgen.geom2d import SplineGeometry
from solver_bdf2_ho import *
import importlib
import pickle
import time as wall_time

ngsglobals.msg_level = 2
SetNumThreads(20)

start_time = wall_time.time()

# -------------------------------- PARAMETERS ---------------------------------
h0 = 0.1                                # Initial mesh size
dt_inv0 = 50                            # Initial time-step
order = 2                               # Finite element order
example = 'data'                        # The library describing the example

rx = (0, 5)                             # Mesh level range to consider
rt = (0, 5)                             # Time-step range to consider

options = {}
options['max_inner_it'] = 10            # Maximum number of sub-iterations
options['omega_0'] = 0.5                # Initial relaxation parameter
options['gamma'] = 0.01                 # Lagrange stabilisation parameter
options['gamma_gp'] = 0.1               # Ghost-penalty parameter
options['c_delta'] = 2                  # Extension strip parameter
options['inverse'] = 'pardiso'          # Sparse direct solver
options['compile_flag'] = True          # Real-compile integrators
options['wait_compile'] = True          # Wait for compilation to complete


filename_data = 'coupled_heat_problem-raw_data-example_{}'.format(example)
filename_data += 'rx({},{})rt({},{})_k{}h{}dtinv{}BDF2HO.data'.format(
    *rx, *rt, order, h0, dt_inv0)

# ------------------------------ LOAD DATA FILE -------------------------------
try:
    raw_data = pickle.load(open(filename_data, "rb"))
    print("loaded the following data:\n", raw_data)
except OSError:
    raw_data = {}

# ---------------------------- LOAD EXAMPLE DATA  -----------------------------
data = importlib.import_module(example)

# --------------------------------- GEOMETRY ----------------------------------
background_domain = SplineGeometry()
background_domain.AddRectangle(data.lowerleft, data.upperright,
                               bc=data.boundary_name)

# ----------------------------- CONVERGENCE STUDY -----------------------------
tm = TaskManager()
tm.__enter__()

rx = (rx[0], rx[1] + 1)
rt = (rt[0], rt[1] + 1)

for lx in range(*rx):

    # Check if this mesh level is already computed
    if len([i for i in range(*rt) if (lx, i) in raw_data]) == rt[1]:
        continue

    hmax = h0 * 0.5**lx
    mesh = Mesh(background_domain.GenerateMesh(maxh=hmax))

    for lt in range(*rt):

        # Check if this mesh/time-step combination is already computed
        if (lx, lt) in raw_data:
            continue
        else:
            raw_data[(lx, lt)] = {}

        dt = 1 / int(dt_inv0 * 2**lt)

        raw_data[(lx, lt)] = solve_coupled_heat_equation(
            example=example, mesh=mesh, hmax=hmax, order=order, dt=dt,
            **options)

        pickle.dump(raw_data, open(filename_data, 'wb'))
        print('')

# ------------------------------ POST-PROCESSING ------------------------------
run_time = wall_time.time() - start_time
tm.__exit__(None, None, None)

print('\n----------- Total time: {:02.0f}:{:02.0f}:{:02.0f}:{:06.3f}'
      ' ----------'.format(run_time // (24 * 60 * 60),
                           run_time % (24 * 60 * 60) // (60 * 60),
                           run_time % 3600 // 60,
                           run_time % 60))
