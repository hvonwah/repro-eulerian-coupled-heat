from ngsolve import *
from ngsolve.solvers import PreconditionedRichardson as PreRic
from xfem import *
from xfem.lsetcurv import *
import importlib


def jump(v):
    return v - v.Other()


def solve_coupled_heat_equation(
        example, mesh, hmax, order, dt, max_inner_it=10, omega_0=0.5,
        ode_tol=1e-8, gamma_n=40, gamma_gp=1.0, c_delta=4.0, inverse='umfpack',
        compile_flag=False, wait_compile=False):
    '''
    Solves the heat equation on a moving domain using an unfitted finite
    element method, where the domain motion is given by the solution of
    an ODE, which in turn depends on the stress from the PDE solution on
    the moving interface. Both the ODE and PDE problem are solved using
    the BDF2 scheme. The scheme is initialised with the BDF1 scheme. To
    achieve higher-order convergence in space, an isoparametric CutFEM
    approach is utilised. The Dirichlet boundary condition on the moving
    interface is implemented using Nitsche's method.

    Parameters
    ----------
    example : str
        name / path of the python module containing the problem specific
        data.
    mesh: ngsolve.Mesh
        The mesh to solve the problem on.
    hmax : float
        Mesh size (to compute extension-strip width).
    order : int
        The polynomial order of the velocity finite element space.
    dt : float
        Time-step size.
    max_inner_it : int
        Maximum number of relaxation sub-steps. Default: 10.
    omega_0 : float (0,1]
        Initial relaxation parameter. Default: 0.5.
    ode_tol : float
        Relative tolerance to solve solid ODE to. Default: 1e-8.
    gamma_n : float
        Nische penalty parameter. Default: 40
    gamma_gp : float
        Ghost-penalty parameter. Default: 1.0.
    c_delta : float
        Extension strip coefficient. Default: 4.0.
    inverse : string
        Direct solver to solve linear systems with. Default: 'umfpack'
    compile: boolean
        Real-compile coefficient functions. Default: False.
    wait_compile : boolean
        Wait for compile to complete. Default: False.
    Returns
    -------
    dictionary('time': [], 'height' [], 'vel': [], 'dragX': [], 'dragY': [])
    '''

    # Load problem specific data ----------------------------------------------
    data = importlib.import_module(example)
    height_0 = data.height_0
    vel_s_0 = data.vel_s_0
    t_end = data.t_end

    height = Parameter(height_0)
    levelset_func = data.levelset_func(height)
    rhs_func_ode = data.rhs_func_ode(height)

    dirichlet = data.dirichlet

    # Set-up finite element space and related variables -----------------------
    V = VectorH1(mesh, order=order, dirichlet=dirichlet, dgjumps=True)

    gfu = GridFunction(V)
    gfu.vec[:] = 0.0
    vel, lagr = gfu.components
    gfu_last, gfu_last2 = GridFunction(V), GridFunction(V)

    gfu_last_on_new_mesh = GridFunction(V)
    gfu_last2_on_new_mesh = GridFunction(V)

    active_dofs = BitArray(V.ndof)

    # Set up mesh-deformation -------------------------------------------------
    lsetmeshadap = LevelSetMeshAdaptation(mesh, order=order, threshold=0.1,
                                          discontinuous_qn=True,
                                          levelset=levelset_func)
    deformation = lsetmeshadap.deform
    deform_last = GridFunction(deformation.space)
    deform_last2 = GridFunction(deformation.space)
    lsetp1 = lsetmeshadap.lset_p1

    # Set-up cut-info and element markers -------------------------------------
    ci_main, ci_inner, ci_outer = CutInfo(mesh), CutInfo(mesh), CutInfo(mesh)

    els_hasneg = ci_main.GetElementsOfType(HASNEG)
    els_if = ci_main.GetElementsOfType(IF)
    els_outer = ci_outer.GetElementsOfType(HASNEG)
    els_inner = ci_inner.GetElementsOfType(NEG)

    els_ext = BitArray(mesh.ne)
    facets_ext = BitArray(mesh.nedge)
    els_outer_old, els_outer_old2 = BitArray(mesh.ne), BitArray(mesh.ne)
    els_test = BitArray(mesh.ne)

    # Set-up (bi-)linear forms ------------------------------------------------
    u, v = V.TnT()

    h = specialcf.mesh_size
    n_lset = Normalize(grad(lsetp1))
    K = Parameter(1)

    vel_solid = Parameter(vel_s_0)
    vel_solid_cf = CoefficientFunction((0, vel_solid))

    mass = InnerProduct(u, v)
    diffusion = InnerProduct(Grad(u), Grad(v))
    nitsche = -InnerProduct(Grad(u) * n_lset, v)
    nitsche += -InnerProduct(Grad(v) * n_lset, u)
    nitsche += (gamma_n * order**2 / h) * InnerProduct(u, v)
    ghost_penalty_ext = gamma_gp * K / h**2 * InnerProduct(jump(u), jump(v))

    mass_old = InnerProduct(gfu_last_on_new_mesh, v)
    mass_old2 = InnerProduct(gfu_last2_on_new_mesh, v)
    nitsche_rhs = -InnerProduct(Grad(v) * n_lset, vel_solid_cf)
    nitsche_rhs += gamma_n * order**2 / h * InnerProduct(vel_solid_cf, v)

    # Integrators -------------------------------------------------------------
    dx = dCut(levelset=lsetp1, domain_type=NEG, definedonelements=els_hasneg,
              deformation=deformation)
    ds = dCut(levelset=lsetp1, domain_type=IF, definedonelements=els_if,
              deformation=deformation)
    dw_ext = dFacetPatch(definedonelements=facets_ext, deformation=deformation)

    compile_opts = {'realcompile': compile_flag, 'wait': wait_compile}

    a1 = RestrictedBilinearForm(V, element_restriction=els_outer,
                                facet_restriction=facets_ext,
                                check_unused=False)
    a1 += (1 / dt * mass + diffusion).Compile(**compile_opts) * dx
    a1 += nitsche.Compile(**compile_opts) * ds
    a1 += ghost_penalty_ext.Compile(**compile_opts) * dw_ext

    f1 = LinearForm(V)
    f1 += (1 / dt * mass_old).Compile(**compile_opts) * dx
    f1 += nitsche_rhs.Compile(**compile_opts) * ds

    a2 = RestrictedBilinearForm(V, element_restriction=els_outer,
                                facet_restriction=facets_ext,
                                check_unused=False)
    a2 += (3 / (2 * dt) * mass + diffusion).Compile(**compile_opts) * dx
    a2 += nitsche.Compile(**compile_opts) * ds
    a2 += ghost_penalty_ext.Compile(**compile_opts) * dw_ext

    f2 = LinearForm(V)
    f2 += ((2 * mass_old - mass_old2 / 2) / dt).Compile(**compile_opts) * dx
    f2 += nitsche_rhs.Compile(**compile_opts) * ds

    # Drag computation --------------------------------------------------------
    ds_2order = ds.order(2 * order)

    stress = Grad(gfu) * n_lset - gamma_n * order**2 / h * (gfu - vel_solid_cf)
    stress = stress.Compile(**compile_opts)

    def comp_drag():
        drag_x = - Integrate(stress[0] * ds_2order, mesh)
        drag_y = - Integrate(stress[1] * ds_2order, mesh)

        return drag_x, drag_y

    # Collect output ----------------------------------------------------------
    data = {key: [] for key in ['time', 'height', 'vel', 'dragX', 'dragY']}

    def append_output_data():
        data['time'].append(time)
        data['height'].append(height.Get())
        data['vel'].append(vel_solid.Get())
        data['dragX'].append(drag_x)
        data['dragY'].append(drag_y)

        return None

    # Time-stepping loop -----------------------------------------------------
    vel_s = vel_solid.Get()
    drag_x, drag_y = 0, 0
    time = 0
    els_outer.Set()

    append_output_data()

    # Fist step using BDF1
    time = dt
    height_last = height.Get()
    gfu_last.vec.data = gfu.vec
    deform_last.vec.data = deformation.vec
    els_outer_old[:] = els_outer
    vel_s_last = vel_solid.Get()

    vel_res = []
    omega = omega_0

    for inner_it in range(max_inner_it):
        # Update velocity with Aitken's Delta^2 relaxation
        vel_s_tmp = vel_s_last + dt * (rhs_func_ode[1] + drag_y)

        vel_res.append(vel_s_tmp - vel_s)
        if inner_it > 0:
            omega *= vel_res[-2] / (vel_res[-1] - vel_res[-2])
        omega = min(1.0, abs(omega))

        # Velocity update
        vel_s_new = (1 - omega) * vel_s + omega * vel_s_tmp

        update = vel_s_new - vel_s
        if inner_it > 0 and abs(update / vel_s_new) < ode_tol:
            print(f' ODE converged after {inner_it} iterations')
            break

        vel_s = vel_s_new
        vel_solid.Set(vel_s)
        height.Set(height_last + dt * vel_s)

        # Mark extension elements and facets
        delta = 2 * c_delta * abs(vel_s) * dt
        K.Set(ceil(delta / hmax))

        InterpolateToP1(levelset_func - delta, lsetp1)
        ci_outer.Update(lsetp1)
        InterpolateToP1(levelset_func + delta, lsetp1)
        ci_inner.Update(lsetp1)

        els_ext[:] = els_outer & ~els_inner
        facets_ext[:] = GetFacetsWithNeighborTypes(mesh, a=els_outer,
                                                   b=els_ext, use_and=True)

        # Update main level set, deformation and project GridFunctions
        deformation = lsetmeshadap.CalcDeformation(levelset_func)
        for _i in range(2):
            gfu_last_on_new_mesh.components[_i].Set(
                shifted_eval(gfu_last.components[_i],
                             back=deform_last, forth=deformation))

        # Update domain markers
        ci_main.Update(lsetp1)

        active_dofs[:] = GetDofsOfElements(V, els_outer)
        active_dofs &= V.FreeDofs()

        # Check element history for method of lines time-derivative approx.
        els_test[:] = els_hasneg & ~els_outer_old
        assert sum(els_test) == 0, 'Some elements do not have a history'

        # Update linear system and solve the resulting system
        a1.Assemble(reallocate=True)
        f1.Assemble()
        inv = a1.mat.Inverse(active_dofs, inverse=inverse)
        gfu.vec.data = PreRic(a1, f1.vec, inv, active_dofs, printing=False)

        drag_x, drag_y = comp_drag()
        del inv
    else:
        str_out = f' WARNING: ODE did not converged after {max_inner_it}'
        str_out += f' iterations: last update {update:4.2e}'
        print(str_out)

    append_output_data()

    print(f't={time:5.3f}, h={height.Get():6.4f}, ', end='')
    print(f'v={vel_s:6.4f}, F_x={drag_x:4.2e}, F_y={drag_y:8.6e} - 1')

    # Clean up some memory
    try:
        del inv
    except NameError:
        pass
    del a1, f1

    # Remaining steps with BDF2
    for it in range(2, int(t_end / dt) + 1):
        time = it * dt

        height_last2 = height_last
        height_last = height.Get()
        gfu_last2.vec.data = gfu_last.vec
        gfu_last.vec.data = gfu.vec
        deform_last2.vec.data = deform_last.vec
        deform_last.vec.data = deformation.vec
        els_outer_old2[:] = els_outer_old
        els_outer_old[:] = els_outer
        vel_s_last2 = vel_s_last
        vel_s_last = vel_solid.Get()

        vel_res = []
        omega = omega_0

        for inner_it in range(max_inner_it):
            # Update velocity with Aitken's Delta^2 relaxation
            vel_s_tmp = 4 / 3 * vel_s_last - 1 / 3 * vel_s_last2
            vel_s_tmp += 2 / 3 * dt * (rhs_func_ode[1] + drag_y)

            vel_res.append(vel_s_tmp - vel_s)
            if inner_it > 0:
                omega *= vel_res[-2] / (vel_res[-1] - vel_res[-2])
            omega = min(1.0, abs(omega))

            # Velocity update
            vel_s_new = (1 - omega) * vel_s + omega * vel_s_tmp

            update = vel_s_new - vel_s
            if inner_it > 0 and abs(update / vel_s_new) < ode_tol:
                print(f' ODE converged after {inner_it} iterations')
                break

            vel_s = vel_s_new
            vel_solid.Set(vel_s)
            height_new = 4 / 3 * height_last - 1 / 3 * height_last2
            height_new += 2 / 3 * dt * vel_s
            height.Set(height_new)

            # Mark extension elements and facets
            delta = 2 * c_delta * abs(vel_s) * dt
            K.Set(ceil(delta / hmax))

            InterpolateToP1(levelset_func - delta, lsetp1)
            ci_outer.Update(lsetp1)
            InterpolateToP1(levelset_func + delta, lsetp1)
            ci_inner.Update(lsetp1)

            els_ext[:] = els_outer & ~els_inner
            facets_ext[:] = GetFacetsWithNeighborTypes(mesh, a=els_outer,
                                                       b=els_ext, use_and=True)
            # Update main level set, deformation and project GridFunctions
            deformation = lsetmeshadap.CalcDeformation(levelset_func)
            for _i in range(2):
                gfu_last2_on_new_mesh.components[_i].Set(
                    shifted_eval(gfu_last2.components[_i],
                                 back=deform_last2, forth=deformation))

                gfu_last_on_new_mesh.components[_i].Set(
                    shifted_eval(gfu_last.components[_i],
                                 back=deform_last, forth=deformation))

            # Update domain markers
            ci_main.Update(lsetp1)

            active_dofs[:] = GetDofsOfElements(V, els_outer) & V.FreeDofs()

            # Check element history for method of lines time-derivative approx.
            els_test[:] = els_hasneg & ~(els_outer_old | els_outer_old2)
            assert sum(els_test) == 0, 'Some elements do not have a history'

            # Update linear system and solve the resulting system
            a2.Assemble(reallocate=True)
            f2.Assemble()
            inv = a2.mat.Inverse(active_dofs, inverse=inverse)
            gfu.vec.data = PreRic(a2, f2.vec, inv, active_dofs, printing=False)

            drag_x, drag_y = comp_drag()
            del inv
        else:
            str_out = f' WARNING: ODE did not converged after {max_inner_it}'
            str_out += f' iterations: last update {update:4.2e}'
            print(str_out)

        append_output_data()

        print(f't={time:5.3f}, h={height.Get():6.4f}, ', end='')
        print(f'v={vel_s:6.4f}, F_x={drag_x:4.2e}, F_y={drag_y:8.6e}')

    # Memory clean up
    try:
        del inv
    except NameError:
        pass
    del a2, f2, gfu, gfu_last, gfu_last2

    return data
