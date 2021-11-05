The reproduction scripts and results data for the paper "Error Estimate for the Heat Equation on a Coupled Moving Domain in a Fully Eulerian Framework by H. v. Wahl and T. Richter.

# Files
```
.
+-- README.md                             # This file
+-- install.txt                           # Installation help
+-- scripts
|   +-- ALE_referenceBDF2.py              # ALE discretisation for reference solution
|   +-- convergence_study.py              # Main driver
|   +-- data.py                           # Contains example specific data
|   +-- solver_bdf1.py                    # Implementation of the BDF1 scheme
|   +-- solver_bdf2.py                    # Implementation of the BDF2 scheme
|   +-- solver_bdf2_ho.py                 # Implementation of the BDF2 scheme with isoparametric cutFEM
|   +-- solver_bdf2_ho_nitsche.py         # Implementation with Nitsche's method, the BDF2 scheme and isoparametric cutFEM
+-- results
|   +-- pickle2latex.py                   # Create LaTeX tables of the full convergence study
|   +-- pickle2plot.py                    # Create txt files of the results for plots 
|   +-- postprocess_convergence_study.py  # Create txt file with full convergence study tables
|   +-- data
|   |   +-- CoupledHeatEquation_h0.005k4dtinv6400bdf2_defset.txt
|   |   +-- coupled_heat_problem-raw_data-example_datarx(0,5)rt(0,5)_k2h0.1dtinv50BDF1.data
|   |   +-- coupled_heat_problem-raw_data-example_datarx(0,5)rt(0,5)_k2h0.1dtinv50BDF2HO.data
|   |   +-- coupled_heat_problem-raw_data-example_datarx(0,5)rt(0,5)_k2h0.1dtinv50BDF2HO_Nitsche.data
```

# Install

See the instructions in `install.txt`

# How to reproduce
The scripts to reproduce the computational results are located in the `scripts` folder. The resulting data is located in the `data` directory.

A convergence study is driven by running `convergence_study.py`. The method (BDF1/2, P1 levelset/isoparametric) is determined in line 15 `from solver_xxx import *`. The implementation of each method is then contained in the individual `solver_xxx.py`

The parameters for which the study is computed, can be tuned in the `PARAMETERS` block in `convergence_study.py`, i.e., mesh sizes, time-step, number of refinements, ect.

By default, the direct solver `pardiso` is used to solve the linear systems resulting from the discretisation. If this is not available, this may be replaced with `umfpack` in `scripts/convergence_study.py`

The results are then the position, velocity and forces acting on the interface over time. The are contained pickled file as a dictionary with the results for each mesh/time-step combination.

# Post-process results
The necessary scripts to compute convergence rates from the convergence studies are located the `results` directory. These python scripts can be used to create convergence tables in txt or LaTeX form, and create text files for convergence plots.