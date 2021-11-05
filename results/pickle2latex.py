from itertools import product
from pickle import load
from pandas import read_csv
from numpy import array, sqrt, sum, square, log2

Lx = (0, 5)
Lt = (0, 5)

dt_inv0 = 50

filename_raw = 'data/coupled_heat_problem-raw_data-example_data'
filename_raw += 'rx(0,5)rt(0,5)_k2h0.1dtinv50BDF2HO_Nitsche.data'

filename_ref = 'data/CoupledHeatEquation_h0.005k4dtinv6400bdf2_defset.txt'
dt_inv_ref = 6400

filename_out = 'convergence_table_example1_bdf2ho_nitsche'


##############################################################################
# Load files
##############################################################################
df_ref = read_csv(filename_ref, sep=' ', index_col=False)
data_conv = load(open(filename_raw, 'rb'))


##############################################################################
# Compute errors and convergence rates
##############################################################################
errors = {key: {} for key in ['vel', 'height']}

for lx, lt in product(range(Lx[0], Lx[1] + 1), range(Lt[0], Lt[1] + 1)):
    dtinv = dt_inv0 * 2**lt
    assert dt_inv_ref % dtinv == 0, 'time-step missmatch w.r.t. reference'

    _n = dt_inv_ref // dtinv
    sample_ref = df_ref[df_ref.index % _n == 0]

    data_lvl = data_conv[(lx, lt)]

    for key in errors:
        err_array = sample_ref[key].to_numpy() - array(data_lvl[key])
        err = sqrt(sum(square(err_array)) / dtinv)
        errors[key][(lx, lt)] = err


for lx, key in product(range(*Lx), errors):
    e1, e2 = errors[key][(lx, Lt[1])], errors[key][(lx + 1, Lt[1])]
    errors[key][(lx + 1, 'eoc_x')] = log2(e1) - log2(e2)

for lx, key in product(range(*Lt), errors):
    e1, e2 = errors[key][(lx, lx)], errors[key][(lx + 1, lx + 1)]
    errors[key][(lx + 1, 'eoc_xt')] = log2(e1) - log2(e2)

for lt, key in product(range(*Lt), errors):
    e1, e2 = errors[key][(Lx[1], lt)], errors[key][(Lx[1], lt + 1)]
    errors[key][('eoc_t', lt + 1)] = log2(e1) - log2(e2)


##############################################################################
# Write LaTeX table data
##############################################################################
x_lvls = [i for i in range(Lx[1] + 1)] + ['eoc_t']
t_lvls = [i for i in range(Lt[1] + 1)] + ['eoc_x', 'eoc_xt']
x_lvls_print = [i for i in range(Lx[1] + 1)] + ['$\\text{eoc}_{t}$']
t_lvls_print = [i for i in range(Lt[1] + 1)] + ['$\\text{eoc}_{x}$',
                                                '$\\text{eoc}_{xt}$']


for key, dic in errors.items():
    fid = open(filename_out + '_' + key + '.tex', 'w')
    fid.write('%!TEX root = ../manuscript_eulerian_coupled.tex\n')

    fid.write('\\begin{table}\n  \\centering\n  \\begin{tabular}{c')
    fid.write('c' * len(x_lvls))
    fid.write('c}\n    \\toprule\n')
    fid.write('    $L_t \\downarrow \\backslash L_x \\rightarrow$')

    fid.write(' '.join([f'& {lx}' for lx in x_lvls_print]))
    fid.write(' \\\\ \n    \\midrule\n')

    for lt, lt_p in zip(t_lvls, t_lvls_print):
        if lt == 'eoc_x':
            fid.write('    \\cmidrule(lr){{1-{:}}}\n'.format(len(x_lvls) + 1))

        fid.write(f'    {lt_p} ')

        for lx in x_lvls:
            if (lx, lt) in dic:
                if isinstance(lx, int) and isinstance(lt, int):
                    num = f'{dic[(lx, lt)]:4.2e}'
                    num = num[:-2] + num[-1:]
                    num = '$' + num.replace('e', ' \\cdot 10^{') + '}$'
                    fid.write(f'& {num} ')
                else:
                    fid.write(f'& ${dic[(lx, lt)]:1.2f}$ ')
            elif isinstance(lt, str) and isinstance(lx, str):
                fid.write('& ')
            else:
                fid.write('& -- ')

        fid.write('\\\\\n')

    fid.write('    \\bottomrule\n  \\end{tabular}\n')
    fid.write('  %\\caption{}\n  %\\label{}\n\\end{table}\n')
    fid.close()
