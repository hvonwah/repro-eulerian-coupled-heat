from itertools import product
from pickle import load
from pandas import read_csv
from numpy import array, sqrt, sum, square, log2

Lx = (0, 5)
Lt = (0, 5)

dt_inv0 = 50

filename_raw = 'data/coupled_heat_problem-raw_data-example_data'
filename_raw += f'rx(0,5)rt(0,5)_k2h0.095dtinv{dt_inv0}BDF2HONitsche.data'

filename_ref = 'data/CoupledHeatEquation_h0.005k4dtinv12800bdf2_defset.txt'
dt_inv_ref = 12800


filename_out = 'convergence_table_bdf2honitsche.txt'

# Load files
df_ref = read_csv(filename_ref, sep=' ', index_col=False)
data_conv = load(open(filename_raw, 'rb'))


# Compute errors
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


# Compute rates
for lx, key in product(range(*Lx), errors):
    e1, e2 = errors[key][(lx, Lt[1])], errors[key][(lx + 1, Lt[1])]
    errors[key][(lx + 1, 'eoc_x')] = log2(e1) - log2(e2)

for lvl, key in product(zip(reversed(range(*Lx)), reversed(range(*Lt))), errors):
    lx, lt = lvl
    e1, e2 = errors[key][(lx, lt)], errors[key][(lx + 1, lt + 1)]
    errors[key][(lvl[0] + 1, 'eoc_xt')] = log2(e1) - log2(e2)

for lt, key in product(range(*Lt), errors):
    e1, e2 = errors[key][(Lx[1], lt)], errors[key][(Lx[1], lt + 1)]
    errors[key][('eoc_t', lt + 1)] = log2(e1) - log2(e2)


# Print convergence table
x_lvls = [i for i in range(Lx[1] + 1)] + ['eoc_t']
t_lvls = [i for i in range(Lt[1] + 1)] + ['eoc_x', 'eoc_xt']

with open(filename_out, 'w') as fid:
    for key, dic in errors.items():
        fid.write(key + ':\nLt\\Lx   ')
        fid.write(' '.join([f'{lx}\t\t' for lx in x_lvls]) + '\n')

        for lt in t_lvls:
            fid.write('{}\t'.format(lt))
            if isinstance(lt, int):
                fid.write('\t')

            for lx in x_lvls:
                if (lx, lt) in dic:
                    if isinstance(lx, int) and isinstance(lt, int):
                        fid.write(f'{dic[(lx, lt)]:2.1e}\t')
                    elif isinstance(lx, str) or isinstance(lt, str):
                        fid.write(f'{dic[(lx, lt)]:2.1f}\t\t')
                elif not (isinstance(lt, str) and isinstance(lx, str)):
                    fid.write('--\t\t')
            fid.write('\n')
        fid.write('\n\n')
