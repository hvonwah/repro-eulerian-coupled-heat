from itertools import product
from pickle import load
from pandas import read_csv
from numpy import array, sqrt, sum, square, log2

Lx = (0, 5)
Lt = (0, 5)

h0 = 0.1
dt_inv0 = 50

filename_raw = 'data/coupled_heat_problem-raw_data-example_data'
filename_raw += 'rx(0,5)rt(0,5)_k2h0.095dtinv50BDF2HONitsche.data'

filename_ref = 'data/CoupledHeatEquation_h0.005k4dtinv12800bdf2_defset.txt'
dt_inv_ref = 12800

filename_out = 'convergence_data_example1_bdf2honitsche'


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


##############################################################################
# Write data to file
##############################################################################
keys = [key for key in errors]

fid = open(filename_out + '_space.txt', 'w')
fid.write(' '.join(key for key in ['hmax'] + keys) + '\n')

for _l in range(Lx[0], Lx[1] + 1):
    fid.write(f'{2**-_l * h0} ')
    fid.write(' '.join(f'{errors[key][(_l, Lt[-1])]:6.3e}' for key in keys))
    fid.write('\n')

fid.close()

keys = [key for key in errors]
fid = open(filename_out + '_time.txt', 'w')
fid.write(' '.join(key for key in ['dt'] + keys) + '\n')

for _l in range(Lt[0], Lt[1] + 1):
    fid.write(f'{2**-_l / dt_inv0} ')
    fid.write(' '.join(f'{errors[key][(Lx[-1], _l)]:6.3e}' for key in keys))
    fid.write('\n')

fid.close()
