"""Small, configurable check of the manuscript Kerr GP diagram.

The two supplied sweep scripts are preserved. This runner uses the reconstructed
core, stores complex amplitudes and numerical diagnostics, and labels finite-cutoff
features as crossovers rather than asserting thermodynamic phase boundaries.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import json
import time
from pathlib import Path
import numpy as np
import qutip
from joblib import Parallel, delayed
from _phase_eta_exact_core import build_kerr_model, steady_state_diagnostics
from gp_geometry import geometric_amplitude


def point(delta, eta, cutoff, U=-1., gamma_1=.1, gamma_2=.1, kappa=.1):
    H, jumps, n = build_kerr_model(delta, eta, cutoff, U, 0., 0., 0., gamma_1, gamma_2, kappa)
    rho = qutip.steadystate(H, jumps)
    diag = steady_state_diagnostics(rho, H, jumps)
    z = geometric_amplitude(rho, n)
    # Positive number angle pi is the manuscript's positive 2pi squeezing loop.
    # The supplied phase_loop_eta.py uses the conjugate (reverse orientation).
    z_half = geometric_amplitude(rho, n, angle=np.pi)
    return {'nbar': float(qutip.expect(n, rho)), 'Z': z, 'Z_half': z_half, **diag}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutoff', type=int, default=50)
    parser.add_argument('--delta', type=float, nargs=3, default=[-15., 30., 31], metavar=('MIN','MAX','COUNT'))
    parser.add_argument('--eta', type=float, nargs=3, default=[.001, 5., 25], metavar=('MIN','MAX','COUNT'))
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--out', type=Path, default=Path('results/kerr_check'))
    args = parser.parse_args()
    if args.cutoff < 4 or not 1 <= args.jobs <= 4:
        parser.error('cutoff must be >=4 and jobs must be 1..4')
    if any(x[2] < 2 or int(x[2]) != x[2] for x in [args.delta,args.eta]):
        parser.error('grid counts must be integers >=2')
    delta = np.linspace(*args.delta[:2], int(args.delta[2]))
    eta = np.linspace(*args.eta[:2], int(args.eta[2]))
    start = time.time()
    rows = Parallel(n_jobs=args.jobs, verbose=5)(delayed(point)(d,e,args.cutoff) for e in eta for d in delta)
    shape = (len(eta),len(delta))
    arrays = {key: np.array([row[key] for row in rows]).reshape(shape) for key in rows[0]}
    arrays['phase'] = np.where(abs(arrays['Z']) > 1e-6, np.angle(arrays['Z']), np.nan)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(str(args.out)+'.npz', delta=delta,eta=eta,cutoff=args.cutoff,**arrays)
    metadata = {'model':'manuscript Kerr oscillator','U':-1.,'gamma_1':.1,'gamma_2':.1,'kappa':.1,
                'E':0.,'phi':0.,'theta':0.,'cutoff':args.cutoff,'shape':shape,'elapsed_seconds':time.time()-start,
                'qutip':qutip.__version__,'numpy':np.__version__,'phase_node_threshold':1e-6,
                'full_loop':'R(phi)=exp(-i phi n), phi=0..2pi; squeezing phase=0..4pi',
                'half_loop':'R(phi)=exp(-i phi n), phi=0..pi; squeezing phase=0..2pi, parity endpoint',
                'interpretation':'finite-cutoff steady-state response; no fitted phase boundary'}
    for key in rows[0]:
        if key not in ('Z','Z_half','nbar'):
            metadata[key+'_range']=[float(np.min(arrays[key])),float(np.max(arrays[key]))]
    Path(str(args.out)+'.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata,indent=2),flush=True)


if __name__ == '__main__':
    main()
