"""Independent model-level checks, in addition to the geometry unit tests."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import unittest
import numpy as np
import qutip as qt
from _phase_eta_exact_core import build_kerr_model,one_point
from dicke_gp_scan import simulate_point
from gp_geometry import geometric_amplitude


class ModelChecks(unittest.TestCase):
    def test_kerr_two_drive_covariance(self):
        pars=dict(Delta=.3,eta=.7,N=12,U=-1.,E=.2,phi=.1,theta=.4,gamma_1=.1,gamma_2=.1,kappa=.1)
        H,c,n=build_kerr_model(**pars)
        angle=.31
        other={**pars,'phi':pars['phi']+angle,'theta':pars['theta']+2*angle}
        H2,c2,_=build_kerr_model(**other)
        R=(-1j*angle*n).expm()
        self.assertLess((H2-R*H*R.dag()).norm('fro'),1e-10)
        rho=qt.steadystate(H,c)
        rho2=qt.steadystate(H2,c2)
        self.assertLess((rho2-R*rho*R.dag()).norm('fro'),1e-10)

    def test_undriven_kerr_has_trivial_number_loop(self):
        n,z=one_point(0.,0.,24,-1.,0.,0.,0.,.1,.1,.1)
        self.assertGreater(n,.1)  # Mixed pumping-loss steady state, not vacuum.
        self.assertAlmostEqual(abs(z-1.),0.,places=10)

    def test_uncoupled_dicke_matches_analytic_spin_precession(self):
        eps=.073
        result=simulate_point(atoms=2,cutoff=4,epsilon=eps,kappa=.1,coupling=0.,periods=4)
        expected=np.cos(2*np.pi*(1+eps)*result['m'])
        np.testing.assert_allclose(result['sigma_x'],expected,atol=1e-11)
        np.testing.assert_allclose(result['nbar'],0.,atol=1e-13)

    def test_dicke_sparse_exponential_matches_independent_ode(self):
        atoms,cutoff,eps,kappa=2,12,.01,.05
        result=simulate_point(atoms=atoms,cutoff=cutoff,epsilon=eps,kappa=kappa,periods=2)
        j=atoms/2
        a=qt.tensor(qt.destroy(cutoff),qt.qeye(atoms+1))
        jx=qt.tensor(qt.qeye(cutoff),qt.jmat(j,'x'))
        jy=qt.tensor(qt.qeye(cutoff),qt.jmat(j,'y'))
        jz=qt.tensor(qt.qeye(cutoff),qt.jmat(j,'z'))
        hoff=(1-eps)*a.dag()*a+(1+eps)*jz
        hon=hoff+2/np.sqrt(atoms)*(a+a.dag())*jx
        rho=qt.ket2dm(qt.tensor(qt.basis(cutoff,0),qt.spin_coherent(j,np.pi/2,0)))
        opts={'method':'dop853','rtol':1e-10,'atol':1e-12,'nsteps':10000,'normalize_output':False}
        for H in [hon,hoff]:
            rho=qt.mesolve(H,rho,[0.,np.pi],[np.sqrt(kappa)*a],options=opts).states[-1]
        self.assertAlmostEqual(qt.expect(jx,rho)/j,result['sigma_x'][1],places=8)
        self.assertAlmostEqual(qt.expect(a.dag()*a,rho),result['nbar'][1],places=8)
        u=1/(4*atoms)
        Q=j*qt.qeye([cutoff,atoms+1])+u*jx+np.sqrt(1-u*u)*jy
        self.assertLess(abs(geometric_amplitude(rho,Q)-result['Z_xy'][1]),2e-7)


if __name__=='__main__':
    unittest.main()
