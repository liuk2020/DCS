#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# isolatedsurf.py

import numpy as np
from tfpy.toroidalField import ToroidalField
from tfpy.toroidalField import changeResolution
from .baseproblem import SurfProblem


class IsolatedSurface(SurfProblem):

    def __init__(self, r: ToroidalField = None, z: ToroidalField = None, omega: ToroidalField = None, mpol: int = None, ntor: int = None, nfp: int = None, iota: float = None, fixIota: bool = False, reverseToroidalAngle: bool = False, reverseOmegaAngle: bool = True) -> None:
        super().__init__(r, z, omega, mpol, ntor, nfp, iota, fixIota, reverseToroidalAngle, reverseOmegaAngle)

    def BoozerResidual(self) -> ToroidalField:
        guu, guv, _ = self.metric
        return guv+self.iota*guu

    def solve(self, nstep: int=5, **kwargs):
        print('============================================================================================')
        print(f'########### The number of DOFs is {self.numsDOF} ')
        if kwargs.get('method') == None:
            kwargs.update({'method': 'BFGS'})
        print('########### The method in the minimization process is ' + kwargs.get('method'))
        # initResidual = self.BoozerResidual()
        init_guu, init_guv, _ = self.metric
        if not self.fixIota:
            self.updateIota(-init_guv.getRe(0,0)/init_guu.getRe(0,0))
        initResidual = init_guv+self.iota*init_guu
        print(f'########### The nfp is {self.nfp} ')
        print(f'########### The resolution of the R and Z:  mpol={self.mpol}, ntor={self.ntor} ')
        print(f'########### The resolution of the residual:  mpol={initResidual.mpol}, ntor={initResidual.ntor}, total={len(initResidual.reArr)} ')
        def cost(dofs):
            self.unpackDOF(dofs)
            residualField = self.BoozerResidual()
            return np.linalg.norm(np.hstack((residualField.reArr, residualField.imArr)))
        from scipy.optimize import minimize
        if kwargs.get('method')=='CG' or kwargs.get('method')=='BFGS':
            if kwargs.get('tol') == None:
                kwargs.update({'tol': 1e-3})
            self.niter = 0
            print("{:>8} {:>16} {:>18}".format('niter', 'iota', 'residual_Boozer'))
            print("{:>8d} {:>16f} {:>18e}".format(0, self.iota, cost(self.initDOFs)))
            def callback(xi):
                self.niter += 1
                if self.niter%nstep == 0:
                    print("{:>8d} {:>16f} {:>18e}".format(self.niter, self.iota, cost(xi)))
            res = minimize(cost, self.initDOFs, callback=callback, **kwargs)
            if self.niter%nstep != 0:
                print("{:>8d} {:>16f} {:>18e}".format(self.niter, self.iota, cost(self.initDOFs)))
        elif 'trust' in kwargs.get('method'):
            kwargs.update({'method': 'trust-constr'})
            res = minimize(cost, self.initDOFs, options={'verbose':3}, **kwargs)
        else:
            if kwargs.get('tol') == None:
                kwargs.update({'tol': 1e-3})
            self.niter = 0
            print("{:>8} {:>16} {:>18}".format('niter', 'iota', 'residual_Boozer'))
            print("{:>8d} {:>16f} {:>18e}".format(0, self.iota, cost(self.initDOFs)))
            def callback(xi):
                self.niter += 1
                if self.niter%nstep == 0:
                    print("{:>8d} {:>16f} {:>18e}".format(self.niter, self.iota, cost(xi)))
            res = minimize(cost, self.initDOFs, callback=callback, **kwargs)
            if self.niter%nstep != 0:
                print("{:>8d} {:>16f} {:>18e}".format(self.niter, self.iota, cost(self.initDOFs)))
        if not res.success:
            print('Warning: ' + res.message)


if __name__ == '__main__':
    pass