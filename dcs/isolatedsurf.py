#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# isolatedsurf.py

import numpy as np
from tfpy.toroidalField import ToroidalField
from .baseproblem import SurfProblem


class IsolatedSurface(SurfProblem):

    def __init__(self, r: ToroidalField = None, z: ToroidalField = None, omega: ToroidalField = None, mpol: int = None, ntor: int = None, nfp: int = None, iota: float = None, fixIota: bool = False, reverseToroidalAngle: bool = False, reverseOmegaAngle: bool = True) -> None:
        super().__init__(r, z, omega, mpol, ntor, nfp, iota, fixIota, reverseToroidalAngle, reverseOmegaAngle)

    def BoozerResidual(self) -> ToroidalField:
        guu, guv, _ = self.metric
        if not self.fixIota:
            self.updateIota(-guv.getRe(0,0)/guu.getRe(0,0))
        return guv + self.iota*guu

    def solve(self, nstep: int=5, **kwargs):
        print('============================================================================================')
        print(f'########### The number of DOFs is {self.numsDOF} ')
        if kwargs.get('method') == None:
            kwargs.update({'method': 'CG'})
        print('########### The method in the minimization process is ' + kwargs.get('method'))
        initResidual = self.BoozerResidual()
        print(f'########### The nfp is {self.nfp} ')
        print(f'########### The resolution of the R and Z:  mpol={self.mpol}, ntor={self.ntor} ')
        print(f'########### The resolution of the residual:  mpol={initResidual.mpol}, ntor={initResidual.ntor}, total={len(initResidual.reArr)} ')
        def cost(dofs):
            self.unpackDOF(dofs)
            residualField = self.BoozerResidual()
            numsTheta, numsZeta = 64, 64
            deltaTheta, deltaZeta = 2*np.pi/numsTheta, 2*np.pi/self.nfp/numsZeta
            thetaArr = np.linspace(deltaTheta/2, 2*np.pi-deltaTheta/2, numsTheta)
            zetaArr = np.linspace(deltaZeta/2, 2*np.pi/self.nfp-deltaZeta/2, numsZeta)
            zetaGrid, thetaGrid = np.meshgrid(thetaArr, zetaArr)
            residualGrid = residualField.getValue(thetaGrid, zetaGrid)
            residual = np.power(deltaTheta*deltaZeta*np.sum(np.power(residualGrid, 2)), 0.5)
            return 4*self.nfp*residual/np.pi/np.pi
        from scipy.optimize import minimize
        if kwargs.get('method') == 'CG':
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
        elif kwargs.get('method') == 'trust-constr':
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