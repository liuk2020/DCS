#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# isolatedsurf.py

import numpy as np
from tfpy.toroidalField import ToroidalField
from tfpy.toroidalField import changeResolution
from .baseproblem import SurfProblem


class IsolatedSurface(SurfProblem):

    def __init__(self, r: ToroidalField = None, z: ToroidalField = None, omega: ToroidalField = None, mpol: int = None, ntor: int = None, nfp: int = None, reverseToroidalAngle: bool = False, reverseOmegaAngle: bool = True) -> None:
        super().__init__(r, z, omega, mpol, ntor, nfp, reverseToroidalAngle, reverseOmegaAngle)

    def BoozerResidual(self, guu, guv, gvv) -> ToroidalField:
        return guv + self.iota*guu    

    def iotaResidual(self) -> float:
        return self._weight['iota']*(self.iota/self._target['iota']-1)**2
    
    # def ratioResidual(self) -> float:
    #     return self._weight['inverse ratio']*(self._target['inverse ratio']-self.inverseRatio)**2
    
    def updateResidual(self):
        self.guu, self.guv, self.gvv = self.metric
        self.Boozer_residual =  self.BoozerResidual(self.guu, self.guv, self.gvv)
        if self._constraint['iota']:
            self.iota_residual = self.iotaResidual()
        # if self._constraint['inverse ratio']:
        #     self.updateInverseRatio()
        #     self.ratio_residual = self.ratioResidual()
            
    def costfunction(self, dofs):
        self.unpackDOF(dofs)
        self.updateResidual()
        cost = np.linalg.norm(np.hstack((self.Boozer_residual.reArr, self.Boozer_residual.imArr)))
        if self._constraint['iota']:
            cost += self.iota_residual
        # if self._constraint['inverse ratio']:
        #     cost += self.ratio_residual
        return cost
    
    def info_print(self) -> str:
        return "{:>8} {:>14} {:>14} {:>14} {:>18} {:>18}".format('niter', 'iota', 'area', 'volume', 'residual_Boozer', 'cost_function')

    def cost_print(self, niter, dofs) -> str:
        cost = self.costfunction(dofs)
        self.updateInverseRatio()
        return "{:>8d} {:>14f} {:>14f} {:>14f} {:>18e} {:>18e}".format(
            niter,
            self.iota,
            self.area,
            self.volume,
            np.linalg.norm(np.hstack((self.Boozer_residual.reArr, self.Boozer_residual.imArr))),
            cost
        )

    def solve(self, nstep: int=5, **kwargs):

        print('============================================================================================')
        print(f'########### The number of DOFs is {self.numsDOF} ')
        if kwargs.get('method') == None:
            kwargs.update({'method': 'BFGS'})
        print('########### The method in the minimization process is ' + kwargs.get('method'))
        # initResidual = self.BoozerResidual()
        init_guu, init_guv, _ = self.metric
        self.updateIota(-init_guv.getRe(0,0)/init_guu.getRe(0,0))
        initResidual = init_guv+self.iota*init_guu
        print(f'########### The nfp is {self.nfp} ')
        print(f'########### The resolution of the R and Z:  mpol={self.mpol}, ntor={self.ntor} ')
        print(f'########### The resolution of the residual:  mpol={initResidual.mpol}, ntor={initResidual.ntor}, total={len(initResidual.reArr)} ')
        
        from scipy.optimize import minimize
        
        if kwargs.get('method')=='CG' or kwargs.get('method')=='BFGS':
            if kwargs.get('tol') == None:
                kwargs.update({'tol': 1e-3})
            self.niter = 0
            print(self.info_print())
            print(self.cost_print(0, self.initDOFs))
            def callback(xi):
                self.niter += 1
                if self.niter%nstep == 0:
                    print(self.cost_print(self.niter, xi))
            res = minimize(self.costfunction, self.initDOFs, callback=callback, **kwargs)
            if self.niter%nstep != 0:
                print(self.cost_print(self.niter, self.initDOFs))
        
        elif 'trust' in kwargs.get('method'):
            kwargs.update({'method': 'trust-constr'})
            res = minimize(self.costfunction, self.initDOFs, options={'verbose':3}, **kwargs)
        
        else:
            if kwargs.get('tol') == None:
                kwargs.update({'tol': 1e-3})
            self.niter = 0
            print(self.info_print())
            print(self.cost_print(0, self.initDOFs))
            def callback(xi):
                self.niter += 1
                if self.niter%nstep == 0:
                    print(self.cost_print(self.niter, xi))
            res = minimize(self.costfunction, self.initDOFs, callback=callback, **kwargs)
            if self.niter%nstep != 0:
                print(self.cost_print(self.niter, self.initDOFs))
        
        if not res.success:
            print('Warning: ' + res.message)


if __name__ == '__main__':
    pass