#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# qssurf.py


import numpy as np
from tfpy.toroidalField import ToroidalField
from tfpy.toroidalField import derivatePol, derivateTor
from .isolatedsurf import IsolatedSurface
from typing import Tuple


class QSSurface(IsolatedSurface):

    def __init__(self, r: ToroidalField = None, z: ToroidalField = None, omega: ToroidalField = None, mpol: int = None, ntor: int = None, nfp: int = None, iota: float = None, fixIota: bool = False, reverseToroidalAngle: bool = False, reverseOmegaAngle: bool = True) -> None:
        super().__init__(r, z, omega, mpol, ntor, nfp, iota, fixIota, reverseToroidalAngle, reverseOmegaAngle)

    def setSymmetry(self, m: int=1, n: int=0):
        self.sym_m = m
        self.sym_n = n

    def _init_paras(self):
        self.mu = 61.8
        super()._init_paras()

    @property
    def numsDOF(self):
        return super().numsDOF + 1

    @property
    def initDOFs(self):
        dofs = super().initDOFs
        if not self.fixIota:
            dofs[-2] = self.iota
            dofs[-1] = self.mu
        else:
            dofs[-1] = self.mu
        return dofs

    def unpackDOF(self, dofs: np.ndarray) -> None:
        super().unpackDOF(dofs)
        if not self.fixIota:
            self.updateIota(dofs[-2])
            self.mu = dofs[-1]
        else:
            self.mu = dofs[-1]

    def QSResidual(self) -> ToroidalField:
        _, guv, gvv = self.metric
        scriptB = gvv + self.iota*guv
        if self.sym_m == 0:
            return derivateTor(scriptB)
        elif self.sym_n == 0:
            return derivatePol(scriptB)
        else:
            return self.sym_m*derivatePol(scriptB) + self.sym_n*derivateTor(scriptB)

    # TODO: reconstruction of this method
    def solve(self, nstep: int = 5, **kwargs):
        
        print('============================================================================================')
        print(f'########### The number of DOFs is {self.numsDOF} ')
        if kwargs.get('method') == None:
            kwargs.update({'method': 'BFGS'})
        print('########### The method in the minimization process is ' + kwargs.get('method'))
        # initResidual = self.BoozerResidual()
        init_guu, init_guv, init_gvv = self.metric
        if not self.fixIota:
            self.updateIota(-init_guv.getRe(0,0)/init_guu.getRe(0,0))
        if (not hasattr(self, 'sym_m')) or (not hasattr(self, 'sym_n')):
            self.setSymmetry()
        initBoozerResidual = init_guv+self.iota*init_guu
        initScriptB = init_gvv+self.iota*init_guv
        if self.sym_m == 0:
            initQSResidual = derivatePol(initScriptB)
        elif self.sym_n == 0:
            initQSResidual = derivateTor(initScriptB)
        else:
            initQSResidual =  self.sym_n*self.nfp*derivatePol(initScriptB) + self.sym_m*derivateTor(initScriptB)
        print(f'########### The nfp is {self.nfp} ')
        print(f'########### The resolution of the R and Z:  mpol={self.mpol}, ntor={self.ntor} ')
        print(f'########### The resolution of the residual:  mpol={initBoozerResidual.mpol}, ntor={initBoozerResidual.ntor}, total={len(initBoozerResidual.reArr)} ')
        if self.sym_n > 0:
            print(f'########### The quasi-symmetric type is: B=B(s,{self.sym_m}*theta-{self.sym_n}*{self.nfp}*zeta) ')
        else:
            print(f'########### The quasi-symmetric type is: B=B(s,{self.sym_m}*theta+{-self.sym_n}*{self.nfp}*zeta) ')
        
        def residual(dofs) -> Tuple[ToroidalField, ToroidalField]:
            self.unpackDOF(dofs)
            guu, guv, gvv = self.metric
            BoozerResidual =  guv + self.iota*guu
            scriptB = gvv + self.iota*guv
            if self.sym_m == 0:
                QSResidual = derivatePol(scriptB)
            elif self.sym_n == 0:
                QSResidual = derivateTor(scriptB)
            else:
                QSResidual = self.sym_n*self.nfp*derivatePol(scriptB) + self.sym_m*derivateTor(scriptB)
            return BoozerResidual, QSResidual
        def cost(dofs):
            BoozerResidual, QSResidual = residual(dofs)
            return (
                np.linalg.norm(np.hstack((BoozerResidual.reArr, BoozerResidual.imArr)))*self.mu
                + np.linalg.norm(np.hstack((QSResidual.reArr, QSResidual.imArr)))
            )
        
        from scipy.optimize import minimize
        
        if kwargs.get('method')=='CG' or kwargs.get('method')=='BFGS':
            if kwargs.get('tol') == None:
                kwargs.update({'tol': 1e-3})
            self.niter = 0
            print("{:>8} {:>16} {:>18} {:>12} {:>18} {:>18}".format('niter', 'iota', 'residual_Boozer', 'mu', 'residual_QS', 'cost function'))
            _initBoozerResidualValue = np.linalg.norm(np.hstack((initBoozerResidual.reArr, initBoozerResidual.imArr)))
            _initQSResidualValue = np.linalg.norm(np.hstack((initQSResidual.reArr, initQSResidual.imArr)))
            print("{:>8d} {:>16f} {:>18e} {:>12f} {:>18e} {:>18e}".format(0, self.iota, _initBoozerResidualValue, self.mu, _initQSResidualValue, _initBoozerResidualValue*self.mu+_initQSResidualValue))
            def callback(xi):
                self.niter += 1
                if self.niter%nstep == 0:
                    _BoozerResidual, _QSResidual = residual(xi)
                    _BoozerResidualValue = np.linalg.norm(np.hstack((_BoozerResidual.reArr, _BoozerResidual.imArr)))
                    _QSResidualValue = np.linalg.norm(np.hstack((_QSResidual.reArr, _QSResidual.imArr)))
                    print("{:>8d} {:>16f} {:>18e} {:>12f} {:>18e} {:>18e}".format(self.niter, self.iota, _BoozerResidualValue, self.mu, _QSResidualValue, _BoozerResidualValue*self.mu+_QSResidualValue))
            res = minimize(cost, self.initDOFs, callback=callback, **kwargs)
            if self.niter%nstep != 0:
                _BoozerResidual, _QSResidual = residual(self.initDOFs)
                _BoozerResidualValue = np.linalg.norm(np.hstack((_BoozerResidual.reArr, _BoozerResidual.imArr)))
                _QSResidualValue = np.linalg.norm(np.hstack((_QSResidual.reArr, _QSResidual.imArr)))
                print("{:>8d} {:>16f} {:>18e} {:>12f} {:>18e} {:>18e}".format(self.niter, self.iota, _BoozerResidualValue, self.mu, _QSResidualValue, _BoozerResidualValue*self.mu+_QSResidualValue))
        
        elif 'trust' in kwargs.get('method'):
            kwargs.update({'method': 'trust-constr'})
            res = minimize(cost, self.initDOFs, options={'verbose':3}, **kwargs)
        else:
            pass
        
        if not res.success:
            print('Warning: ' + res.message)


if __name__ == '__main__':
    pass
