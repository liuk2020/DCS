#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# qssurf.py


import numpy as np
from tfpy.toroidalField import ToroidalField
from tfpy.toroidalField import derivatePol, derivateTor
from .isolatedsurf import IsolatedSurface
from typing import Tuple


class QSSurface(IsolatedSurface):

    def __init__(self, r: ToroidalField = None, z: ToroidalField = None, omega: ToroidalField = None, mpol: int = None, ntor: int = None, nfp: int = None, reverseToroidalAngle: bool = False, reverseOmegaAngle: bool = True) -> None:
        super().__init__(r, z, omega, mpol, ntor, nfp, reverseToroidalAngle, reverseOmegaAngle)
        self.addconstraint('qs', 0.1, 0)

    def setSymmetry(self, m: int=1, n: int=0):
        self.sym_m = m
        self.sym_n = n
        
    def QSResidual(self, guu, guv, gvv) -> ToroidalField:
        scriptB =  self.iota*self.iota*guu + 2*self.iota*guv + gvv
        if self.sym_m == 0:
            residual = derivatePol(scriptB)
        elif self.sym_n == 0:
            residual = derivateTor(scriptB)
        else:
            residual =  self.sym_n*self.nfp*derivatePol(scriptB) + self.sym_m*derivateTor(scriptB)
        return residual
    
    def updateResidual(self):
        super().updateResidual()
        self.qs_residual = self.QSResidual(self.guu, self.guv, self.gvv)
        
    def costfunction(self, dofs):
        cost = super().costfunction(dofs)
        self.updateResidual()
        cost += self._weight['qs'] * np.linalg.norm(np.hstack((self.qs_residual.reArr, self.qs_residual.imArr)))
        return cost
        
    def info_print(self) -> str:
        return "{:>8} {:>14} {:>16} {:>16} {:>18} {:>18} {:>18}".format('niter', 'iota', 'min-crossarea', 'inverse ratio','residual_Boozer', 'residual_qs', 'cost_function')
    
    def cost_print(self, niter, dofs) -> str:
        cost = self.costfunction(dofs)
        if not self._constraint['inverse ratio']:
            self.updateInverseRatio()
        if not self._constraint['min crossarea']:
            self.updateMinCrossArea()
        return "{:>8d} {:>14f} {:>16f} {:>16f} {:>18e} {:>18e} {:>18e}".format(
            niter,
            self.iota,
            self.minCrossArea,
            self.inverseRatio,
            np.linalg.norm(np.hstack((self.Boozer_residual.reArr, self.Boozer_residual.imArr))),
            np.linalg.norm(np.hstack((self.qs_residual.reArr, self.qs_residual.imArr))),
            cost
        )
        
    def solve(self, nstep: int = 5, **kwargs):
        print('==============================================================================================================================')
        if (not hasattr(self, 'sym_m')) or (not hasattr(self, 'sym_n')):
            self.setSymmetry()
        if self.sym_n > 0:
            print(f'########### The quasi-symmetric type is: B=B(s,{int(self.sym_m)}*theta-{int(self.sym_n)}*{self.nfp}*zeta) ')
        else:
            print(f'########### The quasi-symmetric type is: B=B(s,{int(self.sym_m)}*theta+{int(-self.sym_n)}*{self.nfp}*zeta) ')
        super().solve(nstep, **kwargs)

    # TODO
    def solve_condense(self, nstep: int = 5, **kwargs):
        
        print('==============================================================================================================================')
        print(f'########### The total number of DOFs is {self.numsDOF} ')
        if kwargs.get('method') == None:
            kwargs.update({'method': 'BFGS'})
        print('########### The method in the minimization process is ' + kwargs.get('method'))
        # initResidual = self.BoozerResidual()
        init_guu, init_guv, init_gvv = self.metric
        if not self.fixIota:
            self.updateIota(-init_guv.getRe(0,0)/init_guu.getRe(0,0))
        if (not hasattr(self, 'sym_m')) or (not hasattr(self, 'sym_n')):
            self.setSymmetry()
        self.updateResidual(self.initDOFs)
        print(f'########### The nfp is {self.nfp} ')
        print(f'########### The total resolution of the R and Z:  mpol={self.mpol}, ntor={self.ntor} ')
        print(f'########### The total resolution of the residual:  mpol={self.Boozer_residual.mpol}, ntor={self.Boozer_residual.ntor}, total={len(self.Boozer_residual.reArr)} ')
        if self.sym_n > 0:
            print(f'########### The quasi-symmetric type is: B=B(s,{int(self.sym_m)}*theta-{int(self.sym_n)}*{self.nfp}*zeta) ')
        else:
            print(f'########### The quasi-symmetric type is: B=B(s,{int(self.sym_m)}*theta+{int(-self.sym_n)}*{self.nfp}*zeta) ')
        
        def cost(dofs):
            self.updateResidual(dofs)
            aveOmega = self.omega.integrate()
            return (
                np.linalg.norm(np.hstack((self.Boozer_residual.reArr, self.Boozer_residual.imArr)))
                + self.mu_qs * np.linalg.norm(np.hstack((self.QS_residual.reArr, self.QS_residual.imArr)))
                + self.mu_omega * aveOmega
            )
        
        from scipy.optimize import minimize
        
        for index in np.arange(1, max(self.mpol,self.ntor)+1, dtype='int'):
            
            self.freeAll()
            self.fixDOF('rc', 1, 0)
            self.fixDOF('zs', 1, 0)
            if index <= self.mpol and index <= self.ntor:
                self.fixTruncation_rc(index, index)
                self.fixTruncation_zs(index, index)
                self.fixTruncation_omegas(2*index, 2*index)
                print('==============================================================================================================================')
                print(f'########### The temporary resolution of the R and Z:  mpol={index}, ntor={index} ')
                print(f'########### The temporary number of DOFs is {self.numsDOF} ')
            elif index <= self.mpol and index > self.ntor:
                self.fixTruncation_rc(index, self.ntor)
                self.fixTruncation_zs(index, self.ntor)
                self.fixTruncation_omegas(2*index, 2*self.ntor)
                print('==============================================================================================================================')
                print(f'########### The temporary resolution of the R and Z:  mpol={index}, ntor={self.ntor} ')
                print(f'########### The temporary number of DOFs is {self.numsDOF} ')
            elif index > self.mpol and index <= self.ntor:
                self.fixTruncation_rc(self.mpol, index)
                self.fixTruncation_zs(self.mpol, index)
                self.fixTruncation_omegas(2*self.mpol, 2*index)
                print('==============================================================================================================================')
                print(f'########### The temporary resolution of the R and Z:  mpol={self.mpol}, ntor={index} ')
                print(f'########### The temporary number of DOFs is {self.numsDOF} ')
            else:
                break
                
            if kwargs.get('method')=='CG' or kwargs.get('method')=='BFGS':
                if kwargs.get('tol') == None:
                    kwargs.update({'tol': 1e-3})
                self.niter = 0
                print("{:>8} {:>16} {:>18} {:>12} {:>18} {:>12} {:>18} {:>18}".format('niter', 'iota', 'residual_Boozer', 'mu_QS', 'residual_QS', 'mu_omega', 'ave_omega', 'cost function'))
                _initBoozerResidualValue = np.linalg.norm(np.hstack((self.Boozer_residual.reArr, self.Boozer_residual.imArr)))
                _initQSResidualValue = np.linalg.norm(np.hstack((self.QS_residual.reArr, self.QS_residual.imArr)))
                _initaveomega = self.omega.integrate()
                print("{:>8d} {:>16f} {:>18e} {:>12f} {:>18e} {:>12f} {:>18e} {:>18e}".format(0, self.iota, _initBoozerResidualValue, self.mu_qs, _initQSResidualValue, self.mu_omega, _initaveomega, _initBoozerResidualValue+self.mu_qs*_initQSResidualValue+self.mu_omega*_initaveomega))
                def callback(xi):
                    self.niter += 1
                    if self.niter%nstep == 0:
                        _initBoozerResidualValue = np.linalg.norm(np.hstack((self.Boozer_residual.reArr, self.Boozer_residual.imArr)))
                        _initQSResidualValue = np.linalg.norm(np.hstack((self.QS_residual.reArr, self.QS_residual.imArr)))
                        _initaveomega = self.omega.integrate()
                        print("{:>8d} {:>16f} {:>18e} {:>12f} {:>18e} {:>12f} {:>18e} {:>18e}".format(self.niter, self.iota, _initBoozerResidualValue, self.mu_qs, _initQSResidualValue, self.mu_omega, _initaveomega, _initBoozerResidualValue+self.mu_qs*_initQSResidualValue+self.mu_omega*_initaveomega))
                res = minimize(cost, self.initDOFs, callback=callback, **kwargs)
                if self.niter%nstep != 0:
                    _initBoozerResidualValue = np.linalg.norm(np.hstack((self.Boozer_residual.reArr, self.Boozer_residual.imArr)))
                    _initQSResidualValue = np.linalg.norm(np.hstack((self.QS_residual.reArr, self.QS_residual.imArr)))
                    _initaveomega = self.omega.integrate()
                    print("{:>8d} {:>16f} {:>18e} {:>12f} {:>18e} {:>12f} {:>18e} {:>18e}".format(self.niter, self.iota, _initBoozerResidualValue, self.mu_qs, _initQSResidualValue, self.mu_omega, _initaveomega, _initBoozerResidualValue+self.mu_qs*_initQSResidualValue+self.mu_omega*_initaveomega))
            
            elif 'trust' in kwargs.get('method'):
                kwargs.update({'method': 'trust-constr'})
                res = minimize(cost, self.initDOFs, options={'verbose':3}, **kwargs)
            else:
                pass
        
            if not res.success:
                print('Warning: ' + res.message)



if __name__ == '__main__':
    pass
