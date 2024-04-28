#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# qsSurfaceQS.py


import numpy as np
from .vacuumProblem import VacuumSurface
from ..geometry import Surface_BoozerAngle 
from ..toroidalField import ToroidalField 
from ..toroidalField import derivateTor, derivatePol
from scipy.optimize import minimize
from typing import List, Tuple 


class QsSurface(VacuumSurface):
    r"""
    The problem of directly constructing quasisymmetric surface in vacuum. 
        $$ B = B(m\theta - nN_f\zeta)$$
    """

    def __init__(self, surf: Surface_BoozerAngle = None, m: int=1, n: int=0, iota: float = 0.618, freeIota: bool = False, stemSym: bool = True) -> None:
        super().__init__(surf, iota, freeIota, stemSym)
        self.m = m
        self.n = n

    # def solve(self, debug: bool=False, order: int=2, weight: List[float]=[0.8, 0.2], printLog: bool=True, printIter: int=50, **kwargs): 
    #     if kwargs.get("method") == None:
    #         kwargs.update({"method": "BFGS"})
    #     self.iterations = 0
    #     res = minimize(self.cost, self.initValue_DOF, (order, weight, printLog, printIter), **kwargs)
    #     if debug: 
    #         return res
    #     else:
    #         pass

    # def cost(self, dofValue: np.ndarray, order: int, weight: List[float], printLog: bool, printIter: int) -> float: 
    #     self.setValue_DOF(dofValue) 
    #     normalizeFfun, normalizeGfun = self.funErr()
    #     costF = np.linalg.norm(np.hstack((normalizeFfun.reArr, normalizeFfun.imArr)), ord=order) 
    #     costG = np.linalg.norm(np.hstack((normalizeGfun.reArr, normalizeGfun.imArr)), ord=order) 
    #     costValue = weight[0]*costF + weight[1]*costG
    #     # costValue = np.linalg.norm(np.hstack((normalizeFfun.reArr, normalizeFfun.imArr, normalizeGfun.reArr, normalizeGfun.imArr)), ord=order) 
    #     self.iterations += 1
    #     if printLog and (self.iterations%printIter == 0):
    #         print("iter: {:<5d}".format(self.iterations) + 
    #         ", iota: {:+5f}".format(self.iota) + 
    #         ", errF: {:5e}".format(costF) + 
    #         ", errG: {:5e}".format(costG) + 
    #         ", objective value: {:5e}".format(costValue))
    #     return costValue

    def funValue(self) -> Tuple[ToroidalField]:
        g_thetatheta, g_thetazeta, g_zetazeta = self.surf.metric
        fFun = g_thetazeta + g_thetatheta*self.iota
        gFun = g_zetazeta + g_thetazeta*self.iota
        return fFun, gFun

    def funResidual(self) -> Tuple[ToroidalField]:
        fFun, gFun = self.funValue()
        if self.n == 0: 
            deriGfun = derivateTor(gFun)
        elif self.m == 0:
            deriGfun = derivatePol(gFun)
        else: 
            deriGfun = derivatePol(gFun)*(self.n*self.nfp) + derivateTor(gFun)*self.m
        return fFun, gFun, deriGfun


if __name__ == "__main__":
    pass
