#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# qsSurfaceQS.py


import numpy as np
from .vacuumProblem import VacuumSurface
from ..geometry import Surface_BoozerAngle 
from ..toroidalField import derivateTor, derivatePol
from scipy.optimize import minimize
from typing import List


class QsSurface(VacuumSurface):
    r"""
    The problem of directly constructing quasisymmetric surface in vacuum. 
        $$ B = B(m\theta - nN_f\zeta)$$
    """

    def __init__(self, surf: Surface_BoozerAngle = None, m: int=1, n: int=0, iota: float = 0.618, unfixIota: bool = False, stemSym: bool = True) -> None:
        super().__init__(surf, iota, unfixIota, stemSym)
        self.m = m
        self.n = n

    def solve(self, debug: bool=False, order: int=2, weight: List[float]=[0.5,0.5], **kwargs): 
        res = minimize(self.cost, np.zeros(self.numsDOF+self.unfixIota), (order, weight), **kwargs)
        if debug: 
            return res
        else:
            pass

    def cost(self, dofValue: np.ndarray, order: int, weight: List[float]) -> float: 
        self.setDOF(dofValue)
        g_thetatheta, g_thetazeta, g_zetazeta = self.surf.metric
        fFun = g_thetazeta + g_thetatheta*self.iota
        gFun = g_zetazeta + g_thetazeta*self.iota
        if self.n == 0: 
            deriGfun = derivateTor(gFun)
        elif self.m == 0:
            deriGfun = derivatePol(gFun)
        else: 
            deriGfun = derivatePol(gFun)*(self.n*self.nfp) + derivateTor(gFun)*self.m
        normalizeFfun = fFun * (1/g_zetazeta.getRe(0,0))
        normalizeGfun = deriGfun * (1/gFun.getRe(0,0))
        # return (
        #     weight[0] * (np.linalg.norm(np.hstack((normalizeFfun.reArr, normalizeFfun.imArr)), ord=order)) +
        #     weight[1] * (np.linalg.norm(np.hstack((normalizeGfun.reArr, normalizeGfun.imArr)), ord=order))
        # ) 
        return (
            np.linalg.norm(np.hstack((normalizeFfun.reArr, normalizeFfun.imArr, normalizeGfun.reArr, normalizeGfun.imArr)), ord=order)
        )

    def setDOF(self, dofValue: np.ndarray) -> None:
        valueIndex = 0
        while valueIndex < self.numsDOF:
            for dofkey in self.dofkeys:
                for dofIndex, dof in enumerate(self.dof[dofkey]): 
                    if dof:
                        m, n = self.indexReverseMap(dofIndex)
                        if dofkey == "rc": 
                            self.surf.r.setRe(m, n, dofValue[dofIndex])
                            valueIndex += 1
                            continue
                        elif dofkey == "zs": 
                            self.surf.z.setIm(m, n, dofValue[dofIndex])
                            valueIndex += 1
                            continue
                        elif dofkey == "omegas":
                            self.surf.omega.setIm(m, n, dofValue[dofIndex])
                            valueIndex += 1 
                            continue
                        elif dofkey == "rs": 
                            self.surf.r.setIm(m, n, dofValue[dofIndex])
                            valueIndex += 1
                            continue
                        elif dofkey == "zc": 
                            self.surf.z.setRe(m, n, dofValue[dofIndex])
                            valueIndex += 1
                            continue
                        elif dofkey == "omegac": 
                            self.surf.omega.setRe(m, n, dofValue[dofIndex])
                            valueIndex += 1
                            continue
                        else: 
                            continue
                    else: 
                        continue
        if self.unfixIota:
            self.changeIota(dofValue[-1])
        return


if __name__ == "__main__":
    pass
