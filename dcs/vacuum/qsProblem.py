#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# qsSurfaceQS.py


import logging
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

    def __init__(self, surf: Surface_BoozerAngle = None, m: int=1, n: int=0, stemSym: bool = True) -> None:
        super().__init__(surf, stemSym)
        self.m = m
        self.n = n

    def funValue(self, targetIota: float) -> Tuple[ToroidalField]:
        self.g_thetatheta, self.g_thetazeta, self.g_zetazeta = self.surf.metric
        self.iota = self.getIota(self.g_thetazeta, self.g_thetatheta)
        fFun = self.g_thetazeta + self.g_thetatheta*targetIota
        gFun = self.g_zetazeta + self.g_thetazeta*targetIota
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

    def solve(self, 
        targetIota: float=0.309, 
        mode: str="biobject", 
        logfile: str="log.txt", 
        logscreen: bool=True, 
        weight: List=[0.99, 0.01], 
        vmecinput: str="vacuum", 
        surfName: str="boozerSurf",
        figname: str="value", 
        **kwargs
    ):
        # log #########################################################################################
        logger = logging.getLogger('my logger')
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(logfile, mode='w', encoding='utf-8')
        fmt = logging.Formatter(fmt = "%(asctime)s  - %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        fh.setLevel(logging.INFO)
        if logscreen:
            sh = logging.StreamHandler() 
            sh.setFormatter(fmt)
            logger.addHandler(sh)
            sh.setLevel(logging.INFO)
        # solver ######################################################################################
        if kwargs.get("method") == None: 
            kwargs.update({"method": "CG"})
        if mode == "biobject": 
            self.solve_biobject(targetIota, logger, weight, **kwargs)
        elif mode == "Lagrange" or mode == "lagrange":
            self.solve_Lagrange(targetIota, logger, **kwargs)
        else:
            logger.error("Incorrect Mode! ")
        # vmec input ###################################################################################
        if vmecinput is not None:
            surfCylinder = self.surf.toCylinder()
            surfCylinder.toVacuumVMEC(vmecinput) 
        # write surface ################################################################################ 
        if surfName is not None:
            self.surf.writeH5(surfName)
            fig = self.surf.plot_plt()
            fig.savefig(surfName+".pdf")
        # plot #########################################################################################
        if figname is not None:
            import matplotlib.pyplot as plt
            try:
                import matplotlib
                matplotlib.rcParams['text.usetex'] = True
            except:
                pass
            fFun, gFun, gResidual = self.funResidual(targetIota)
            fig, ax = plt.subplots()
            _ = fFun.plot_plt(fig=fig, ax=ax)
            fig.tight_layout()
            fig.savefig("fFun_"+figname+".jpg")
            fig, ax = plt.subplots()
            _ = (gFun*(1/gFun.getRe(0,0))).plot_plt(fig=fig, ax=ax)
            fig.tight_layout()
            fig.savefig("gFun_"+figname+".jpg")
            fig, ax = plt.subplots()
            _ = (gResidual*(1/gFun.getRe(0,0))).plot_plt(fig=fig, ax=ax)
            fig.tight_layout()
            fig.savefig("gDeri_"+figname+".jpg")

    def solve_biobject(self, targetIota, logger, weight: List[float], **kwargs):
        def precost(dofs):
            self.unpackDOF(dofs)
            fFun, gFun, deriGfun = self.funResidual(targetIota)
            normalizeGDeri = deriGfun * (1/gFun.getRe(0,0))
            costF = np.linalg.norm(np.hstack((fFun.reArr, fFun.imArr))) 
            costG = np.linalg.norm(np.hstack((normalizeGDeri.reArr, normalizeGDeri.imArr)))
            return costF, costG
        def cost(dofs): 
            costF, costG = precost(dofs)
            return weight[0]*costF + weight[1]*costG
        self.niter = 0 
        def callbackFun(xi):
            self.niter += 1 
            if self.niter%10 == 0:
                costF, costG = precost(xi)
                ans = weight[0]*costF + weight[1]*costG
                logger.info("{:>8d} {:>16e} {:>16e} {:>16e} {:>16e}".format(self.niter, costF, costG, self.iota, ans))
        _costF, _costG = precost(self.initValue_DOF)
        _ans = weight[0]*_costF + weight[1]*_costG
        logger.info("{:>8} {:>16} {:>16} {:>16} {:>16}".format('niter', 'fFun', 'gFun', '-<g_uv>/<g_uu>', 'costFunction')) 
        logger.info("{:>8d} {:>16e} {:>16e} {:>16e} {:>16e}".format(0, _costF, _costG, self.iota, _ans)) 
        res = minimize(cost, self.initValue_DOF, callback=callbackFun, **kwargs)
        _costF, _costG = precost(res.x)
        _ans = weight[0]*_costF + weight[1]*_costG
        logger.info("{:>8d} {:>16e} {:>16e} {:>16e} {:>16e}".format(self.niter, _costF, _costG, self.iota, _ans)) 
        
    def solve_Lagrange(self, targetIota, logger, muInit: float=99.0, **kwargs): 
        def precost(dofs):
            self.unpackDOF(dofs[:-1])
            fFun, gFun, deriGfun = self.funResidual(targetIota)
            normalizeGDeri = deriGfun * (1/gFun.getRe(0,0))
            costF = np.linalg.norm(np.hstack((fFun.reArr, fFun.imArr))) 
            costG = np.linalg.norm(np.hstack((normalizeGDeri.reArr, normalizeGDeri.imArr)))
            return costF, costG
        def cost(dofs): 
            costF, costG = precost(dofs)
            return (dofs[-1]*costF+costG) / (dofs[-1]+1)
        self.niter = 0 
        def callbackFun(xi):
            self.niter += 1 
            if self.niter%10 == 0:
                costF, costG = precost(xi)
                ans = (xi[-1]*costF + costG) / (xi[-1]+1)
                logger.info("{:>8d} {:>16e} {:>16e} {:>16e} {:>16e} {:>16e}".format(self.niter, costF, costG, self.iota, xi[-1], ans))
        _costF, _costG = precost(np.append(self.initValue_DOF,muInit))
        _ans = (muInit*_costF+_costG) / (muInit+1)
        logger.info("{:>8} {:>16} {:>16} {:>16} {:>16} {:>16}".format('niter', 'fFun', 'gFun', '-<g_uv>/<g_uu>', 'mu', 'costFunction')) 
        logger.info("{:>8d} {:>16e} {:>16e} {:>16e} {:>16e} {:>16e}".format(0, _costF, _costG, self.iota, muInit, _ans)) 
        res = minimize(cost, np.append(self.initValue_DOF,muInit), callback=callbackFun, **kwargs)
        # self.unpackDOF(res.x)
        _costF, _costG = precost(res.x)
        _ans = (res.x[-1]*_costF+_costG) / (res.x[-1]+1)
        logger.info("{:>8d} {:>16e} {:>16e} {:>16e} {:>16e} {:>16e}".format(self.niter, _costF, _costG, self.iota, res.x[-1], _ans)) 


if __name__ == "__main__":
    pass
