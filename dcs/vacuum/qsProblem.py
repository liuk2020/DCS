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

    def __init__(self, surf: Surface_BoozerAngle = None, m: int=1, n: int=0, iota: float = 0.618, freeIota: bool = False, stemSym: bool = True) -> None:
        super().__init__(surf, iota, freeIota, stemSym)
        self.m = m
        self.n = n

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

    def solve(self, 
        mode: str="biobject", 
        log: str="log.txt", 
        weight: List=[0.99, 0.01], 
        muInit: float = 20.0,
        vmecinput: str="vacuum", 
        surfH5: str="boozerSurf",
        figname: str="value", 
        **kwargs
    ):
        # log #########################################################################################
        logger = logging.getLogger('my logger')
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log, mode='w', encoding='utf-8')
        sh = logging.StreamHandler()
        fmt = logging.Formatter(fmt = "%(asctime)s  - %(message)s")
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
        fh.setLevel(logging.INFO)
        sh.setLevel(logging.INFO)
        # solver ######################################################################################
        if mode == "biobject": 
            self.solve_biobject(logger, weight, **kwargs)
        elif mode == "Lagrange" or mode == "lagrange":
            self.solve_Lagrange(logger, muInit, **kwargs)
        else:
            logger.error("Incorrect Mode! ")
        # vmec input ###################################################################################
        if vmecinput is not None:
            surfCylinder = self.surf.toCylinder()
            surfCylinder.toVacuumVMEC(vmecinput) 
        # write surface ################################################################################ 
        if surfH5 is not None:
            self.surf.writeH5(surfH5)
        # plot #########################################################################################
        if figname is not None:
            import matplotlib.pyplot as plt
            try:
                import matplotlib
                matplotlib.rcParams['text.usetex'] = True
            except:
                pass
            fFun, gFun, gResidual = self.funResidual()
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

    def solve_biobject(self, logger, weight: List[float], **kwargs):
        def precost(dofs):
            self.setValue_DOF(dofs)
            fFun, gFun, deriGfun = self.funResidual()
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
                if self.freeIota:
                    logger.info("{:>8d} {:>14e} {:>14e} {:>14e} {:>14e}".format(self.niter, costF, costG, xi[-1], ans))
                else:
                    logger.info("{:>8d} {:>14e} {:>14e} {:>14e}".format(self.niter, costF, costG, ans))
        _costF, _costG = precost(self.initValue_DOF)
        _ans = weight[0]*_costF + weight[1]*_costG
        if self.freeIota:
            logger.info("{:>8} {:>14} {:>14} {:>14} {:>14}".format('niter', 'fFun', 'gFun', 'iota', 'costFunction')) 
            logger.info("{:>8d} {:>14e} {:>14e} {:>14e} {:>14e}".format(0, _costF, _costG, self.iota, _ans)) 
        else:
            logger.info("{:>8} {:>14} {:>14} {:>14}".format('niter', 'fFun', 'gFun', 'costFunction')) 
            logger.info("{:>8d} {:>14e} {:>14e} {:>14e}".format(0, _costF, _costG, _ans)) 
        res = minimize(cost, self.initValue_DOF, callback=callbackFun, **kwargs)
        self.setValue_DOF(res.x)
        
    def solve_Lagrange(self, logger, muInit, **kwargs): 
        def precost(dofs):
            self.setValue_DOF(dofs[:-1])
            fFun, gFun, deriGfun = self.funResidual()
            normalizeGDeri = deriGfun * (1/gFun.getRe(0,0))
            costF = np.linalg.norm(np.hstack((fFun.reArr, fFun.imArr))) 
            costG = np.linalg.norm(np.hstack((normalizeGDeri.reArr, normalizeGDeri.imArr)))
            return costF, costG
        def cost(dofs): 
            costF, costG = precost(dofs)
            return dofs[-1]*costF + costG
        self.niter = 0 
        def callbackFun(xi):
            self.niter += 1 
            if self.niter%10 == 0:
                costF, costG = precost(xi)
                ans = xi[-1]*costF + costG
                if self.freeIota:
                    logger.info("{:>8d} {:>14e} {:>14e} {:>14e} {:>14e} {:>14e}".format(self.niter, costF, costG, xi[-2], xi[-1], ans))
                else:
                    logger.info("{:>8d} {:>14e} {:>14e} {:>14e} {:>14e}".format(self.niter, costF, costG, xi[-1], ans))
        _costF, _costG = precost(np.append(self.initValue_DOF,muInit))
        _ans = muInit*_costF + _costG
        if self.freeIota:
            logger.info("{:>8} {:>14} {:>14} {:>14} {:>14} {:>14}".format('niter', 'fFun', 'gFun', 'iota', 'mu', 'costFunction')) 
            logger.info("{:>8d} {:>14e} {:>14e} {:>14e} {:>14e} {:>14e}".format(0, _costF, _costG, self.iota, muInit, _ans)) 
        else:
            logger.info("{:>8} {:>14} {:>14} {:>14} {:>14}".format('niter', 'fFun', 'gFun', 'mu', 'costFunction')) 
            logger.info("{:>8d} {:>14e} {:>14e} {:>14e} {:>14e}".format(0, _costF, _costG, muInit, _ans)) 
        res = minimize(cost, np.append(self.initValue_DOF,muInit), callback=callbackFun, **kwargs)
        self.setValue_DOF(res.x)


if __name__ == "__main__":
    pass
