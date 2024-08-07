#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vacuumProblem.py


import logging
import numpy as np
from .vacuum import VacuumField 
from ..toroidalField import ToroidalField
from ..geometry import Surface_cylindricalAngle 
from ..toroidalField import changeResolution
from scipy.optimize import root, minimize


class VacuumProblem(VacuumField):

    def __init__(
        self, 
        surf: Surface_cylindricalAngle, 
        mpol: int, 
        ntor: int,
        iota: float=0.0, 
        stellSym: bool=True,
        fixIota: bool=False,
        logfile: str="log",
        logscreen: bool=True,
    ) -> None:
        _lam = ToroidalField.constantField(0, nfp=surf.nfp, mpol=mpol, ntor=ntor)
        _lam.reIndex, _lam.imIndex = False, True
        _omega = ToroidalField.constantField(0, nfp=surf.nfp, mpol=mpol, ntor=ntor)
        _omega.reIndex, _omega.imIndex = False, True
        super().__init__(surf, _lam, _omega, iota, stellSym) 
        self.updateStellSym(stellSym)
        self.fixIota = fixIota
        self._init_paras()
        self._init_log(logfile, logscreen)
        self.logger.info("Problem initialization is done... ")

    def _init_log(self, logfile, logscreen) -> None:
        import time
        self.logger = logging.getLogger(str(time.time()))
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(logfile+".txt", mode='w', encoding='utf-8')
        fmt = logging.Formatter(fmt = "%(asctime)s  - %(message)s")
        fh.setFormatter(fmt)
        self.logger.addHandler(fh)
        fh.setLevel(logging.INFO)
        if logscreen:
            sh = logging.StreamHandler() 
            sh.setFormatter(fmt)
            self.logger.addHandler(sh)
            sh.setLevel(logging.INFO)

    def _init_paras(self): 
        self._powerIndex = 0.97
        self._iotaIndex = 50

    @property
    def mpol(self) -> int:
        return self.lam.mpol
    
    @property
    def ntor(self) -> int:
        return self.lam.ntor

    @property
    def initDOFs(self) -> np.ndarray:
        # dofs = [self.iota/self._iotaIndex]
        dofs = list()
        for index, value in enumerate(self.lam.imArr[1: ]):
            m, n = self.lam.indexReverseMap(index+1)
            dofs.append(value / pow(self._powerIndex,abs(m)+abs(n)))
        for index, value in enumerate(self.omega.imArr[1: ]):
            m, n = self.omega.indexReverseMap(index+1)
            dofs.append(value / pow(self._powerIndex,abs(m)+abs(n)))
        if not self.stellSym:
            for index, value in enumerate(self.lam.reArr[1: ]):
                m, n = self.lam.indexReverseMap(index+1)
                dofs.append(value / pow(self._powerIndex,abs(m)+abs(n)))
            for index, value in enumerate(self.omega.reArr[1: ]):
                m, n = self.omega.indexReverseMap(index+1)
                dofs.append(value / pow(self._powerIndex,abs(m)+abs(n)))
        if  not self.fixIota:
            dofs.append(self.iota/self._iotaIndex)
        return np.array(dofs)

    def updateResolution(self, mpol: int, ntor: int):
        # self.mpol = mpol
        # self.ntor = ntor
        self.lam = changeResolution(self.lam, mpol=mpol, ntor=ntor)
        self.omega = changeResolution(self.omega, mpol=mpol, ntor=ntor)

    def unpackDOFs(self, dofs: np.ndarray):
        length = (2*self.ntor+1)*self.mpol + self.ntor
        if self.stellSym:
            if not self.fixIota:
                assert dofs.size == 2*length + 1
            else:
                dofs.size == 2*length
        else:
            if not self.fixIota:
                assert dofs.size == 4*length + 1
            else:
                assert dofs.size == 4*length
        if not self.fixIota:
            self.updateIota(dofs[-1]*self._iotaIndex)
        for index in range(0, length):
            m, n = self.lam.indexReverseMap(index+1)
            self.lam.setIm(m, n, dofs[index]*pow(self._powerIndex,abs(m)+abs(n)))
        for index in range(length, 2*length):
            m, n = self.omega.indexReverseMap(index-length+1)
            self.omega.setIm(m, n, dofs[index]*pow(self._powerIndex,abs(m)+abs(n)))
        if not self.stellSym:
            for index in range(2*length, 3*length):
                m, n = self.lam.indexReverseMap(index-2*length+1)
                self.lam.setRe(m, n, dofs[index]*pow(self._powerIndex,abs(m)+abs(n)))
            for index in range(3*length, 4*length):
                m, n = self.omega.indexReverseMap(index-3*length+1)
                self.omega.setRe(m,n,dofs[index]*pow(self._powerIndex,abs(m)+abs(n)))

    def solve(self, 
        type: str="minimize",
        nStep: int=10,
        **kwargs
    ):
        r"""
        Solve the equation of the constrain of the vacuum field to get the lambda and omega. 
        Args:
            `type`: type of solver.  Should be "minimize" or "root" 
            `nStep`: number of the steps at log output
        """
        self.logger.info(f"The resolution of the geometry:     mpol={max(self.surf.r.mpol,self.surf.z.mpol)}, ntor={max(self.surf.r.ntor,self.surf.z.ntor)}")
        self.logger.info(f"The resolution of lambda and omega: mpol={self.mpol}, ntor={self.ntor}")
        if  max(self.surf.r.ntor,self.surf.z.ntor) == 0:
            self.logger.info("The surface is axisymmetrical...")
            self.solveAxisymmetry()
            return
        def cost(dofs, norm: bool=True):
            self.unpackDOFs(dofs)
            error = self.vacuumError()
            if not norm:
                if self.stellSym:
                    constraints = [error.reArr[0]]
                    for m in range(1, 2*self.mpol+1):
                        constraints.append(error.getRe(m,0))
                    for n in range(1, 2*self.ntor+1):
                        constraints.append(error.getRe(0,n))
                    for m in range(1, 2*self.mpol+1):
                        for n in range(1, 2*self.ntor+1):
                            constraints.append(error.getRe(m,n))
                    return np.array(constraints)
                # TODO: the constraints without the stellarator symmetry 
                else:
                    pass
            else:
                if self.stellSym:
                    return np.linalg.norm(error.reArr)
                else:
                    return np.linalg.norm(np.hstack((error.reArr, error.imArr)))
        self.niter = 0 
        def callbackFun(xi):
            self.niter += 1
            if self.niter%nStep == 0:
                self.logger.info("{:<10d} {:<16f} {:<16e}".format(self.niter, self.iota, cost(xi,norm=True)))
        g_thetatheta, g_thetaphi, g_phiphi = self.surf.metric 
        if not self.fixIota:
            _iota = - g_thetaphi.getRe(0,0) / g_thetatheta.getRe(0,0)
            self.updateIota(_iota)
        self.logger.info(f"Initial iota: {self.iota}")
        self.logger.info("{:<10} {:<16} {:<16}".format('niter', 'iota', 'residuals')) 
        self.logger.info("{:<10d} {:<16f} {:<16e}".format(0, self.iota, cost(self.initDOFs,norm=True)))
        if type == "minimize":
            res = minimize(cost, self.initDOFs, callback=callbackFun, **kwargs)
        elif type == "root":
            res = root(cost, self.initDOFs, **kwargs)
        if self.niter != 0 and self.niter%nStep != 0:
            self.logger.info("{:<10d} {:<16f} {:<16e}".format(self.niter, self.iota, cost(res.x,norm=True)))
        return

    def solveAxisymmetry(self):
        _lam = ToroidalField.constantField(0, nfp=self.nfp, mpol=self.mpol, ntor=self.ntor)
        _omega = ToroidalField.constantField(0, nfp=self.nfp, mpol=self.mpol, ntor=self.ntor)
        self.updateLambda(_lam)
        self.updateOmega(_omega)
        self.updateStellSym(self.stellSym)
        self.updateIota(0)
        return


if __name__ == "__main__": 
    pass
