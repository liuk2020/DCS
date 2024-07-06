#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vacuumProblem.py


import logging
import numpy as np
from .vacuum import VacuumField 
from ..toroidalField import ToroidalField
from ..geometry import Surface_cylindricalAngle 
from ..toroidalField import changeResolution
from scipy.optimize import root


class VacuumProblem(VacuumField):

    def __init__(
        self, 
        surf: Surface_cylindricalAngle, 
        mpol: int, 
        ntor: int,
        targetIota: float=0.618, 
        stellSym: bool=True
    ) -> None:
        self.mpol, self.ntor = mpol, ntor
        _lam = self._init_lam(surf.nfp, self.mpol, self.ntor)
        _omega = self._init_omega(surf.nfp, self.mpol, self.ntor)
        super().__init__(surf, _lam, _omega, targetIota, stellSym) 

    def _init_lam(self, nfp: int, mpol: int, ntor: int): 
        return ToroidalField.constantField(0, nfp=nfp, mpol=mpol, ntor=ntor)

    def _init_omega(self, nfp: int, mpol: int, ntor: int):
        return ToroidalField.constantField(0, nfp=nfp, mpol=mpol, ntor=ntor)
    
    @property
    def initDOFs(self) -> np.ndarray:
        dofs = np.append(self.iota, self.lam.imArr[1: ])
        dofs = np.append(dofs, self.omega.imArr[1: ])
        if not self.stellSym:
            dofs = np.append(dofs, self.lam.reArr[1: ])
            dofs = np.append(dofs, self.omega.reArr[1: ])
        return dofs

    def updateResolution(self, mpol: int, ntor: int):
        self.mpol = mpol
        self.ntor = ntor
        self.lam = changeResolution(self.lam, mpol=mpol, ntor=ntor)
        self.omega = changeResolution(self.omega, mpol=mpol, ntor=ntor)

    def unpackDOFs(self, dofs: np.ndarray):
        length = (2*self.ntor+1)*self.mpol + self.ntor
        if self.stellSym:
            assert dofs.size == 2*length + 1
        else:
            assert dofs.size == 4*length + 1
        self.updateIota(dofs[0])
        for index in range(1, length+1):
            m, n = self.lam.indexReverseMap(index-1)
            self.lam.setIm(m, n, dofs[index])
        for index in range(length+1, 2*length+1):
            m, n = self.omega.indexReverseMap(index-1-length)
            self.omega.setIm(m, n, dofs[index])
        if not self.stellSym:
            for index in range(2*length+1, 3*length+1):
                m, n = self.lam.indexReverseMap(index-1-2*length)
                self.lam.setRe(m, n, dofs[index])
            for index in range(3*length+1, 4*length+1):
                m, n = self.omega.indexReverseMap(index-1-3*length)
                self.omega.setRe(m,n,dofs[index])

    def solve(self, 
        logfile: str="log", 
        logscreen: bool=True, 
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
        # solve #######################################################################################
        def cost(dofs, norm: bool=False):
            self.unpackDOFs(dofs)
            error = self.errerField()
            if not norm:
                return np.hstack(error.reArr, error.imArr)
            else:
                return np.linalg.norm(np.hstack(error.reArr, error.imArr))
        self.niter = 0 
        def callbackFun(xi):
            self.niter += 1
            if self.niter%50 == 0:
                logger.info("{:>8d} {:>16e} {:>16e}".format(0, self.iota, cost(xi,norm=True)))
        logger.info("{:>8} {:>16} {:>16}".format('niter', 'iota', 'residuals')) 
        logger.info("{:>8d} {:>16e} {:>16e}".format(0, self.iota, cost(self.initDOFs,norm=True)))
        res = root(cost, self.initDOFs, callback=callbackFun, **kwargs)
        self.unpackDOFs(res.x)
        if self.niter != 0 and self.niter%50 != 0:
            logger.info("{:>8d} {:>16e} {:>16e}".format(0, self.iota, cost(self.initDOFs,norm=True)))


if __name__ == "__main__": 
    pass
