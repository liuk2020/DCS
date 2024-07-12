#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vacuumProblem.py


import logging
import h5py
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
        stellSym: bool=True
    ) -> None:
        self.mpol, self.ntor = mpol, ntor
        _lam = ToroidalField.constantField(0, nfp=surf.nfp, mpol=mpol, ntor=ntor)
        _lam.reIndex, _lam.imIndex = False, True
        _omega = ToroidalField.constantField(0, nfp=surf.nfp, mpol=mpol, ntor=ntor)
        _omega.reIndex, _omega.imIndex = False, True
        super().__init__(surf, _lam, _omega, iota, stellSym) 
        self.updateStellSym(stellSym)
    
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
        nStep: int=10,
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
        def cost(dofs, norm: bool=True):
            self.unpackDOFs(dofs)
            error = self.errerField()
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
                logger.info("{:>8d} {:>16e} {:>16e}".format(self.niter, self.iota, cost(xi,norm=True)))
        logger.info("{:>8} {:>16} {:>16}".format('niter', 'iota', 'residuals')) 
        logger.info("{:>8d} {:>16e} {:>16e}".format(0, self.iota, cost(self.initDOFs,norm=True)))
        # res = root(cost, self.initDOFs, callback=callbackFun, **kwargs)
        res = minimize(cost, self.initDOFs, callback=callbackFun, **kwargs)
        if self.niter != 0 and self.niter%nStep != 0:
            logger.info("{:>8d} {:>16e} {:>16e}".format(self.niter, self.iota, cost(res.x,norm=True)))

    def writeH5(self, filename: str="vacuumSurf"):
        with h5py.File(filename+".h5", 'w') as f:
            f.create_dataset(
                "resolution", 
                data = (self.nfp, int(self.stellSym), self.mpol, self.ntor, self.surf.r.mpol, self.surf.r.ntor, self.surf.z.mpol, self.surf.z.ntor, self.iota), 
                dtype = "int32"
            )
            f.create_group("r")
            f.create_group("z")
            f.create_group("lam")
            f.create_group("omega")
            f["r"].create_dataset("re", self.surf.r.reArr)
            f["z"].create_dataset("im", self.surf.z.imArr)
            f["lam"].create_dataset("im", self.lam.imArr)
            f["omega"].create_dataset("im", self.omega.imArr)
            if not self.stellSym:
                f["r"].create_dataset("im", self.surf.r.imArr)
                f["z"].create_dataset("re", self.surf.z.reArr)
                f["lam"].create_dataset("re", self.lam.reArr)
                f["omega"].create_dataset("re", self.omega.reArr)

    @classmethod
    def readH5(cls, filename):
        with h5py.File(filename, 'r') as f:
            nfp = int(f["resolution"][0])
            stellsym = bool(f["resolution"][1])
            mpol = int(f["resolution"][2])
            ntor = int(f["resolution"][3])
            r_mpol = int(f["resolution"][4])
            r_ntor = int(f["resolution"][5])
            z_mpol = int(f["resolution"][6])
            z_ntor = int(f["resolution"][7])
            iota = float(f["resolution"][8])
            if stellsym:
                _r = ToroidalField(
                    nfp=nfp, mpol=r_mpol, ntor=r_ntor, 
                    reArr=f["r"]["re"][:], 
                    imArr=np.zeros_like(f["r"]["re"][:]), 
                    imIndex=False
                )
                _z = ToroidalField(
                    nfp=nfp, mpol=z_mpol, ntor=z_ntor, 
                    reArr=np.zeros_like(f["z"]["im"][:]),
                    imArr=f["z"]["im"][:],  
                    reIndex=False
                )
                _lam = ToroidalField(
                    nfp=nfp, mpol=mpol, ntor=ntor, 
                    reArr=np.zeros_like(f["lam"]["im"][:]),
                    imArr=f["lam"]["im"][:],  
                    reIndex=False
                )
                _omega = ToroidalField(
                    nfp=nfp, mpol=mpol, ntor=ntor, 
                    reArr=np.zeros_like(f["omega"]["im"][:]),
                    imArr=f["omega"]["im"][:],  
                    reIndex=False
                )
            else:
                _r = ToroidalField(
                    nfp=nfp, mpol=r_mpol, ntor=r_ntor, 
                    reArr=f["r"]["re"][:], 
                    imArr=f["r"]["im"][:]
                )
                _z = ToroidalField(
                    nfp=nfp, mpol=z_mpol, ntor=z_ntor, 
                    reArr=f["z"]["re"][:],
                    imArr=f["z"]["im"][:] 
                )
                _lam = ToroidalField(
                    nfp=nfp, mpol=mpol, ntor=ntor, 
                    reArr=f["lam"]["re"][:],
                    imArr=f["lam"]["im"][:] 
                )
                _omega = ToroidalField(
                    nfp=nfp, mpol=mpol, ntor=ntor, 
                    reArr=f["omega"]["re"][:],
                    imArr=f["omega"]["im"][:]
                )
        _vaccumSurf = cls(Surface_cylindricalAngle(_r,_z), mpol=mpol, ntor=ntor, iota=iota, stellsym=stellsym)
        _vaccumSurf.updateLam(_lam)
        _vaccumSurf.updateOmega(_omega)
        _vaccumSurf.updateIota(iota)
        return _vaccumSurf


if __name__ == "__main__": 
    pass
