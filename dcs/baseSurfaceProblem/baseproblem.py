#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# baseproblem.py


import numpy as np
from ..toroidalField import ToroidalField
from ..geometry import Surface_cylindricalAngle 
from ..toroidalField import derivatePol, derivateTor
from typing import List


class BaseProblem_Cylinder: 
    
    def __init__(self, surf: Surface_cylindricalAngle, lambdaField: ToroidalField=None, omegaField: ToroidalField=None, iota: float=0.0, stellSym: bool=True) -> None:
        self.surf = surf 
        self.updateLambda(lambdaField) 
        self.updateOmega(omegaField) 
        self.updateIota(iota) 
        self.updateStellSym(stellSym)

    def updateLambda(self, lambdaField: ToroidalField) -> None:
        assert lambdaField.nfp == self.surf.nfp 
        self.lam = lambdaField 
    
    def updateOmega(self, omegaField: ToroidalField) -> None:
        assert self.surf.nfp == omegaField.nfp
        self.omega = omegaField
    
    @property
    def iota(self): 
        return self._iota

    @property
    def stellSym(self) -> bool:
        return self.surf.stellSym and (not self.omega.reIndex) and (not self.lam.reIndex)

    @property
    def nfp(self): 
        return self.surf.nfp

    @property
    def dthetadvartheta(self): 
        return 1 + derivatePol(self.lam) + self.iota*derivatePol(self.omega)

    @property
    def dthetadphi(self): 
        return derivateTor(self.lam) + self.iota*derivateTor(self.omega)

    @property 
    def dzetadvartheta(self): 
        return derivatePol(self.omega)

    @property
    def dzetadphi(self): 
        return 1 + derivateTor(self.omega)
    
    def updateIota(self, iota: float): 
        self._iota = iota

    def updateStellSym(self, stellSym: bool):
        if stellSym:
            self.surf.changeStellSym(stellSym)
            self.omega.reIndex, self.lam.reIndex = False, False 
        else:
            self.surf.changeStellSym(stellSym)
            self.omega.reIndex, self.lam.reIndex = True, True

    def updateNFP(self, nfp: int): 
        self.surf.r.nfp = nfp
        self.surf.z.nfp = nfp
        self.lam.nfp = nfp
        self.omega.nfp = nfp

    def transBoozer(self, valueField: ToroidalField or List[ToroidalField], mpol: int=None, ntor: int=None, **kwargs) -> ToroidalField:

        from collections import Iterable
        if mpol is None and ntor is None: 
            if isinstance(valueField, ToroidalField):
                mpol = valueField.mpol + max(self.omega.mpol, self.lam.mpol)
                ntor = valueField.ntor + max(self.omega.ntor, self.lam.ntor)
            elif isinstance(valueField, Iterable) and isinstance(valueField[0], ToroidalField):
                mpol = valueField[0].mpol + max(self.omega.mpol, self.lam.mpol) 
                ntor = valueField[0].ntor + max(self.omega.ntor, self.lam.ntor)
            else:
                print("Wrong type of the valuefield... ")
        sampleTheta = np.linspace(0, 2*np.pi, 2*mpol+1, endpoint=False) 
        sampleZeta = -np.linspace(0, 2*np.pi/self.nfp, 2*ntor+1, endpoint=False) 
        gridSampleZeta, gridSampleTheta = np.meshgrid(sampleZeta, sampleTheta) 

        # find the fixed point of vartheta and varphi 
        def varthetaphiValue(inits, theta, zeta):
            vartheta, varphi = inits[0], inits[1]
            lamValue = self.lam.getValue(vartheta, varphi) 
            omegaValue = self.omega.getValue(vartheta, varphi)
            return np.array([
                theta - lamValue - self.iota*omegaValue, 
                zeta - omegaValue
            ])

        from scipy.optimize import fixed_point
        gridVartheta, gridVarphi = np.zeros_like(gridSampleTheta), np.zeros_like(gridSampleZeta) 
        for i in range(len(gridVartheta)): 
            for j in range(len(gridVartheta[0])): 
                try:
                    varthetaphi = fixed_point(
                        varthetaphiValue, [gridSampleTheta[i,j],gridSampleZeta[i,j]], args=(gridSampleTheta[i,j],gridSampleZeta[i,j]), **kwargs
                    )
                except:
                    varthetaphi = fixed_point(
                        varthetaphiValue, [gridSampleTheta[i,j],gridSampleZeta[i,j]], args=(gridSampleTheta[i,j],gridSampleZeta[i,j]), method="iteration",**kwargs
                    )
                gridVartheta[i,j] = float(varthetaphi[0,0])
                gridVarphi[i,j] = float(varthetaphi[1,0])
        
        from ..toroidalField import fftToroidalField
        if isinstance(valueField, ToroidalField):
            sampleValue = valueField.getValue(gridVartheta, gridVarphi)
            return fftToroidalField(sampleValue, nfp=self.nfp)
        elif isinstance(valueField, Iterable) and isinstance(valueField[0], ToroidalField):
            ans = list()
            for _field in valueField:
                _sampleValue = _field.getValue(gridVartheta, gridVarphi)
                ans.append(fftToroidalField(_sampleValue, nfp=self.nfp))
            return ans
        else:
            print("Wrong type of the valuefield... ")

    def writeH5(self, filename: str="vacuumSurf"):
        import h5py
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
        import h5py
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

    @classmethod
    def init_VMECOutput(cls, vmec_out: str, surfaceIndex: int=-1, ssym: bool=True):
        from ..vmec import VMECOut
        vmeclib = VMECOut(vmec_out, ssym=ssym)
        surf, lam = vmeclib.getSurface(surfaceIndex)
        omega = vmeclib.getOmega(surfaceIndex)
        iota = vmeclib.iota(surfaceIndex)
        surfProblem = cls(surf, lam, omega, iota, ssym)
        return surfProblem



if __name__ == "__main__": 
    pass
