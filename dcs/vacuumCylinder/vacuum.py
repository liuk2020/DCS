#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vacuum.py


import numpy as np 
from ..toroidalField import ToroidalField
from ..geometry import Surface_cylindricalAngle 
from ..toroidalField import derivatePol, derivateTor
from ..toroidalField import fftToroidalField
from typing import List


class VacuumField: 
    
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

    def updateCoefficient(self): 
        self.coefficient = (
            self.dthetadvartheta*self.dzetadphi 
            - self.dthetadphi*self.dzetadvartheta
        )

    def errerField(self) -> ToroidalField: 
        g_thetatheta, g_thetaphi, g_phiphi = self.surf.metric 
        return (
            self.dzetadphi * (self.dthetadphi-self.iota*self.dzetadphi) * g_thetatheta 
            - (self.dthetadvartheta*self.dzetadphi + self.dthetadphi*self.dzetadvartheta -2*self.iota*self.dzetadvartheta*self.dzetadphi) * g_thetaphi 
            + self.dzetadvartheta * (self.dthetadvartheta-self.iota*self.dzetadvartheta) * g_phiphi 
        )

    def gField(self) -> ToroidalField: 
        g_thetatheta, g_thetaphi, g_phiphi = self.surf.metric 
        return (
            self.dthetadphi * (self.dthetadphi-self.iota*self.dzetadphi) * g_thetatheta 
            - (2*self.dthetadvartheta*self.dthetadphi - self.iota*(self.dthetadvartheta*self.dzetadphi + self.dthetadphi*self.dzetadvartheta)) * g_thetaphi 
            + self.dthetadvartheta * (self.dthetadvartheta-self.iota*self.dzetadvartheta) * g_phiphi 
        )

    def bField(self, mpol: int=None, ntor: int=None) -> ToroidalField:
        if mpol is None:
            mpol = 2*self.omega.mpol+1
        if ntor is None:
            ntor = 2*self.omega.ntor+1
        deltaTheta = 2*np.pi / (2*mpol+1)
        deltaZeta = 2*np.pi / self.nfp / (2*ntor+1)
        sampleTheta, sampleZeta = np.arange(2*mpol+1)*deltaTheta, np.arange(2*ntor+1)*deltaZeta
        gridSampleZeta, gridSampleTheta = np.meshgrid(sampleZeta, sampleTheta)
        gField = self.gField()
        self.updateCoefficient()
        sampleB = np.power(
            gField.getValue(gridSampleTheta, gridSampleZeta) / np.power(self.coefficient.getValue(gridSampleTheta, gridSampleZeta), 2), 0.5
        )
        return fftToroidalField(sampleB, nfp=self.nfp)


    def transBoozer(self, valueField: ToroidalField or List[ToroidalField], mpol: int=None, ntor: int=None, **kwargs) -> ToroidalField:

        if mpol is None and ntor is None: 
            mpol = valueField.mpol + max(self.omega.mpol, self.lam.mpol) 
            ntor = valueField.ntor + max(self.omega.ntor, self.lam.ntor) 
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
        
        from collections import Iterable
        from ..toroidalField import fftToroidalField
        if isinstance(valueField, ToroidalField):
            sampleValue = valueField.getValue(gridVartheta, gridVarphi)
            return fftToroidalField(sampleValue, nfp=self.nfp)
        elif isinstance(valueField, Iterable) and isinstance(valueField[0], ToroidalField):
            ans = list()
            for _field in valueField:
                _sampleValue = _field.getValue(gridVartheta, gridVarphi)
                ans.append(fftToroidalField(sampleValue, nfp=self.nfp))
            return ans
        else:
            print("Wrong type of the valuefield... ")


if __name__ == "__main__": 
    pass
