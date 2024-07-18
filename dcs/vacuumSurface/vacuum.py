#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vacuum.py


import numpy as np 
from ..baseSurfaceProblem import BaseProblem_Cylinder
from ..toroidalField import ToroidalField
from ..geometry import Surface_cylindricalAngle 
from ..toroidalField import derivatePol, derivateTor
from ..toroidalField import fftToroidalField
from typing import List


class VacuumField(BaseProblem_Cylinder):

    def __init__(self, surf: Surface_cylindricalAngle, lambdaField: ToroidalField = None, omegaField: ToroidalField = None, iota: float = 0, stellSym: bool = True) -> None:
        super().__init__(surf, lambdaField, omegaField, iota, stellSym) 

    def updateCoefficient(self): 
        self.coefficient = (
            self.dthetadvartheta*self.dzetadphi 
            - self.dthetadphi*self.dzetadvartheta
        )

    def vacuumError(self) -> ToroidalField: 
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


if __name__ == "__main__": 
    pass
