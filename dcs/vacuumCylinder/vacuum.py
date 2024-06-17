#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vacuum.py


from ..toroidalField import ToroidalField
from ..geometry import Surface_cylindricalAngle 
from ..toroidalField import derivatePol, derivateTor


class VacuumField: 
    
    def __init__(self, surf: Surface_cylindricalAngle, lambdaField: ToroidalField=None, omegaField: ToroidalField=None, iota: float=0.0) -> None:
        self.surf = surf 
        self.initLambda(lambdaField) 
        self.initOmega(omegaField) 
        self._iota = iota 

    def initLambda(self, lambdaField: ToroidalField) -> None:
        self.lam = lambdaField 
    
    def initOmega(self, omegaField: ToroidalField) -> None:
        self.omega = omegaField
    
    @property
    def iota(self): 
        return self._iota

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

    def updateCoefficient(self): 
        self.coefficient = (
            self.dthetadvartheta*self.dzetadphi 
            - self.dthetadphi*self.dzetadvartheta
        )

    def errerField(self) -> ToroidalField: 
        g_thetatheta, g_thetazeta, g_zetazeta = self.surf.metric 
        return (
            self.dzetadphi * (self.dthetadphi-self.iota*self.dzetadphi) * g_thetatheta 
            - (self.dthetadvartheta*self.dzetadphi + self.dthetadphi*self.dzetadvartheta -2*self.iota*self.dzetadvartheta*self.dzetadphi) * g_thetazeta 
            + self.dzetadvartheta * (self.dthetadvartheta-self.iota*self.dzetadvartheta) * g_zetazeta
        )



if __name__ == "__main__": 
    pass
