#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# surfaceBoozerAngle.py


from .surface import Surface
from ..toroidalField import ToroidalField
from ..toroidalField import derivatePol, derivateTor
from ..toroidalField import changeResolution 


class Surface_BoozerAngle(Surface):
    r"""
    The magnetic surface in Boozer coordinates $(\theta, \zeta)$.  
        $$ \phi = -\zeta + \omega(\theta,\zete) $$ 
    There is a mapping with coordinates corrdinates $(R, \phi, Z)$
        $$ R = R(\theta, \zeta) $$ 
        $$ \phi = -\zeta + \omega(\theta,\zete) $$ 
        $$ Z = Z(\theta, \zeta) $$ 
    """

    def __init__(self, r: ToroidalField, z: ToroidalField, omega: ToroidalField) -> None:
        super().__init__(r, z)
        self.omega = omega

    def changeResolution(self, mpol: int, ntor: int): 
        self.r = changeResolution(self.r, mpol, ntor)
        self.z = changeResolution(self.z, mpol, ntor)
        self.omega = changeResolution(self.omega, mpol, ntor)


    @property
    def metric(self): 
        dPhidTheta = derivatePol(self.omega)
        dPhidZeta = derivateTor(self.omega)+ToroidalField.constantField(-1,self.omega.nfp,self.omega.mpol,self.omega.ntor)
        g_thetatheta = (
            self.dRdTheta*self.dRdTheta + 
            self.r*self.r*dPhidTheta*dPhidTheta +
            self.dZdTheta*self.dZdTheta
        )
        g_thetazeta = (
            self.dRdTheta*self.dRdZeta + 
            self.r*self.r*dPhidTheta*dPhidZeta +
            self.dZdTheta*self.dZdZeta
        )
        g_zetazeta = (
            self.dRdZeta*self.dRdZeta + 
            self.r*self.r*dPhidZeta*dPhidZeta +
            self.dZdZeta*self.dZdZeta
        )
        return g_thetatheta, g_thetazeta, g_zetazeta


if __name__ == "__main__":
    pass
