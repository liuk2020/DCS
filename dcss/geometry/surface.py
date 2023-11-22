#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# surface.py


from ..toroidalField import ToroidalField 
from ..toroidalField import derivatePol, derivateTor


class Surface:

    def __init__(self, r: ToroidalField, z: ToroidalField) -> None:
        self.r = r
        self.z = z

    @property
    def dRdTheta(self) -> ToroidalField:
        return derivatePol(self.r)

    @property
    def dRdZeta(self) -> ToroidalField:
        return derivateTor(self.r)

    @property
    def dZdTheta(self) -> ToroidalField:
        return derivatePol(self.z)

    @property
    def dZdZeta(self) -> ToroidalField:
        return derivateTor(self.z)


class Surface_cylindricalAngle(Surface):

    def __init__(self, r: ToroidalField, z: ToroidalField, reverseToroidalAngle: bool=True) -> None:
        super().__init__(r, z)
        self.reverseToroidalAngle = reverseToroidalAngle

    @property
    def metric(self):
        g_thetatheta = self.dRdTheta*self.dRdTheta + self.dZdTheta*self.dZdTheta
        g_thetaphi = self.dRdTheta*self.dRdZeta + self.dZdTheta*self.dZdZeta
        g_zetazeta = self.dRdZeta*self.dRdZeta + self.r*self.r + self.dZdZeta*self.dZdZeta
        return g_thetatheta, g_thetaphi, g_zetazeta


if __name__ == "__main__":
    pass
