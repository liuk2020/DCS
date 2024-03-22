#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# surfaceCylindricalAngle.py


from .surface import Surface
from ..toroidalField import ToroidalField 


class Surface_cylindricalAngle(Surface):

    def __init__(self, r: ToroidalField, z: ToroidalField, reverseToroidalAngle: bool=True) -> None:
        super().__init__(r, z)
        self.reverseToroidalAngle = reverseToroidalAngle 

    @property
    def metric(self):
        g_thetatheta = self.dRdTheta*self.dRdTheta + self.dZdTheta*self.dZdTheta
        g_thetazeta = self.dRdTheta*self.dRdZeta + self.dZdTheta*self.dZdZeta
        g_zetazeta = self.dRdZeta*self.dRdZeta + self.r*self.r + self.dZdZeta*self.dZdZeta
        return g_thetatheta, g_thetazeta, g_zetazeta 


if __name__ == "__main__":
    pass
