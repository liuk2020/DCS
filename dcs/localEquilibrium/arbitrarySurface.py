#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# arbitrarySurface.py


import numpy as np 
from .straightSurface import StraightSurfaceField
from ..geometry import Surface_cylindricalAngle
from ..toroidalField import ToroidalField
from ..vmec import VMECOut
from ..toroidalField import derivatePol, derivateTor, changeResolution


class ArbitrarySurfaceField(StraightSurfaceField):
    r"""
    Field on a specified toroidal surface arbitrary coordinates. 
    $$ \theta = \vartheta + \lambda $$
    where $\theta$ is the straight-field-line poloidal angle, $\vartheta$ is the arbitrary poloidal angle.  
    """

    def __init__(self, surf: Surface_cylindricalAngle, lam: ToroidalField, iota: float) -> None:
        assert surf.r.nfp == lam.nfp
        self.iota = iota 
        self.nfp = surf.r.nfp
        self.mpol = 2*surf.r.mpol + lam.mpol 
        self.ntor = 2*surf.r.ntor + lam.ntor
        self.ininSurfLambda(surf, lam)
        self.g_thetatheta, self.g_thetazeta, self.g_zetazeta = self.surf.metric 
        self.P = (
            (ToroidalField.constantField(self.iota, self.nfp, self.mpol, self.ntor) - derivateTor(self.lambdaField)) * self.g_thetazeta +
            ToroidalField.constantField(1, self.nfp, self.mpol, self.ntor) + derivatePol(self.lambdaField) * self.g_zetazeta
        )
        self.Q = (
            (ToroidalField.constantField(self.iota, self.nfp, self.mpol, self.ntor) - derivateTor(self.lambdaField)) * self.g_thetatheta +
            ToroidalField.constantField(1, self.nfp, self.mpol, self.ntor) + derivatePol(self.lambdaField) * self.g_thetazeta
        )
        self.D = derivatePol(self.P) - derivateTor(self.Q)

    def ininSurfLambda(self, surf: Surface_cylindricalAngle, lam: ToroidalField) -> None: 
        """
        Change the resolution of the surface and lambda field! 
        """
        _r = changeResolution(surf.r, self.mpol, self.ntor)
        _z = changeResolution(surf.z, self.mpol, self.ntor)
        self.surf = Surface_cylindricalAngle(_r, _z)
        self.lambdaField = changeResolution(lam, self.mpol, self.ntor)

    @classmethod
    def readVMEC(cls, vmecfile: str, surfaceIndex: int=-1, ssym: bool=True):
        vmecData = VMECOut(vmecfile,ssym=ssym)
        iota = vmecData.iotaf[surfaceIndex]
        surf, lam = vmecData.getSurface()
        return cls(surf, lam, iota)


if __name__ == "__main__":
    pass
