#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vmecOut.py


import numpy as np 
from booz_xform import Booz_xform
from ..toroidalField import ToroidalField
from ..geometry import Surface_BoozerAngle


class BoozerForm(Booz_xform): 
    
    def __init__(self) -> None:
        super().__init__()

    def surface(self, surfaceIndex: int=-1, asym: bool=True, reverseToroidal: bool=False, reverseOmegaAngle:bool=True) -> Surface_BoozerAngle: 
        nfp = int(self.nfp)
        mpol = int(self.mboz) - 1
        ntor = int(self.nboz) 
        rbc = self.rmnc_b[:, surfaceIndex].copy()
        zbs = self.zmns_b[:, surfaceIndex].copy()
        if not asym: 
            rbs = self.rmns_b[:, surfaceIndex].copy()
            zbc = self.zmnc_b[:, surfaceIndex].copy()
        else: 
            rbs = np.zeros_like(rbc)
            zbc = np.zeros_like(zbs)
        rbc[1:-1] = rbc[1:-1] / 2
        rbs[1:-1] = rbs[1:-1] / 2
        zbc[1:-1] = zbc[1:-1] / 2
        zbs[1:-1] = zbs[1:-1] / 2
        _rfield = ToroidalField(
            nfp = nfp, 
            mpol = mpol, 
            ntor = ntor,
            reArr = rbc, 
            imArr = -rbs, 
            reIndex = True, 
            imIndex = not asym
        ) 
        _zfield = ToroidalField(
            nfp = nfp, 
            mpol = mpol, 
            ntor = ntor,
            reArr = zbc, 
            imArr = -zbs, 
            reIndex = not asym, 
            imIndex = True
        ) 
        nus = self.numns_b[:, surfaceIndex].copy()
        if not asym:
            nuc = self.numnc_b[:, surfaceIndex].copy()
        else: 
            nuc = np.zeros_like(nus)
        nuc[1:-1] = nuc[1:-1] / 2
        nus[1:-1] = nus[1:-1] / 2
        _omegafield = ToroidalField(
            nfp = nfp,
            mpol = mpol, 
            ntor = ntor,
            reArr = -nuc, 
            imArr = -nus, 
            reIndex = not asym, 
            imIndex = True
        )
        return Surface_BoozerAngle(_rfield, _zfield, _omegafield, reverseToroidalAngle=reverseToroidal, reverseOmegaAngle=reverseOmegaAngle) 

    def getIota(self, surfaceIndex: int=-1):
        return self.iota[surfaceIndex]


if __name__ == "__main__": 
    pass
