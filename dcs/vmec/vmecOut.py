#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vmecOut.py


import xarray
import numpy as np
from ..toroidalField import ToroidalField
from ..geometry import Surface_cylindricalAngle
from typing import Tuple


class VMECOut():
    
    def __init__(self, fileName: str, ssym: bool=True) -> None:
        _vmeclib = xarray.open_dataset(fileName)
        self.keys = [key for key in _vmeclib.data_vars] 
        for key in self.keys:
            setattr(self, key, _vmeclib[key].values)
        self.nfp = int(self.nfp) 
        self.ssym = ssym

    def getSurface(self, surfaceIndex: int=-1, reverseToroidal: bool=True) -> Tuple:
        """
        returns:
            surface, lambda
        """
        mpol = int(self.mpol) - 1 
        ntor = int(self.ntor) 
        rbc = self.rmnc[surfaceIndex, :] 
        zbs = self.zmns[surfaceIndex, :] 
        lams = self.lmns[surfaceIndex, :]
        if not self.ssym: 
            rbs = self.rmns[surfaceIndex, :] 
            zbc = self.zmnc[surfaceIndex, :] 
            lamc = self.lmnc[surfaceIndex, :]
        else:
            rbs = np.zeros_like(rbc) 
            zbc = np.zeros_like(zbs) 
            lamc = np.zeros_like(lams)
        rbc[1:-1] = rbc[1:-1] / 2 
        zbs[1:-1] = zbs[1:-1] / 2 
        rbs[1:-1] = rbs[1:-1] / 2 
        zbc[1:-1] = zbc[1:-1] / 2 
        lams[1:-1] = lams[1:-1] / 2
        lamc[1:-1] = lamc[1:-1] / 2
        _rField = ToroidalField(
            nfp = self.nfp, 
            mpol = mpol, 
            ntor = ntor, 
            reArr = rbc, 
            imArr = -rbs 
        )
        _zField = ToroidalField(
            nfp = self.nfp, 
            mpol = mpol, 
            ntor = ntor, 
            reArr = zbc, 
            imArr = -zbs 
        )
        lam = ToroidalField(
            nfp = self.nfp, 
            mpol = mpol, 
            ntor = ntor, 
            reArr = lamc, 
            imArr = -lams 
        )
        surf = Surface_cylindricalAngle(_rField, _zField, reverseToroidalAngle=reverseToroidal)
        return surf, lam

    def getJacobian(self, surfaceIndex: int=-1) -> ToroidalField:
        mpol_nyq = int(np.max(self.xm_nyq))
        ntor_nyq = int(np.max(self.xn_nyq/self.nfp))
        gc = self.gmnc[surfaceIndex, :] 
        if not self.ssym:
            gs = self.gmns[surfaceIndex, :]
        else:
            gs = np.zeros_like(gc)
        gc[1:-1] = gc[1:-1] / 2
        gs[1:-1] = gs[1:-1] / 2
        _Jacobian = ToroidalField(
            nfp = self.nfp,
            mpol = mpol_nyq, 
            ntor = ntor_nyq,
            reArr = gc,
            imArr = -gs
        )
        return _Jacobian
    
    def getBsup(self, surfaceIndex: int=-1) -> Tuple:
        """
        returns:
            bsupu, bsupv
        """
        mpol_nyq = int(np.max(self.xm_nyq))
        ntor_nyq = int(np.max(self.xn_nyq/self.nfp))
        bSupUc = self.bsupumnc[surfaceIndex, :] 
        bSupVc = self.bsupvmnc[surfaceIndex, :] 
        if not self.ssym:
            bSupUs = self.bsupumns[surfaceIndex, :]
            bSupVs = self.bsupvmns[surfaceIndex, :]
        else:
            bSupUs = np.zeros_like(bSupUc)
            bSupVs = np.zeros_like(bSupVc)
        bSupUc[1: -1] = bSupUc[1: -1] / 2
        bSupVc[1: -1] = bSupVc[1: -1] / 2
        bSupUs[1: -1] = bSupUs[1: -1] / 2
        bSupVs[1: -1] = bSupVs[1: -1] / 2
        _bSupU = ToroidalField(
            nfp = self.nfp,
            mpol = mpol_nyq, 
            ntor = ntor_nyq,
            reArr = bSupUc,
            imArr = -bSupUs
        )
        _bSupV = ToroidalField(
            nfp = self.nfp,
            mpol = mpol_nyq, 
            ntor = ntor_nyq,
            reArr = bSupVc,
            imArr = -bSupVs
        )
        return _bSupU, _bSupV

    def getB(self, surfaceIndex: int=-1) -> ToroidalField:
        """
        returns:
            B
        """
        mpol_nyq = int(np.max(self.xm_nyq))
        ntor_nyq = int(np.max(self.xn_nyq/self.nfp))
        bc = self.bmnc[surfaceIndex, :] 
        if not self.ssym:
            bs = self.bmns[surfaceIndex, :]
        else:
            bs = np.zeros_like(bc)
        bc[1:-1] = bc[1:-1] / 2
        bs[1:-1] = bs[1:-1] / 2
        _bField = ToroidalField(
            nfp = self.nfp,
            mpol = mpol_nyq, 
            ntor = ntor_nyq,
            reArr = bc,
            imArr = -bs
        )
        return _bField


if __name__ == "__main__":
    pass
