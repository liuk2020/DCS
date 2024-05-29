#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# _FieldLine.py


import h5py
import numpy as np
from .specField import SPECField
from typing import List


class FieldLine:
    """
    Field line in SPEC coordinates and cylindrical coordinates! 
    """

    def __init__(self, nfp: int, nZeta: int, sArr: np.ndarray, thetaArr: np.ndarray, zetaArr: np.ndarray, rArr: np.ndarray, zArr: np.ndarray, equalZeta: bool=True) -> None:
        self.nfp = nfp
        self.nZeta = nZeta
        self.sArr = sArr
        self.thetaArr = thetaArr
        self.zetaArr = zetaArr
        self.rArr = rArr
        self.zArr = zArr
        self.equalZeta = equalZeta

    @classmethod
    def getLine_tracing(cls, bField: SPECField, nZeta: int, sArr: np.ndarray, thetaArr: np.ndarray, zetaArr: np.ndarray, **kwargs):
        rGrid, r_s, r_theta, r_zeta, zGrid, z_s, z_theta, z_zeta = bField.getGrid()
        rArr = bField.interpValue(baseData=rGrid, sValue=sArr, thetaValue=thetaArr, zetaValue=zetaArr)
        zArr = bField.interpValue(baseData=zGrid, sValue=sArr, thetaValue=thetaArr, zetaValue=zetaArr)
        return cls(
            nfp = bField.nfp, 
            nZeta = nZeta,
            sArr = sArr,
            thetaArr = thetaArr,
            zetaArr = zetaArr,
            rArr = rArr,
            zArr = zArr,
            **kwargs
        )

    @classmethod
    def readH5(cls, h5File: str, **kwargs):
        with h5py.File(h5File, 'r') as f:
            nfp = int(f["grid"][0])
            nZeta = int(f["grid"][1])
            sArr = f["sArr"][:]
            thetaArr = f["thetaArr"][:]
            zetaArr = f["zetaArr"][:]
            rArr = f["rArr"][:]
            zArr = f["zArr"][:]
        return cls(
            nfp = nfp,
            nZeta = nZeta,
            sArr = sArr,
            thetaArr = thetaArr,
            zetaArr = zetaArr,
            rArr = rArr,
            zArr = zArr, 
            **kwargs
        )

    def writeH5(self, h5File: str) -> None:
        with h5py.File(h5File, 'w') as f:
            f.create_dataset("grid", data=np.array([self.nfp, self.nZeta]))
            f.create_dataset("sArr", data=self.sArr)
            f.create_dataset("thetaArr", data=self.thetaArr) 
            f.create_dataset("zetaArr", data=self.zetaArr)
            f.create_dataset("rArr", data=self.rArr)
            f.create_dataset("zArr", data=self.zArr)



if __name__ == "__main__":
    pass
