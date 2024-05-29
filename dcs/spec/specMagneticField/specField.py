#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# _SPECField.py


import h5py
import numpy as np
from scipy.interpolate import interpn
from ..specOut import SPECOut


deltaS = 1e-10


class SPECField:
    """
    Magnetic field in SPEC coordinates! 
    """

    def __init__(self, specData: SPECOut, lvol: int=0,
    sResolution: int=2, thetaResolution: int=2, zetaResolution: int=2) -> None:
        """
        Args:
            specData: the `mpy.SPECOut` or `py_spec.SPECout` class 
            lvol: the number of the volume 
            sResolution: the resolution in the s direction 
            thetaResolution: the resolution in the poloidal direction
            zetaResolution: the resolution in the toroidal direction 
        """
        self.specData = specData
        self.lvol = lvol
        self.nfp = specData.input.physics.Nfp
        self.sArr = np.linspace(-1+deltaS, 1-deltaS, sResolution)
        self.thetaArr = np.linspace(0, 2*np.pi, thetaResolution)
        self.zetaArr = np.linspace(0, 2*np.pi/self.nfp, zetaResolution)

    @classmethod
    def read(cls, file: str): 
        return cls(
            specData=SPECOut(file), 
            lvol=0, 
            sResoltion=128, thetaResoltion=128, zetaResoltion=128
        )

    def interpValue(self, baseData: np.ndarray, sValue: float or np.ndarray, thetaValue: float or np.ndarray, zetaValue: float or np.ndarray, **kwargs):
        thetaValue = thetaValue %  (2*np.pi)
        zetaValue = zetaValue % (2*np.pi/self.nfp)
        if kwargs.get("sArr") is None:
            sArr = self.sArr
        else:
            sArr = kwargs.get("sArr")
        if kwargs.get("thetaArr") is None:
            thetaArr = self.thetaArr
        else:
            thetaArr = kwargs.get("thetaArr")
        if kwargs.get("zetaArr") is None:
            zetaArr = self.zetaArr
        else:
            zetaArr = kwargs.get("zetaArr")
        grid = (sArr, thetaArr, zetaArr)
        point = (sValue, thetaValue, zetaValue)
        return interpn(grid, baseData, point)

    def changeResolution(self, sResolution: int=2, thetaResolution: int=2,zetaResolution: int=2) -> None:
        self.sArr = np.linspace(-1+deltaS, 1-deltaS, sResolution)
        self.thetaArr = np.linspace(0, 2*np.pi, thetaResolution)
        self.zetaArr = np.linspace(0, 2*np.pi/self.nfp, zetaResolution)

    def getGrid(self, writeH5: str=None):
        """
        return:
            rGrid, r_s, r_theta, r_zeta, zGrid, z_s, z_theta, z_zeta
        """
        rGrid, r_s, r_theta, r_zeta, zGrid, z_s, z_theta, z_zeta = self.specData.get_RZ_derivatives(
            lvol = self.lvol, 
            sarr = self.sArr,
            tarr = self.thetaArr,
            zarr = self.zetaArr
        )
        if writeH5 is not None:
            with h5py.File(writeH5, 'w') as f:
                f.create_dataset("sArr", data=self.sArr)
                f.create_dataset("thetaArr", data=self.thetaArr)
                f.create_dataset("zetaArr", data=self.zetaArr)
                f.create_dataset("rGrid", data=rGrid)
                f.create_dataset("r_s", data=r_s)
                f.create_dataset("r_theta", data=r_theta)
                f.create_dataset("r_zeta", data=r_zeta)
                f.create_dataset("zGrid", data=zGrid)
                f.create_dataset("z_s", data=z_s)
                f.create_dataset("z_theta", data=z_theta)
                f.create_dataset("z_zeta", data=z_zeta)
        return rGrid, r_s, r_theta, r_zeta, zGrid, z_s, z_theta, z_zeta

    def getB(self, writeH5: str=None):
        """
        return:
            bSupS, bSupTheta, bSupZeta
        """
        field = self.specData.get_B(
            lvol = self.lvol, 
            sarr = self.sArr,
            tarr = self.thetaArr,
            zarr = self.zetaArr
        )
        bSupS = field[:,:,:,0]
        bSupTheta = field[:,:,:,1]
        bSupZeta = field[:,:,:,2]
        if writeH5 is not None:
            with h5py.File(writeH5, 'w') as f:
                f.create_dataset("sArr", data=self.sArr)
                f.create_dataset("thetaArr", data=self.thetaArr)
                f.create_dataset("zetaArr", data=self.zetaArr)
                f.create_dataset("bSupS", data=bSupS)
                f.create_dataset("bSupTheta", data=bSupTheta)
                f.create_dataset("bSupZeta", data=bSupZeta)
        return bSupS, bSupTheta, bSupZeta
    
    def getJacobian(self, writeH5: str=None):
        jacobian = self.specData.jacobian(
            lvol = self.lvol, 
            sarr = self.sArr,
            tarr = self.thetaArr,
            zarr = self.zetaArr
        )
        if writeH5 is not None:
            with h5py.File(writeH5, 'w') as f:
                f.create_dataset("sArr", data=self.sArr)
                f.create_dataset("thetaArr", data=self.thetaArr)
                f.create_dataset("zetaArr", data=self.zetaArr)
                f.create_dataset("jacobian", data=jacobian)
        return jacobian

    def getMetric(self, writeH5: str=None):
        metric = self.specData.metric(
            lvol = self.lvol, 
            sarr = self.sArr,
            tarr = self.thetaArr,
            zarr = self.zetaArr
        )
        if writeH5 is not None:
            with h5py.File(writeH5, 'w') as f:
                f.create_dataset("sArr", data=self.sArr)
                f.create_dataset("thetaArr", data=self.thetaArr)
                f.create_dataset("zetaArr", data=self.zetaArr)
                f.create_dataset("metric", data=metric)
        return metric


if __name__ == "__main__":
    pass
