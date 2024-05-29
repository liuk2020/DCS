#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# readData.py


import h5py


def readGrid(h5File: str):
    """
    return:
        sArr, thetaArr, zetaArr, rGrid, r_s, r_theta, r_zeta, zGrid, z_s, z_theta, z_zeta
    """
    with h5py.File(h5File, 'r') as f:
        sArr = f["sArr"][:]
        thetaArr = f["thetaArr"][:]
        zetaArr = f["zetaArr"][:]
        rGrid = f["rGrid"][:]
        r_s = f["r_s"][:]
        r_theta = f["r_theta"][:]
        r_zeta = f["r_zeta"][:]
        zGrid = f["zGrid"][:]
        z_s = f["z_s"][:]
        z_theta = f["z_theta"][:]
        z_zeta = f["z_zeta"][:]
    return sArr, thetaArr, zetaArr, rGrid, r_s, r_theta, r_zeta, zGrid, z_s, z_theta, z_zeta

def readB(h5File: str):
    """
    return:
        sArr, thetaArr, zetaArr, bSupS, bSupTheta, bSupZeta
    """
    with h5py.File(h5File, 'r') as f:
        sArr = f["sArr"][:]
        thetaArr = f["thetaArr"][:]
        zetaArr = f["zetaArr"][:]
        bSupS = f["bSupS"][:]
        bSupTheta = f["bSupTheta"][:]
        bSupZeta = f["bSupZeta"][:]
    return sArr, thetaArr, zetaArr, bSupS, bSupTheta, bSupZeta

def readJacobian(h5File: str):
    with h5py.File(h5File, 'r') as f:
        sArr = f["sArr"][:]
        thetaArr = f["thetaArr"][:]
        zetaArr = f["zetaArr"][:]
        jacobian = f["jacobian"][:]
    return sArr, thetaArr, zetaArr, jacobian

def readMetric(h5File: str):
    with h5py.File(h5File, 'r') as f:
        sArr = f["sArr"][:]
        thetaArr = f["thetaArr"][:]
        zetaArr = f["zetaArr"][:]
        metric = f["metric"][:]
    return sArr, thetaArr, zetaArr, metric

if __name__ == "__main__":
    pass
