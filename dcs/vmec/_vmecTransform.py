#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vmecTransform.py 


import numpy as np 
from .vmecOut import VMECOut
from ..toroidalField import ToroidalField
from ..toroidalField import changeResolution
from typing import Tuple


# TODO: This function was not tested. 
def vmec2straight(vmecData: VMECOut, surfaceIndex: int=-1) -> Tuple[ToroidalField]:
    r"""
    Return: 
        r, z, lambda, bSupU, bSupV, bField, Jacobian 
    $$ \theta = \vartheta + \lambda(\varthea, \varphi) $$ 
    where $\vartheta, \varphi$ are poloidal and toroidal angles in VMEC coordinates, $\theta$ is the poloidla angle in straight-field-line coordinates. 
    """
    nfp = int(vmecData.nfp) 
    mpol = int(vmecData.mpol) - 1 
    ntor = int(vmecData.ntor) 
    mpol_nyq = int(np.max(vmecData.xm_nyq))
    ntor_nyq = int(np.max(vmecData.xn_nyq/nfp))
    iota = vmecData.iotaf[surfaceIndex]
    rbc = vmecData.rmnc[surfaceIndex, :] 
    zbs = vmecData.zmns[surfaceIndex, :] 
    try: 
        rbs = vmecData.rmns[surfaceIndex, :] 
        zbc = vmecData.zmnc[surfaceIndex, :] 
    except:
        rbs = np.zeros_like(rbc) 
        zbc = np.zeros_like(zbs) 
    rbc[1:-1] = rbc[1:-1] / 2 
    zbs[1:-1] = zbs[1:-1] / 2 
    rbs[1:-1] = rbs[1:-1] / 2 
    zbs[1:-1] = zbs[1:-1] / 2 
    _rField = ToroidalField(
        nfp = nfp, 
        mpol = mpol, 
        ntor = ntor, 
        reArr = rbc, 
        imArr = -rbs 
    )
    _zField = ToroidalField(
        nfp = nfp, 
        mpol = mpol, 
        ntor = ntor, 
        reArr = zbc, 
        imArr = -zbs 
    )
    bSupUc = vmecData.bsupumnc[surfaceIndex, :] 
    try:
        bSupUs = vmecData.bsupumns[surfaceIndex, :]
    except:
        bSupUs = np.zeros_like(bSupUc)
    _bSupU = ToroidalField(
        nfp = nfp,
        mpol = mpol_nyq, 
        ntor = ntor_nyq,
        reArr = bSupUc,
        imArr = -bSupUs
    )
    bSupVc = vmecData.bsupvmnc[surfaceIndex, :] 
    try:
        bSupVs = vmecData.bsupvmns[surfaceIndex, :]
    except:
        bSupVs = np.zeros_like(bSupVc)
    _bSupV = ToroidalField(
        nfp = nfp,
        mpol = mpol_nyq, 
        ntor = ntor_nyq,
        reArr = bSupVc,
        imArr = -bSupVs
    )
    bc = vmecData.bmnc[surfaceIndex, :] 
    try:
        bs = vmecData.bmns[surfaceIndex, :] 
    except:
        bs = np.zeros_like(bc)
    _bField = ToroidalField(
        nfp = nfp,
        mpol = mpol_nyq, 
        ntor = ntor_nyq,
        reArr = bc,
        imArr = -bs
    )
    gc = vmecData.gmnc[surfaceIndex, :] 
    try:
        gs = vmecData.gmns[surfaceIndex, :]
    except:
        gs = np.zeros_like(gc)
    _Jacobian = ToroidalField(
        nfp = nfp,
        mpol = mpol_nyq, 
        ntor = ntor_nyq,
        reArr = gc,
        imArr = -gs
    )
    rField = changeResolution(_rField, mpol=2*mpol_nyq, ntor=2*ntor_nyq) 
    zField = changeResolution(_zField, mpol=2*mpol_nyq, ntor=2*ntor_nyq) 
    bSupU = changeResolution(_bSupU, mpol=2*mpol_nyq, ntor=2*ntor_nyq) 
    bSupV = changeResolution(_bSupV, mpol=2*mpol_nyq, ntor=2*ntor_nyq) 
    bField = changeResolution(_bField, mpol=2*mpol_nyq, ntor=2*ntor_nyq)
    Jacobian = changeResolution(_Jacobian, mpol=2*mpol_nyq, ntor=2*ntor_nyq)
    dLambdadU = Jacobian*bSupV - ToroidalField.constantField(1, nfp, 2*mpol_nyq, 2*ntor_nyq) 
    dLambdadV = ToroidalField.constantField(iota, nfp, 2*mpol_nyq, 2*ntor_nyq) - Jacobian*bSupU 
    lambdaField = ToroidalField.constantField(0, nfp, 2*mpol_nyq, 2*ntor_nyq)
    return rField, zField, lambdaField, bSupU, bSupV, bField, Jacobian 


if __name__ == "__main__":
    pass
