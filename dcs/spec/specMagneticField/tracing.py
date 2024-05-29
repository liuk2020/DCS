#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tracing.py


import numpy as np 
from scipy.integrate import solve_ivp 
from .specField import SPECField
from .fieldLine import FieldLine 
from .readData import readB, readJacobian
from dcs.misc import print_progress
from typing import List


def traceLine(
    bField: SPECField, 
    s0: np.ndarray, theta0: np.ndarray, zeta0: np.ndarray, 
    niter: int=128, nstep: int=32,
    bMethod: str="calculate", 
    bData: str=None, jacobianData: str=None, 
    sResolution: int=128, thetaResolution: int=128, zetaResolution: int=128, 
    printControl: bool=True, writeControl: str=None, **kwargs
) -> List[FieldLine]:
    r"""
    Working in SPEC coordintes (s, \theta, \zeta), compute magnetic field lines by solving
        $$ \frac{ds}{d\zeta} = \frac{B^s}{B^\zeta} $$
        $$ \frac{d\theta}{d\zeta} = \frac{B^\theta}{B^\zeta} $$
    Args:
        bField: the toroidal magnetic field. 
        s0: list of s components of initial points. 
        theta0: list of theta components of initial points. 
        zeta0: list of zeta components of initial points. 
        niter: Number of toroidal periods. 
        nstep: Number of intermediate step for one period
        bMethod: should be `"calculate"` or "`interpolate`" , the method to get the magnetic field. 
    """
    if isinstance(s0, float):
        s0, theta0, zeta0 = np.array([s0]), np.array([theta0]), np.array([zeta0])
    elif isinstance(s0, list):
        s0, theta0, zeta0 = np.array(s0), np.array(theta0), np.array(zeta0)
    assert s0.shape == theta0.shape == zeta0.shape
    if kwargs.get("method") is None:
        kwargs.update({"method": "LSODA"}) 
    if kwargs.get("rtol") is None:
        kwargs.update({"rtol": 1e-10}) 
    
    print("Change the resolution of the field... ")
    bField.changeResolution(sResolution=sResolution, thetaResolution=thetaResolution, zetaResolution=zetaResolution)
    if bMethod == "calculate":
        from pyoculus.problems import SPECBfield
        pyoculusField = SPECBfield(bField.specData, bField.lvol+1)
        if jacobianData is None:
            base_Jacobian = bField.getJacobian()
            base_sArr = bField.sArr
            base_thetaArr = bField.thetaArr
            base_zetaArr = bField.zetaArr
        else:
            base_sArr, base_thetaArr, base_zetaArr, base_Jacobian = readJacobian(jacobianData)
    elif bMethod == "interpolate":
        if bData is None:
            base_bSupS, base_bSupTheta, base_bSupZeta = bField.getB()
            base_sArr = bField.sArr
            base_thetaArr = bField.thetaArr
            base_zetaArr = bField.zetaArr
        else:
            base_sArr, base_thetaArr, base_zetaArr, base_bSupS, base_bSupTheta, base_bSupZeta = readB(bData)
    else:
        raise ValueError(
            "`bMethod` should be `calculate` or `interpolate`. "
        )

    def getB_calculate(zeta, s_theta):
        # field = pyoculusField.B_many(s_theta[0], s_theta[1], zeta) / bField.interpValue(base_Jacobian, s_theta[0], s_theta[1], zeta, sArr=base_sArr, thetaArr=base_thetaArr, zetaArr=base_zetaArr)
        # bSupS = field[0, 0]
        # bSupTheta = field[0, 1]
        # bSupZeta = field[0, 2]
        field = pyoculusField.B([s_theta[0], s_theta[1], zeta]) / bField.interpValue(base_Jacobian, s_theta[0], s_theta[1], zeta, sArr=base_sArr, thetaArr=base_thetaArr, zetaArr=base_zetaArr)
        bSupS = field[0]
        bSupTheta = field[1]
        bSupZeta = field[2]
        return [bSupS/bSupZeta, bSupTheta/bSupZeta]
    
    def getB_interpolate(zeta, s_theta):
        bSupS = bField.interpValue(baseData=base_bSupS, sValue=s_theta[0], thetaValue=s_theta[1], zetaValue=zeta, sArr=base_sArr, thetaArr=base_thetaArr, zetaArr=base_zetaArr)
        bSupTheta = bField.interpValue(baseData=base_bSupTheta, sValue=s_theta[0], thetaValue=s_theta[1], zetaValue=zeta, sArr=base_sArr, thetaArr=base_thetaArr, zetaArr=base_zetaArr)
        bSupZeta = bField.interpValue(baseData=base_bSupZeta, sValue=s_theta[0], thetaValue=s_theta[1], zetaValue=zeta, sArr=base_sArr, thetaArr=base_thetaArr, zetaArr=base_zetaArr)
        return [bSupS/bSupZeta, bSupTheta/bSupZeta]
    
    lines = list()
    nLine = len(s0)
    if printControl:
        print("Begin field-line tracing: ")
    for i in range(nLine):              # loop over each field-line 
        s_theta = [s0[i], theta0[i]]
        zetaStart = zeta0[i]
        dZeta = 2 * np.pi / bField.nfp / nstep
        sArr = [s0[i]]
        thetaArr = [theta0[i]]
        zetaArr = [zeta0[i]]
        for j in range(niter):          # loop over each toroidal iteration
            if printControl:
                print_progress(i*niter+j+1, nLine*niter)
            for k in range(nstep):      # loop inside one iteration
                if bMethod == "calculate":
                    sol = solve_ivp(
                        getB_calculate, 
                        (zetaStart, zetaStart+dZeta), 
                        s_theta, **kwargs
                    )
                elif bMethod == "interpolate":
                    sol = solve_ivp(
                        getB_interpolate, 
                        (zetaStart, zetaStart+dZeta), 
                        s_theta, **kwargs
                    )
                else:
                    raise ValueError("bMethod should be calculate or interpolate. ")
                sArr.append(sol.y[0,-1])
                thetaArr.append(sol.y[1,-1])
                zetaArr.append(zetaStart+dZeta)
                s_theta = [sArr[-1], thetaArr[-1]]
                zetaStart = zetaArr[-1]
        lines.append(FieldLine.getLine_tracing(bField, nstep, np.array(sArr), np.array(thetaArr), np.array(zetaArr)))
        if writeControl:
            lines[-1].writeH5(writeControl+str(i)+".h5")
    
    return lines


def traceLine_byLength(
    bField: SPECField, 
    s0: np.ndarray, theta0: np.ndarray, zeta0: np.ndarray, 
    oneLength: float, 
    niter: int=128, nstep: int=32, 
    sResolution: int=128, thetaResolution: int=128, zetaResolution: int=128, 
    writeControl: str=None, **kwargs
) -> List[FieldLine]:
    r"""
    Working in SPEC coordintes (s, \theta, \zeta), compute magnetic field lines by solving
        $$ \frac{ds}{dl} = \frac{B^s}{B} $$
        $$ \frac{d\theta}{dl} = \frac{B^\theta}{B} $$ 
        $$ \frac{d\zeta}{dl} = \frac{B^\zeta}{B} $$
    Args:
        bField: the toroidal magnetic field. 
        s0: list of s components of initial points. 
        theta0: list of theta components of initial points. 
        zeta0: list of zeta components of initial points. 
        niter: Number of toroidal periods. 
        nstep: Number of intermediate step for one period
    """

    if isinstance(s0, float):
        s0, theta0, zeta0 = np.array([s0]), np.array([theta0]), np.array([zeta0])
    elif isinstance(s0, list):
        s0, theta0, zeta0 = np.array(s0), np.array(theta0), np.array(zeta0)
    assert s0.shape == theta0.shape == zeta0.shape
    if kwargs.get("method") is None:
        kwargs.update({"method": "LSODA"}) 
    if kwargs.get("rtol") is None:
        kwargs.update({"rtol": 1e-10}) 
    print("Change the resolution of the field... ")
    bField.changeResolution(sResolution=sResolution, thetaResolution=thetaResolution, zetaResolution=zetaResolution)
    print("Get the Jacobian and metric of the field... ")
    # baseJacobian = bField.getJacobian()
    # baseMetric = bField.getMetric()
    Rarr0, Zarr0, baseJacobian, baseMetric = bField.specData.get_grid_and_jacobian_and_metric(
        lvol = bField.lvol, 
        sarr = bField.sArr, 
        tarr = bField.thetaArr,
        zarr = bField.zetaArr
    )

    from pyoculus.problems import SPECBfield
    pyoculusField = SPECBfield(bField.specData, bField.lvol+1)
    def getB(dLength, point):
        field = pyoculusField.B([point[0], point[1], point[2]]) / bField.interpValue(baseJacobian, point[0], point[1], point[2])
        metric = bField.interpValue(baseMetric, point[0], point[1], point[2])
        # bSupS = field[0]
        # bSupTheta = field[1]
        # bSupZeta = field[2]
        bPow = 0
        for i in range(3):
            for j in range(3): 
                bPow += (field[i]*field[j]*metric[0,i,j]) 
        b = np.power(bPow, 0.5)
        return [field[0]/b, field[1]/b, field[2]/b]

    print("Begin field line tracing: ")
    lines = list()
    for lineIndex in range(len(s0)):           # loop over each field line
        point = [s0[lineIndex], theta0[lineIndex], zeta0[lineIndex]]
        initLength = 0
        deltaLength = oneLength / nstep
        sArr = [s0[lineIndex]]
        thetaArr = [theta0[lineIndex]]
        zetaArr = [zeta0[lineIndex]]
        for j in range(niter):              # loop over each toroidal iteration
            print_progress(lineIndex*niter+j+1, len(s0)*niter)
            for k in range(nstep):          # loop inside one iteration
                sol = solve_ivp(getB, (initLength, initLength+deltaLength), point, **kwargs)            # solve ODEs
                sArr.append(sol.y[0,-1])
                thetaArr.append(sol.y[1,-1])
                zetaArr.append(sol.y[2,-1])
                point = [sArr[-1], thetaArr[-1], zetaArr[-1]]
                initLength += deltaLength
        lines.append(FieldLine.getLine_tracing(bField, nstep, np.array(sArr), np.array(thetaArr), np.array(zetaArr), equalZeta=False))
        if writeControl:
            lines[-1].writeH5(writeControl+str(lineIndex)+".h5")
    
    return lines


if __name__ == "__main__":
    pass
