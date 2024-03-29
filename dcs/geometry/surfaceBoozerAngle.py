#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# surfaceBoozerAngle.py


import numpy as np 
import matplotlib.pyplot as plt 
from .surface import Surface
from .surfaceCylindricalAngle import Surface_cylindricalAngle
from ..toroidalField import ToroidalField
from ..toroidalField import derivatePol, derivateTor
from ..toroidalField import changeResolution, fftToroidalField 
from ..misc import print_progress
from scipy.optimize import fixed_point
from scipy.integrate import dblquad


class Surface_BoozerAngle(Surface):
    r"""
    The magnetic surface in Boozer coordinates $(\theta, \zeta)$.  
        $$ \phi = -\zeta + \omega(\theta,\zete) $$ 
    There is a mapping with coordinates corrdinates $(R, \phi, Z)$
        $$ R = R(\theta, \zeta) $$ 
        $$ \phi = -\zeta + \omega(\theta,\zete) $$ 
        $$ Z = Z(\theta, \zeta) $$ 
    """

    def __init__(self, r: ToroidalField, z: ToroidalField, omega: ToroidalField) -> None:
        super().__init__(r, z)
        self.omega = omega

    def changeResolution(self, mpol: int, ntor: int): 
        self.mpol = mpol
        self.ntor = ntor
        self.r = changeResolution(self.r, mpol, ntor)
        self.z = changeResolution(self.z, mpol, ntor)
        self.omega = changeResolution(self.omega, mpol, ntor)

    @property
    def metric(self): 
        dPhidTheta = derivatePol(self.omega)
        dPhidZeta = derivateTor(self.omega)+ToroidalField.constantField(-1,self.omega.nfp,self.omega.mpol,self.omega.ntor)
        g_thetatheta = (
            self.dRdTheta*self.dRdTheta + 
            self.r*self.r*dPhidTheta*dPhidTheta +
            self.dZdTheta*self.dZdTheta
        )
        g_thetazeta = (
            self.dRdTheta*self.dRdZeta + 
            self.r*self.r*dPhidTheta*dPhidZeta +
            self.dZdTheta*self.dZdZeta
        )
        g_zetazeta = (
            self.dRdZeta*self.dRdZeta + 
            self.r*self.r*dPhidZeta*dPhidZeta +
            self.dZdZeta*self.dZdZeta
        )
        return g_thetatheta, g_thetazeta, g_zetazeta


    def toCylinder(self, method: str="DFT", **kwargs) -> Surface_cylindricalAngle:
        if method == "DFT":
            return self.toCylinder_dft(**kwargs)
        else:
            return self.toCylinder_integrate()

    def toCylinder_dft(self, mpol: int=12, ntor: int=12, xtol: float=1e-13) -> Surface_cylindricalAngle:
        
        deltaTheta = 2*np.pi / (2*mpol+1) 
        deltaPhi = 2*np.pi / self.nfp / (2*ntor+1) 
        sampleTheta, samplePhi = np.arange(2*mpol+1)*deltaTheta, -np.arange(2*ntor+1)*deltaPhi 
        gridPhi, gridTheta = np.meshgrid(samplePhi, sampleTheta) 
        
        # Find fixed point of zeta. 
        def zetaValue(zeta, theta, phi):
            return (
                self.omega.getValue(float(theta), float(zeta)) - phi
            )
        gridZeta = np.zeros_like(gridPhi)
        for i in range(len(gridZeta)): 
            for j in range(len(gridZeta[0])): 
                gridZeta[i,j] = float(
                    fixed_point(zetaValue, gridZeta[i,j], args=(gridTheta[i,j], gridPhi[i,j]), xtol=xtol)
                )
        
        sampleR = self.r.getValue(gridTheta, gridZeta)
        sampleZ = self.z.getValue(gridTheta, gridZeta)
        _fieldR = fftToroidalField(sampleR, nfp=self.nfp) 
        _fieldZ = fftToroidalField(sampleZ, nfp=self.nfp)
        return Surface_cylindricalAngle(
            _fieldR, 
            _fieldZ, 
            reverseToroidalAngle = False
        )

    # TODO: An untested function! 
    def toCylinder_integrate(self) -> Surface_cylindricalAngle:

        def substitutionFactor(theta, zeta) -> float:
            return float(
                derivateTor(self.omega).getValue(theta, zeta) - 1 
            )
        
        def helicalAngle(theta, zeta, m, n, _m, _n):
            return (
                m*theta - n*self.nfp*zeta - _m*theta + _n*self.nfp*(-zeta+self.omega.getValue(theta, zeta))
            )

        def rRe(zeta, theta, m, n, _m, _n): 
            factor = substitutionFactor(theta, zeta)
            _angle = helicalAngle(theta, zeta, m, n, _m, _n)
            return ((
                self.r.getRe(m,n) * np.cos(_angle) - 
                self.r.getIm(m,n) * np.sin(_angle)
            )*factor).flatten() 

        def rIm(zeta, theta, m, n, _m, _n): 
            factor = substitutionFactor(theta, zeta)
            _angle = helicalAngle(theta, zeta, m, n, _m, _n)
            return ((
                self.r.getRe(m,n) * np.sin(_angle) + 
                self.r.getIm(m,n) * np.cos(_angle)
            )*factor).flatten() 

        def zRe(zeta, theta, m, n, _m, _n): 
            factor = substitutionFactor(theta, zeta)
            _angle = helicalAngle(theta, zeta, m, n, _m, _n)
            return ((
                self.z.getRe(m,n) * np.cos(_angle) - 
                self.z.getIm(m,n) * np.sin(_angle)
            )*factor).flatten() 

        def zIm(zeta, theta, m, n, _m, _n): 
            factor = substitutionFactor(theta, zeta)
            _angle = helicalAngle(theta, zeta, m, n, _m, _n)
            return ((
                self.z.getRe(m,n) * np.sin(_angle) + 
                self.z.getIm(m,n) * np.cos(_angle)
            )*factor).flatten() 

        print("Transformation from Boozer to Cylindrical Coordinates... ")

        print("Compute the R component: ")
        rReArr = np.zeros((2*self.ntor+1)*self.mpol+self.ntor+1)
        rImArr = np.zeros((2*self.ntor+1)*self.mpol+self.ntor+1)
        for index in range((2*self.ntor+1)*self.mpol+self.ntor+1): 
            _m, _n = self.r.indexReverseMap(index)
            for _i ,m in enumerate(range(-self.mpol, self.mpol+1)):
                for _j, n in enumerate(range(-self.ntor, self.ntor+1)): 
                    rReArr[index] += dblquad(
                        rRe, 
                        0, 2*np.pi, 
                        0, 2*np.pi, 
                        args = (m, n, _m, _n)
                    )[0] * (1/4/np.pi/np.pi)
                    rImArr[index] += dblquad(
                        rIm, 
                        0, 2*np.pi, 
                        0, 2*np.pi, 
                        args = (m, n, _m, _n)
                    )[0] * (1/4/np.pi/np.pi)
                    print_progress(
                        index*((2*self.mpol+1)*(2*self.ntor+1)) + 
                        _i * (2*self.ntor+1) + _j+1, 
                        ((2*self.ntor+1)*self.mpol+self.ntor+1) * ((2*self.mpol+1)*(2*self.ntor+1))
                    ) 

        print("Compute the Z component: ")
        zReArr = np.zeros((2*self.ntor+1)*self.mpol+self.ntor+1)
        zImArr = np.zeros((2*self.ntor+1)*self.mpol+self.ntor+1)
        for index in range((2*self.ntor+1)*self.mpol+self.ntor+1): 
            _m, _n = self.r.indexReverseMap(index)
            for _i ,m in enumerate(range(-self.mpol, self.mpol+1)):
                for _j, n in enumerate(range(-self.ntor, self.ntor+1)): 
                    zReArr[index] += dblquad(
                        zRe, 
                        0, 2*np.pi, 
                        0, 2*np.pi, 
                        args = (m, n, _m, _n)
                    )[0] * (1/4/np.pi/np.pi)
                    zImArr[index] += dblquad(
                        zIm, 
                        0, 2*np.pi, 
                        0, 2*np.pi, 
                        args = (m, n, _m, _n)
                    )[0] * (1/4/np.pi/np.pi)
                    print_progress(
                        index*((2*self.mpol+1)*(2*self.ntor+1)) + 
                        _i * (2*self.ntor+1) + _j+1, 
                        ((2*self.ntor+1)*self.mpol+self.ntor+1) * ((2*self.mpol+1)*(2*self.ntor+1))
                    )

        _rField = ToroidalField(
            nfp = self.nfp, 
            mpol = self.mpol, 
            ntor = self.ntor,
            reArr = rReArr,
            imArr = rImArr
        )
        _zField = ToroidalField(
            nfp = self.nfp, 
            mpol = self.mpol, 
            ntor = self.ntor,
            reArr = zReArr,
            imArr = zImArr
        )
        return Surface_cylindricalAngle(
            _rField, 
            _zField, 
            reverseToroidalAngle = False
        )

    def plot_plt(self, ntheta: int=360, nzeta: int=360, fig=None, ax=None, **kwargs):
        if ax is None: 
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        plt.sca(ax) 
        thetaArr = np.linspace(0, 2*np.pi, ntheta) 
        zetaArr = np.linspace(0, 2*np.pi, nzeta) 
        thetaGrid, zetaGrid = np.meshgrid(thetaArr, zetaArr) 
        rArr = self.r.getValue(thetaGrid, zetaGrid)
        zArr = self.z.getValue(thetaGrid, zetaGrid)
        omegaArr = self.omega.getValue(thetaGrid, zetaGrid)
        phiArr = - zetaGrid + omegaArr
        xArr = rArr * np.cos(phiArr)
        yArr = rArr * np.sin(phiArr)
        ax.plot_surface(xArr, yArr, zArr, color="coral") 
        plt.axis("equal")
        return fig
                

if __name__ == "__main__":
    pass
