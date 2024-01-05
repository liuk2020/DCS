#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# surface.py


import numpy as np 
import matplotlib.pyplot as plt 
from ..toroidalField import ToroidalField 
from ..toroidalField import derivatePol, derivateTor 
from ..toroidalField import changeResolution 


class Surface:

    def __init__(self, r: ToroidalField, z: ToroidalField) -> None:
        self.r = r
        self.z = z

    @property
    def dRdTheta(self) -> ToroidalField:
        return derivatePol(self.r)

    @property
    def dRdZeta(self) -> ToroidalField:
        return derivateTor(self.r)

    @property
    def dZdTheta(self) -> ToroidalField:
        return derivatePol(self.z)

    @property
    def dZdZeta(self) -> ToroidalField:
        return derivateTor(self.z) 

    def changeResolution(self, mpol: int, ntor: int): 
        self.r = changeResolution(self.r, mpol, ntor)
        self.z = changeResolution(self.z, mpol, ntor)


class Surface_cylindricalAngle(Surface):

    def __init__(self, r: ToroidalField, z: ToroidalField, reverseToroidalAngle: bool=True) -> None:
        super().__init__(r, z)
        self.reverseToroidalAngle = reverseToroidalAngle 

    def plot_plt(self, ntheta: int=360, nzeta: int=360, fig=None, ax=None, **kwargs): 
        if ax is None: 
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        plt.sca(ax) 
        thetaArr = np.linspace(0, 2*np.pi, ntheta) 
        zetaArr = np.linspace(0, 2*np.pi, nzeta) 
        if self.reverseToroidalAngle:
            zetaArr = -zetaArr
        thetaGrid, zetaGrid = np.meshgrid(thetaArr, zetaArr) 
        rArr = self.r.getValue(thetaGrid, zetaGrid)
        zArr = self.z.getValue(thetaGrid, zetaGrid)
        xArr = rArr * np.cos(zetaGrid)
        yArr = rArr * np.sin(zetaGrid)
        ax.plot_surface(xArr, yArr, zArr, color="coral") 
        plt.axis("equal")
        return fig


    @property
    def metric(self):
        g_thetatheta = self.dRdTheta*self.dRdTheta + self.dZdTheta*self.dZdTheta
        g_thetazeta = self.dRdTheta*self.dRdZeta + self.dZdTheta*self.dZdZeta
        g_zetazeta = self.dRdZeta*self.dRdZeta + self.r*self.r + self.dZdZeta*self.dZdZeta
        return g_thetatheta, g_thetazeta, g_zetazeta


if __name__ == "__main__":
    pass
