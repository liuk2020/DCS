#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# surfaceCylindricalAngle.py


import numpy as np 
import matplotlib.pyplot as plt 
from .surface import Surface
from ..toroidalField import ToroidalField 
from typing import Tuple, Dict


class Surface_cylindricalAngle(Surface):

    def __init__(self, r: ToroidalField, z: ToroidalField, reverseToroidalAngle: bool=True) -> None:
        super().__init__(r, z)
        self.reverseToroidalAngle = reverseToroidalAngle 

    @property
    def metric(self):
        g_thetatheta = self.dRdTheta*self.dRdTheta + self.dZdTheta*self.dZdTheta
        g_thetazeta = self.dRdTheta*self.dRdZeta + self.dZdTheta*self.dZdZeta
        g_zetazeta = self.dRdZeta*self.dRdZeta + self.r*self.r + self.dZdZeta*self.dZdZeta
        return g_thetatheta, g_thetazeta, g_zetazeta 

    def getRZ(self, thetaGrid: np.ndarray, zetaGrid: np.ndarray) -> Tuple[np.ndarray]: 
        if self.reverseToroidalAngle: 
            zetaGrid = -zetaGrid
        rArr = self.r.getValue(thetaGrid, zetaGrid)
        zArr = self.z.getValue(thetaGrid, zetaGrid)
        return rArr, zArr
    
    def getXYZ(self, thetaGrid: np.ndarray, zetaGrid: np.ndarray) -> Tuple[np.ndarray]: 
        if self.reverseToroidalAngle: 
            zetaGrid = -zetaGrid
        rArr = self.r.getValue(thetaGrid, zetaGrid)
        zArr = self.z.getValue(thetaGrid, zetaGrid)
        xArr = rArr * np.cos(zetaGrid)
        yArr = rArr * np.sin(zetaGrid)
        return xArr, yArr, zArr

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

    def toVacuumVMEC(self, fileName: str, params: Dict=dict()) -> None: 
        
        if "lasym" not in params.keys():
            params["lasym"] = True 
        if "phiedge" not in params.keys():
            params["phiedge"] = 1.0 
        if "delt" not in params.keys():
            params["delt"] = 0.75 
        if "niter" not in params.keys():
            params["niter"] = 20000 
        if "nstep" not in params.keys():
            params["nstep"] = 200 
        if "ns_array" not in params.keys():
            params["ns_array"] = [32, 64, 128, 256] 
        if "ftol_array" not in params.keys():
            params["ftol_array"] = [1e-8, 1e-10, 1e-12, 1e-14] 
        
        with open("input."+fileName, 'w+') as f:
            f.write("! This &INDATA namelist was generated by dcs.geometry.Surface_cylindricalAngle \n")
            f.write("&INDATA \n")
            f.write("   LFREEB = F \n") 
            f.write("!----- Grid Parameters ----- \n") 
            if params["lasym"]:
                f.write("   LASYM = T \n") 
            else:
                f.write("   LASYM = F \n") 
            f.write("   NFP = " + str(self.nfp) + " \n")
            f.write("   MPOL = " + str(self.mpol+1) + " \n")
            f.write("   NTOR = " + str(self.ntor) + " \n")
            f.write("   PHIEDGE = " + str(params["phiedge"]) + " \n")
            f.write("!----- Runtime Parameters ----- \n") 
            f.write("   DELT = " + str(params["delt"]) + " \n") 
            f.write("   NITER = " + str(params["niter"]) + " \n") 
            f.write("   NSTEP = " + str(params["nstep"]) + " \n") 
            f.write("   NS_ARRAY = " + str(params["ns_array"])[1:-1] + " \n") 
            f.write("   FTOL_ARRAY = " + str(params["ftol_array"])[1:-1] + " \n") 
            f.write("!----- Pressure Parameters ----- \n") 
            f.write("   AM = 0.0 0.0 \n")
            f.write("!----- Current/Iota Parameters ----- \n") 
            f.write("   NCURR = 1 \n") 
            f.write("   CURTOR = 0.0 \n") 
            f.write("   AC = 0.0 0.0 \n") 
            f.write("!----- Boundary Parameters ----- \n") 
            for i in range((2*self.ntor+1)*self.mpol+self.ntor+1): 
                m, n = self.r.indexReverseMap(i)
                if m==0 and n==0:
                    f.write("   RBC("+str(n)+","+str(m)+")="+"{:.14e}".format(self.r.getRe(m,n)))
                    f.write("   ZBS("+str(n)+","+str(m)+")="+"{:.14e}".format(-self.z.getIm(m,n)))
                    if params["lasym"]:
                        f.write(" \n") 
                    else:
                        f.write("   RBS("+str(n)+","+str(m)+")="+"{:.14e}".format(-self.r.getIm(m,n)))
                        f.write("   ZBC("+str(n)+","+str(m)+")="+"{:.14e}".format(self.z.getRe(m,n)) + " \n")
                else:
                    f.write("   RBC("+str(n)+","+str(m)+")="+"{:.14e}".format(2*self.r.getRe(m,n)))
                    f.write("   ZBS("+str(n)+","+str(m)+")="+"{:.14e}".format(-2*self.z.getIm(m,n)))
                    if params["lasym"]:
                        f.write(" \n") 
                    else:
                        f.write("   RBS("+str(n)+","+str(m)+")="+"{:.14e}".format(-2*self.r.getIm(m,n)))
                        f.write("   ZBC("+str(n)+","+str(m)+")="+"{:.14e}".format(2*self.z.getRe(m,n)) + " \n")
            f.write("/ \n")


if __name__ == "__main__":
    pass