#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# straightSurface.py


import numpy as np 
from .solver import Solver
from .anotherSolver import AnotherSolver
from ..geometry import Surface_cylindricalAngle
from ..toroidalField import ToroidalField
from ..toroidalField import derivatePol, derivateTor, changeResolution


class StraightSurfaceField:
    """
    Field on a specified toroidal surface in straight-field-line coordinates. 
    """

    def __init__(self, surf: Surface_cylindricalAngle, iota: float) -> None:
        self.initSurf(surf)
        self.iota = iota
        self.nfp = self.surf.r.nfp
        self.mpol = self.surf.r.mpol
        self.ntor = self.surf.r.ntor
        self.g_thetatheta, self.g_thetazeta, self.g_zetazeta = self.surf.metric
        self.P = self.g_thetazeta*iota + self.g_zetazeta
        self.Q = self.g_thetatheta*iota + self.g_thetazeta

    def initSurf(self, surf: Surface_cylindricalAngle) -> None:
        """
        Change the resolution of the surface! 
        """
        _r = changeResolution(surf.r, 2*surf.r.mpol, 2*surf.r.ntor)
        _z = changeResolution(surf.z, 2*surf.z.mpol, 2*surf.z.ntor)
        self.surf = Surface_cylindricalAngle(_r, _z)

    def solveJacobian(self, method: str="linear") -> None:
        equation = AnotherSolver(pField=self.P, qField=self.Q)
        if method == "linear":
            self._Jacobian = equation.solve_linearEquation()

    def solveJacobianReciprocal(self, method: str="linear") -> None:
        equation = Solver(pField=self.P, qField=self.Q)
        if method == "linear":
            self._JacobianReciprocal = equation.solve_linearEquation()

    @property
    def JacobianReciprocal(self):
        r"""
        Retunrn:
            $\frac{1}{\sqrt{g}}$
        """
        try:
            return self._JacobianReciprocal
        except AttributeError:
            self.solveJacobianReciprocal() 
            return self._JacobianReciprocal
        except:
            print("An error occurred while calculating the reciprocal of Jacobian... ")

    @property
    def Jacobian(self):
        r"""
        Retunrn:
            $\sqrt{g}$
        """
        try:
            return self._Jacobian
        except AttributeError:
            self.solveJacobian() 
            return self._Jacobian
        except:
            print("An error occurred while calculating the Jacobian... ")

    def getB(self, thetaArr: np.ndarray, zetaArr: np.ndarray) -> np.ndarray:
        JacobianReciprocalGrid = self.JacobianReciprocal.getValue(thetaArr, zetaArr)
        g_thetathetaGrid = self.g_thetatheta.getValue(thetaArr, zetaArr)
        g_thetazetaGrid = self.g_thetazeta.getValue(thetaArr, zetaArr)
        g_zetazetaGrid = self.g_zetazeta.getValue(thetaArr, zetaArr)
        B2Grid = (
            np.power(JacobianReciprocalGrid, 2) *
            (g_zetazetaGrid + 2*self.iota*g_thetazetaGrid + self.iota*self.iota*g_thetathetaGrid)
        )
        return np.power(B2Grid, 0.5)

    def plotB(self, ntheta: int=360, nzeta: int=360, ax=None, fig=None, onePeriod: bool=True, **kwargs):
        from matplotlib import cm
        import matplotlib.pyplot as plt 
        thetaArr = np.linspace(0, 2*np.pi, ntheta)
        thetaValue =  np.linspace(0, 2*np.pi, 3)
        if onePeriod:
            zetaArr = np.linspace(0, 2*np.pi/self.nfp, nzeta)
            zetaValue =  np.linspace(0, 2*np.pi/self.nfp, 3)
        else:
            zetaArr = np.linspace(0, 2*np.pi, nzeta) 
            zetaValue =  np.linspace(0, 2*np.pi, 3)
        if ax is None: 
            fig, ax = plt.subplots() 
        plt.sca(ax) 
        thetaGrid, zetaGrid = np.meshgrid(thetaArr, zetaArr) 
        valueGrid = self.getB(thetaGrid, zetaGrid) 
        ctrig = ax.contourf(zetaGrid, thetaGrid, valueGrid, cmap=cm.rainbow)
        colorbar = fig.colorbar(ctrig)
        colorbar.ax.tick_params(labelsize=18)
        if kwargs.get("toroidalLabel") == None:
            kwargs.update({"toroidalLabel": r"$\zeta$"})
        if kwargs.get("poloidalLabel") == None:
            kwargs.update({"poloidalLabel": r"$\theta$"})
        ax.set_xlabel(kwargs.get("toroidalLabel"), fontsize=18)
        ax.set_ylabel(kwargs.get("poloidalLabel"), fontsize=18)
        ax.set_xticks(zetaValue)
        if onePeriod and self.nfp!=1:
            ax.set_xticklabels(["$0$", r"$\pi/"+str(self.nfp)+"$", r"$2\pi/"+str(self.nfp)+"$"], fontsize=18) 
        else:
            ax.set_xticklabels(["$0$", r"$\pi$", r"$2\pi$"], fontsize=18) 
        ax.set_yticks(thetaValue)
        ax.set_yticklabels(["$0$", r"$\pi$", r"$2\pi$"], fontsize=18)
        return


if __name__ == "__main__": 
    pass
