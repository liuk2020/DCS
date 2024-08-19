#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# isolatedsurf.py

import numpy as np
from tfpy.toroidalField import ToroidalField
from .baseproblem import SurfProblem


class IsolatedSurface(SurfProblem):

    def __init__(self, r: ToroidalField = None, z: ToroidalField = None, omega: ToroidalField = None, mpol: int = None, ntor: int = None, nfp: int = None, iota: float = None, fixIota: bool = False, reverseToroidalAngle: bool = True, reverseOmegaAngle: bool = False) -> None:
        super().__init__(r, z, omega, mpol, ntor, nfp, iota, fixIota, reverseToroidalAngle, reverseOmegaAngle)

    def BoozerResidual(self) -> ToroidalField:
        guu, guv, _ = self.metric
        if not self.fixIota:
            self.updateIota(-guv.getRe(0,0)/guu.getRe(0,0))
        return guv + self.iota*guu

    def solve(self, **kwargs):
        self.niter = 0
        print("{:>8} {:>16} {:>16}".format('niter', 'iota', 'residual'))
        def cost(dofs):
            self.unpackDOF(dofs)
            residualField = self.BoozerResidual()
            return np.linalg.norm(np.hstack((residualField.reArr, residualField.imArr)))
        def callback(xi):
            self.niter += 1
            print("{:>8d} {:>16f} {:>16e}".format(self.niter, self.iota, cost(xi)))
        print("{:>8d} {:>16f} {:>16e}".format(0, self.iota, cost(self.initDOFs)))
        from scipy.optimize import minimize
        res = minimize(cost, self.initDOFs, callback=callback, **kwargs)
        print(res.message)


if __name__ == '__main__':
    pass