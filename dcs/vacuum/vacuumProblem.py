#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vacuumSurfaceQS.py


from ..toroidalField import ToroidalField
from ..geometry import Surface_BoozerAngle 
from typing import Tuple


class VacuumSurface(): 
    r"""
    The problem of directly constructing flux surface in vacuum. 
    """

    def __init__(self, 
        surf: Surface_BoozerAngle=None, 
        iota: float=0.618, 
        unfixIota: bool = False, 
        stemSym: bool = True
    ) -> None:
        if surf is None:
            self._init_surf()
        else: 
            self.surf = surf
            self.nfp = surf.r.nfp
            self.mpol = surf.r.mpol
            self.ntor = surf.r.ntor
        self._iota = iota
        self.unfixIota = unfixIota
        self.stemSym = stemSym
        self._initDOF()

    def _init_surf(self): 
        self.nfp = 3
        self.mpol = 2
        self.ntor = 2
        self.surf = Surface_BoozerAngle(
            r = ToroidalField.constantField(1, nfp=self.nfp, mpol=self.mpol, ntor=self.ntor), 
            z = ToroidalField.constantField(0, nfp=self.nfp, mpol=self.mpol, ntor=self.ntor), 
            omega = ToroidalField.constantField(0, nfp=self.nfp, mpol=self.mpol, ntor=self.ntor)
        )

    def _initDOF(self):
        length = (2*self.ntor+1)*self.mpol + self.ntor + 1
        self.dof = {}
        if self.stemSym:
            self.dofkeys = ["rc", "zs", "omegas"]
        else:
            self.dofkeys = ["rc", "zs", "omegas", "rs", "zc", "omegac"]
        for index in self.dofkeys:
            self.dof[index] = [False for i in range(length)] 

    @property
    def iota(self):
        return self._iota

    def changeIota(self, iota: float):
        self._iota = iota

    def changeResolution(self, mpol: int, ntor: int):
        length = (2*ntor+1)*mpol + ntor + 1
        _dof = {}
        for dof in self.dofkeys:
            _dof[dof] = [False for i in range(length)]
            for index, constrain in enumerate(self.dof[dof]): 
                m, n = self.indexReverseMap(index)
                if abs(m) <= mpol and abs(n) <= ntor: 
                    _dof[dof][ntor+(2*ntor+1)*(m-1)+(n+ntor+1)] = constrain 
        self.dof = _dof
        self.mpol, self.ntor = mpol, ntor 
        self.surf.changeResolution(mpol=mpol, ntor=ntor) 

    def changeNfp(self, nfp: int):
        self.nfp = nfp
        self.surf.r.nfp = nfp
        self.surf.z.nfp = nfp
        self.surf.omega.nfp = nfp
    
    @property
    def numsDOF(self): 
        nums = 0
        for key in self.dofkeys:
            for dof in self.dof[key]:
                nums = nums + (dof)
        return nums

    def fixDOF(self, dof: str, m: int=0, n: int=0): 
        if dof not in self.dof.keys():
            print("Incorrect Argument! ")
            return
        else:
            self.dof[dof][self.indexMap(m,n)] = False

    def unfixDOF(self, dof: str, m: int=0, n: int=0): 
        if dof not in self.dof.keys():
            print("Incorrect Argument! ")
            return
        else:
            self.dof[dof][self.indexMap(m,n)] = True

    def fixAll(self):
        for dof in self.dofkeys:
            for index, constain in enumerate(self.dof[dof]):
                self.dof[dof][index] = False

    def unfixAll(self):
        for dof in self.dofkeys:
            for index, constain in enumerate(self.dof[dof]):
                self.dof[dof][index] = True
        self.dof["omegas"][0] = False
        if not self.stemSym:
            self.dof["omegac"][0] = False 

    # TODO
    def toVMEC(self):
        pass

    def indexMap(self, m: int, n: int) -> int:
        assert abs(m) <= self.mpol and abs(n) <= self.ntor
        return self.ntor + (2*self.ntor+1)*(m-1) + (n+self.ntor+1)

    def indexReverseMap(self, index: int) -> Tuple[int]: 
        assert index < (self.mpol*(2*self.ntor+1)+self.ntor+1)
        if index <= self.ntor:
            return 0, index
        else:
            return (index-self.ntor-1)//(2*self.ntor+1)+1, (index-self.ntor-1)%(2*self.ntor+1)-self.ntor


if __name__ == "__main__": 
    pass
