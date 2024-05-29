#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vacuumSurfaceQS.py


import numpy as np
from ..toroidalField import ToroidalField
from ..geometry import Surface_BoozerAngle 
from typing import Tuple


class VacuumSurface(): 
    r"""
    The problem of directly constructing flux surface in vacuum. 
    """

    def __init__(self, 
        surf: Surface_BoozerAngle=None, 
        iota : float=0.618, 
        iotafree: bool=True, 
        stellSym: bool = True
    ) -> None:
        if surf is None:
            self._init_surf()
        else: 
            self.surf = surf
        self.iota = iota
        self.iotafree = iotafree
        self.setStellSym(stellSym)
        self._initDOF()
        self._init_paras()

    def _init_surf(self): 
        nfp, mpol, ntor = 3, 2, 2
        self.surf = Surface_BoozerAngle(
            r = ToroidalField.constantField(1, nfp=nfp, mpol=mpol, ntor=ntor), 
            z = ToroidalField.constantField(0, nfp=nfp, mpol=mpol, ntor=ntor), 
            omega = ToroidalField.constantField(0, nfp=nfp, mpol=mpol, ntor=ntor)
        )
        self.surf.r.reIndex, self.surf.r.imIndex = True, True 
        self.surf.z.reIndex, self.surf.z.imIndex = True, True 
        self.surf.omega.reIndex, self.surf.omega.imIndex = True, True
        self.surf.r.setRe(1, 0, 0.2)
        self.surf.z.setIm(1, 0, 0.2)

    def _init_paras(self): 
        self._powerIndex = 1.25

    def _initDOF(self):
        length = (2*self.ntor+1)*self.mpol + self.ntor + 1
        self._dofGeometry = {}
        for index in ["rc", "zs", "omegas", "rs", "zc", "omegac"]:
            self._dofGeometry[index] = [False for i in range(length)] 

    @property
    def nfp(self) -> int:
        return self.surf.nfp 

    @property
    def mpol(self) -> int:
        return self.surf.mpol 

    @property
    def ntor(self) -> int: 
        return self.surf.ntor 

    @property
    def stellSym(self) -> bool:
        return (not self.surf.r.imIndex) and (not self.surf.z.reIndex) and (not self.surf.omega.reIndex)

    @property
    def dofkeys(self):
        if self.stellSym:
            _dofkeys = ["rc", "zs", "omegas"]
        else:
            _dofkeys = ["rc", "zs", "omegas", "rs", "zc", "omegac"]
        return _dofkeys

    @property
    def dofGeometry(self):
        return self._dofGeometry 

    def setStellSym(self, stellSym: bool):
        if stellSym: 
            self.surf.r.imIndex, self.surf.z.reIndex, self.surf.omega.reIndex = False, False, False
        else:
            self.surf.r.imIndex, self.surf.z.reIndex, self.surf.omega.reIndex = True, True, True

    def setResolution(self, mpol: int, ntor: int):
        length = (2*ntor+1)*mpol + ntor + 1
        _dofGeometry = {}
        for dofName in ["rc", "zs", "omegas", "rs", "zc", "omegac"]:
            _dofGeometry[dofName] = [False for _i in range(length)] 
            for index, constrain in enumerate(self._dofGeometry[dofName]): 
                m, n = self.indexReverseMap(index)
                if abs(m) <= mpol and abs(n) <= ntor: 
                    _dofGeometry[dofName][ntor+(2*ntor+1)*(m-1)+(n+ntor+1)] = constrain 
        self._dofGeometry = _dofGeometry
        self.surf.changeResolution(mpol=mpol, ntor=ntor) 

    def setNfp(self, nfp: int):
        self.surf.r.nfp = nfp
        self.surf.z.nfp = nfp
        self.surf.omega.nfp = nfp

    def setIota(self, iota: float):
        self.iota = iota
    
    @property
    def numsDOF(self): 
        nums = 0
        for key in self.dofkeys:
            for dof in self.dofGeometry[key]:
                nums = nums + (dof)
        return nums

    def fixDOF(self, dof: str, m: int=0, n: int=0): 
        if dof not in self.dofkeys:
            print("Incorrect Argument! ")
            return
        else:
            self._dofGeometry[dof][self.indexMap(m,n)] = False

    def freeDOF(self, dof: str, m: int=0, n: int=0): 
        assert (m != 0) and (n != 0)
        if dof not in self.dofkeys:
            print("Incorrect Argument! ")
            return
        else:
            self._dofGeometry[dof][self.indexMap(m,n)] = True

    def fixAll_rc(self):
        for index, constrain in enumerate(self._dofGeometry['rc']): 
            self._dofGeometry['rc'][index] = False

    def fixAll_zs(self):
        for index, constrain in enumerate(self._dofGeometry['zs']): 
            self._dofGeometry['zs'][index] = False

    def fixAll_omegas(self):
        for index, constrain in enumerate(self._dofGeometry['omegas']): 
            self._dofGeometry['omegas'][index] = False

    def freeAll_rc(self):
        for index, constrain in enumerate(self._dofGeometry['rc']): 
            if index != 0:
                self._dofGeometry['rc'][index] = True
    
    def freeAll_zs(self):
        for index, constrain in enumerate(self._dofGeometry['zs']): 
            if index != 0:
                self._dofGeometry['zs'][index] = True

    def freeAll_omegas(self):
        for index, constrain in enumerate(self._dofGeometry['omegas']): 
            if index != 0:
                self._dofGeometry['omegas'][index] = True

    def fixAll(self):
        for dofName in ["rc", "zs", "omegas", "rs", "zc", "omegac"]:
            for index, constrain in enumerate(self._dofGeometry[dofName]): 
                self._dofGeometry[dofName][index] = False

    def freeAll(self):
        for dofName in ["rc", "zs", "omegas", "rs", "zc", "omegac"]:
            for index, constrain in enumerate(self._dofGeometry[dofName]): 
                if index != 0:
                    self._dofGeometry[dofName][index] = True

    def unpackDOF(self, dofValue: np.ndarray) -> None:
        assert dofValue.size == self.numsDOF+self.iotafree
        valueIndex = 0 
        while valueIndex < self.numsDOF:
            for dofkey in self.dofkeys:
                for dofIndex, dof in enumerate(self.dofGeometry[dofkey]): 
                    if dof:
                        m, n = self.indexReverseMap(dofIndex)
                        value = dofValue[valueIndex] / pow(self._powerIndex,abs(m)+abs(n))
                        if dofkey == "rc": 
                            self.surf.r.setRe(m, n, value)
                            valueIndex += 1
                            continue
                        elif dofkey == "zs": 
                            self.surf.z.setIm(m, n, value)
                            valueIndex += 1
                            continue
                        elif dofkey == "omegas":
                            self.surf.omega.setIm(m, n, value)
                            valueIndex += 1 
                            continue
                        elif dofkey == "rs": 
                            self.surf.r.setIm(m, n, value)
                            valueIndex += 1
                            continue
                        elif dofkey == "zc": 
                            self.surf.z.setRe(m, n, value)
                            valueIndex += 1
                            continue
                        elif dofkey == "omegac": 
                            self.surf.omega.setRe(m, n, value)
                            valueIndex += 1
                            continue
                        else: 
                            continue
                    else: 
                        continue
        if self.iotafree:
            self.setIota(dofValue[-1])
        return 

    @property
    def initValue_DOF(self) -> np.ndarray:
        initValue = np.zeros(self.numsDOF+self.iotafree) 
        valueIndex = 0
        while valueIndex < self.numsDOF:
            for dofkey in self.dofkeys:
                for dofIndex, dof in enumerate(self.dofGeometry[dofkey]): 
                    if dof:
                        m, n = self.indexReverseMap(dofIndex)
                        if dofkey == "rc": 
                            initValue[valueIndex] = self.surf.r.getRe(m,n) * pow(self._powerIndex,abs(m)+abs(n))
                            valueIndex += 1
                            continue
                        elif dofkey == "zs": 
                            initValue[valueIndex] = self.surf.z.getIm(m,n) * pow(self._powerIndex,abs(m)+abs(n))
                            valueIndex += 1
                            continue
                        elif dofkey == "omegas":
                            initValue[valueIndex] = self.surf.omega.getIm(m,n) * pow(self._powerIndex,abs(m)+abs(n))
                            valueIndex += 1 
                            continue
                        elif dofkey == "rs": 
                            initValue[valueIndex] = self.surf.r.getIm(m,n) * pow(self._powerIndex,abs(m)+abs(n))
                            valueIndex += 1
                            continue
                        elif dofkey == "zc": 
                            initValue[valueIndex] = self.surf.z.getRe(m,n) * pow(self._powerIndex,abs(m)+abs(n))
                            valueIndex += 1
                            continue
                        elif dofkey == "omegac": 
                            initValue[valueIndex] = self.surf.omega.getRe(m,n) * pow(self._powerIndex,abs(m)+abs(n))
                            valueIndex += 1
                            continue
                        else: 
                            continue
                    else: 
                        continue
        if self.iotafree:
            initValue[-1] = self.iota
        return initValue

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
