#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# baseproblem.py


import numpy as np
from tfpy.toroidalField import ToroidalField, changeResolution
from tfpy.geometry import Surface_BoozerAngle
from typing import Tuple


class SurfProblem(Surface_BoozerAngle):
    
    def __init__(self, 
        r: ToroidalField=None,
        z: ToroidalField=None,
        omega: ToroidalField=None,
        mpol: int=None,
        ntor: int=None,
        nfp: int=None,
        reverseToroidalAngle: bool = False,
        reverseOmegaAngle: bool = True
    ) -> None:
        if (r is None) or (z is None) or (omega is None):
            assert mpol and ntor and nfp
            self._mpol, self._ntor = mpol, ntor
            r = ToroidalField.constantField(1, nfp=nfp, mpol=mpol, ntor=ntor)
            z = ToroidalField.constantField(0, nfp=nfp, mpol=mpol, ntor=ntor)
            omega = ToroidalField.constantField(0, nfp=nfp, mpol=mpol, ntor=ntor)
            super().__init__(r, z, omega, reverseToroidalAngle, reverseOmegaAngle)
            self.changeStellSym(True)
            self._iota = 0.0
        else:
            if mpol is None or ntor is None:
                mpol, ntor = r.mpol, r.ntor
            self._mpol, self._ntor = mpol, ntor
            super().__init__(changeResolution(r,mpol,ntor), changeResolution(z,mpol,ntor), changeResolution(omega,mpol,ntor), reverseToroidalAngle, reverseOmegaAngle)
            self._init_iota()
        self._init_dofs()
        self._init_paras()
        self._init_constraint()
        self._init_weight()
        self._init_target()

    def _init_iota(self):
        guu, guv, gvv = self.metric
        self._iota = -guv.getRe(0,0)/guu.getRe(0,0)
    
    def _init_dofs(self):
        self.doflabels = {}
        self.doflabels['rc'] = [False for i in range(self.mpol*(2*self.ntor+1)+self.ntor)]
        self.doflabels['zs'] = [False for i in range(self.mpol*(2*self.ntor+1)+self.ntor)]
        self.doflabels['omegas'] = [False for i in range(self.mpol*(2*self.ntor+1)+self.ntor)]
        if not self.stellSym:
            self.doflabels['rs'] = [False for i in range(self.mpol*(2*self.ntor+1)+self.ntor)]
            self.doflabels['zc'] = [False for i in range(self.mpol*(2*self.ntor+1)+self.ntor)]
            self.doflabels['omegac'] = [False for i in range(self.mpol*(2*self.ntor+1)+self.ntor)]

    def _init_paras(self):
        self.paras = {
            'powerIndex': 0.97
        }
    
    def updateparas(self, key: str, value):
        if key not in self.paras.keys():
            print(f'Warning: there is no parameter named {key}')
        else:
            self.paras[key] = value
            
    def _init_constraint(self):
        self._constraint = {
            'iota': False,
            'inverse ratio': False
        }
        
    def updateconstraint(self, key: str, value: bool):
        if key not in self._constraint.keys():
            print(f'Warning: there is no constraint named {key}')
        else:
            self._constraint[key] = value
            
    def _init_weight(self):
        self._weight = {
            'iota': 0.05,
            'inverse ratio': 0.01
        }
        
    def updateweight(self, key: str, value: float):
        if key not in self._weight.keys():
            print(f'Warning: there is no constraint named {key}')
        else:
            self._weight[key] = value
            
    def _init_target(self):
        self._target = {
            'iota': 0.309,
            'inverse ratio': 0.1
        }
        
    def updatetarget(self, key: str, value):
        if key not in self._target.keys():
            print(f'Warning: there is no constraint named {key}')
        else:
            self._target[key] = value
            
    def addconstraint(self, key: str, default_weight: float, default_target: float):
        self._constraint[key] = False
        self._weight[key] = default_weight
        self._target[key] = default_target
    
    @property
    def iota(self):
        return self._iota
    
    def updateIota(self, iota):
        self._iota = iota

    def updateInverseRatio(self):
        _area, _volume = self.getAreaVolume(npol=128, ntor=128)
        self.area, self.volume = _area, _volume
        self.majorR = self.area*self.area/8/np.pi/np.pi/self.volume
        self.minorR = 2*self.volume/self.area
        self.inverseRatio = self.minorR / self.majorR

    def updateMinCrossArea(self, npol: int=128, ntor: int=64):
        try:
            crossArea = np.abs([self.getCrossArea(phi, npol=npol) for phi in np.linspace(0,2*np.pi/self.nfp,ntor)])
        except:
            crossArea = 1e-14 
        self.minCrossArea = np.min(crossArea)
    
    def changeResolution(self, mpol: int, ntor: int):
        self._mpol, self._ntor = mpol, ntor
        self.r = changeResolution(self.r, mpol, ntor)
        self.z = changeResolution(self.z, mpol, ntor)
        self.omega = changeResolution(self.omega, 2*mpol, 2*ntor)
        self._init_dofs()

    def changeNfp(self, nfp: int):
        self.nfp = nfp
        self.r.nfp = nfp
        self.z.nfp = nfp
        self.omega.nfp = nfp

    @property
    def numsDOF(self):
        nums = 0
        for key in ['rc', 'zs', 'omegas']:
            nums += np.sum(self.doflabels[key])
        if not self.stellSym:
            for key in ['rs', 'zc', 'omegac']:
                nums += np.sum(self.doflabels[key])
        return nums + 1

    def fixAll(self):
        for key in ['rc', 'zs', 'omegas']:
            for i, label in enumerate(self.doflabels[key]):
                self.doflabels[key][i] = False
        if not self.stellSym:
            for key in ['rs', 'zc', 'omegac']:
                for i, label in enumerate(self.doflabels[key]):
                    self.doflabels[key][i] = False

    def freeAll(self):
        for key in ['rc', 'zs', 'omegas']:
            for i, label in enumerate(self.doflabels[key]):
                self.doflabels[key][i] = True
        if not self.stellSym:
            for key in ['rs', 'zc', 'omegac']:
                for i, label in enumerate(self.doflabels[key]):
                    self.doflabels[key][i] = True
                    
    def fixAll_rc(self):
        for i, label in enumerate(self.doflabels['rc']):
            self.doflabels['rc'][i] = False

    def freeAll_rc(self):
        for i, label in enumerate(self.doflabels['rc']):
            self.doflabels['rc'][i] = True

    def fixAll_zs(self):
        for i, label in enumerate(self.doflabels['zs']):
            self.doflabels['zs'][i] = False

    def freeAll_zs(self):
        for i, label in enumerate(self.doflabels['zs']):
            self.doflabels['zs'][i] = True

    def fixAll_omegas(self):
        for i, label in enumerate(self.doflabels['omegas']):
            self.doflabels['omegas'][i] = False

    def freeAll_omegas(self):
        for i, label in enumerate(self.doflabels['omegas']):
            self.doflabels['omegas'][i] = True

    def fixTruncation_rc(self, m: int, n: int):
        assert m<=self.r.mpol and n<=self.r.ntor
        for i, label in enumerate(self.doflabels['rc']):
            _m ,_n = self.r.indexReverseMap(i+1)
            if abs(_m)>abs(m) or abs(_n)>abs(n):
                self.doflabels['rc'][i] = False

    def fixTruncation_zs(self, m: int, n: int):
        assert m<=self.z.mpol and n<=self.z.ntor
        for i, label in enumerate(self.doflabels['zs']):
            _m ,_n = self.z.indexReverseMap(i+1)
            if abs(_m)>abs(m) or abs(_n)>abs(n):
                self.doflabels['zs'][i] = False

    def fixTruncation_omegas(self, m: int, n: int):
        assert m<=self.omega.mpol and n<=self.omega.ntor
        for i, label in enumerate(self.doflabels['omegas']):
            _m ,_n = self.omega.indexReverseMap(i+1)
            if abs(_m)>abs(m) or abs(_n)>abs(n):
                self.doflabels['omegas'][i] = False

    def fixDOF(self, dof: str, m: int=0, n: int=0):
        if self.stellSym:
            assert dof in ['rc', 'zs', 'omegas']
        else:
            assert dof in ['rc', 'zs', 'omegas', 'rs', 'zc', 'omegac']
        if dof=='rc' or dof=='rs':
            self.doflabels[dof][self.r.indexMap(m,n)-1] = False
        if dof=='zc' or dof=='zs':
            self.doflabels[dof][self.z.indexMap(m,n)-1] = False
        if dof=='omegac' or dof=='omegas':
            self.doflabels[dof][self.omega.indexMap(m,n)-1] = False

    def freeDOF(self, dof: str, m: int=0, n: int=0):
        if self.stellSym:
            assert dof in ['rc', 'zs', 'omegas']
        else:
            assert dof in ['rc', 'zs', 'omegas', 'rs', 'zc', 'omegac']
        if dof=='rc' or dof=='rs':
            self.doflabels[dof][self.r.indexMap(m,n)-1] = True
        if dof=='zc' or dof=='zs':
            self.doflabels[dof][self.z.indexMap(m,n)-1] = True
        if dof=='omegac' or dof=='omegas':
            self.doflabels[dof][self.omega.indexMap(m,n)-1] = True

    @property
    def initDOFs(self):
        dofs = np.zeros(self.numsDOF)
        dofindex = 0
        for key in ['rc', 'zs', 'omegas']:
            for i, label in enumerate(self.doflabels[key]):
                if label:
                    if key == 'rc':
                        m, n = self.r.indexReverseMap(i+1)
                        dofs[dofindex] = self.r.getRe(m,n) * pow(self.paras['powerIndex'],abs(m)+abs(n))
                    elif key == 'zs':
                        m, n = self.z.indexReverseMap(i+1)
                        dofs[dofindex] = self.z.getIm(m,n) * pow(self.paras['powerIndex'],abs(m)+abs(n))
                    elif key == 'omegas':
                        m, n = self.omega.indexReverseMap(i+1)
                        dofs[dofindex] = self.omega.getIm(m,n) * pow(self.paras['powerIndex'],abs(m)+abs(n))
                    dofindex += 1
        if not self.stellSym:
            for key in ['rs', 'zc', 'omegac']:
                for i, label in enumerate(self.doflabels[key]):
                    if label:
                        if key == 'rs':
                            m, n = self.r.indexReverseMap(i+1)
                            dofs[dofindex] = self.r.getIm(m,n) * pow(self.paras['powerIndex'],abs(m)+abs(n))
                        elif key == 'zc':
                            m, n = self.z.indexReverseMap(i+1)
                            dofs[dofindex] = self.z.getRe(m,n) * pow(self.paras['powerIndex'],abs(m)+abs(n))
                        elif key == 'omegac':
                            m, n = self.omega.indexReverseMap(i+1)
                            dofs[dofindex] = self.omega.getRe(m,n) * pow(self.paras['powerIndex'],abs(m)+abs(n))
                        dofindex += 1
        dofs[-1] = self.iota
        return dofs

    def unpackDOF(self, dofs: np.ndarray) -> None:
        assert dofs.size == self.numsDOF
        dofindex = 0
        for key in ['rc', 'zs', 'omegas']:
            for i, label in enumerate(self.doflabels[key]):
                if label:
                    if key == 'rc':
                        m, n = self.r.indexReverseMap(i+1)
                        self.r.setRe(m,n,dofs[dofindex] / pow(self.paras['powerIndex'],abs(m)+abs(n)))
                    elif key == 'zs':
                        m, n = self.z.indexReverseMap(i+1)
                        self.z.setIm(m,n,dofs[dofindex] / pow(self.paras['powerIndex'],abs(m)+abs(n)))
                    elif key == 'omegas':
                        m, n = self.omega.indexReverseMap(i+1)
                        self.omega.setIm(m,n,dofs[dofindex] / pow(self.paras['powerIndex'],abs(m)+abs(n)))
                    dofindex += 1
        if not self.stellSym:
            for key in ['rs', 'zc', 'omegac']:
                for i, label in enumerate(self.doflabels[key]):
                    if label:
                        if key == 'rs':
                            m, n = self.r.indexReverseMap(i+1)
                            self.r.setIm(m,n,dofs[dofindex] / pow(self.paras['powerIndex'],abs(m)+abs(n)))
                        elif key == 'zc':
                            m, n = self.z.indexReverseMap(i+1)
                            self.z.setRe(m,n,dofs[dofindex] / pow(self.paras['powerIndex'],abs(m)+abs(n)))
                        elif key == 'omegac':
                            m, n = self.omega.indexReverseMap(i+1)
                            self.omega.setRe(m,n,dofs[dofindex] / pow(self.paras['powerIndex'],abs(m)+abs(n)))
                        dofindex += 1
        self.updateIota(dofs[-1])
        return


if __name__ == "__main__":
    pass
