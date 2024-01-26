#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# solver.py 


import numpy as np 
from ..toroidalField import ToroidalField 
from ..toroidalField import derivatePol, derivateTor, changeResolution
from typing import Tuple


class Solver:
    r"""
    Solve the equation
        $$ \frac{\partial(PG)}{\partial\theta} = \frac{\partial(QG)}{\partial\zeta} $$ 
    or
        $$ G(\frac{\partial P}{\partial\theta}-\frac{\partial Q}{\partial\zeta}) + P\frac{\partial G}{\partial\theta} - Q\frac{\partial G}{\partial\zeta} = 0 $$ 
    """


    def __init__(self, pField: ToroidalField, qField: ToroidalField) -> None:
        assert pField.nfp == qField.nfp 
        self.nfp = pField.nfp 
        self.mpol = pField.mpol + qField.mpol 
        self.ntor = pField.ntor + qField.ntor 
        self.P = changeResolution(pField, mpol=self.mpol, ntor=self.ntor) 
        self.Q = changeResolution(qField, mpol=self.mpol, ntor=self.ntor) 
        self.D = derivatePol(self.P) - derivateTor(self.Q) 


    def solve_linearEquation(self) -> ToroidalField:
        
        from scipy.linalg import solve 
    
        def getVectorB() -> np.ndarray:
            vectorB = np.zeros(2*self.ntor+2*self.mpol*(2*self.ntor+1))
            for i in range(2*self.ntor+2*self.mpol*(2*self.ntor+1)): 
                m, n, label = self.indexMap(i+1)
                if label == "re":
                    vectorB[i] = self.getRe_CoefMN(m,n,0,0)
                elif label == "im":
                    vectorB[i] = self.getIm_CoefMN(m,n,0,0)
            vectorB *= -1
            return vectorB

        def getMatrixCoef() -> np.ndarray:
            matrixCoef = np.zeros([2*self.ntor+2*self.mpol*(2*self.ntor+1), 2*self.ntor+2*self.mpol*(2*self.ntor+1)])
            for i in range(2*self.ntor+2*self.mpol*(2*self.ntor+1)):
                m, n, equationLabel = self.indexMap(i+1) 
                for j in range(2*self.ntor+2*self.mpol*(2*self.ntor+1)):
                    _m, _n, variableLabel = self.indexMap(j+1) 
                    if equationLabel == "re":
                        if variableLabel == "re":
                            matrixCoef[i,j] = self.getRe_CoefMN(m,n,_m,_n) + self.getRe_CoefMN(m,n,-_m,-_n)
                        elif variableLabel == "im":
                            matrixCoef[i,j] = - self.getIm_CoefMN(m,n,_m,_n) + self.getIm_CoefMN(m,n,-_m,-_n)
                    elif equationLabel == "im":
                        if variableLabel == "re":
                            matrixCoef[i,j] = self.getIm_CoefMN(m,n,_m,_n) + self.getIm_CoefMN(m,n,-_m,-_n)
                        elif variableLabel == "im":
                            matrixCoef[i,j] = self.getRe_CoefMN(m,n,_m,_n) - self.getRe_CoefMN(m,n,-_m,-_n)
            return matrixCoef

        matrixCoef = getMatrixCoef() 
        vectorB = getVectorB()
        vectorG = solve(matrixCoef, vectorB)
        reArr = np.zeros(self.ntor+1+self.mpol*(2*self.ntor+1))
        imArr = np.zeros(self.ntor+1+self.mpol*(2*self.ntor+1))
        reArr[0] = 1
        _field = ToroidalField(
            nfp = self.nfp, 
            mpol = self.mpol, 
            ntor = self.ntor, 
            reArr = reArr,
            imArr = imArr
        )
        for i in range(self.ntor+self.mpol*(2*self.ntor+1)):
            m, n, label = self.indexMap(i+1)
            if label == "re":
                _field.setRe(m, n, vectorG[i])
            elif label == "im":
                _field.setIm(m, n, vectorG[i])
        return _field 


    def getRe_CoefMN(self, m: int, n : int, _m : int, _n: int) -> float:
        return (
            self.D.getRe(m-_m,n-_n) 
            - _m*self.P.getIm(m-_m,n-_n) 
            - _n*self.nfp*self.Q.getIm(m-_m,n-_n) 
        )


    def getIm_CoefMN(self, m: int, n : int, _m : int, _n: int) -> float:
        return (
            self.D.getIm(m-_m,n-_n) 
            + _m*self.P.getRe(m-_m,n-_n) 
            + _n*self.nfp*self.Q.getRe(m-_m,n-_n) 
        )


    def indexMap(self, index: int) -> Tuple:
        assert 1 <= index <= 2*self.ntor+2*self.mpol*(2*self.ntor+1)
        if index <= self.ntor+self.mpol*(2*self.ntor+1):
            if 1 <= index <= self.ntor:
                m = 0
                n = index
            elif self.ntor+1 <= index <= self.ntor+self.mpol*(2*self.ntor+1):
                m = (index-self.ntor-1) // (2*self.ntor+1) + 1
                n = (index-self.ntor-1) % (2*self.ntor+1) - self.ntor 
            label = "re"
        elif self.ntor+self.mpol*(2*self.ntor+1)+1 <= index <= 2*self.ntor+2*self.mpol*(2*self.ntor+1):
            if self.ntor+self.mpol*(2*self.ntor+1)+1 <= index <= 2*self.ntor+self.mpol*(2*self.ntor+1):
                m = 0
                n = index - self.ntor - self.mpol*(2*self.ntor+1)
            elif 2*self.ntor+self.mpol*(2*self.ntor+1)+1 <= index <= 2*self.ntor+2*self.mpol*(2*self.ntor+1):
                m = (index-2*self.ntor-self.mpol*(2*self.ntor+1)-1) // (2*self.ntor+1) + 1
                n = (index-2*self.ntor-self.mpol*(2*self.ntor+1)-1) % (2*self.ntor+1) - self.ntor 
            label = "im"
        return m, n, label


if __name__ == "__main__": 
    pass
