import numpy as np
from dcs import QSSurface


def QA_configuration(nfp: int, inverseRatio: float, iota: float):
    
    surfProblem = QSSurface(
        mpol=1,
        ntor=1,
        nfp=nfp
    )
    surfProblem.changeStellSym(True)
    surfProblem.r.setRe(0, 0, 1)
    surfProblem.r.setRe(1, 0, inverseRatio/2)
    surfProblem.z.setIm(1, 0, -inverseRatio/2)
    surfProblem.r.setRe(1, 1, inverseRatio/20)
    surfProblem.z.setIm(1, 1, -inverseRatio/20)
    
    surfProblem.freeAll()
    surfProblem.fixDOF('rc', 1, 0)
    surfProblem.fixDOF('zs', 1, 0)
    surfProblem.updateconstraint('iota', True)
    surfProblem.updateweight('iota', 0.1)
    surfProblem.updateweight('qs', 0)
    surfProblem.solve(nstep=10)
    surfProblem.writeH5('step1')
    
    surfProblem.changeResolution(2, 2)
    surfProblem.freeAll()
    surfProblem.fixDOF('rc', 1, 0)
    surfProblem.fixDOF('zs', 1, 0)
    surfProblem.updateweight('iota', 0.01)
    surfProblem.updateweight('qs', 0.01)
    surfProblem.solve(nstep=10)
    surfProblem.writeH5('step2')
    
    surfProblem.changeResolution(3, 3)
    surfProblem.freeAll()
    surfProblem.fixDOF('rc', 1, 0)
    surfProblem.fixDOF('zs', 1, 0)
    surfProblem.updateweight('iota', 0.001)
    surfProblem.updateweight('qs', 0.005)
    surfProblem.solve(nstep=5)
    surfProblem.writeH5('step3')
    surfProblem.toVTK('surf')
    

if __name__ == '__main__':
    QA_configuration(2, 0.1, (np.sqrt(5)-1)/4)