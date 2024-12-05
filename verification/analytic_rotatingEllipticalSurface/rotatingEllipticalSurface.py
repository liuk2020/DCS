import sys
from dcs import IsolatedSurface


def rotatingEllip(nfp: int, a: float, Delta:float):
    
    casename = f'{nfp}_{a}_{Delta}'
    
    surfProblem = IsolatedSurface(
        mpol=4,
        ntor=4,
        nfp=nfp
    )
    surfProblem.changeStellSym(True)
    surfProblem.updateMajorRadius(1)
    surfProblem.r.setRe(1, 0, a)
    surfProblem.z.setIm(1, 0, -a)
    surfProblem.r.setRe(1, 1, a*Delta)
    surfProblem.z.setIm(1, 1, a*Delta)
    surfProblem.fixAll()
    surfProblem.freeAll_omegas()
    
    print(f'============== This case is {nfp}_{a}_{Delta} ')
    surfProblem.solve()
    surfProblem.writeH5('./rawDatas/finalSurf_' + casename)
    

if __name__ == '__main__':
    print('========================================================================================================================================')
    print(f'The NFP is {sys.argv[1]}')
    print(f'The minor radius a is {sys.argv[2]}')
    print(f'The Delta is {sys.argv[3]}')
    rotatingEllip(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]))
