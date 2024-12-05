import sys
from tfpy.toroidalField import ToroidalField
from tfpy.boozXform import BoozerForm
from dcs import IsolatedSurface



def onecase(resolution: int):

    datalib = BoozerForm('./LandremanQA/boozer_wout_LandremanPaul2021_QA.nc')
    surf = datalib.getSurface(-1)
    _omega=ToroidalField.constantField(0, nfp=surf.r.nfp, mpol=2*resolution, ntor=2*resolution)
    _omega.reIndex, _omega.imIndex = False, True
    
    surfProblem = IsolatedSurface(
        r = surf.r,
        z = surf.z,
        omega=_omega,
        mpol=resolution,
        ntor=resolution
    )
    surfProblem.fixAll()
    surfProblem.freeAll_omegas()
    
    print(f'============== This case is {resolution} ')
    surfProblem.solve()
    surfProblem.writeH5(f'./rawDatas/finalSurf_{resolution}')
    
    

if __name__ == '__main__':
    print('========================================================================================================================================')
    onecase(int(sys.argv[1]))