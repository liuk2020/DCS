from dcs.vacuumBoozer import QsSurface

nfp = 7
r0 = 2
a = 0.1
delta = 0.1
qsProblem = QsSurface()
qsProblem.setResolution(3, 3)
qsProblem.setNfp(nfp)
qsProblem.surf.r.setRe(0, 0, r0)
qsProblem.surf.r.setRe(1, 0, a/2)
qsProblem.surf.r.setRe(0, 1, a*delta/2/nfp)
qsProblem.surf.z.setIm(0, 0, 0)
qsProblem.surf.z.setIm(1, 0, a/2)
qsProblem.surf.z.setIm(0, -1, a*delta/2/nfp)
qsProblem.freeAll()
qsProblem.fixAll_rc()
qsProblem.fixAll_zs()

qsProblem.solve(
    weight=[1, 0]
)
