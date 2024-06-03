from dcs.vacuumBoozer import QsSurface

nfp = 1
r0 = 1
a = 0.2
qsProblem = QsSurface()
qsProblem.setResolution(2, 0)
qsProblem.setNfp(nfp)
qsProblem.surf.r.setRe(0, 0, r0)
qsProblem.surf.r.setRe(1, 0, a)
qsProblem.surf.z.setIm(0, 0, 0)
qsProblem.surf.z.setIm(1, 0, a)
qsProblem.freeAll()
qsProblem.fixAll_rc()
qsProblem.fixAll_zs()

qsProblem.solve(
    weight=[1, 0]
)
