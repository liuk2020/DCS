from dcs.vacuumBoozer import QsSurface

nfp = 3
r0 = 1.5
a = 0.4
delta = 0.2
qsProblem = QsSurface()
qsProblem.setResolution(5, 5)
qsProblem.setNfp(nfp)
qsProblem.surf.r.setRe(0, 0, r0)
qsProblem.surf.r.setRe(1, 0, a/2)
qsProblem.surf.r.setRe(1, -1, a*delta/2)
qsProblem.surf.z.setIm(0, 0, 0)
qsProblem.surf.z.setIm(1, 0, a/2)
qsProblem.surf.z.setIm(1, -1, -a*delta/2)
qsProblem.freeAll()
qsProblem.fixAll_rc()
qsProblem.fixAll_zs()

qsProblem.solve(
    weight=[1, 0]
)
