from dcs.vacuum import QsSurface


qsProblem = QsSurface()
qsProblem.setResolution(mpol=4, ntor=4)
qsProblem.setNfp(nfp=2)
qsProblem.setIota(0.4158)
qsProblem.surf.r.setRe(0, 0, 1)
qsProblem.surf.r.setRe(1, 0, 0.2)
qsProblem.surf.z.setIm(1, 0, 0.2)
qsProblem.surf.r.setRe(1, 1, 0.05)
qsProblem.surf.z.setIm(1, 1, 0.05)
qsProblem.freeAll()
qsProblem.freeIota = False
qsProblem.fixDOF('rc', m=1, n=0)
qsProblem.fixDOF('zs', m=1, n=0)


qsProblem.solve(
    mode = "lagrange",
    logfile = "fixiota.txt", 
    vmecinput = "fixiota", 
    surfH5 = "fixiota", 
    figname = "fixiota"
)
