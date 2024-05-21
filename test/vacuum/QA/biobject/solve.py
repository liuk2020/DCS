from dcs.vacuum import QsSurface


qsProblem = QsSurface()
qsProblem.setResolution(mpol=3, ntor=3)
qsProblem.setNfp(nfp=2)
qsProblem.surf.r.setRe(0, 0, 1)
qsProblem.surf.r.setRe(1, 0, 0.15)
qsProblem.surf.z.setIm(1, 0, 0.15)
qsProblem.surf.r.setRe(1, 1, 0.025)
qsProblem.surf.z.setIm(1, 1, 0.025)
qsProblem.freeAll()
qsProblem.fixDOF('rc', m=1, n=0)


qsProblem.solve(
    mode = "biobject",
    logfile = "fixiota.txt", 
    vmecinput = "fixiota", 
    surfName = "fixiota", 
    figname = "fixiota"
)
