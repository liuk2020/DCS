&physicslist
    igeometry    = 3
    istellsym    = 1
    lfreebound   = 0
    phiedge      = 1.00000e+00
    curtor       = 0.00000e+00
    curpol       = 0.00000e+00
    gamma        = 0.00000e+00
    nfp          = 1
    nvol         = 1
    mpol         = 5
    ntor         = 1
    lrad         = 8
    lconstraint  = 0
    tflux        = 1.00000e+00
    rbc(0, 0)    = 1.00000e+00    zbs(0, 0)    = -0.00000e+00    rbs(0, 0)    = -0.00000e+00    zbc(0, 0)    = 1.17751e-17    rbc(1, 0)    = 0.00000e+00    zbs(1, 0)    = -0.00000e+00    rbs(1, 0)    = 0.00000e+00    zbc(1, 0)    = -0.00000e+00
    rbc(-1, 1)   = 0.00000e+00    zbs(-1, 1)   = 0.00000e+00    rbs(-1, 1)   = -0.00000e+00    zbc(-1, 1)   = -0.00000e+00
    rbc(0, 1)    = 4.00000e-01    zbs(0, 1)    = -4.00000e-01    rbs(0, 1)    = 0.00000e+00    zbc(0, 1)    = -2.80520e-17
    rbc(1, 1)    = 0.00000e+00    zbs(1, 1)    = -0.00000e+00    rbs(1, 1)    = 0.00000e+00    zbc(1, 1)    = -0.00000e+00
    rbc(-1, 2)   = 0.00000e+00    zbs(-1, 2)   = 0.00000e+00    rbs(-1, 2)   = -0.00000e+00    zbc(-1, 2)   = -0.00000e+00
    rbc(0, 2)    = -5.38290e-17    zbs(0, 2)    = -0.00000e+00    rbs(0, 2)    = 0.00000e+00    zbc(0, 2)    = 9.09410e-18
    rbc(1, 2)    = 0.00000e+00    zbs(1, 2)    = -0.00000e+00    rbs(1, 2)    = 0.00000e+00    zbc(1, 2)    = -0.00000e+00
    rbc(-1, 3)   = 0.00000e+00    zbs(-1, 3)   = 0.00000e+00    rbs(-1, 3)   = -0.00000e+00    zbc(-1, 3)   = -0.00000e+00
    rbc(0, 3)    = 2.69145e-17    zbs(0, 3)    = -1.34572e-17    rbs(0, 3)    = 0.00000e+00    zbc(0, 3)    = 1.31472e-17
    rbc(1, 3)    = 0.00000e+00    zbs(1, 3)    = -0.00000e+00    rbs(1, 3)    = 0.00000e+00    zbc(1, 3)    = -0.00000e+00
    rbc(-1, 4)   = 0.00000e+00    zbs(-1, 4)   = 0.00000e+00    rbs(-1, 4)   = -0.00000e+00    zbc(-1, 4)   = -0.00000e+00
    rbc(0, 4)    = -9.42007e-17    zbs(0, 4)    = -1.34572e-17    rbs(0, 4)    = 0.00000e+00    zbc(0, 4)    = 1.37680e-17
    rbc(1, 4)    = 0.00000e+00    zbs(1, 4)    = -0.00000e+00    rbs(1, 4)    = 0.00000e+00    zbc(1, 4)    = -0.00000e+00
    rbc(-1, 5)   = 0.00000e+00    zbs(-1, 5)   = 0.00000e+00    rbs(-1, 5)   = -0.00000e+00    zbc(-1, 5)   = -0.00000e+00
    rbc(0, 5)    = 1.34572e-17    zbs(0, 5)    = 5.38290e-17    rbs(0, 5)    = 0.00000e+00    zbc(0, 5)    = -1.97324e-17
    rbc(1, 5)    = 0.00000e+00    zbs(1, 5)    = -0.00000e+00    rbs(1, 5)    = 0.00000e+00    zbc(1, 5)    = -0.00000e+00
    mupftol      = 1.00000e-14
    mupfits      = 8
/

&numericlist
    ndiscrete    = 2
    nquad        = -1
    impol        = -4
    intor        = -4
    lsparse      = 0
    lsvdiota     = 0
    imethod      = 3
    iorder       = 2
    iprecon      = 0
    iotatol      = -1.00000e+00
/

&locallist
    lbeltrami    = 4
    linitgues    = 1
/

&globallist
    lfindzero    = 2
    escale       = 0.00000e+00
    pcondense    = 4.00000e+00
    forcetol     = 1.00000e-10
    c05xtol      = 1.00000e-12
    c05factor    = 1.00000e-02
    lreadgf      = .true.
    opsilon      = 1.00000e+00
    epsilon      = 0.00000e+00
    upsilon      = 1.00000e+00
/

&diagnosticslist
    odetol       = 1.00000e-07
    absreq       = 1.00000e-08
    relreq       = 1.00000e-08
    absacc       = 1.00000e-04
    epsr         = 1.00000e-08
    nppts        = 400
    nptrj        = -1
    lhevalues    = .false.
    lhevectors   = .false.
/

&screenlist
    wpp00aa      = .true.
/

