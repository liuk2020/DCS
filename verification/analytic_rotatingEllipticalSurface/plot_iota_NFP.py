# import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from tfpy.geometry import Surface_BoozerAngle


def plot_iota_NFP():
    
    iota005_01, iota007_01 = list(), list()
    nfp005_01, nfp007_01 = list(), list()
    iota005_02, iota007_02 = list(), list()
    nfp005_02, nfp007_02 = list(), list()
    
    a = 0.05
    for nfp in [2, 3, 4, 5, 6]:
        for index, delta in enumerate([0.1, 0.2]):
            casename = f'{nfp}_{a}_{delta}'
            filename = './rawDatas/finalSurf_'+casename+'.h5'
            # print('get data from '+filename)
            surf = Surface_BoozerAngle.readH5(filename)
            guu, guv, _ = surf.metric
            if index == 0:
                iota005_01.append(-guv.getRe(0,0) / guu.getRe(0,0))
                nfp005_01.append(nfp)
            else:
                iota005_02.append(-guv.getRe(0,0) / guu.getRe(0,0))
                nfp005_02.append(nfp)
    a = 0.07
    for nfp in [2, 3, 4, 5, 6]:
        for index, delta in enumerate([0.1, 0.2]):
            casename = f'{nfp}_{a}_{delta}'
            filename = './rawDatas/finalSurf_'+casename+'.h5'
            # print('get data from '+filename)
            surf = Surface_BoozerAngle.readH5(filename)
            guu, guv, _ = surf.metric
            if index == 0:
                iota007_01.append(-guv.getRe(0,0) / guu.getRe(0,0))
                nfp007_01.append(nfp)
            else:
                iota007_02.append(-guv.getRe(0,0) / guu.getRe(0,0))
                nfp007_02.append(nfp)
    fig, ax = plt.subplots()
    ax.scatter(nfp005_01, iota005_01, label='$a=0.05,\Delta=0.1$')
    ax.scatter(nfp005_02, iota005_02, label='$a=0.05,\Delta=0.2$')
    ax.scatter(nfp007_01, iota007_01, label='$a=0.07,\Delta=0.1$')
    ax.scatter(nfp007_02, iota007_02, label='$a=0.07,\Delta=0.2$')
    ax.plot([1.7,6.3], [0.01*1.7,0.01*6.3], c='k', ls='--')
    ax.plot([1.7,6.3], [0.04*1.7,0.04*6.3], c='k', ls='--', label='$\mathrm{Boozer}$')
    ax.set_xlabel("$\mathrm{NFP}$", fontsize=16)
    ax.set_ylabel(r"$\iota$", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.legend(fontsize=16)
    fig.savefig('iota_nfp.png', dpi=1000)
    fig.savefig('iota_nfp.pdf')


if __name__ == '__main__':
    plot_iota_NFP()
    