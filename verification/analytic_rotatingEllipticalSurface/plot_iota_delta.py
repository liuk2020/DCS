import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from tfpy.geometry import Surface_BoozerAngle


def plot_iota_delta():
    
    iota005_2, iota007_2 = list(), list()
    delta005_2, delta007_2 = list(), list()
    iota005_3, iota007_3 = list(), list()
    delta005_3, delta007_3 = list(), list()
    
    a = 0.05
    for index, nfp in enumerate([2, 3]):
        for delta in [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]:
            casename = f'{nfp}_{a}_{delta}'
            filename = './rawDatas/finalSurf_'+casename+'.h5'
            # print('get data from '+filename)
            surf = Surface_BoozerAngle.readH5(filename)
            guu, guv, _ = surf.metric
            if index == 0:
                iota005_2.append(-guv.getRe(0,0) / guu.getRe(0,0))
                delta005_2.append(delta)
            else:
                iota005_3.append(-guv.getRe(0,0) / guu.getRe(0,0))
                delta005_3.append(delta)
    a = 0.07
    for index, nfp in enumerate([2, 3]):
        for delta in [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]:
            casename = f'{nfp}_{a}_{delta}'
            filename = './rawDatas/finalSurf_'+casename+'.h5'
            # print('get data from '+filename)
            surf = Surface_BoozerAngle.readH5(filename)
            guu, guv, _ = surf.metric
            if index == 0:
                iota007_2.append(-guv.getRe(0,0) / guu.getRe(0,0))
                delta007_2.append(delta)
            else:
                iota007_3.append(-guv.getRe(0,0) / guu.getRe(0,0))
                delta007_3.append(delta)
    fig, ax = plt.subplots()
    ax.scatter(delta005_2, iota005_2, label='$a=0.05,\mathrm{NFP}=2$')
    ax.scatter(delta005_3, iota005_3, label='$a=0.05,\mathrm{NFP}=3$')
    ax.scatter(delta007_2, iota007_2, label='$a=0.07,\mathrm{NFP}=2$')
    ax.scatter(delta007_3, iota007_3, label='$a=0.07,\mathrm{NFP}=3$')
    ax.plot(np.linspace(0.02,0.21,100), 2*np.power(np.linspace(0.02,0.21,100),2), c='k', ls='--')
    ax.plot(np.linspace(0.02,0.21,100), 3*np.power(np.linspace(0.02,0.21,100),2), c='k', ls='--', label='$\mathrm{Boozer}$')
    ax.set_xlabel("$\Delta$", fontsize=16)
    ax.set_ylabel(r"$\iota$", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.legend(fontsize=16)
    fig.savefig('iota_delta.png', dpi=1000)
    fig.savefig('iota_delta.pdf')


if __name__ == '__main__':
    plot_iota_delta()
    