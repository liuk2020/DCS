from tfpy.geometry import Surface_BoozerAngle
from tfpy.boozXform import BoozerForm
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import numpy as np


def plot_all():

    fig, ax = plt.subplots()

    for resolution in [2, 3, 4, 5, 6, 7]:
        filename = f'./rawDatas/finalSurf_{resolution}.h5'
        surf = Surface_BoozerAngle.readH5(filename)
        guu, guv, gvv = surf.metric
        iota = - guv.getRe(0,0)/guu.getRe(0,0)
        scriptB = gvv + iota*guv
        thetaArr = np.linspace(0,2*np.pi,100)
        zetaArr = np.zeros(100)
        ax.plot(
            thetaArr,
            scriptB.getValue(thetaArr, zetaArr),
            label=r'$(\mathrm{mpol,ntor})=('+str(2*resolution)+','+str(2*resolution)+')$'
        )
        
    datalib = BoozerForm('./LandremanQA/boozer_wout_LandremanPaul2021_QA.nc')
    B = datalib.getB()
    G = datalib.Boozer_G_all[-1]
    ax.scatter(
        np.linspace(0,2*np.pi,20),
        G*G/np.power(B.getValue(np.linspace(0,2*np.pi,20),np.zeros(20)),2),
        c='violet', label=r'$\mathrm{Landreman}$'
    )
    
    ax.set_xlabel(r"$\theta$", fontsize=16)
    ax.set_ylabel(r"$\mathcal{B}$", fontsize=16)
    # plt.xticks(fontsize=16)
    xValues = [0, np.pi, 2*np.pi]
    ax.set_xticks(xValues)
    ax.set_xticklabels(["$0$", "$\pi$", "$2\pi$"], fontsize=16)
    plt.yticks(fontsize=16)
    ax.legend(fontsize=16)
    fig.tight_layout()
    fig.savefig('scriptB.png', dpi=1000)
    fig.savefig('scriptB.pdf')
    
    
def plot_local():
    
    fig, ax = plt.subplots(figsize=(4.7,5.0))

    for resolution in [2, 3, 4, 5, 6, 7]:
        filename = f'./rawDatas/finalSurf_{resolution}.h5'
        surf = Surface_BoozerAngle.readH5(filename)
        guu, guv, gvv = surf.metric
        iota = - guv.getRe(0,0)/guu.getRe(0,0)
        scriptB = gvv + iota*guv
        thetaArr = np.linspace(2*np.pi-0.7,2*np.pi,100)
        zetaArr = np.zeros(100)
        ax.plot(
            thetaArr,
            scriptB.getValue(thetaArr, zetaArr),
            label=r'$(\mathrm{mpol,ntor})=('+str(2*resolution)+','+str(2*resolution)+')$',
            linewidth=2
        )
        
    datalib = BoozerForm('./LandremanQA/boozer_wout_LandremanPaul2021_QA.nc')
    B = datalib.getB()
    G = datalib.Boozer_G_all[-1]
    ax.scatter(
        np.linspace(2*np.pi-0.7,2*np.pi,10),
        G*G/np.power(B.getValue(np.linspace(2*np.pi-0.7,2*np.pi,10),np.zeros(10)),2),
        c='violet', label=r'$\mathrm{Landreman}$'
    )

    ax.set_xlabel(r"$\theta$", fontsize=26)
    ax.set_ylabel(r"$\mathcal{B}$", fontsize=26)
    ax.set_xticks([9*np.pi/5, 2*np.pi])
    ax.set_xticklabels([r"$\frac{9\pi}{5}$", r"$2\pi$"], fontsize=26)
    # plt.xticks(fontsize=26)
    ax.set_yticks([1.3, 1.4])
    ax.set_yticklabels(["$1.3$", "$1.4$"], fontsize=26)
    # plt.yticks(fontsize=26)
    # ax.legend(fontsize=16)
    fig.tight_layout()
    fig.savefig('localscriptB.png', dpi=1000)
    fig.savefig('localscriptB.pdf')


if __name__ == '__main__':
    plot_all()
    plot_local()
