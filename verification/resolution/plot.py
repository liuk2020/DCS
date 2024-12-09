from tfpy.geometry import Surface_BoozerAngle
from tfpy.boozXform import BoozerForm
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
# from tfpy.vmec import VMECOut
# from tfpy.spec import SPECOut
# from tfpy.toroidalField import fftToroidalField

datalib = BoozerForm('./LandremanQA/boozer_wout_LandremanPaul2021_QA.nc')
iotaLandreman = datalib.getIota()

iota = list()
for resolution in [2, 3, 4, 5, 6, 7]:
    
    filename = f'./rawDatas/finalSurf_{resolution}.h5'
    surf = Surface_BoozerAngle.readH5(filename)
    guu, guv, _ = surf.metric
    iota.append(-guv.getRe(0,0) / guu.getRe(0,0))

fig, ax = plt.subplots(figsize=(7.0,5.0))
ax.scatter([4,6,8,10,12,14], iota, c='coral', label=r'$\mathrm{Numerical}$')
ax.plot([3.5, 14.5], [iotaLandreman, iotaLandreman], ls='--', c='violet', label=r'$\mathrm{Landreman}$')
ax.set_xlabel(r"$\mathrm{resolution}$", fontsize=16)
ax.set_ylabel(r"$\iota$", fontsize=16)
plt.xticks(fontsize=16)
ax.set_yticks([0.414, 0.415, 0.416])
ax.set_yticklabels(["$0.414$", "$0.415$", "$0.416$"], fontsize=16)
# plt.yticks(fontsize=16)
ax.legend(fontsize=16, loc='right')
fig.tight_layout()
fig.savefig('resolution.png', dpi=1000)
fig.savefig('resolution.pdf')