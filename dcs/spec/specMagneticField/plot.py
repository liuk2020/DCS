#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# plot.py


import numpy as np
import matplotlib.pyplot as plt
from .fieldLine import FieldLine
from typing import List


def plotPoincare(lines: List[FieldLine], toroidalIdx: int=0, ax=None, **kwargs):

    if ax is None:
        fig, ax = plt.subplots()
    plt.sca(ax)
    if kwargs.get("marker") == None:
        kwargs.update({"marker": "."})
    if kwargs.get("s") == None:
        kwargs.update({"s": 1.4})

    for line in lines:
        rArr = list()
        zArr = list()
        if line.equalZeta:
            for i in range(len(line.rArr)):
                if (i-toroidalIdx) % line.nZeta == 0:
                    rArr.append(line.rArr[i])
                    zArr.append(line.zArr[i])
        else:
            zetaPeriod = 2*np.pi/line.nfp
            # zetaPeriod = 2 * np.pi
            for i in range(len(line.zetaArr)-1):
                if line.zetaArr[i]//zetaPeriod+1 == line.zetaArr[i+1]//zetaPeriod:
                    rArr.append(
                        ((line.rArr[i] * (line.zetaArr[i+1]%zetaPeriod)) + (line.rArr[i+1]) * (zetaPeriod-line.zetaArr[i]%zetaPeriod))
                        / (line.zetaArr[i+1] - line.zetaArr[i])
                    )
                    zArr.append(
                        ((line.zArr[i] * (line.zetaArr[i+1]%zetaPeriod)) + (line.zArr[i+1]) * (zetaPeriod-line.zetaArr[i]%zetaPeriod))
                        / (line.zetaArr[i+1] - line.zetaArr[i])
                    )
                elif line.zetaArr[i]//zetaPeriod-1 == line.zetaArr[i+1]//zetaPeriod: 
                    rArr.append(
                        ((line.rArr[i+1] * (line.zetaArr[i]%zetaPeriod)) + (line.rArr[i]) * (zetaPeriod-line.zetaArr[i+1]%zetaPeriod))
                        / (line.zetaArr[i] - line.zetaArr[i+1])
                    )
                    zArr.append(
                        ((line.zArr[i+1] * (line.zetaArr[i]%zetaPeriod)) + (line.zArr[i]) * (zetaPeriod-line.zetaArr[i+1]%zetaPeriod))
                        / (line.zetaArr[i] - line.zetaArr[i+1])
                    )
        dots = ax.scatter(rArr, zArr, **kwargs)
    plt.axis("equal")

    return


if __name__ == "__main__":
    pass