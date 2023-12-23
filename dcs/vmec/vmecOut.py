#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vmecOut.py


import xarray


class VMECOut():
    
    def __init__(self, fileName: str) -> None:
        vmeclib = xarray.open_dataset(fileName)
        self.keys = [key for key in vmeclib.data_vars] 
        for key in self.keys:
            setattr(self, key, vmeclib[key].values)


if __name__ == "__main__":
    pass
