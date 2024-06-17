#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vacuumProblem.py


from .vacuum import VacuumField 
from ..toroidalField import ToroidalField
from ..geometry import Surface_cylindricalAngle 
from scipy.optimize import minimize 


class VacuumProblem(VacuumField):

    def __init__(self, surf: Surface_cylindricalAngle, lambdaField: ToroidalField=None, omegaField: ToroidalField=None, iota: float=0) -> None: 
        super().__init__(surf, lambdaField, omegaField, iota)

    def solve(): 
        pass


if __name__ == "__main__": 
    pass
