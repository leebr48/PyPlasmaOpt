from .curve import RotatedCurve
import numpy as np
from math import pi


class CoilCollection():
    """
    Given some input coils and currents, this performs the reflection and
    rotation to generate a full set of stellerator coils.
    """

    def __init__(self, coils, currents, nfp, stellerator_symmetrie):
        self.__base_coils = coils
        self.__base_currents = currents
        self.coils = []
        self.currents = []
        flip_list = [False, True] if stellerator_symmetrie else [False] 
        self.map = []
        for k in range(0, nfp):
            for flip in flip_list:
                for i in range(len(coils)):
                    if k == 0 and not flip:
                        self.coils.append(self.__base_coils[i])
                        self.currents.append(self.__base_currents[i])
                    else:
                        rotcoil = RotatedCurve(coils[i], 2*pi*k/nfp, flip)
                        self.coils.append(rotcoil)
                        self.currents.append(-self.__base_currents[i] if flip else currents[i])
                    self.map.append(i)
        dof_ranges = [(0, len(self.__base_coils[0].get_dofs()))]
        for i in range(1, len(self.__base_coils)):
            dof_ranges.append((dof_ranges[-1][1], dof_ranges[-1][1] + len(self.__base_coils[i].get_dofs())))
        self.dof_ranges = dof_ranges

    def set_dofs(self, dofs):
        assert len(dofs) == self.dof_ranges[-1][1]
        for i in range(len(self.__base_coils)):
            self.__base_coils[i].set_dofs(dofs[self.dof_ranges[i][0]:self.dof_ranges[i][1]])

    def get_dofs(self):
        return np.concatenate([coil.get_dofs() for coil in self.__base_coils])

    def reduce_derivatives(self, derivatives, axis=0):
        """
        Add derivatives for all those coils that were obtained by rotation and
        reflection of the initial coils.
        """
        assert len(derivatives) == len(self.coils)
        res = len(self.__base_coils) * [None]
        for i in range(len(derivatives)):
            if res[self.map[i]] is None:
                res[self.map[i]]  = derivatives[i]
            else:
                res[self.map[i]] += derivatives[i]
        return np.concatenate(res, axis=axis)