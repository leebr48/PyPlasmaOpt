from .curve import RotatedCurve,ControlCoil
import numpy as np
from math import pi

class CoilCollection():
    """
    Given some input coils and currents, this performs the reflection and
    rotation to generate a full set of stellerator coils.
    """

    def __init__(self, coils, currents, nfp, stellerator_symmetrie, frzMod=False):
        self._base_coils = coils
        self._base_currents = currents
        self.frzMod = frzMod
        self.coils = []
        self.currents = []
        self.all_control_coil = []
        flip_list = [False, True] if stellerator_symmetrie else [False] 
        self.map = []
        self.current_sign = []
        for k in range(0, nfp):
            for flip in flip_list:
                for i in range(len(coils)):
                    if k == 0 and not flip:
                        self.coils.append(self._base_coils[i])
                        iscontrol = True if isinstance(self._base_coils[i],ControlCoil) else False
                        self.all_control_coil.append(iscontrol)
                        self.currents.append(self._base_currents[i])
                    else:
                        rotcoil = RotatedCurve(coils[i], 2*pi*k/nfp, flip)
                        self.coils.append(rotcoil)
                        iscontrol = True if isinstance(coils[i],ControlCoil) else False
                        self.all_control_coil.append(iscontrol)
                        self.currents.append(-self._base_currents[i] if flip else currents[i])
                    self.map.append(i)
                    self.current_sign.append(-1 if flip else +1)
        dof_ranges = [(0, len(self._base_coils[0].get_dofs()))]
        for i in range(1, len(self._base_coils)):
            dof_ranges.append((dof_ranges[-1][1], dof_ranges[-1][1] + len(self._base_coils[i].get_dofs())))
        self.dof_ranges = dof_ranges
        self.base_control_coil_count = 0
        for i,x in enumerate(self._base_coils):
            if isinstance(x,ControlCoil):
                self.base_control_coil_count += 1

    def set_dofs(self, dofs):
        assert len(dofs) == self.dof_ranges[-1][1]
        for i in range(len(self._base_coils)):
            self._base_coils[i].set_dofs(dofs[self.dof_ranges[i][0]:self.dof_ranges[i][1]])

    def get_dofs(self):
        return np.concatenate([coil.get_dofs() for coil in self._base_coils])
    
    def set_currents(self, currents):
        self._base_currents = currents
        for i in range(len(self.currents)):
            self.currents[i] = self.current_sign[i] * currents[self.map[i]]

    def get_currents(self):
        return np.asarray(self._base_currents)

    def reduce_coefficient_derivatives(self, derivatives, axis=0):
        """
        Add derivatives for all those coils that were obtained by rotation and
        reflection of the initial coils.
        """
        lenDer = len(derivatives)
        assert lenDer == len(self.coils) or lenDer == len(self._base_coils) or lenDer == self.base_control_coil_count
        if self.frzMod and (lenDer == self.base_control_coil_count):
            diff = len(self._base_coils) - self.base_control_coil_count
            addor = [None]*diff
            [addor.append(i) for i in derivatives]
            derivatives = addor
        res = len(self._base_coils) * [None]
        for i in range(len(derivatives)):
            if self.frzMod and (not self.all_control_coil[i]):
                continue
            if res[self.map[i]] is None:
                res[self.map[i]]  = derivatives[i]
            else:
                res[self.map[i]] += derivatives[i]
        res = [item for item in res if not (item is None)]
        return np.concatenate(res, axis=axis)

    def reduce_current_derivatives(self, derivatives):
        """
        Combine derivatives with respect to current for all those coils that
        were obtained by rotation and reflection of the initial coils.
        """
        assert len(derivatives) == len(self.coils) or len(derivatives) == len(self._base_coils)
        res = len(self._base_coils) * [None]
        for i in range(len(derivatives)):
            if res[self.map[i]] is None:
                res[self.map[i]]  = self.current_sign[i] * derivatives[i]
            else:
                res[self.map[i]] += self.current_sign[i] * derivatives[i]
        return np.asarray(res)
