from .logging import info
import numpy as np

def J_evaluate(obj,x):
    try:
        obj.update(x)
        info(f'RES: {obj.res}')
        info(f'NORM(DRES): {np.linalg.norm(obj.dres)}')
        return obj.res, obj.dres
    except RuntimeError as ex:
        info(ex)
        obj.res = 2 * obj.res # For each failure, the error gets progressively larger. 
        obj.dres = 2 * obj.dres
        info(f'RES: {obj.res}')
        info(f'NORM(DRES): {np.linalg.norm(obj.dres)}')
        return obj.res, -obj.dres
