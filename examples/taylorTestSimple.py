from pyplasmaopt import *
from get_objective import get_objective
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import pathlib as pl

np.set_printoptions(floatmode='unique') # Ensures data is saved in full detail

obj, args = get_objective()

outdir = obj.outdir

def taylor_test(obj, x, order=6, export=False, nrando=1):
    for randind in range(nrando):
        #np.random.seed(1)
        h = np.random.rand(*(x.shape))
        np.savetxt(str(pl.Path(outdir).joinpath('taylor_test_direction-%d.txt'%randind)), h)
        obj.update(x)
        dj0 = obj.dres
        djh = sum(dj0*h)
        djhnorm = np.linalg.norm(djh)
        info('djh norm: ', djhnorm)
        if order == 1:
            shifts = [0, 1]
            weights = [-1, 1]
        elif order == 2:
            shifts = [-1, 1]
            weights = [-0.5, 0.5]
        elif order == 4:
            shifts = [-2, -1, 1, 2]
            weights = [1/12, -2/3, 2/3, -1/12]
        elif order == 6:
            shifts = [-3, -2, -1, 1, 2, 3]
            weights = [-1/60, 3/20, -3/4, 3/4, -3/20, 1/60]
        epsvec = []
        errvec = []
        for i in range(10, 40):
            eps = 0.5**i
            epsvec.append(eps)
            obj.update(x + shifts[0]*eps*h)
            fd = weights[0] * obj.res
            for i in range(1, len(shifts)):
                obj.update(x + shifts[i]*eps*h)
                fd += weights[i] * obj.res
            err = abs(fd/eps - djh)
            errnorm = err/djhnorm
            errvec.append(errnorm)
            info("%.6e, %.6e, %.6e", eps, err, errnorm)
        obj.update(x)
        info("-----")
        plt.figure()
        yminind = np.argmin(errvec)
        refline = np.asarray([item**(order) for item in epsvec])
        split = errvec[yminind]-refline[yminind]
        upshift = split
        refline2 = refline/upshift
        plt.loglog(epsvec,errvec,marker='o')
        plt.loglog(epsvec,refline2,linestyle='--',marker='o')
        plt.xlim([epsvec[yminind]/1e5,np.max(epsvec)])
        plt.ylim([errvec[yminind]/1e30,np.max(errvec)])
        plt.xlabel('eps')
        plt.ylabel('err (normalized)')
        plt.savefig(str(pl.Path(outdir).joinpath('taylorPlot-%d.png'%randind)),bbox_inches='tight')
        np.savetxt(str(pl.Path(outdir).joinpath('epsvec-%d.txt'%randind)),epsvec)
        np.savetxt(str(pl.Path(outdir).joinpath('errvec-%d.txt'%randind)),errvec)

x = obj.x0
obj.update(x)
obj.callback(x)

maxiter = args.iter
memory = 200
maxfun = args.iter * 100 
iprint = -1
maxls = 50 #Elizabeth's suggestion

J_eval = lambda y: J_evaluate(obj,y)

res = minimize(J_eval, x, jac=True, method='l-bfgs-b', tol=1e-20, 
        options={"maxiter": maxiter, "maxcor": memory, "ftol":1e-20, "gtol":1e-16, "maxfun":maxfun, "iprint":iprint, "maxls":maxls},
               callback=obj.callback) 

info("%s" % res)
xmin = res.x
info('XMIN_currents before perturb: ',xmin[obj.current_dof_idxs[0]:obj.current_dof_idxs[1]])


multiplier = xmin[obj.current_dof_idxs[0]]/10
randos = multiplier*np.random.rand(len(xmin[obj.current_dof_idxs[0]:obj.current_dof_idxs[1]]))
xmin[obj.current_dof_idxs[0]:obj.current_dof_idxs[1]] += randos
info('XMIN_currents after perturb: ',xmin[obj.current_dof_idxs[0]:obj.current_dof_idxs[1]])

J_distance = MinimumDistance(obj.stellarator_group[0].coils, 0)
info("Minimum distance = %f" % J_distance.min_dist())

iteration = len(obj.xiterates)-1
Checkpoint(obj,iteration=iteration)
np.savetxt(outdir + "xmin.txt", xmin)

if args.Taylor:
    taylor_test(obj, xmin, nrando=3)
