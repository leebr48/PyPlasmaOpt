from pyplasmaopt import *
from get_objective import get_objective
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import pathlib as pl
from pyplasmaopt.checkpoint import Checkpoint

obj, args = get_objective()
obj.plot('tmp.png') #This will only plot the coils and the first magnetic axis. 

outdir = obj.outdir

def taylor_test(obj, x, order=6, export=False, nrando=1):
    for randind in range(nrando):
        #np.random.seed(1)
        h = np.random.rand(*(x.shape))
        np.savetxt(str(pl.Path(outdir).joinpath('taylor_test_direction-%d.txt'%randind)), h)
        #print('h: ',h)
        '''
        if export:
            obj.update(h)
            obj.save_to_matlab('h')
            obj.update(x+h)
            obj.save_to_matlab('xplush')
            print('x+h', obj.res)
            obj.update(x)
            obj.save_to_matlab('x')
            print('x', obj.res)
        else:
        '''
        obj.update(x)
        dj0 = obj.dres
        djh = sum(dj0*h)
        djhnorm = np.linalg.norm(djh)
        print('djh norm: ', djhnorm)
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

if False:
    taylor_test(obj, x, order=1, export=True)
    taylor_test(obj, x, order=2)
    taylor_test(obj, x, order=4)
    taylor_test(obj, x, order=6)
    import sys; sys.exit()

maxiter = args.iter
memory = 200
maxfun = args.iter * 100 
iprint = -1
maxls = 50 #Elizabeth's suggestion

def J_scipy(x):
    try:
        obj.update(x)
        return obj.res, obj.dres
    except RuntimeError as ex:
        info(ex)
        return 2*obj.res, -obj.dres

res = minimize(J_scipy, x, jac=True, method='l-bfgs-b', tol=1e-20, 
        options={"maxiter": maxiter, "maxcor": memory, "ftol":1e-20, "gtol":1e-16, "maxfun":maxfun, "iprint":iprint, "maxls":maxls},
               callback=obj.callback) 

info("%s" % res)
xmin = res.x
J_distance = MinimumDistance(obj.stellarator_group[0].coils, 0)
info("Minimum distance = %f" % J_distance.min_dist())

iteration = len(obj.xiterates)-1
Checkpoint(obj,iteration=iteration)
np.savetxt(outdir + "xmin.txt", xmin)
if args.Taylor:
    taylor_test(obj, xmin, nrando=1)
