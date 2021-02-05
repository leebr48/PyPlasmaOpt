from pyplasmaopt import *
from example3_get_objective import example3_get_objective
from scipy.optimize import minimize
import numpy as np
import pathlib as pl

obj, args = example3_get_objective()
obj.plot('tmp.png')

outdir = obj.outdir

def taylor_test(obj, x, order=6, export=False):
    np.random.seed(1)
    h = np.random.rand(*(x.shape))
    np.savetxt(str(pl.Path(outdir).joinpath('taylor_test_direction.txt')), h)
    print(h)
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
        obj.update(x)
    dj0 = obj.dres
    djh = sum(dj0*h)
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
    for i in range(10, 40):
        eps = 0.5**i
        obj.update(x + shifts[0]*eps*h)
        fd = weights[0] * obj.res
        for i in range(1, len(shifts)):
            obj.update(x + shifts[i]*eps*h)
            fd += weights[i] * obj.res
        err = abs(fd/eps - djh)
        info("%.6e, %.6e, %.6e", eps, err, err/np.linalg.norm(djh))
    obj.update(x)
    info("-----")

x = obj.x0
obj.update(x)
obj.callback(x)
obj.save_to_matlab('matlab_init')
if False:
    taylor_test(obj, x, order=1, export=True)
    taylor_test(obj, x, order=2)
    taylor_test(obj, x, order=4)
    taylor_test(obj, x, order=6)
    import sys; sys.exit()

maxiter = 10000 #for real
#maxiter = 1 #for testing
memory = 200

def J_scipy(x):
    obj.update(x)
    res = obj.res
    dres = obj.dres
    return res, dres

res = minimize(J_scipy, x, jac=True, method='l-bfgs-b', tol=1e-15,
               options={"maxiter": maxiter, "maxcor": memory},
               callback=obj.callback)

info("%s" % res)
xmin = res.x
obj.save_to_matlab('matlab_optim')
J_distance = MinimumDistance(obj.stellarator_group[0].coils, 0)
info("Minimum distance = %f" % J_distance.min_dist())
'''
for i,stell in enumerate(obj.stellarator_group):
    obj.stellarator_group[i].savetotxt(outdir,i) #Maybe works?
    np.savetxt(str(pl.Path(obj.outdir).joinpath('currents_%d.txt'%i)), obj.stellarator_group[i]._base_currents) #Maybe works?
'''
for stellind,stellarator in enumerate(obj.stellarator_group):
    np.savetxt(str(pl.Path(obj.outdir).joinpath('currents_%d.txt'%stellind)), obj.stellarator_group[stellind]._base_currents) #Maybe works?
    for coilind,coil in enumerate(obj.stellarator_group[stellind].coils):
        np.savetxt(str(pl.Path(obj.outdir).joinpath('coil-%d.txt'%coilind)),obj.stellarator_group[0].coils[coilind].gamma) #Maybe works?
        np.savetxt(str(pl.Path(obj.outdir).joinpath('current-%d_%d.txt'%(coilind,stellind))),[obj.stellarator_group[stellind].currents[coilind]]) #Maybe works?

matlabcoils = [c.tomatlabformat() for c in obj.stellarator_group[0]._base_coils] #Should be fine
np.savetxt(str(pl.Path(obj.outdir).joinpath('coilsmatlab.txt')), np.hstack(matlabcoils)) #Should be fine

save = obj.stellarator_group[0]._base_coils[0].coefficients
for i in range(1,len(obj.stellarator_group[0]._base_coils)):
    save = np.append(save,obj.stellarator_group[0]._base_coils[i].coefficients,axis=0)
np.savetxt(str(pl.Path(obj.outdir).joinpath('coilCoeffs.txt')), save,fmt='%.20f') #Should be fine

for i,ma in enumerate(obj.ma_group):
    save = []
    for item in obj.ma_group[i].coefficients:
        save.append(item.tolist())
    with open(str(pl.Path(obj.outdir).joinpath('maCoeffs_%d.txt'%i)), "w") as f: #Maybe works?
        for line in save:
            for ind,item in enumerate(line):
                f.write(str(item))
                if ind!=len(line)-1:
                    f.write(' ')
            f.write('\n')

for i,qsf in enumerate(obj.qsf_group):
    save1 = obj.qsf_group[i].eta_bar
    np.savetxt(str(pl.Path(obj.outdir).joinpath('eta_bar_%d.txt'%i)), [save1],fmt='%.20f') 
    save2 = obj.qsf_group[i].iota
    np.savetxt(str(pl.Path(obj.outdir).joinpath('iota_ma_%d.txt'%i)), [save2],fmt='%.20f')

np.savetxt(outdir + "xmin.txt", xmin)
np.savetxt(outdir + "Jvals.txt", obj.Jvals)
np.savetxt(outdir + "dJvals.txt", obj.dJvals)
np.savetxt(outdir + "xiterates.txt", obj.xiterates)
np.savetxt(outdir + "Jvals_individual.txt", obj.Jvals_individual)

if False:
    taylor_test(obj, xmin)
