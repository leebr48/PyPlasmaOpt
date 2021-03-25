from pyplasmaopt import *
import numpy as np
from math import pi
import argparse
import os 
import pathlib as pl
from mpi4py import MPI
comm = MPI.COMM_WORLD

def get_objective():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--nfp", type=int, default=3)
    parser.add_argument("--num_coils", type=int, default=3)
    parser.add_argument("--Nt_ma", type=int, default=6)
    parser.add_argument("--Nt_coils", type=int, default=6)
    parser.add_argument("--ppp", type=int, default=20)
    parser.add_argument("--curv", type=float, default=0.)
    parser.add_argument("--tors", type=float, default=0.)
    parser.add_argument("--arclen", type=float, default=0.)
    parser.add_argument("--min_dist", type=float, default=0.04)
    parser.add_argument("--dist_wt", type=float, default=0.)
    parser.add_argument("--iota_targ", type=float, nargs='*', default=[-0.395938929522566])
    parser.add_argument("--iota_wt", type=float, default=1.0)
    parser.add_argument("--QS_wt", type=float, default=100.0) 
    parser.add_argument("--QFM_wt", type=float, default=0)
    parser.add_argument("--flat", action='store_true', default=False)
    parser.add_argument("--frzCoils", action='store_true', default=False)
    parser.add_argument("--tanMap", action='store_true', default=False) #Compute iota using tangent map method. 
    parser.add_argument("--rld", type=str, required=False)
    parser.add_argument("--stellID", type=int, default=0)
    parser.add_argument("--iter", type=int, default=10000)
    parser.add_argument("--Taylor", action='store_true', default=False)
    parser.add_argument("--maj_rad", type=float, default=1.4)
    parser.add_argument("--min_rad", type=float, default=0.33) # Minor radius of coils, not plasma 
    parser.add_argument("--qfm_vol", type=float, default=2.959) # target volume
    parser.add_argument("--mmax", type=int, default=3) # maximum poloidal mode number for surface
    parser.add_argument("--nmax", type=int, default=3) # maximum toroidal mode number for surface
    parser.add_argument("--ntheta", type=int, default=20) # number of poloidal grid points for integration
    parser.add_argument("--nphi", type=int, default=20) # number of toroidal grid points for integration
    parser.add_argument("--at_opt", dest="at_optimum", default=False,action="store_true")
    parser.add_argument("--tik", type=float, default=0.)
    parser.add_argument("--sob", type=float, default=0.)
    parser.add_argument("--ftol_abs", type=float, default=1e-15)
    parser.add_argument("--ftol_rel", type=float, default=1e-15)
    parser.add_argument("--xtol_abs", type=float, default=1e-15)
    parser.add_argument("--xtol_rel", type=float, default=1e-15)
    parser.add_argument("--package", type=str, default='nlopt') # For QFM surface finder
    parser.add_argument("--method", type=str, default='LBFGS') # For QFM surface finder 
    parser.add_argument("--renorm", action='store_true', default=False) # Use renormalized objective function
    parser.add_argument("--image", type=int, default=250) # How often images of stellarator should be written
    parser.add_argument("--kick", action='store_true', default=False) # Add a perturbation to the currents when loading flat coils
    args = parser.parse_args()

    if args.QFM_wt != float(0):
        args.kick = True #FIXME - an initial perturbation seems helpful for making the QFM solver behave properly

    keys = list(args.__dict__.keys())
    cutoff_key = 'rld'
    cutoff_ind = keys.index(cutoff_key)
    cutoff_number = len(keys) - cutoff_ind - 1 
    assert keys[0] == "out"
    if not args.__dict__[keys[0]] == "":
        outdir = "output-%s" % args.__dict__[keys[0]]
    else:
        outdir = "output"
    #if args.__dict__[keys[1]]:
    #    outdir += "_atopt"
    if not args.rld:
        for i in range(1, len(keys)-cutoff_number):
            k = keys[i]
            outdir += "_%s-%s" % (k, args.__dict__[k])
    if args.rld:
        for i in range(1, len(keys)-cutoff_number-1):
            k = keys[i]
            outdir += "_%s-%s" % (k, args.__dict__[k])
        outdir += "_rld-True"
        #outdir += "_stellID-%d"%args.stellID
    outdir = outdir.replace(".", "p")
    outdir = outdir.replace(' ','')
    outdir = outdir.replace(',','_')
    outdir += "/"

    os.makedirs(outdir, exist_ok=True)
    set_file_logger(outdir + "log.txt")
    info("Configuration: \n%s", args.__dict__)

    if isinstance(args.iota_targ,list):
        iota_target = args.iota_targ
    else:
        iota_target = [args.iota_targ]
    
    num_stell = len(iota_target)

    np.savetxt(str(pl.Path(outdir).joinpath('ppp.txt')),[args.ppp])
    np.savetxt(str(pl.Path(outdir).joinpath('Nt_ma.txt')),[args.Nt_ma])
    np.savetxt(str(pl.Path(outdir).joinpath('Nt_coils.txt')),[args.Nt_coils])
    np.savetxt(str(pl.Path(outdir).joinpath('num_coils.txt')),[args.num_coils]) 
    np.savetxt(str(pl.Path(outdir).joinpath('mmax.txt')),[args.mmax]) 
    np.savetxt(str(pl.Path(outdir).joinpath('nmax.txt')),[args.nmax])
    np.savetxt(str(pl.Path(outdir).joinpath('nfp.txt')),[args.nfp])
    np.savetxt(str(pl.Path(outdir).joinpath('ntheta.txt')),[args.ntheta])
    np.savetxt(str(pl.Path(outdir).joinpath('nphi.txt')),[args.nphi])
    np.savetxt(str(pl.Path(outdir).joinpath('maj_rad.txt')),[args.maj_rad])
    np.savetxt(str(pl.Path(outdir).joinpath('min_rad.txt')),[args.min_rad])
    np.savetxt(str(pl.Path(outdir).joinpath('ftol_abs.txt')),[args.ftol_abs])
    np.savetxt(str(pl.Path(outdir).joinpath('ftol_rel.txt')),[args.ftol_rel])
    np.savetxt(str(pl.Path(outdir).joinpath('xtol_abs.txt')),[args.xtol_abs])
    np.savetxt(str(pl.Path(outdir).joinpath('xtol_rel.txt')),[args.xtol_rel])
    np.savetxt(str(pl.Path(outdir).joinpath('qfm_volume.txt')),[args.qfm_vol]) #This will be overwritten if the QFM surface is included in the optimization. 

    with open(str(pl.Path(outdir).joinpath('package.txt')),'w') as f:
        f.write(args.package)
    with open(str(pl.Path(outdir).joinpath('method.txt')),'w') as f:
        f.write(args.method)

    if args.rld:
        sourcedir = str(pl.Path.cwd().joinpath(args.rld).resolve())
        with open(str(pl.Path(outdir).joinpath('reload_source.txt')),'w') as f:
            f.write('{:}\n'.format(sourcedir))
            f.write('stellID: {:}'.format(args.stellID))
        (coils, mas, currents, eta_bar) = reload_stell(sourcedir=sourcedir,ppp=args.ppp,Nt_ma=args.Nt_ma,Nt_coils=args.Nt_coils,nfp=args.nfp,stellID=args.stellID,num_coils=args.num_coils,copies=num_stell) 
        eta_bar = np.repeat(eta_bar,num_stell)
    elif args.flat:
        (coils, mas, currents) = make_flat_stell(Nt_ma=args.Nt_ma, Nt_coils=args.Nt_coils, ppp=args.ppp, copies=num_stell, nfp=args.nfp, num_coils=args.num_coils, major_radius=args.maj_rad, minor_radius=args.min_rad, kick=args.kick)
        eta_bar = np.repeat(1,num_stell)
    else:
        (coils, mas, currents) = get_ncsx_data(Nt_ma=args.Nt_ma, Nt_coils=args.Nt_coils, ppp=args.ppp, copies=num_stell)
        eta_bar = np.repeat(0.685,num_stell)

    stellarators = [CoilCollection(coils, currents, args.nfp, True) for i in range(num_stell)]

    coil_length_target = None
    magnetic_axis_length_target = None

    obj = NearAxisQuasiSymmetryObjective(
        stellarators, mas, iota_target, eta_bar=eta_bar,
        coil_length_target=coil_length_target, magnetic_axis_length_target=magnetic_axis_length_target,
        curvature_weight=args.curv, torsion_weight=args.tors,
        tikhonov_weight=args.tik, arclength_weight=args.arclen, sobolev_weight=args.sob,
        minimum_distance=args.min_dist, distance_weight=args.dist_wt,
        mode='deterministic', outdir=outdir, freezeCoils=args.frzCoils, tanMap=args.tanMap, iota_weight=args.iota_wt,
        quasisym_weight=args.QS_wt, qfm_weight=args.QFM_wt, mmax=args.mmax, nmax=args.nmax, nfp=args.nfp,
        qfm_volume=args.qfm_vol, ntheta=args.ntheta, nphi=args.nphi, ftol_abs=args.ftol_abs, ftol_rel=args.ftol_rel,
        xtol_abs=args.xtol_abs,xtol_rel=args.xtol_rel,package=args.package,method=args.method,major_radius=args.maj_rad,
        renorm=args.renorm,image_freq=args.image)
    return obj, args
