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
    parser.add_argument("--contNum", type=int, default=0) # Number of control coils in the problem from the beginning. 
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
    parser.add_argument("--QS_wt", type=float, default=1) 
    parser.add_argument("--QFM_wt", type=float, default=0)
    parser.add_argument("--flat", action='store_true', default=False)
    parser.add_argument("--frzCoils", action='store_true', default=False)
    parser.add_argument("--frzMod", action='store_true', default=False)
    parser.add_argument("--tanMap", action='store_true', default=False) # Compute iota using tangent map method. 
    parser.add_argument("--newCont", type=int, default=0) # For adding control coils to previous optimization. 
    parser.add_argument("--rld", type=str, required=False) 
    parser.add_argument("--cons", action='store_false', default=True) # Controls the 'constrained' switch in the tangent map class.
    parser.add_argument("--keepAx", action='store_false', default=True) # Keep the magnetic axis in the parameter space
    parser.add_argument("--stellID", type=int, default=-1) 
    parser.add_argument("--iter", type=int, default=100000)
    parser.add_argument("--Taylor", action='store_true', default=False)
    parser.add_argument("--maj_rad", type=float, default=1.5) # Based (approximately) on NCSX
    parser.add_argument("--min_rad", type=float, default=0.815) # Minor radius of coils, not plasma; default value based (approximately) on NCSX
    parser.add_argument("--qfm_vol", type=float, default=2.959) # target volume
    parser.add_argument("--mmax", type=int, default=3) # maximum poloidal mode number for surface
    parser.add_argument("--nmax", type=int, default=3) # maximum toroidal mode number for surface
    parser.add_argument("--ntheta", type=int, default=20) # number of poloidal grid points for integration
    parser.add_argument("--nphi", type=int, default=20) # number of toroidal grid points for integration
    parser.add_argument("--renorm", action='store_true', default=False) # Use renormalized objective function
    parser.add_argument("--image", type=int, default=250) # How often images of stellarator should be written
    parser.add_argument("--kick", action='store_true', default=False) # Add a perturbation to the currents when loading flat coils
    parser.add_argument("--mag", type=float, default=0.05) # Perturbation (kick) magnitude
    parser.add_argument("--z0factr", type=float, default=4) # Additional perturbation in the z direction
    parser.add_argument("--contRad", type=float, default=1.3)
    parser.add_argument("--N", type=int, default=0) # 0 for QA, 1 for QH
    parser.add_argument("--oldFormat", action='store_true', required=False, default=False) # Included for backwards compatibility operation in the reload_stell function
    args = parser.parse_args()

    if args.frzMod and args.contNum == 0 and args.newCont == 0:
        raise AssertionError('Do not use frzMod with no control coils - use frzCoils instead!')

    keys = list(args.__dict__.keys())
    cutoff_key = 'rld'
    cutoff_ind = keys.index(cutoff_key)
    cutoff_number = len(keys) - cutoff_ind - 1 
    assert keys[0] == "out"
    if not args.__dict__[keys[0]] == "":
        outdir = "out-%s" % args.__dict__[keys[0]]
    else:
        outdir = "out"
    if not args.rld:
        for i in range(1, len(keys)-cutoff_number):
            k = keys[i]
            outdir += "_%s-%s" % (k, args.__dict__[k])
    if args.rld:
        for i in range(1, len(keys)-cutoff_number-1):
            k = keys[i]
            outdir += "_%s-%s" % (k, args.__dict__[k])
        outdir += "_rld-True"
    outdir = outdir.replace(".", "p")
    outdir = outdir.replace(' ','')
    outdir = outdir.replace(',','_')
    outdir = outdir.replace('True','T')
    outdir = outdir.replace('False','F')
    outdir = outdir.replace('None','F')
    outdir += "/"

    os.makedirs(outdir, exist_ok=True)
    set_file_logger(outdir + "log.txt")
    info("Configuration: \n%s", args.__dict__)

    if args.stellID < 0:
        stellID = None
    else:
        stellID = args.stellID

    if isinstance(args.iota_targ,list):
        iota_target = args.iota_targ
    else:
        iota_target = [args.iota_targ]
    
    num_stell = len(iota_target)
    qs_N = args.N * args.nfp

    np.savetxt(str(pl.Path(outdir).joinpath('ppp.txt')),[args.ppp])
    np.savetxt(str(pl.Path(outdir).joinpath('Nt_ma.txt')),[args.Nt_ma])
    np.savetxt(str(pl.Path(outdir).joinpath('Nt_coils.txt')),[args.Nt_coils])
    np.savetxt(str(pl.Path(outdir).joinpath('num_coils.txt')),[args.num_coils])
    np.savetxt(str(pl.Path(outdir).joinpath('contNum.txt')),[args.contNum + args.newCont])
    np.savetxt(str(pl.Path(outdir).joinpath('mmax.txt')),[args.mmax]) 
    np.savetxt(str(pl.Path(outdir).joinpath('nmax.txt')),[args.nmax])
    np.savetxt(str(pl.Path(outdir).joinpath('nfp.txt')),[args.nfp])
    np.savetxt(str(pl.Path(outdir).joinpath('ntheta.txt')),[args.ntheta])
    np.savetxt(str(pl.Path(outdir).joinpath('nphi.txt')),[args.nphi])
    np.savetxt(str(pl.Path(outdir).joinpath('maj_rad.txt')),[args.maj_rad])
    np.savetxt(str(pl.Path(outdir).joinpath('min_rad.txt')),[args.min_rad])
    np.savetxt(str(pl.Path(outdir).joinpath('qfm_volume.txt')),[args.qfm_vol]) #This will be overwritten if the QFM surface is included in the optimization. 
    np.savetxt(str(pl.Path(outdir).joinpath('contRad.txt')),[args.contRad])

    xopt_rld = None
    coil_length_targets = None
    magnetic_axis_length_targets = None
    tanMap_axis = None

    if args.rld:
        sourcedir = str(pl.Path.cwd().joinpath(args.rld).resolve())
        with open(str(pl.Path(outdir).joinpath('reload_source.txt')),'w') as f:
            f.write('{:}\n'.format(sourcedir))
            f.write('stellID: {:}'.format(stellID))
        (coils, mas, currents, eta_bar) = reload_stell(sourcedir=sourcedir,ppp=args.ppp,Nt_ma=args.Nt_ma,Nt_coils=args.Nt_coils,nfp=args.nfp,stellID=stellID,num_coils=args.num_coils,contNum=args.contNum,newCont=args.newCont,contRad=args.contRad,copies=num_stell,oldFormat=args.oldFormat) # The 'copies' attribute is only used if stellID != None, and 'contRad' applies only to 'newCont'. 
        try:
            coil_length_targets = np.loadtxt(str(pl.Path(sourcedir).joinpath('coil_length_targets.txt')),ndmin=1)
        except OSError:
            pass
        try:
            magnetic_axis_length_targets = np.loadtxt(str(pl.Path(sourcedir).joinpath('magnetic_axis_length_targets.txt')),ndmin=1)
            if (num_stell != 1) and (stellID != None):
                ma_len_targ = magnetic_axis_length_targets[stellID]
                magnetic_axis_length_targets = np.repeat(ma_len_targ,num_stell)
        except OSError:
            pass
        if args.QFM_wt > 0:
            try:
                xopt_rld = []
                if stellID != None:
                    xopt_old = np.loadtxt(str(pl.Path(sourcedir).joinpath('xopt_{:}.txt'.format(stellID))))
                    [xopt_rld.append(xopt_old) for i in range(num_stell)]
                else:
                    for stell in range(num_stell):
                        xopt_rld.append(np.loadtxt(str(pl.Path(sourcedir).joinpath('xopt_{:}.txt'.format(stell)))))
            except OSError:
                xopt_rld = None # Allows us to add a QFM surface when there was not one originally
        if args.tanMap:
            if stellID != None:
                tanMap_axis,_,_ = NoncoilReload('tanMap_axis_coeffs',sourcedir,args.Nt_ma,args.nfp,args.ppp,copy=num_stell,stellID=stellID)
            else:
                tanMap_axis,_,_ = NoncoilReload('tanMap_axis_coeffs',sourcedir,args.Nt_ma,args.nfp,args.ppp)
            if tanMap_axis == []:
                tanMap_axis = None

    elif args.flat:
        (coils, mas, currents) = make_flat_stell(Nt_ma=args.Nt_ma, Nt_coils=args.Nt_coils, ppp=args.ppp, copies=num_stell, nfp=args.nfp, num_coils=args.num_coils, major_radius=args.maj_rad, minor_radius=args.min_rad, kick=args.kick, magnitude=args.mag, z0factr=args.z0factr, contNum=args.contNum, contRad=args.contRad)
        eta_bar = np.repeat(1,num_stell)
    
    else:
        (coils, mas, currents) = get_ncsx_data(Nt_ma=args.Nt_ma, Nt_coils=args.Nt_coils, ppp=args.ppp, copies=num_stell, contNum=args.contNum, contRad=args.contRad)
        eta_bar = np.repeat(0.685,num_stell)

    stellarators = [CoilCollection(coils, currents[i], args.nfp, True, frzMod=args.frzMod) for i in range(num_stell)]

    obj = NearAxisQuasiSymmetryObjective(
        stellarators, mas, iota_target, eta_bar=eta_bar, Nt_ma=args.Nt_ma,
        coil_length_targets=coil_length_targets, magnetic_axis_length_targets=magnetic_axis_length_targets,
        curvature_weight=args.curv, torsion_weight=args.tors,
        tikhonov_weight=0, arclength_weight=args.arclen, sobolev_weight=0,
        minimum_distance=args.min_dist, distance_weight=args.dist_wt,
        mode='deterministic', outdir=outdir, freezeCoils=args.frzCoils, freezeMod=args.frzMod,
        tanMap=args.tanMap, constrained=args.cons, keepAxis=args.keepAx, tanMap_axis=tanMap_axis,
        iota_weight=args.iota_wt, quasisym_weight=args.QS_wt, nfp=args.nfp, 
        qfm_weight=args.QFM_wt, mmax=args.mmax, nmax=args.nmax, qfm_volume=args.qfm_vol, 
        ntheta=args.ntheta, nphi=args.nphi, xopt_rld=xopt_rld,
        renorm=args.renorm, image_freq=args.image, qs_N=qs_N)
    return obj, args
