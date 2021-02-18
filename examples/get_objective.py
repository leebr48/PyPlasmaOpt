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
    parser.add_argument("--at_opt", dest="at_optimum", default=False,
                        action="store_true")
    parser.add_argument("--nfp", type=int, default=3)
    parser.add_argument("--ppp", type=int, default=20)
    parser.add_argument("--Nt_ma", type=int, default=4)
    parser.add_argument("--Nt_coils", type=int, default=4)
    parser.add_argument("--curv", type=float, default=0.)
    parser.add_argument("--tors", type=float, default=0.)
    parser.add_argument("--tik", type=float, default=0.)
    parser.add_argument("--sob", type=float, default=0.)
    parser.add_argument("--arclen", type=float, default=0.)
    parser.add_argument("--min_dist", type=float, default=0.04)
    parser.add_argument("--dist_wt", type=float, default=0.)
    parser.add_argument("--iota_targ", type=float, nargs='*', default=[-0.395938929522566])
    parser.add_argument("--iota_wt", type=float, default=1.0)
    parser.add_argument("--QS_wt", type=float, default=100.0) #Might switch to 10 if 100 causes issues.
    parser.add_argument("--frzCoils", action='store_true', default=False)
    parser.add_argument("--rld", type=str, required=False)
    parser.add_argument("--stellID", type=int, default=0)
    parser.add_argument("--iter", type=int, default=10000)
    parser.add_argument("--Taylor", action='store_true', default=False)
    args = parser.parse_args()

    keys = list(args.__dict__.keys())
    assert keys[0] == "out"
    if not args.__dict__[keys[0]] == "":
        outdir = "output-%s" % args.__dict__[keys[0]]
    else:
        outdir = "output"
    if args.__dict__[keys[1]]:
        outdir += "_atopt"
    if not args.rld:
        for i in range(2, len(keys)-4):
            k = keys[i]
            outdir += "_%s-%s" % (k, args.__dict__[k])
    if args.rld:
        for i in range(2, len(keys)-4):
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

    if args.rld:
        sourcedir = str(pl.Path.cwd().joinpath(args.rld).resolve())
        with open(str(pl.Path(outdir).joinpath('reload_source.txt')),'w') as f:
            f.write('{:}\n'.format(sourcedir))
            f.write('stellID: {:}'.format(args.stellID))
        (coils, mas, currents, eta_bar) = reload_ncsx(sourcedir=sourcedir,ppp=args.ppp,Nt_ma=args.Nt_ma,Nt_coils=args.Nt_coils,nfp=args.nfp,stellID=args.stellID,num_coils=3,copies=num_stell) 
        eta_bar = np.repeat(eta_bar,num_stell)
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
        mode='deterministic', outdir=outdir, freezeCoils=args.frzCoils, iota_weight=args.iota_wt,
        quasisym_weight=args.QS_wt)
    return obj, args
