from pyplasmaopt import *
import numpy as np
from math import pi
import argparse
import os 
import pathlib as pl
from mpi4py import MPI
comm = MPI.COMM_WORLD

def example3_get_objective():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--at-optimum", dest="at_optimum", default=False,
                        action="store_true")
    parser.add_argument("--nfp", type=int, default=3)
    parser.add_argument("--ppp", type=int, default=20)
    parser.add_argument("--Nt_ma", type=int, default=4)
    parser.add_argument("--Nt_coils", type=int, default=4)
    parser.add_argument("--curvature", type=float, default=0.)
    parser.add_argument("--torsion", type=float, default=0.)
    parser.add_argument("--tikhonov", type=float, default=0.)
    parser.add_argument("--sobolev", type=float, default=0.)
    parser.add_argument("--arclength", type=float, default=0.)
    parser.add_argument("--min-dist", type=float, default=0.04)
    parser.add_argument("--dist-weight", type=float, default=0.)
    parser.add_argument("--iota_target", type=float, nargs='*', default=[-0.395938929522566])
    parser.add_argument("--iota_weight", type=float, default=1.0)
    parser.add_argument("--quasisym_weight", type=float, default=100.0) #Might switch to 10 if 100 causes issues.
    parser.add_argument("--freezeCoils", action='store_true', default=False)
    parser.add_argument("--reload", type=str, required=False)
    args = parser.parse_args()

    keys = list(args.__dict__.keys())
    assert keys[0] == "output"
    if not args.__dict__[keys[0]] == "":
        outdir = "output-%s" % args.__dict__[keys[0]]
    else:
        outdir = "output"
    if args.__dict__[keys[1]]:
        outdir += "_atopt"
    if not args.reload:
        for i in range(2, len(keys)):
            k = keys[i]
            outdir += "_%s-%s" % (k, args.__dict__[k])
    if args.reload:
        for i in range(2, len(keys)-1):
            k = keys[i]
            outdir += "_%s-%s" % (k, args.__dict__[k])
        outdir += "_reload-True"
    outdir = outdir.replace(".", "p")
    outdir += "/"
    
    os.makedirs(outdir, exist_ok=True)
    set_file_logger(outdir + "log.txt")
    info("Configuration: \n%s", args.__dict__)
    
    if args.reload:
        sourcedir = str(pl.Path.cwd().joinpath(args.reload).resolve())
        with open(str(pl.Path(outdir).joinpath('reload_source.txt')),'w') as f:
            f.write('{:}\n'.format(sourcedir))

    if args.reload:
        (coils, ma, currents, eta_bar) = reload_ncsx(sourcedir=sourcedir,ppp=args.ppp,Nt_ma=args.Nt_ma,Nt_coils=args.Nt_coils,nfp=args.nfp,num_coils=3) 
    else:
        (coils, ma, currents) = get_ncsx_data(Nt_ma=args.Nt_ma, Nt_coils=args.Nt_coils, ppp=args.ppp)
        eta_bar = 0.685
    stellarator = CoilCollection(coils, currents, args.nfp, True)
    if isinstance(args.iota_target,list):
        iota_target = args.iota_target
    else:
        iota_target = [args.iota_target]
    coil_length_target = None
    magnetic_axis_length_target = None

    obj = NearAxisQuasiSymmetryObjective(
        stellarator, ma, iota_target, eta_bar=eta_bar,
        coil_length_target=coil_length_target, magnetic_axis_length_target=magnetic_axis_length_target,
        curvature_weight=args.curvature, torsion_weight=args.torsion,
        tikhonov_weight=args.tikhonov, arclength_weight=args.arclength, sobolev_weight=args.sobolev,
        minimum_distance=args.min_dist, distance_weight=args.dist_weight,
        mode='deterministic', outdir=outdir, freezeCoils=args.freezeCoils, iota_weight=args.iota_weight,
        quasisym_weight=args.quasisym_weight)
    return obj, args
