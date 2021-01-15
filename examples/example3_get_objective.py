from pyplasmaopt import *
import numpy as np
from math import pi
import argparse
import os 
from mpi4py import MPI
comm = MPI.COMM_WORLD

def example3_get_objective():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--at-optimum", dest="at_optimum", default=False,
                        action="store_true")
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
    parser.add_argument("--iota_target", type=float, default=-0.395938929522566)
    parser.add_argument("--freezeCoils", action='store_true', default=False)
    parser.add_argument("--reload", type=str, required=False)
    args, _ = parser.parse_known_args()

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
    # print(f"lr {args.lr}, tau {args.tau}, c {args.c}, lam {args.lam}")
    # os.system('tail -n 1 voyager-output/' + outdir + 'out_of_sample_means.txt')
    # import sys; sys.exit()

    if args.reload:
        if os.path.isabs(args.reload):
            sourcedir = args.reload
        else: 
            sourcedir = os.path.join(os.path.dirname(__file__),args.reload)

    os.makedirs(outdir, exist_ok=True)
    set_file_logger(outdir + "log.txt")
    info("Configuration: \n%s", args.__dict__)
    
    nfp = 3
    if args.reload:
        (coils, ma, currents, eta_bar) = reload_ncsx(sourcedir=sourcedir,ppp=args.ppp,Nt_ma=args.Nt_ma,Nt_coils=args.Nt_coils,nfp=nfp,num_coils=3) 
    else:
        (coils, ma, currents) = get_ncsx_data(Nt_ma=args.Nt_ma, Nt_coils=args.Nt_coils, ppp=args.ppp)
        eta_bar = 0.685
    stellarator = CoilCollection(coils, currents, nfp, True)
    iota_target = args.iota_target
    coil_length_target = None
    magnetic_axis_length_target = None

    obj = NearAxisQuasiSymmetryObjective(
        stellarator, ma, iota_target, eta_bar=eta_bar,
        coil_length_target=coil_length_target, magnetic_axis_length_target=magnetic_axis_length_target,
        curvature_weight=args.curvature, torsion_weight=args.torsion,
        tikhonov_weight=args.tikhonov, arclength_weight=args.arclength, sobolev_weight=args.sobolev,
        minimum_distance=args.min_dist, distance_weight=args.dist_weight,
        mode='deterministic', outdir=outdir, freezeCoils=args.freezeCoils)
    return obj, args
