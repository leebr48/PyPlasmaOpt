"""
This script is used to compute a quadratic flux minimizing surface with 
a given volume for the input coils, as well as information about the coils.
This all is then fed into VMEC and BOOZXFORM.  
"""

# Options
## All
image_filetype = 'pdf' #Choose something that MatPlotLib can handle. 
font_size = 18
## Poincare plot
qfm_max_tries = 5
poincare_max_tries = 5
nperiods = 700 #200 
batch_size = 4
delta = 0.01
spp = 120
marker_size = 0.01 #0.04
marker_symbol = '.'
poincare_w_qfm_name = 'poincare_w_qfm'
poincare_w_vmec_name = 'poincare_w_vmec'
poincare_w_both_name = 'poincare_w_both' 
## CoilPy
coils_file_suffix = 'pyplasmaopt'
## MAKEGRID
makegrid_input_file = 'input_xgrid.dat'  
makegrid_output_file = 'makegrid_output.txt'
current_format = 's' #'s' for scaled current, 'r' for raw current
stellarator_symmetry = 'y' #'y' for stellarator symmetry true, 'n' for false
toroidal_cutplanes = 36 #MAKEGRID setting
radial_grid_points = 201 #MAKEGRID setting
vertical_grid_points = 201 #MAKEGRID setting
## VMEC  
vmec_output_file = 'VMEC_output.txt'
vmec_DELT = 9.00E-01 #This is typically 0.7-0.9 and may have to be tweaked to make runs converge properly.
vmec_NITER = 10000
vmec_NSTEP = 200
vmec_TCON0 = 2.00E+00
vmec_NS_ARRAY = '3 9 29 49 99'
vmec_FTOL_ARRAY = '1.000000E-05 1.000000E-06 1.000000E-08 1.000000E-10 1.000000E-12'
vmec_LASYM = 'F'
#vmec_MPOL is set with an argument below
vmec_NTOR = 6
vmec_LFREEB = 'T'
vmec_NVACSKIP = 6
vmec_AM = 0
vmec_CURTOR = 0
vmec_NCURR = 1
vmec_AC = 0
## VMEC/PyPlasmaOpt Comparison
iota_compare_fig_name = 'iota_compare'
profile_compare_name = 'profile_compare'
profile_compare_ind_freq = 2 #Integer describing the spacing between profile images (higher number -> fewer images). 
## BOOZXFORM
booz_output_file = 'booz_output.txt'
booz_toroidal_harmonics = 51
booz_poloidal_harmonics = 51
booz_surface_indices = '2 25 50 75 99' #Surface indices on which to compute booz transform
max_m = 10 # maximum poloidal mode number to plot
max_n = 10 # maximum toroidal mode number to plot
booz_harmonicsplot_name = 'booz_harmonics_plot'
booz_QAplot_name = 'booz_QA_plot'

#####################################################################################################################

# Load packages
import pyplasmaopt as ppo
from pyplasmaopt import *
import numpy as np
from scipy.io import netcdf
import argparse 
import os 
import pathlib as pl 
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from coilpy import * 
import subprocess as sp
ppo_path = os.path.dirname(ppo.__file__)
with open(str(pl.Path(ppo_path).joinpath('ALPOpt_dir.txt')),'r') as f:
        ALPOpt_dir = f.read().strip()
sys.path.append(ALPOpt_dir)
#from vmec_input import init_modes #FIXME
from vmec_output import VmecOutput

plt.rcParams['font.size'] = str(font_size)

# Sort out the command line arguments 
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--sourcedir", nargs='+', required=True)
parser.add_argument("--outdir", type=str, required=False, default='') 
parser.add_argument("--qfm_vol", type=float, required=False, default=None)
parser.add_argument("--ppp", type=int, default=None)
parser.add_argument("--Nt_ma", type=int, default=None)
parser.add_argument("--Nt_coils", type=int, default=None)
parser.add_argument("--num_coils", type=int, default=None)
parser.add_argument("--contNum", type=int, default=None)
parser.add_argument("--nfp", type=int, default=None)
parser.add_argument("--mmax", type=int, default=None)
parser.add_argument("--nmax", type=int, default=None)
parser.add_argument("--ntheta", type=int, default=None)
parser.add_argument("--nphi", type=int, default=None)
parser.add_argument("--min_rad", type=float, default=None)
parser.add_argument("--stellID", type=int, default=0)
parser.add_argument("--MPOL", type=int, default=6) #11 is also a good choice 
parser.add_argument("--saveQFM", action='store_true', required=False, default=False)
parser.add_argument("--oldFormat", action='store_true', required=False, default=False) # Included for backwards compatibility operation in the reload_stell function
parser.add_argument("--noPoincare", action='store_true', required=False, default=False) # These options shut down parts of the code, which run in the given order. 
parser.add_argument("--noMAKEGRID", action='store_true', required=False, default=False)
parser.add_argument("--noVMEC", action='store_true', required=False, default=False)
parser.add_argument("--noCompare", action='store_true', required=False, default=False)
parser.add_argument("--noBoozRun", action='store_true', required=False, default=False)
parser.add_argument("--noBoozProc", action='store_true', required=False, default=False)
args = parser.parse_args() 

def var_assign(load,arg):
    if arg == None:
        try:
            fileToLoad = '{:}.txt'.format(load)
            loaded = np.loadtxt(str(pl.Path(sourcedir).joinpath(fileToLoad)))
            return loaded
        except IOError:
            print('File {:} not found, you must specify this parameter as an argument!'.format(fileToLoad))
            quit()
    else:
        return arg

vmec_MPOL = args.MPOL

for sourceitem in args.sourcedir:
    sourcedir = str(pl.Path.cwd().joinpath(sourceitem).resolve())

    if args.outdir and len(args.sourcedir)==1: #For ease of programming, only allow special output folder when there is one input folder. 
        outdir = str(pl.Path.cwd().joinpath(args.outdir).resolve())
        if not os.path.exists(outdir):
            os.mkdir(outdir)
    else:
        outdir = sourcedir
    
    Nt_ma = int(var_assign('Nt_ma',args.Nt_ma))
    Nt_coils = int(var_assign('Nt_coils',args.Nt_coils))
    num_coils = int(var_assign('num_coils',args.num_coils))
    contNum = int(var_assign('contNum',args.contNum))
    ppp = int(var_assign('ppp',args.ppp))
    volume = var_assign('qfm_volume',args.qfm_vol)
    nfp = int(var_assign('nfp',args.nfp))
    mmax = int(var_assign('mmax',args.mmax))
    nmax = int(var_assign('nmax',args.nmax))
    ntheta = int(var_assign('ntheta',args.ntheta))
    nphi = int(var_assign('nphi',args.nphi))
    min_rad = float(var_assign('min_rad',args.min_rad))
    stellID = args.stellID
    oldFormat = args.oldFormat
   
    old_xopt = []
    try:
        old_vol = np.loadtxt(str(pl.Path(sourcedir).joinpath('qfm_volume.txt')))
        if volume == old_vol:
            try:
                old_xopt = np.loadtxt(str(pl.Path(sourcedir).joinpath('xopt_{:}.txt'.format(args.stellID))))
            except:
                pass
    except:
        pass

    print('Processing {:}, stellID {:}'.format(sourcedir,str(stellID)))

    with open(str(pl.Path(outdir).joinpath('postprocess_source.txt')),'w') as f:
        f.write('{:}\n'.format(sourcedir))
        f.write('StellID: {:}'.format(str(stellID)))

    # Get the QFM surface
    (unique_coils, mas, unique_currents, eta_bar) = reload_stell(sourcedir=sourcedir,ppp=ppp,Nt_ma=Nt_ma,Nt_coils=Nt_coils,nfp=nfp,num_coils=num_coils,contNum=contNum,copies=1,stellID=stellID,oldFormat=oldFormat)
    ma = mas[0] #You should only be loading one stellarator at a time!
    unique_currents = unique_currents[0]
    stellarator = CoilCollection(unique_coils, unique_currents, nfp, True)

    bs = BiotSavart(stellarator.coils, stellarator.currents)
        
    magnetic_axis_radius = np.sum(ma.coefficients[0]) #First group is for R, second is for Z. First series is cosine and we calculate R at phi=0, so we can sum the coefficients to get R.  

    # Initialize parameters, or just load old xopt
    if len(old_xopt)==0:
        print('Using generic initial guess for QFM surface.')
        runs = 1
        while runs < qfm_max_tries:
            qfm = QfmSurface(mmax, nmax, nfp, stellarator, ntheta, nphi, volume, outdir=outdir, stellID=stellID)
            paramsInit = qfm.ParamsInit(ma)
            qfm_full = qfm.DetermineFull(paramsInit)
            
            print('Beginning QFM surface optimization - attempt %d.'%runs)
            try:
                fopt = qfm.qfm_metric(paramsInit=paramsInit,full=qfm_full)
                xopt = qfm.paramsPrev
                success = True
                if args.saveQFM:
                    qfm.SaveState()
                    np.savetxt(str(pl.Path(outdir).joinpath('qfm_volume.txt')),[qfm.volume_target]) #Overwrite old file if this portion of the code runs.
                break
            except RuntimeError:
                print('Optimization for given volume failed.')
                volume = volume/2
                runs += 1 
        if not (success):
            print('QFM surface not found!')
            quit()
    else:
        print('Loading QFM surface from xopt file.')
        qfm = QfmSurface(mmax, nmax, nfp, stellarator, ntheta, nphi, volume, outdir=outdir, stellID=stellID)
        xopt = old_xopt
        qfm_full = qfm.DetermineFull(old_xopt)
    print('Final QFM surface volume: ', volume)
    
    R,Z = qfm.position(xopt,full=qfm_full) # R and Z for the surface over ONE field period
    X,Y = qfm.Cyl_to_Cart(R) # X and Y for the surface over ONE field period

    # Create boundary.txt file
    Rbc, Zbs, xn, xm = qfm.ft_surface(params=xopt,mmax=mmax,nmax=nmax,outdir=outdir,full=qfm_full)

    # Save Poincare plot with QFM surface.
    if not args.noPoincare:
        max_thickness = min_rad 
        
        runs = 1
        while runs < poincare_max_tries: 
            try:
                rphiz, xyz, absB, phi_no_mod = compute_field_lines(bs, nperiods=nperiods, batch_size=batch_size, magnetic_axis_radius=magnetic_axis_radius, max_thickness=max_thickness, delta=delta, steps_per_period=spp) 
                break 
            except ValueError:
                max_thickness = max_thickness/2
                delta = delta/2
                runs += 1
                print('Poincare plotting failed - rerunning with new parameters.')
                continue

        nparticles = rphiz.shape[0]

        data0 = np.zeros((nperiods, nparticles*2))
        data1 = np.zeros((nperiods, nparticles*2))
        data2 = np.zeros((nperiods, nparticles*2))
        data3 = np.zeros((nperiods, nparticles*2))
        for i in range(nparticles):
            data0[:, 2*i+0] = rphiz[i, range(0, nperiods*spp, spp), 0]
            data0[:, 2*i+1] = rphiz[i, range(0, nperiods*spp, spp), 2]
            data1[:, 2*i+0] = rphiz[i, range(1*spp//(nfp*4), nperiods*spp, spp), 0]
            data1[:, 2*i+1] = rphiz[i, range(1*spp//(nfp*4), nperiods*spp, spp), 2]
            data2[:, 2*i+0] = rphiz[i, range(2*spp//(nfp*4), nperiods*spp, spp), 0]
            data2[:, 2*i+1] = rphiz[i, range(2*spp//(nfp*4), nperiods*spp, spp), 2]
            data3[:, 2*i+0] = rphiz[i, range(3*spp//(nfp*4), nperiods*spp, spp), 0]
            data3[:, 2*i+1] = rphiz[i, range(3*spp//(nfp*4), nperiods*spp, spp), 2]

        fig_w_qfm,ax_w_qfm = plt.subplots()
        fig_w_vmec,ax_w_vmec = plt.subplots()
        for i in range(nparticles):
            ax_w_qfm.scatter(rphiz[i, range(0, nperiods*spp, spp), 0], rphiz[i, range(0, nperiods*spp, spp), 2], s=marker_size, marker=marker_symbol, linewidth=1) 
            ax_w_vmec.scatter(rphiz[i, range(0, nperiods*spp, spp), 0], rphiz[i, range(0, nperiods*spp, spp), 2], s=marker_size, marker=marker_symbol, linewidth=1) 
        Ruse = np.append(R,np.reshape(R[:,0],(R.shape[0],1)),axis=1)
        Zuse = np.append(Z,np.reshape(Z[:,0],(Z.shape[0],1)),axis=1)
        ax_w_qfm.plot(Ruse[0,:],Zuse[0,:])
        ax_w_qfm.set_aspect('equal','box') 
        ax_w_qfm.set_xlabel(r'$R$ (m)')
        ax_w_qfm.set_ylabel(r'$Z$ (m)')
        ax_w_vmec.set_aspect('equal','box') 
        ax_w_vmec.set_xlabel(r'$R$ (m)')
        ax_w_vmec.set_ylabel(r'$Z$ (m)')
        fig_w_qfm.savefig(str(pl.Path(outdir).joinpath(poincare_w_qfm_name+'_'+str(stellID)+'.'+image_filetype)),bbox_inches='tight',dpi=400)
        print('Poincare plots created.')

    # Load in coil and current information 
    Ncoils = (num_coils+contNum)*nfp*2 #Not true in general, but fine in our case
    currents = []
    names = []
    groups = []
    xx = [[]]; yy = [[]]; zz = [[]]
    for icoil in range(Ncoils):
        filename = str(pl.Path(sourcedir).joinpath('current-'+str(icoil)+'_'+str(stellID)+'.txt')) 
        xx.append([]); yy.append([]); zz.append([])
        if not os.path.exists(filename) :
            raise IOError ("File does not exist. Please check again!")
        with open(filename,'r') as currentfile:
            currents.append(float(currentfile.readline()))
        filename = str(pl.Path(sourcedir).joinpath('coil-'+str(icoil)+'.txt'))
        with open(filename,'r') as coilfile:
            if not os.path.exists(filename) :
                raise IOError ("File does not exist. Please check again!")
            for line in coilfile:
                linelist = line.split()
                xx[icoil].append(float(linelist[0]))
                yy[icoil].append(float(linelist[1]))
                zz[icoil].append(float(linelist[2]))
    xx.pop()
    yy.pop()
    zz.pop()
                
    for icoil in range(Ncoils):
        groups.append(icoil % int(Ncoils/(2*nfp)))
        names.append('Mod_'+str(groups[icoil]))
    
    coilObject = coils.Coil(xx,yy,zz,currents,names,groups)

    # Make the coils.* file for VMEC
    coils_file_name = 'coils.' + coils_file_suffix
    coilObject.save_makegrid(str(pl.Path(outdir).joinpath(coils_file_name)),nfp=nfp)

    # Make the infile for MAKEGRID 
    def extend(arr, factr=0.1):
        smaller = np.min(arr)
        larger = np.max(arr)

        diff = larger - smaller
        add = factr * diff
        
        newSmaller = smaller - add
        newLarger = larger + add

        return newSmaller, newLarger

    R_min, R_max = extend(R)
    Z_min, Z_max = extend(Z)

    with open(str(pl.Path(outdir).joinpath(makegrid_input_file)),'w') as f:
        f.write(coils_file_suffix+' ! coils.* file suffix\n')
        f.write('{:} ! Current format (S/R : Scale current to unit current or use raw currents, old version did not prompt the user for this and assumed scaled current)\n'.format(current_format)) 
        f.write('{:} ! Stellarator Symmetry (y/n)\n'.format(stellarator_symmetry))
        f.write('{:} ! Minimum radial extent (Rmin)\n'.format(str(R_min)))
        f.write('{:} ! Maximum radial extent (Rmax)\n'.format(str(R_max)))
        f.write('{:} ! Minimum vertical extent (Zmin)\n'.format(str(Z_min)))
        f.write('{:} ! Maximum vertical extent (Zmax)\n'.format(str(Z_max)))
        f.write('{:} ! Number of toroidal cutplanes (per half period if stellarator symmetry is assumed)\n'.format(str(toroidal_cutplanes)))
        f.write('{:} ! Number of radial grid points\n'.format(str(radial_grid_points)))
        f.write('{:} ! Number of vertical grid points\n'.format(str(vertical_grid_points)))

    # Run MAKEGRID
    if not args.noMAKEGRID:
        cmd = 'cd '+outdir+' && xgrid < '+makegrid_input_file+' | tee '+makegrid_output_file
        process = sp.Popen(cmd, stdout=sp.PIPE, shell=True)
        makegrid_output, makegrid_error = process.communicate()
        print('MAKEGRID ran.')

    # Get estimate for VMEC's PHIEDGE parameter 
    phi_edge = qfm.toroidal_flux(np.concatenate((Rbc,Zbs)),full=qfm_full)

    # Write the VMEC input.* script
    def list2str(invec):
        outstr = ''
        for ind,item in enumerate(invec):
            outstr += str(item)
            if ind != len(invec)-1:
                outstr += ' '
        return outstr

    extcur = list2str(unique_currents)
    r_axis = list2str(ma.coefficients[0])
    z_axis = list2str(-1*ma.coefficients[1]) #Note that the 'list items' of ma.coefficients are actually stored as numpy arrays, which is why this works. The -1 is due to a sign convention difference between PyPlasmaopt and VMEC.  

    with open(str(pl.Path(outdir).joinpath('input.'+coils_file_suffix)),'w') as f:
        f.write('&INDATA\n')
        f.write('!----- Runtime Parameters ----------\n')
        f.write('  DELT       = {:}\n'.format(str(vmec_DELT)))  
        f.write('  NITER      = {:}\n'.format(str(vmec_NITER)))
        f.write('  NSTEP      = {:}\n'.format(str(vmec_NSTEP)))
        f.write('  TCON0      = {:}\n'.format(str(vmec_TCON0)))
        f.write('  NS_ARRAY   = {:}\n'.format(vmec_NS_ARRAY))
        f.write('  FTOL_ARRAY = {:}\n'.format(vmec_FTOL_ARRAY))
        f.write('!----- Grid Parameters -------------\n')
        f.write('  LASYM      = {:}\n'.format(vmec_LASYM))
        f.write('  NFP        = {:}\n'.format(str(nfp)))
        f.write('  MPOL       = {:}\n'.format(str(vmec_MPOL)))
        f.write('  NTOR       = {:}\n'.format(str(vmec_NTOR)))
        f.write('  NZETA      = {:}\n'.format(str(toroidal_cutplanes)))
        f.write('  PHIEDGE    = {:}\n'.format(str(phi_edge)))
        f.write('!----- Free Boundary Parameters ----\n')
        f.write('  LFREEB     = {:}\n'.format(str(vmec_LFREEB)))
        f.write("  MGRID_FILE = 'mgrid_{:}.nc'\n".format(coils_file_suffix))
        f.write('  NVACSKIP   = {:}\n'.format(str(vmec_NVACSKIP)))
        f.write('  extcur     = {:}\n'.format(extcur)) 
        f.write('!----- Pressure Parameters ---------\n')
        #f.write('  GAMMA      = 0.000000E+00\n')
        #f.write('  BLOAT      = 1.000000E+00\n')
        #f.write('  SPRES_PED  = 1.00000000000000E+00\n')
        f.write('  AM         = {:}\n'.format(str(vmec_AM)))
        f.write('!----- Current/Iota Parameters -----\n')
        f.write('  CURTOR     = {:}\n'.format(str(vmec_CURTOR)))
        f.write('  NCURR      = {:}\n'.format(str(vmec_NCURR)))
        f.write('  AC         = {:}\n'.format(str(vmec_AC)))
        f.write('!----- Axis Parameters -------------\n')
        f.write('  RAXIS      = {:}\n'.format(r_axis))
        f.write('  ZAXIS      = {:}\n'.format(z_axis))
        f.write('!----- Boundary Parameters ---------\n')
        for im in range(qfm.mnmax):
            f.write('rbc(%d,%d) = %0.12f\n' % (xn[im]/nfp,xm[im],Rbc[im]))
            f.write('zbs(%d,%d) = %0.12f\n' % (xn[im]/nfp,xm[im],Zbs[im]))
        f.write('/\n')
        f.write('&END\n')

    # Run VMEC
    if not args.noVMEC:
        VMEC_input_file = 'input.'+coils_file_suffix 
        cmd = 'cd '+outdir+' && xvmec2000 '+VMEC_input_file+' | tee '+ vmec_output_file
        process = sp.Popen(cmd, stdout=sp.PIPE, shell=True)
        VMEC_output, VMEC_error = process.communicate()
        print('VMEC ran.')

    # Compare iota from VMEC with iota from PyPlasmaOpt
    if not args.noCompare:
        wout_filename = str(pl.Path(outdir).joinpath('wout_'+coils_file_suffix+'.nc'))
        vmecOutput = VmecOutput(wout_filename)

        iota_half = vmecOutput.iota
        iota_full = np.zeros(vmecOutput.ns_half)
        iota_full[0:-1] = (vmecOutput.iota[0:-1]+vmecOutput.iota[1::])*0.5
        iota_full[-1] = 1.5*vmecOutput.iota[-1]-0.5*vmecOutput.iota[-2]

        V_vmec = 4*np.pi**2/vmecOutput.ns_half*np.cumsum(vmecOutput.vp) #FIXME

        iota_pyplasmaopt = -1*np.loadtxt(str(pl.Path(sourcedir).joinpath('iota_ma_%d.txt'%stellID))) #VMEC and PyPlasmaOpt define iota differently, hence the negative sign.

        volume_xlabel = r'Plasma Volume $\mathrm{\left(m^{3}\right)}$'

        plt.figure()
        #plt.plot(vmecOutput.s_full[1::],iota_full) #FIXME
        plt.plot(V_vmec,iota_full) #FIXME
        plt.axhline(iota_pyplasmaopt,linestyle='--')
        #plt.xlabel(r'$\Psi_T/\Psi_T^{\mathrm{edge}}$') #FIXME
        plt.xlabel(volume_xlabel) #FIXME
        plt.ylabel(r'$\iota$')
        plt.legend(['VMEC','PyPlasmaOpt'])
        plt.savefig(str(pl.Path(outdir).joinpath(iota_compare_fig_name+'_'+str(stellID)+'.'+image_filetype)),bbox_inches='tight')

        index = np.argmin((iota_full-iota_pyplasmaopt)**2)
        VMEC_iota = iota_full[index]
        print('iota from VMEC = '+str(VMEC_iota))
        print('iota from pyplasmaopt = '+str(iota_pyplasmaopt))

    # Compare plasma profiles between VMEC and PyPlasmaOpt
    if not args.noCompare:
        nphi = len(R[:,0])
        ntheta = nphi
        phi_grid = np.linspace(0,2*np.pi/nfp,nphi+1)
        phi_grid = np.delete(phi_grid,-1,0)
        theta_grid = np.linspace(0,2*np.pi,ntheta+1)
        theta_grid = np.delete(theta_grid,-1,0)
        [phi_2d,theta_2d] = np.meshgrid(phi_grid,theta_grid)

        [X_vmec,Y_vmec,Z_vmec,R_vmec] = vmecOutput.position(isurf=-1,theta=theta_2d,zeta=phi_2d)
        
        R_switch = R_vmec.T
        Z_switch = Z_vmec.T
        Ruse = np.append(R_switch,np.reshape(R_switch[:,0],(R_switch.shape[0],1)),axis=1)
        Zuse = np.append(Z_switch,np.reshape(Z_switch[:,0],(Z_switch.shape[0],1)),axis=1)
        
        ax_w_vmec.plot(Ruse[0,:],Zuse[0,:])
        fig_w_vmec.savefig(str(pl.Path(outdir).joinpath(poincare_w_vmec_name+'_'+str(stellID)+'.'+image_filetype)),bbox_inches='tight',dpi=400)

        ax_w_qfm.plot(Ruse[0,:],Zuse[0,:])
        fig_w_qfm.savefig(str(pl.Path(outdir).joinpath(poincare_w_both_name+'_'+str(stellID)+'.'+image_filetype)),bbox_inches='tight',dpi=400)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X,Y,Z)
        plt.title(coils_file_suffix)
        plt.savefig(str(pl.Path(outdir).joinpath(profile_compare_name+'_'+coils_file_suffix+'_'+str(stellID)+'.'+image_filetype)),bbox_inches='tight')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X_vmec,Y_vmec,Z_vmec)
        plt.title('VMEC')
        plt.savefig(str(pl.Path(outdir).joinpath(profile_compare_name+'_VMEC'+'_'+str(stellID)+'.'+image_filetype)),bbox_inches='tight')

        figind = 1
        for iphi in range(0,nphi,profile_compare_ind_freq):
            fig = plt.figure()
            plt.plot(R[iphi,:],Z[iphi,:]) 
            plt.plot(R_vmec[:,iphi],Z_vmec[:,iphi])
            plt.xlabel(r'$R$ (m)')
            plt.ylabel(r'$Z$ (m)')
            plt.legend(['PyPlasmaOpt','VMEC'])
            plt.savefig(str(pl.Path(outdir).joinpath(profile_compare_name+str(figind)+'_'+str(stellID)+'.'+image_filetype)),bbox_inches='tight')
            plt.close('all')
            figind += 1

    # Make BOOZXFORM input file
    booz_input_file = 'in_booz.'+coils_file_suffix

    with open(str(pl.Path(outdir).joinpath(booz_input_file)),'w') as f:
        f.write('{:} {:} ! number of toroidal and poloidal harmonics\n'.format(booz_toroidal_harmonics,booz_poloidal_harmonics))
        f.write("'{:}' ! file extension of wout_*.nc filename\n".format(coils_file_suffix))
        f.write('{:} ! index of surface on which to compute transform\n'.format(booz_surface_indices))

    # Run BOOZXFORM
    if not args.noBoozRun:
        cmd = 'cd '+outdir+' && xbooz_xform '+booz_input_file+' | tee '+booz_output_file
        process = sp.Popen(cmd, stdout=sp.PIPE, shell=True)
        booz_output, booz_error = process.communicate()
        print('BOOZXFORM ran.')

    # Process the BOOZXFORM outputs
    if not args.noBoozRun:
        filename = str(pl.Path(outdir).joinpath('boozmn_'+coils_file_suffix+'.nc'))

        f = netcdf.netcdf_file(filename,mode='r',mmap=False)
        phi_b = f.variables['phi_b'][()]
        #ns_b = f.variables['ns_b'][()] #FIXME
        nfp_b = f.variables['nfp_b'][()]
        ixn_b = f.variables['ixn_b'][()]
        ixm_b = f.variables['ixm_b'][()]
        bmnc_b = f.variables['bmnc_b'][()]
        jlist = f.variables['jlist'][()]
        f.close()

        nmodes = len(ixn_b)

        fig = plt.figure()

        #s = (jlist-1.5)/(ns_b-1.0) #FIXME

        V_vmec_booz = [V_vmec[i-2] for i in jlist] #FIXME this is almost surely wrong...

        backgroundColor='b'
        QAColor=[0,0.7,0]
        mirrorColor=[0.7,0.5,0]
        helicalColor=[1,0,1]

        scale_factor = np.max(np.abs(bmnc_b))

        for imode in range(nmodes): # First, plot just the 1st mode of each type, so the legend looks nice.
            if ixn_b[imode]==0 and ixm_b[imode]==0:
                #plt.semilogy(s,abs(bmnc_b[:,imode])/scale_factor, color=backgroundColor,label=r'$m = 0, n = 0$ (Background)') #FIXME
                plt.semilogy(V_vmec_booz,abs(bmnc_b[:,imode])/scale_factor, color=backgroundColor,label=r'$m = 0, n = 0$ (Background)') #FIXME
                break
        for imode in range(nmodes):
            if ixn_b[imode]==0 and ixm_b[imode]!=0:
                #plt.semilogy(s,abs(bmnc_b[:,imode])/scale_factor, color=QAColor,label=r'$m \ne 0, n = 0$ (Quasiaxisymmetric)') #FIXME
                plt.semilogy(V_vmec_booz,abs(bmnc_b[:,imode])/scale_factor, color=QAColor,label=r'$m \ne 0, n = 0$ (Quasiaxisymmetric)') #FIXME
                break
        for imode in range(nmodes):
            if ixn_b[imode]!=0 and ixm_b[imode]==0:
                #plt.semilogy(s,abs(bmnc_b[:,imode])/scale_factor, color=mirrorColor,label=r'$m = 0, n \ne 0$ (Mirror)') #FIXME
                plt.semilogy(V_vmec_booz,abs(bmnc_b[:,imode])/scale_factor, color=mirrorColor,label=r'$m = 0, n \ne 0$ (Mirror)') #FIXME
                break
        for imode in range(nmodes):
            if ixn_b[imode]!=0 and ixm_b[imode]!=0:
                #plt.semilogy(s,abs(bmnc_b[:,imode])/scale_factor, color=helicalColor,label=r'$m \ne 0, n \ne 0$ (Helical)') #FIXME
                plt.semilogy(V_vmec_booz,abs(bmnc_b[:,imode])/scale_factor, color=helicalColor,label=r'$m \ne 0, n \ne 0$ (Helical)') #FIXME
                break
        plt.legend(fontsize=9,loc='best')
        for imode in range(nmodes):
            if np.abs(ixm_b[imode]) > max_m:
                continue
            if np.abs(ixn_b[imode]) > max_n * nfp_b:
                continue
            if ixn_b[imode]==0:
                if ixm_b[imode]==0:
                    mycolor = backgroundColor
                else:
                    mycolor = QAColor
            else:
                if ixm_b[imode]==0:
                    mycolor = mirrorColor
                else:
                    mycolor = helicalColor
            #plt.semilogy(s,abs(bmnc_b[:,imode])/scale_factor, color=mycolor) #FIXME
            plt.semilogy(V_vmec_booz,abs(bmnc_b[:,imode])/scale_factor, color=mycolor) #FIXME

        #plt.xlabel(r'$\Psi_T/\Psi_T^{\mathrm{edge}}$') #FIXME
        plt.xlabel(volume_xlabel) #FIXME
        plt.title(r'Fourier Harmonics of $|B|$ in Boozer Coordinates')

        plt.savefig(str(pl.Path(outdir).joinpath(booz_harmonicsplot_name+'_'+str(stellID)+'.'+image_filetype)),bbox_inches='tight')

        plt.figure()
        QA_metric = np.zeros(len(jlist))
        for index in range(len(jlist)):
            summed_total = 0
            summed_nonQA = 0
            for imode in range(nmodes):
                if ixn_b[imode]!=0:
                    summed_nonQA += bmnc_b[index,imode]**2
                summed_total += bmnc_b[index,imode]**2
            # Normalize by total sum
            QA_metric[index] = np.sqrt(summed_nonQA/summed_total)
            
        #plt.plot(s,QA_metric,marker='o') #FIXME
        plt.semilogy(V_vmec_booz,QA_metric,marker='o') #FIXME
        #plt.xlabel(r'$\Psi_T/\Psi_T^{\mathrm{edge}}$') #FIXME
        plt.xlabel(volume_xlabel) #FIXME
        plt.ylabel('QA Metric')
        #plt.ylim(bottom=0) #FIXME

        plt.savefig(str(pl.Path(outdir).joinpath(booz_QAplot_name+'_'+str(stellID)+'.'+image_filetype)),bbox_inches='tight')
