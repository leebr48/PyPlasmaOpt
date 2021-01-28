"""
This script is used to compute a quadratic flux minimizing surface with 
a given volume (default 1.0) for the NCSX coils, as well as information
about the coils. This all is then fed into VMEC. 
"""
from pyplasmaopt import *
import numpy as np
from scipy.io import netcdf
from pyplasmaopt.grad_optimizer import GradOptimizer
from pyplasmaopt.qfm_surface import QfmSurface
import argparse 
import os #FIXME - can you turn everything into pathlib? Prolly not, because of running scripts...
import pathlib as pl #FIXME - you might not need this, check! 
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #FIXME - might be able to get rid of this - are you plotting anything 3d in the final version?
from pyplasmaopt.poincareplot import compute_field_lines
from coilpy import * 
import subprocess as sp
sys.path.append('/home/leebr48/ALPOpt/')
from vmec_input import init_modes 
from vmec_output import VmecOutput
#FIXME - this program should be able to take a vector input for sourcedir
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--sourcedir", type=str, required=True)
parser.add_argument("--outdir", type=str, required=False, default='') 
parser.add_argument("--volume", type=float, required=False, default=1.0) # target volume
parser.add_argument("--ppp", type=int, default=20)
parser.add_argument("--Nt_ma", type=int, default=6)
parser.add_argument("--Nt_coils", type=int, default=6)
parser.add_argument("--num_coils", type=int, default=3)
parser.add_argument("--nfp", type=int, default=3)
parser.add_argument("--mmax", type=int, default=3) # maximum poloidal mode number for surface
parser.add_argument("--nmax", type=int, default=3) # maximum toroidal mode number for surface
parser.add_argument("--ntheta", type=int, default=20) # number of poloidal grid points for integration
parser.add_argument("--nphi", type=int, default=20) # number of toroidal grid points for integration
args, _ = parser.parse_known_args()

sourcedir = str(pl.Path.cwd().joinpath(args.sourcedir).resolve())

if args.outdir:
    outdir = str(pl.Path.cwd().joinpath(args.outdir).resolve())
    if not os.path.exists(outdir):
        os.mkdir(outdir)
else:
    outdir = sourcedir

Nt_ma = args.Nt_ma
Nt_coils = args.Nt_coils
num_coils = args.num_coils
ppp = args.ppp
volume = args.volume
nfp = args.nfp
mmax = args.mmax
nmax = args.nmax
ntheta = args.ntheta
nphi = args.nphi

#(unique_coils, ma, unique_currents) = get_ncsx_data(Nt_ma=Nt_ma, Nt_coils=Nt_coils, ppp=ppp)
(unique_coils, ma, unique_currents, eta_bar) = reload_ncsx(sourcedir=sourcedir,ppp=ppp,Nt_ma=Nt_ma,Nt_coils=Nt_coils,nfp=nfp,num_coils=num_coils)
stellarator = CoilCollection(unique_coils, unique_currents, nfp, True)
bs = BiotSavart(stellarator.coils, stellarator.currents)

qfm = QfmSurface(mmax, nmax, nfp, bs, ntheta, nphi, volume)

objective = qfm.quadratic_flux
d_objective = qfm.d_quadratic_flux

# Initialize parameters - circular cross section torus
paramsInitR = np.zeros((qfm.mnmax))
paramsInitZ = np.zeros((qfm.mnmax))

paramsInitR[(qfm.xm==1)*(qfm.xn==0)]=0.188077
paramsInitZ[(qfm.xm==1)*(qfm.xn==0)]=-0.188077

paramsInit = np.hstack((paramsInitR[1::],paramsInitZ))

optimizer = GradOptimizer(nparameters=len(paramsInit),outdir=outdir)
optimizer.add_objective(objective,d_objective,1)
(xopt, fopt, result) = optimizer.optimize(paramsInit,package='scipy',method='BFGS') 
R,Z = qfm.position(xopt) # R and Z for the surface

# Create boundary.txt file
Rbc, Zbs, xn, xm = qfm.ft_surface(params=xopt,mmax=mmax,nmax=nmax,outdir=outdir)

# Save Poincare plot with QFM surface.
magnetic_axis_radius=np.sum(R[0,:])/np.size(R[0,:])
nperiods = 200 #FIXME - should this be a changeable option?
spp = 120 #FIXME - should this be a changeable option?
rphiz, xyz, absB, phi_no_mod = compute_field_lines(bs, nperiods=nperiods, batch_size=4, magnetic_axis_radius=magnetic_axis_radius, max_thickness=0.9, delta=0.01, steps_per_period=spp)
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
    
plt.figure()
for i in range(nparticles):
    plt.scatter(rphiz[i, range(0, nperiods*spp, spp), 0], rphiz[i, range(0, nperiods*spp, spp), 2], s=0.1)
    plt.plot(R[0,:],Z[0,:])
plt.savefig(os.path.join(outdir,'poincare_w_QFM.png'),bbox_inches='tight') #FIXME - image name can be moved to the header
print('Poincare plot created.')

# Load in coil and current information 
Ncoils = num_coils*nfp*2 #FIXME?
currents = []
names = []
groups = []
xx = [[]]; yy = [[]]; zz = [[]]
for icoil in range(Ncoils):
    filename = os.path.join(sourcedir,'current-'+str(icoil)+'.txt')
    xx.append([]); yy.append([]); zz.append([])
    if not os.path.exists(filename) :
        raise IOError ("File not existed. Please check again!")
    with open(filename,'r') as currentfile:
        currents.append(float(currentfile.readline()))
    filename = os.path.join(sourcedir,'coil-'+str(icoil)+'.txt')   
    with open(filename,'r') as coilfile:
        if not os.path.exists(filename) :
            raise IOError ("File not existed. Please check again!")
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

# Save the coils.* file and get some other necessary info for VMEC
coils_file_suffix = 'pyplasmaopt'
coils_file_name = 'coils.' + coils_file_suffix
coilObject.save_makegrid(os.path.join(outdir,coils_file_name),nfp=nfp)

R_arr = np.sqrt(np.array(xx)**2 + np.array(yy)**2)
Z_arr = np.array(zz)
R_min = np.min(R_arr)
R_max = np.max(R_arr)
Z_min = np.min(Z_arr)
Z_max = np.max(Z_arr)

makegrid_input_file = 'input_xgrid.dat' #FIXME - when ~finished you can move all these names up to the top of the code. 
toroidal_cutplanes = 36
with open(os.path.join(outdir,makegrid_input_file),'w') as f: #FIXME - does any of this need to be generalized?
    f.write(coils_file_suffix+' ! coils.* file suffix\n')
    f.write('s'+' ! Current format (S/R : Scale current to unit current or use raw currents, old version did not prompt the user for this and assumed scaled current)\n') 
    f.write('y'+' ! Stellarator Symmetry (y/n)\n')
    f.write(str(R_min)+' ! Minimum radial extent (Rmin)\n')
    f.write(str(R_max)+' ! Maximum radial extent (Rmax)\n')
    f.write(str(Z_min)+' ! Minimum vertical extent (Zmin)\n')
    f.write(str(Z_max)+' ! Maximum vertical extent (Zmax)\n')
    f.write(str(toroidal_cutplanes)+' ! Number of toroidal cutplanes (per half period if stellarator symmetry is assumed)\n')
    f.write('201'+' ! Number of radial grid points\n')
    f.write('201'+' ! Number of vertical grid points')

# Run MAKEGRID
makegrid_output_file = 'makegrid_output.txt'
cmd = 'cd '+outdir+' && xgrid < '+makegrid_input_file+' | tee '+makegrid_output_file
process = sp.Popen(cmd, stdout=sp.PIPE, shell=True)
makegrid_output, makegrid_error = process.communicate()
print('MAKEGRID ran.')

# Get estimate for VMEC's PHIEDGE parameter 
phi_edge = qfm.toroidal_flux(np.concatenate((Rbc[1:],Zbs))) #Looks odd, but this is how the function takes inputs.

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
z_axis = list2str(-1*ma.coefficients[1]) #Note that the 'list items' of ma.coefficients are actually stored as numpy arrays, which is why this works. 

with open(str(pl.Path(outdir).joinpath('input.'+coils_file_suffix)),'w') as f:
    f.write('&INDATA\n')
    f.write('!----- Runtime Parameters ----------\n')
    f.write('  DELT       = 9.00E-01\n') #NOTE: This is typically 0.7-0.9 and may have to be tweaked to make runs converge properly. 
    f.write('  NITER      = 5000\n')
    f.write('  NSTEP      = 200\n')
    f.write('  TCON0      = 2.00E+00\n')
    f.write('  NS_ARRAY   = 9 29 49 99\n')
    f.write('  FTOL_ARRAY = 1.000000E-06 1.000000E-08 1.000000E-10 1.000000E-12\n')
    f.write('!----- Grid Parameters -------------\n')
    f.write('  LASYM      = F\n')
    f.write('  NFP        = {:}\n'.format(str(nfp)))
    f.write('  MPOL       = 11\n')
    f.write('  NTOR       = 6\n')
    f.write('  NZETA      = {:}\n'.format(str(toroidal_cutplanes)))
    f.write('  PHIEDGE    = {:}\n'.format(str(phi_edge)))
    f.write('!----- Free Boundary Parameters ----\n')
    f.write('  LFREEB     = T\n')
    f.write("  MGRID_FILE = 'mgrid_{:}.nc'\n".format(coils_file_suffix))
    f.write('  NVACSKIP   = 6\n')
    f.write('  extcur     = {:}\n'.format(extcur)) 
    f.write('!----- Pressure Parameters ---------\n')
    #f.write('  GAMMA     = 0.000000E+00\n')
    #f.write('  BLOAT     = 1.000000E+00\n')
    #f.write('  SPRES_PED = 1.00000000000000E+00\n')
    f.write('  AM         = 0\n')
    f.write('!----- Current/Iota Parameters -----\n')
    f.write('  CURTOR     = 0\n')
    f.write('  NCURR      = 1\n')
    f.write('  AC         = 0\n')
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
VMEC_output_file = 'VMEC_output.txt'
VMEC_input_file = 'input.'+coils_file_suffix #FIXME - this name should stay here when you are moving them all up to the header. 
cmd = 'cd '+outdir+' && xvmec2000 '+VMEC_input_file+' | tee '+VMEC_output_file
process = sp.Popen(cmd, stdout=sp.PIPE, shell=True)
VMEC_output, VMEC_error = process.communicate()
print('VMEC ran.')

# Compare iota from VMEC with iota from PyPlasmaOpt
wout_filename = str(pl.Path(outdir).joinpath('wout_'+coils_file_suffix+'.nc')) #FIXME - this should stay here when moving stuff to header
vmecOutput = VmecOutput(wout_filename)

iota_half = vmecOutput.iota
iota_full = np.zeros(vmecOutput.ns_half)
iota_full[0:-1] = (vmecOutput.iota[0:-1]+vmecOutput.iota[1::])*0.5
iota_full[-1] = 1.5*vmecOutput.iota[-1]-0.5*vmecOutput.iota[-2]

iota_pyplasmaopt = np.loadtxt(str(pl.Path(sourcedir).joinpath('iota_ma.txt')))

plt.figure()
plt.plot(vmecOutput.s_full[1::],iota_full)
plt.axhline(iota_pyplasmaopt,linestyle='--')
plt.xlabel('$\Psi_T/\Psi_T^{\mathrm{edge}}$')
plt.ylabel('$\iota$')
plt.legend(['VMEC','pyplasmaopt'])
plt.savefig(str(pl.Path(outdir).joinpath('iota_compare.png')),bbox_inches='tight') #FIXME - image name can be moved to the header

index = np.argmin((iota_full-iota_pyplasmaopt)**2)
VMEC_iota = iota_full[index]
print('iota from VMEC = '+str(VMEC_iota))
print('iota from pyplasmaopt = '+str(iota_pyplasmaopt))
index += 1 # need to add 1 to account for axis #FIXME - might be able to get rid of this - check below

#FIXME - you will need to change some stuff around to make this section of the code work if need be
# Compare plasma profiles between VMEC and PyPlasmaOpt
nphi = len(R[:,0]) # R is square
ntheta = nphi
phi_grid = np.linspace(0,2*np.pi,nphi+1)
phi_grid = np.delete(phi_grid,-1,0)
theta_grid = np.linspace(0,2*np.pi,ntheta+1)
theta_grid = np.delete(theta_grid,-1,0)
[phi_2d,theta_2d] = np.meshgrid(phi_grid,theta_grid)

[X_vmec,Y_vmec,Z_vmec,R_vmec] = vmecOutput.position(isurf=-1,theta=theta_2d,zeta=phi_2d)

figind = 1
profile_compare_name = 'profile_compare'#FIXME - image name can be moved to the header
'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_interp,Y_interp,Z_interp)
plt.title(coils_file_suffix)
plt.savefig(str(pl.Path(outdir).joinpath(profile_compare_name+str(figind)+'.png')),bbox_inches='tight')
figind += 1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_vmec,Y_vmec,Z_vmec)
plt.title('vmec')
plt.savefig(str(pl.Path(outdir).joinpath(profile_compare_name+str(figind)+'.png')),bbox_inches='tight')
figind += 1
'''
Rflip = np.flip(R_vmec.T,axis=0)
Zflip = np.flip(Z_vmec.T,axis=0)
profile_compare_ind_freq = 2 #FIXME - you can move this to the header
for iphi in range(0,nphi,profile_compare_ind_freq):
    fig = plt.figure()
    plt.plot(R[iphi,:],Z[iphi,:]) #FIXME - fix negatives depending on how VMEC treats phi 
    plt.plot(R_vmec[:,iphi],Z_vmec[:,iphi])
    plt.xlabel('R')
    plt.ylabel('Z')
    plt.legend(['pyplasmaopt','VMEC'])
    #plt.savefig(str(pl.Path(outdir).joinpath(profile_compare_name+str(figind)+'.png')),bbox_inches='tight')
    plt.savefig(str(pl.Path(outdir).joinpath(profile_compare_name+str(iphi)+'.png')),bbox_inches='tight')
    plt.close('all')
    figind += 1

# Prep to call BOOZXFORM
booz_input_file = 'in_booz.'+coils_file_suffix #FIXME - keep this here when moving stuff to header
booz_output_file = 'booz_output.txt'

with open(str(pl.Path(outdir).joinpath(booz_input_file)),'w') as f: #FIXME - does any of this need to be generalized?
    f.write('51 51 ! number of toroidal and poloidal harmonics\n')
    f.write("'{:}' ! file extension of wout_*.nc filename\n".format(coils_file_suffix))
    f.write('1 25 50 75 99 ! index of surface on which to compute transform\n')

cmd = 'cd '+outdir+' && xbooz_xform '+booz_input_file+' | tee '+booz_output_file
process = sp.Popen(cmd, stdout=sp.PIPE, shell=True)
booz_output, booz_error = process.communicate()
print('BOOZXFORM ran.')

# Process the BOOZXFORM outputs
max_m = 10 # maximum poloidal mode number to plot
max_n = 10 # maximum toroidal mode number to plot
filename = str(pl.Path(outdir).joinpath('boozmn_'+coils_file_suffix+'.nc'))

f = netcdf.netcdf_file(filename,mode='r',mmap=False)
phi_b = f.variables['phi_b'][()]
ns_b = f.variables['ns_b'][()]
nfp_b = f.variables['nfp_b'][()]
ixn_b = f.variables['ixn_b'][()]
ixm_b = f.variables['ixm_b'][()]
bmnc_b = f.variables['bmnc_b'][()]
jlist = f.variables['jlist'][()]
f.close()

nmodes = len(ixn_b)

fig = plt.figure()

s = (jlist-1.5)/(ns_b-1.0)

backgroundColor='b'
QAColor=[0,0.7,0]
mirrorColor=[0.7,0.5,0]
helicalColor=[1,0,1]

scale_factor = np.max(np.abs(bmnc_b))

for imode in range(nmodes): # First, plot just the 1st mode of each type, so the legend looks nice.
    if ixn_b[imode]==0 and ixm_b[imode]==0:
        plt.semilogy(s,abs(bmnc_b[:,imode])/scale_factor, color=backgroundColor,label='m = 0, n = 0 (Background)')
        break
for imode in range(nmodes):
    if ixn_b[imode]==0 and ixm_b[imode]!=0:
        plt.semilogy(s,abs(bmnc_b[:,imode])/scale_factor, color=QAColor,label=r'm $\ne$ 0, n = 0 (Quasiaxisymmetric)')
        break
for imode in range(nmodes):
    if ixn_b[imode]!=0 and ixm_b[imode]==0:
        plt.semilogy(s,abs(bmnc_b[:,imode])/scale_factor, color=mirrorColor,label=r'm = 0, n $\ne$ 0 (Mirror)')
        break
for imode in range(nmodes):
    if ixn_b[imode]!=0 and ixm_b[imode]!=0:
        plt.semilogy(s,abs(bmnc_b[:,imode])/scale_factor, color=helicalColor,label=r'm $\ne$ 0, n $\ne$ 0 (Helical)')
        break
plt.legend(fontsize=9,loc=2)
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
    plt.semilogy(s,abs(bmnc_b[:,imode])/scale_factor, color=mycolor)

plt.xlabel('Normalized toroidal flux')
plt.title('Fourier harmonics of |B| in Boozer coordinates')

booz_harmonicsplot_name = 'booz_harmonics_plot.png'
plt.savefig(str(pl.Path(outdir).joinpath(booz_harmonicsplot_name)),bbox_inches='tight')

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
    
plt.plot(s,QA_metric)
plt.xlabel('s')
plt.ylabel('QA metric')

booz_QAplot_name = 'booz_QA_plot.png'
plt.savefig(str(pl.Path(outdir).joinpath(booz_QAplot_name)),bbox_inches='tight')

#FIXME - get rid of os.path.join in all your scripts! 
#FIXME - you should be able to disable parts of this code to speed stuff up.  
