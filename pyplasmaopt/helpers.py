import os
import fnmatch
import numpy as np
from .curve import CartesianFourierCurve, StellaratorSymmetricCylindricalFourierCurve, ControlCoil

def coil_spacing(num_coils,nfp):
    field_period_angle = 2*np.pi/nfp #The angle occupied by each field period.
    total_coils = 2*nfp*num_coils #Total number of coils in device assuming stellarator symmetry. 
    shift = np.pi/total_coils #Half the angle between each coil in the device. This is the angle between the first coil in a field period and the \ 
            # beginning of the field period itself, as well as the angle between the last field period and the end of the field period itself.  
    phi_vals = np.linspace(shift, field_period_angle-shift, 2*num_coils, endpoint=True) #This gets the proper angles of each coil in the field period.  
    phi_vals = phi_vals[:len(phi_vals)//2] #Due to stellarator symmetry, we only need the first half of the list - the other coils are generated \
            # using stellaratory symmetry downstream. 
    return phi_vals,total_coils

def interp(gammaax,phieval):
    xax = gammaax[:,0]
    yax = gammaax[:,1]
    zax = gammaax[:,2]
    phiax = np.arctan(yax/xax)
    Rax = (xax**2+yax**2)**(1/2)
    Reval = np.interp(phieval,phiax,Rax)
    zeval = np.interp(phieval,phiax,zax)
    return (Reval,zeval)

def count_files(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    out = len(result)
    return out

def get_ncsx_data(Nt_coils=25, Nt_ma=25, ppp=10, copies=1, contNum=0, contRad=0.5):
    assert (contNum==0 or contNum==3), 'NCSX was designed with 3 toroidal field control coils, and B!=1 on the axis if you use anything other than 0 or 3'

    dir_path = os.path.dirname(os.path.realpath(__file__))

    coil_data = np.loadtxt(os.path.join(dir_path, "data", "NCSX_coil_coeffs.dat"), delimiter=',')
    nfp = 3
    num_coils = 3
    points = np.linspace(0, 1, Nt_coils*ppp, endpoint=False)
    coils = [CartesianFourierCurve(Nt_coils, points) for i in range(num_coils)]
    for ic in range(num_coils):
        coils[ic].coefficients[0][0] = coil_data[0, 6*ic + 1]
        coils[ic].coefficients[1][0] = coil_data[0, 6*ic + 3]
        coils[ic].coefficients[2][0] = coil_data[0, 6*ic + 5]
        for io in range(0, Nt_coils):
            coils[ic].coefficients[0][2*io+1] = coil_data[io+1, 6*ic + 0]
            coils[ic].coefficients[0][2*io+2] = coil_data[io+1, 6*ic + 1]
            coils[ic].coefficients[1][2*io+1] = coil_data[io+1, 6*ic + 2]
            coils[ic].coefficients[1][2*io+2] = coil_data[io+1, 6*ic + 3]
            coils[ic].coefficients[2][2*io+1] = coil_data[io+1, 6*ic + 4]
            coils[ic].coefficients[2][2*io+2] = coil_data[io+1, 6*ic + 5]
        coils[ic].update()

    numpoints = Nt_ma*ppp+1 if ((Nt_ma*ppp) % 2 == 0) else Nt_ma*ppp
    mas = [StellaratorSymmetricCylindricalFourierCurve(Nt_ma, nfp, np.linspace(0, 1/nfp, numpoints, endpoint=False)) for i in range(copies)]
    cR = [1.471415400740515, 0.1205306261840785, 0.008016125223436036, -0.000508473952304439, -0.0003025251710853062, -0.0001587936004797397, 3.223984137937924e-06, 3.524618949869718e-05, 2.539719080181871e-06, -9.172247073731266e-06, -5.9091166854661e-06, -2.161311017656597e-06, -5.160802127332585e-07, -4.640848016990162e-08, 2.649427979914062e-08, 1.501510332041489e-08, 3.537451979994735e-09, 3.086168230692632e-10, 2.188407398004411e-11, 5.175282424829675e-11, 1.280947310028369e-11, -1.726293760717645e-11, -1.696747733634374e-11, -7.139212832019126e-12, -1.057727690156884e-12, 5.253991686160475e-13]
    sZ = [0.06191774986623827, 0.003997436991295509, -0.0001973128955021696, -0.0001892615088404824, -2.754694372995494e-05, -1.106933185883972e-05, 9.313743937823742e-06, 9.402864564707521e-06, 2.353424962024579e-06, -1.910411249403388e-07, -3.699572817752344e-07, -1.691375323357308e-07, -5.082041581362814e-08, -8.14564855367364e-09, 1.410153957667715e-09, 1.23357552926813e-09, 2.484591855376312e-10, -3.803223187770488e-11, -2.909708414424068e-11, -2.009192074867161e-12, 1.775324360447656e-12, -7.152058893039603e-13, -1.311461207101523e-12, -6.141224681566193e-13, -6.897549209312209e-14]
    for j in range(copies):
        for i in range(Nt_ma):
            mas[j].coefficients[0][i] = cR[i]
            mas[j].coefficients[1][i] = sZ[i]
        mas[j].coefficients[0][Nt_ma] = cR[Nt_ma]
        mas[j].update()
    
    # Add control coils if desired
    if contNum > 0:
        phi_vals,total_control_coils = coil_spacing(contNum,nfp)
        for phi in phi_vals:
            R0,zc = interp(mas[0].gamma,phi)
            CC = ControlCoil(points)
            CC.set_dofs([R0,phi,zc,phi+np.pi/2,np.pi/2,contRad])
            coils.append(CC)
    else:
        total_control_coils = 0

    # Set currents
    if contNum > 0:
        normfac = 1.584
        cont_coil_current = -1*4.551387376532E+04/normfac # -1 is for PPO sign convention, normfac normalizes B to be about 1 on the axis
    else:
        normfac = 1.474

    currents_part = [c/normfac for c in [6.52271941985300E+05, 6.51868569367400E+05, 5.37743588647300E+05]] # normalise to get a magnetic field of around 1 on the axis
    [currents_part.append(cont_coil_current) for i in range(contNum)]

    currents = [currents_part for i in range(copies)]

    return (coils, mas, currents)

def make_flat_stell(Nt_coils=6, Nt_ma=6, nfp=3, ppp=20, num_coils=3, major_radius=1.4, minor_radius=0.33, copies=1, kick=False, magnitude=0.05, z0factr=4, contNum=0, contRad=0.5, Bc=0.1): #kwargs are based on NCSX specs
    assert Bc <= 1, 'The fraction of the field magnitude on the axis attributable to the control coils cannot be greater than 1'

    points = np.linspace(0, 1, Nt_coils*ppp, endpoint=False)
    coils = [CartesianFourierCurve(Nt_coils, points) for i in range(num_coils)] #This is for the modular coils
    
    phi_vals,total_mod_coils = coil_spacing(num_coils,nfp)
    assert len(coils)==len(phi_vals) #Sanity check. 

    #These Fourier coefficients come from expressing the coils in cylindrical coordinates. 
    X1 = minor_radius*np.cos(phi_vals)
    Y1 = minor_radius*np.sin(phi_vals)
    Z1 = np.repeat(minor_radius,len(phi_vals))
    if kick:
        X0 = (major_radius + (minor_radius*magnitude)*np.cos(1*nfp*phi_vals))*np.cos(phi_vals)
        Y0 = (major_radius + (minor_radius*magnitude)*np.cos(1*nfp*phi_vals))*np.sin(phi_vals)
        Z0 = z0factr*magnitude*(minor_radius)*np.sin(1*nfp*phi_vals)
    else:
        X0 = major_radius*np.cos(phi_vals)
        Y0 = major_radius*np.sin(phi_vals)
        Z0 = np.repeat(0,len(phi_vals))

    for ic in range(num_coils):
        coils[ic].coefficients[0][0] = X0[ic]
        coils[ic].coefficients[1][0] = Y0[ic]
        coils[ic].coefficients[2][0] = Z0[ic]
        coils[ic].coefficients[0][1] = 0 
        coils[ic].coefficients[0][2] = X1[ic]
        coils[ic].coefficients[1][1] = 0
        coils[ic].coefficients[1][2] = Y1[ic]
        coils[ic].coefficients[2][1] = Z1[ic]
        coils[ic].coefficients[2][2] = 0
        for io in range(2, Nt_coils):
            coils[ic].coefficients[0][2*io-1] = 0
            coils[ic].coefficients[0][2*io] = 0
            coils[ic].coefficients[1][2*io-1] = 0
            coils[ic].coefficients[1][2*io] = 0
            coils[ic].coefficients[2][2*io-1] = 0
            coils[ic].coefficients[2][2*io] = 0
        coils[ic].update()
    
    numpoints = Nt_ma*ppp+1 if ((Nt_ma*ppp) % 2 == 0) else Nt_ma*ppp
    mas = [StellaratorSymmetricCylindricalFourierCurve(Nt_ma, nfp, np.linspace(0, 1/nfp, numpoints, endpoint=False)) for i in range(copies)]
    for j in range(copies):
        mas[j].coefficients[0][0] = major_radius
        mas[j].coefficients[1][0] = 0
        if kick:
            mas[j].coefficients[0][1] =   minor_radius*magnitude
            mas[j].coefficients[1][0] =   z0factr*minor_radius*magnitude
        mas[j].update()
    
    # Add control coils if desired
    if contNum > 0:
        phi_vals,total_control_coils = coil_spacing(contNum,nfp)
        for phi in phi_vals:
            R0,zc = interp(mas[0].gamma,phi)
            CC = ControlCoil(points)
            CC.set_dofs([R0,phi,zc,phi+np.pi/2,np.pi/2,contRad])
            coils.append(CC)
    else:
        total_control_coils = 0
    
    # Set currents 
    mu_nought = 4*np.pi*1e-7 #SI units 
    if contNum > 0:
        cont_coil_current = 2*np.pi*major_radius*Bc/mu_nought/total_control_coils
    else:
        Bc = 0
    mod_coil_current = 2*np.pi*major_radius*(1-Bc)/mu_nought/total_mod_coils # Normalized to give B=1 on the axis.

    currents_part = []
    [currents_part.append(-1*mod_coil_current) for i in range(num_coils)] # The -1 is for a PPO sign convention.
    [currents_part.append(-1*cont_coil_current) for i in range(contNum)]

    currents = [currents_part for i in range(copies)]

    return (coils, mas, currents)

def reload_stell(sourcedir,Nt_coils=25,Nt_ma=25,ppp=10,nfp=3,stellID=None,num_coils=3,contNum=0,copies=1,oldFormat=False):
    '''
    Data for coils, currents, and the magnetic axis is pulled from sourcedir. 
    There is only need to input *unique* coils - the others will be created using CoilCollection as usual.
    Note that Nt_coils, Nt_ma, ppp, and nfp MUST be the same as in the original stellarator.
    '''
    if not oldFormat:
        shaped_coil_data = []
        passed_initial = False
        with open(os.path.join(sourcedir,'coilCoeffs.txt'),'r') as f:
            for line in f:
                listofstr = line.strip().split()
                if listofstr[0] == 'COIL':
                    if passed_initial == True:
                        shaped_coil_data.append(coil_data)
                    coil_data = [] # Reset data for individual coil
                    passed_initial = True
                else:
                    coil_data.append([float(element) for element in listofstr])
        shaped_coil_data.append(coil_data) # Ensures the last line in the file is included 
    else:
        coil_data = np.loadtxt(os.path.join(sourcedir,'coilCoeffs.txt'))

        repeat_factor = len(coil_data)/num_coils #How many consecutive lines of coil_data belong to each coil.  

        shaped_coil_data = [] #List. Indices: shaped_coil_data[unique coil index][coefficients sublist index][coefficient index]
        for vecind,vec in enumerate(coil_data):
            if vecind%repeat_factor==0:
                intermed = []
                intermed.append(vec.tolist())
            else:
                intermed.append(vec.tolist())
            if len(intermed)==repeat_factor:
                shaped_coil_data.append(intermed)

    points = np.linspace(0, 1, Nt_coils*ppp, endpoint=False)
    coils = [CartesianFourierCurve(Nt_coils, points) for i in range(num_coils)] #Create blank coils object of the proper class.
    for coilind in range(num_coils):
        for sublistind in range(len(coils[coilind].coefficients)):
            for coeffind in range(len(coils[coilind].coefficients[sublistind])):
                coils[coilind].coefficients[sublistind][coeffind] = shaped_coil_data[coilind][sublistind][coeffind]
        coils[coilind].update()

    if contNum > 0:
        for coilind in range(num_coils,num_coils+contNum):
            CC = ControlCoil(points)
            CC.set_dofs(shaped_coil_data[coilind][0])
            coils.append(CC)
    
    ma_raw = []
    numpoints = Nt_ma*ppp+1 if ((Nt_ma*ppp) % 2 == 0) else Nt_ma*ppp
    
    if stellID != None:
        single_ma = []
        with open(os.path.join(sourcedir,'maCoeffs_%d.txt'%stellID),'r') as f:
            for line in f:
                linelist = [float(coeff) for coeff in line.strip().split()]
                ma_raw.append(linelist)
        mas = [StellaratorSymmetricCylindricalFourierCurve(Nt_ma, nfp, np.linspace(0, 1/nfp, numpoints, endpoint=False)) for i in range(copies)]
        for j in range(copies):
            for ind1 in range(len(ma_raw)):
                for ind2 in range(len(ma_raw[ind1])):
                        mas[j].coefficients[ind1][ind2] = ma_raw[ind1][ind2]
            mas[j].update() #The magnetic axes should now be ready to go. 
        currents = [np.loadtxt(os.path.join(sourcedir,'currents_%d.txt'%stellID)).tolist() for i in range(copies)] #Only the currents for the three unique coils need to be imported.
        eta_bar = [np.loadtxt(os.path.join(sourcedir,'eta_bar_%d.txt'%stellID)) for i in range(copies)] #Reload eta_bar from previous run as a starting point. 
    else:
        num_stell = count_files('maCoeffs_*',sourcedir)
        currents = []
        eta_bar = []
        for stell in range(num_stell):
            single_ma = []
            with open(os.path.join(sourcedir,'maCoeffs_%d.txt'%stell),'r') as f:
                for line in f:
                    linelist = [float(coeff) for coeff in line.strip().split()]
                    single_ma.append(linelist)
            ma_raw.append(single_ma)
            currents.append(np.loadtxt(os.path.join(sourcedir,'currents_%d.txt'%stell)).tolist())
            eta_bar.append(np.loadtxt(os.path.join(sourcedir,'eta_bar_%d.txt'%stell)))
        eta_bar = np.array(eta_bar)
        mas = [StellaratorSymmetricCylindricalFourierCurve(Nt_ma, nfp, np.linspace(0, 1/nfp, numpoints, endpoint=False)) for i in range(num_stell)]
        for j in range(num_stell):
            for ind1 in range(len(ma_raw[j])):
                for ind2 in range(len(ma_raw[j][ind1])):
                        mas[j].coefficients[ind1][ind2] = ma_raw[j][ind1][ind2]
            mas[j].update() 

    return (coils, mas, currents, eta_bar)
