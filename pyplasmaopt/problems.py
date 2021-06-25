from .biotsavart import BiotSavart
from .quasi_symmetric_field import QuasiSymmetricField
from .objective import BiotSavartQuasiSymmetricFieldDifference, BiotSavartQuasiSymmetricFieldDifferenceRenormalized, CurveLength, CurveTorsion, CurveCurvature, SobolevTikhonov, UniformArclength, MinimumDistance
from .curve import GaussianSampler
from .stochastic_objective import StochasticQuasiSymmetryObjective, CVaR
from .logging import info
from .qfm_surface import QfmSurface
from .tangent_map import TangentMap
from pyplasmaopt.checkpoint import Checkpoint
from .curve import ControlCoil

from mpi4py import MPI
from math import pi, sin, cos
import numpy as np
import os
import pathlib as pl
import copy

class NearAxisQuasiSymmetryObjective():
    def __init__(self, stellarators, mas, iota_target, eta_bar=-2.25, Nt_ma=6,
                 coil_length_targets=None, magnetic_axis_length_targets=None,
                 coil_length_weight=1, ma_length_weight=1, curvature_weight=1e-6, 
                 torsion_weight=1e-4, tikhonov_weight=0., arclength_weight=0., 
                 sobolev_weight=0., minimum_distance=0.04, distance_weight=1.,
                 ninsamples=0, noutsamples=0, sigma_perturb=1e-4, 
                 length_scale_perturb=0.2, mode="deterministic",
                 outdir="output/", seed=1, freezeCoils=False, freezeMod=False,
                 tanMap=False, constrained=True, keepAxis=True, tanMap_axis=None,
                 iota_weight=1, quasisym_weight=1, qfm_weight=0,
                 qfm_max_tries=5, qfm_volume=1, mmax=3, nmax=3, nfp=3, ntheta=20, nphi=20, 
                 ftol_abs=1e-15, ftol_rel=1e-15,xtol_abs=1e-15,xtol_rel=1e-15,package='nlopt',method='LBFGS',xopt_rld=None,major_radius=1.4,
                 renorm=False, image_freq=250, qs_N=0, res_axis_weight=1000):
        if (tanMap is True) and (keepAxis is False):
            raise NotImplementedError('New derivatives and logic switches are required to use keepAxis=False!')
        assert not (freezeCoils and freezeMod), 'Do NOT simultaneously use frzCoils and frzMod!'
        num_stellarators = len(iota_target)
        self.num_stellarators = num_stellarators
        stellList = range(num_stellarators)
        self.stellList = stellList
        self.freezeCoils = freezeCoils
        self.freezeMod = freezeMod
        self.tanMap = tanMap
        self.Nt_ma = Nt_ma
        self.iota_weight = iota_weight
        self.quasisym_weight = quasisym_weight
        self.stellarator_group = stellarators
        self.seed = seed
        self.ma_group = mas
        self.res_axis_weight = res_axis_weight #Extra weight for the res_axis terms in the tanMap
        self.biotsavart_group = [BiotSavart(self.stellarator_group[i].coils, self.stellarator_group[i].currents) for i in stellList] 
        for i in stellList:
            self.biotsavart_group[i].set_points(self.ma_group[i].gamma)
            # Check that magnetic field has correct sign
            gamma = self.ma_group[i].gamma
            X = gamma[...,0]
            Y = gamma[...,1]
            R = np.sqrt(X**2 + Y**2)
            BX = self.biotsavart_group[i].B[...,0]
            BY = self.biotsavart_group[i].B[...,1]
            BZ = self.biotsavart_group[i].B[...,2]
            BP = -Y*BX/R + X*BY/R
            assert(np.sign(BP[0])==1)
        self.qsf_group = [QuasiSymmetricField(eta_bar[i], self.ma_group[i], qs_N) for i in stellList] 
        self.ninsamples = ninsamples
        self.noutsamples = noutsamples
        self.constrained = constrained
        self.keepAxis = keepAxis
        if tanMap and (tanMap_axis != None):
            old_coeffs = copy.deepcopy([self.ma_group[i].coefficients for i in stellList])
            for i in stellList:
                for ind1 in range(len(self.ma_group[i].coefficients)):
                    for ind2 in range(len(self.ma_group[i].coefficients[ind1])):
                        self.ma_group[i].coefficients[ind1][ind2] = tanMap_axis[i].coefficients[ind1][ind2]
                self.ma_group[i].update()
            self.tangentMap_group = [TangentMap(self.stellarator_group[i],self.ma_group[i],constrained=constrained) for i in stellList]
            for i in stellList:
                for ind1 in range(len(self.ma_group[i].coefficients)):
                    for ind2 in range(len(self.ma_group[i].coefficients[ind1])):
                        self.ma_group[i].coefficients[ind1][ind2] = old_coeffs[i][ind1][ind2]
                self.ma_group[i].update()
        else:
            self.tangentMap_group = [TangentMap(self.stellarator_group[i],self.ma_group[i],constrained=constrained) for i in stellList]

        if renorm:
            self.J_BSvsQS      = [BiotSavartQuasiSymmetricFieldDifferenceRenormalized(self.qsf_group[i], self.biotsavart_group[i]) for i in stellList]
        else:
            self.J_BSvsQS      = [BiotSavartQuasiSymmetricFieldDifference(self.qsf_group[i], self.biotsavart_group[i]) for i in stellList]
        
        if not self.freezeMod:
            coils = self.stellarator_group[0]._base_coils
        else:
            assert any([isinstance(self.stellarator_group[0]._base_coils[i],ControlCoil) for i in range(len(self.stellarator_group[0]._base_coils))]), 'Do NOT use the frzMod option if you only have modular coils - use frzCoils instead.'
            self.coils_all_inds = [i for i,x in enumerate(self.stellarator_group[0]._base_coils) if isinstance(x,ControlCoil)]
            self.coils_inds = [self.coils_all_inds[0],self.coils_all_inds[-1]+1]
            coils = self.stellarator_group[0]._base_coils[self.coils_inds[0]:self.coils_inds[1]]
        
        self.J_coil_lengths    = [CurveLength(coil) for coil in coils]
        self.J_axis_lengths    = [CurveLength(self.ma_group[i]) for i in stellList]
        
        if coil_length_targets is not None:
            if len(coil_length_targets) != len(self.J_coil_lengths):
                for i in range(len(coil_length_targets),len(self.J_coil_lengths)):
                    coil_length_targets = np.append(coil_length_targets,self.J_coil_lengths[i].J())
            self.coil_length_targets = coil_length_targets
        else:
            self.coil_length_targets = [J.J() for J in self.J_coil_lengths]
            np.savetxt(str(pl.Path(outdir).joinpath('coil_length_targets.txt')),self.coil_length_targets)
        if self.freezeMod and (len(self.coil_length_targets) != len(self.J_coil_lengths)):
            self.coil_length_targets = self.coil_length_targets[self.coils_inds[0]:self.coils_inds[1]]
        
        if magnetic_axis_length_targets is not None:
            self.magnetic_axis_length_targets = magnetic_axis_length_targets
        else:
            self.magnetic_axis_length_targets = [J.J() for J in self.J_axis_lengths]
            np.savetxt(str(pl.Path(outdir).joinpath('magnetic_axis_length_targets.txt')),self.magnetic_axis_length_targets)

        self.J_coil_curvatures = [CurveCurvature(coil, length) for (coil, length) in zip(coils, self.coil_length_targets)]
        self.J_coil_torsions   = [CurveTorsion(coil, p=2) for coil in coils]
        self.J_sobolev_weights = [SobolevTikhonov(coil, weights=[1., .1, .1, .1]) for coil in coils] + [SobolevTikhonov(self.ma_group[0], weights=[1., .1, .1, .1])] #Wrong, but we're ignoring these anyway
        self.J_arclength_weights = [UniformArclength(coil, length) for (coil, length) in zip(coils, self.coil_length_targets)]
        self.J_distance = MinimumDistance(self.stellarator_group[0].coils, minimum_distance)

        self.coil_length_weight = coil_length_weight
        self.ma_length_weight = ma_length_weight
        self.iota_target                 = iota_target #This is a LIST of floats now. 
        self.curvature_weight            = curvature_weight 
        self.torsion_weight              = torsion_weight
        self.num_ma_dofs = len(self.ma_group[0].get_dofs())  
        self.current_fak = 1./(4 * pi * 1e-7)
        
        self.eta_bar_idxs = (0,self.num_stellarators)
        self.current_dof_idxs = (self.eta_bar_idxs[1], self.eta_bar_idxs[1] + self.num_stellarators*len(self.stellarator_group[0].get_currents()))
        self.ma_dof_idxs = (self.current_dof_idxs[1], self.current_dof_idxs[1] + self.num_stellarators*self.num_ma_dofs)
        if not self.freezeMod:
            self.coil_dof_idxs = (self.ma_dof_idxs[1], self.ma_dof_idxs[1] + len(self.stellarator_group[0].get_dofs()))
        else:
            coil_dofs = np.array([])
            for i in range(len(coils)):
                coil_dofs = np.concatenate((coil_dofs,coils[i].get_dofs()))
            count_dofs = len(coil_dofs)
            self.coil_dof_idxs = (self.ma_dof_idxs[1], self.ma_dof_idxs[1] + count_dofs)
            self.frozen_coil_dofs = self.stellarator_group[0].get_dofs()[0:len(self.stellarator_group[0].get_dofs())-count_dofs]

        if mode in ["deterministic", "stochastic"]:
            eta_bar_cat = [self.qsf_group[i].eta_bar for i in stellList]
            stellarator_cat = np.asarray([self.stellarator_group[i].get_currents()/self.current_fak for i in stellList]).flatten()
            ma_dofs_cat = np.asarray([self.ma_group[i].get_dofs() for i in stellList]).flatten()
        
            varlist = [eta_bar_cat]
            varlist.append(stellarator_cat)
            varlist.append(ma_dofs_cat)
            if not self.freezeCoils and not self.freezeMod:
                varlist.append(self.stellarator_group[0].get_dofs())
            elif self.freezeMod:
                varlist.append(coil_dofs)
            self.x0 = np.concatenate(tuple(varlist))
            
        elif mode[0:4] == "cvar":
            if self.freezeCoils or self.num_stellarators != 1:
                raise NotImplementedError
            self.x0 = np.concatenate(([qsf.eta_bar], self.ma.get_dofs(), self.stellarator.get_currents()/self.current_fak, self.stellarator.get_dofs(), [0.]))
        else:
            raise NotImplementedError
        self.x = self.x0.copy()
        self.sobolev_weight = sobolev_weight
        self.tikhonov_weight = tikhonov_weight
        self.arclength_weight = arclength_weight
        self.distance_weight = distance_weight
        self.qfm_weight = qfm_weight
        self.initial_qfm_opt = False
        self.qfm_max_tries = qfm_max_tries
        self.qfm_volume = qfm_volume
        self.mmax = mmax
        self.nmax = nmax
        self.nfp = nfp
        self.ntheta = ntheta
        self.nphi = nphi
        self.ftol_abs = ftol_abs 
        self.ftol_rel = ftol_rel
        self.xtol_abs = xtol_abs
        self.xtol_rel = xtol_rel
        self.package = package
        self.method = method
        self.major_radius = major_radius
        self.xopt_rld = xopt_rld
        self.ignore_tol = 0 #Cutoff weight for determining if res and dres contributions will be computed in the update() function
        self.image_freq = image_freq
        self.old_points = copy.deepcopy([self.ma_group[i].points for i in stellList])

        sampler = GaussianSampler(coils[0].points, length_scale=length_scale_perturb, sigma=sigma_perturb) #I think I can ignore this
        self.sampler = sampler#I think I can ignore this

        self.stochastic_qs_objective = StochasticQuasiSymmetryObjective(self.stellarator_group[0], sampler, ninsamples, self.qsf_group[0], self.seed) #I think I can ignore this 
        self.stochastic_qs_objective_out_of_sample = None #I think I can ignore this

        if mode in ["deterministic", "stochastic"]:
            self.mode = mode 
        elif mode[0:4] == "cvar": 
            self.mode = "cvar" 
            self.cvar_alpha = float(mode[4:])
            self.cvar = CVaR(self.cvar_alpha, .01)
        else:
            raise NotImplementedError 

        self.stochastic_qs_objective.set_magnetic_axis(self.ma_group[0].gamma)#I think this should be okay as we don't use the stochastic optimization?

        self.Jvals_perturbed = [] 
        self.Jvals_quantiles = []
        self.Jvals_no_noise = []
        self.xiterates = []
        self.Jvals_individual = []
        self.QSvsBS_perturbed = []
        self.Jvals = []
        self.dJvals = []
        self.out_of_sample_values = []
        self.outdir = outdir
    
    def split_list(self, alist, wanted_parts=None):  
        ''' Splits a 1D list or NumPy array evenly into the desired number of parts and returns a list of lists.''' 
        if wanted_parts is None:
            wanted_parts = self.num_stellarators
        length = len(alist)
        if length % wanted_parts != 0:
            raise IOError('The list is not evenly divisible into the requested number of parts.')
        return [alist[i*length // wanted_parts: (i+1)*length // wanted_parts] for i in range(wanted_parts) ]
    
    def set_dofs(self, x):
        x_etabar = x[self.eta_bar_idxs[0]:self.eta_bar_idxs[1]]
        x_current = x[self.current_dof_idxs[0]:self.current_dof_idxs[1]] 
        x_current_split = self.split_list(x_current)
        x_ma = x[self.ma_dof_idxs[0]:self.ma_dof_idxs[1]] 
        x_ma_split = self.split_list(x_ma)
        if (not self.freezeCoils) or (not self.freezeMod):
            x_coil = x[self.coil_dof_idxs[0]:self.coil_dof_idxs[1]]
            self.t = x[-1] #Not sure what this does. (Doesn't seem important.)
            if self.freezeMod:
                x_coil = np.concatenate((self.frozen_coil_dofs,x_coil))
        
        for i in self.stellList:
            self.qsf_group[i].eta_bar = x_etabar[i]
            self.stellarator_group[i].set_currents(self.current_fak * x_current_split[i])
            self.ma_group[i].set_dofs(x_ma_split[i])
            self.biotsavart_group[i].set_points(self.ma_group[i].gamma)
            if not self.freezeCoils:
                self.stellarator_group[i].set_dofs(x_coil)
            self.biotsavart_group[i].clear_cached_properties()
            self.qsf_group[i].clear_cached_properties() 
            if self.tanMap:
                self.tangentMap_group[i].update_solutions()

    def update(self, x):
        self.x[:] = x
        J_BSvsQS          = self.J_BSvsQS #This is a list now! 
        J_coil_lengths    = self.J_coil_lengths 
        J_axis_lengths    = self.J_axis_lengths #List now! 
        J_coil_curvatures = self.J_coil_curvatures 
        J_coil_torsions   = self.J_coil_torsions 

        iota_target                  = self.iota_target #A list now
        magnetic_axis_length_targets = self.magnetic_axis_length_targets 
        curvature_weight             = self.curvature_weight 
        torsion_weight               = self.torsion_weight
        qsf_group = self.qsf_group #List
        tanMap_group = self.tangentMap_group #List
        self.set_dofs(x) 
        
        self.dresetabar  = np.zeros(self.eta_bar_idxs[1]-self.eta_bar_idxs[0]) 
        self.drescurrent = np.zeros(self.current_dof_idxs[1]-self.current_dof_idxs[0])
        self.dresma      = np.zeros(self.ma_dof_idxs[1]-self.ma_dof_idxs[0]) 
        if not self.freezeCoils:
            self.drescoil    = np.zeros(self.coil_dof_idxs[1]-self.coil_dof_idxs[0])

        """ Objective values """
        if not self.freezeCoils:
            self.res2      = 0.5 * self.coil_length_weight * sum( (1/l)**2 * (J2.J() - l)**2 for (J2, l) in zip(J_coil_lengths, self.coil_length_targets))
            self.drescoil += self.coil_length_weight * self.stellarator_group[0].reduce_coefficient_derivatives([
                (1/l)**2 * (J_coil_lengths[i].J()-l) * J_coil_lengths[i].dJ_by_dcoefficients() for (i, l) in zip(list(range(len(J_coil_lengths))), self.coil_length_targets)])
        else:
            self.res2 = 0

        self.res3    = self.ma_length_weight * np.sum([0.5 * (1/magnetic_axis_length_targets[i])**2 * (J_axis_lengths[i].J() - magnetic_axis_length_targets[i])**2 for i in self.stellList])
        self.dresma += self.ma_length_weight * np.concatenate(([(1/magnetic_axis_length_targets[i])**2 * (J_axis_lengths[i].J()-magnetic_axis_length_targets[i]) * J_axis_lengths[i].dJ_by_dcoefficients() for i in self.stellList]))
        
        if not self.tanMap:
            self.res4        = np.sum([0.5 * self.iota_weight * (1/iota_target[i]**2) * (qsf_group[i].iota-iota_target[i])**2 for i in self.stellList])
            self.dresetabar += np.concatenate(([self.iota_weight * (1/iota_target[i]**2) * (qsf_group[i].iota - iota_target[i]) * qsf_group[i].diota_by_detabar[:,0] for i in self.stellList])) 
            self.dresma     += np.concatenate(([self.iota_weight * (1/iota_target[i]**2) * (qsf_group[i].iota - iota_target[i]) * qsf_group[i].diota_by_dcoeffs[:,0] for i in self.stellList]))
            self.calc_iotas = [qsf_group[i].iota for i in self.stellList]
        else:
            tanMap_iota = [tanMap_group[i].compute_iota() for i in self.stellList]
            self.calc_iotas = tanMap_iota
            
            self.res4         = np.sum([0.5 * self.iota_weight * (1/iota_target[i]**2) * ((tanMap_iota[i] - iota_target[i])**2 + self.res_axis_weight*tanMap_group[i].res_axis()) for i in self.stellList]) 
            self.dresma      += np.concatenate(([self.iota_weight * (1/iota_target[i]**2) * (0.5 * self.res_axis_weight*tanMap_group[i].d_res_axis_d_magneticaxiscoeffs()) for i in self.stellList]))
            self.drescurrent += np.concatenate(([self.current_fak * self.iota_weight * (1/iota_target[i]**2) * ((tanMap_iota[i] - iota_target[i]) * tanMap_group[i].d_iota_dcoilcurrents() + 0.5 * self.res_axis_weight*tanMap_group[i].d_res_axis_d_coil_currents()) for i in self.stellList]),)

            if not self.freezeCoils:
                self.drescoil    += np.sum([self.iota_weight * (1/iota_target[i]**2) * ((tanMap_iota[i] - iota_target[i]) * tanMap_group[i].d_iota_dcoilcoeffs() + 0.5 * self.res_axis_weight*tanMap_group[i].d_res_axis_d_coil_coeffs()) for i in self.stellList],axis=0)

            for i in self.stellList: #The tangent map sets the points to have length 1, so we have to reset them for the rest of the code to work properly.  
                self.ma_group[i].points = self.old_points[i]
                self.ma_group[i].update()
                self.biotsavart_group[i].set_points(self.ma_group[i].gamma)

        if curvature_weight > self.ignore_tol and not self.freezeCoils:
            self.res5      = sum(curvature_weight * J.J() for J in J_coil_curvatures)
            self.drescoil += self.curvature_weight * self.stellarator_group[0].reduce_coefficient_derivatives([J.dJ_by_dcoefficients() for J in J_coil_curvatures])
        else:
            self.res5 = 0
        
        if torsion_weight > self.ignore_tol and not self.freezeCoils:
            self.res6      = sum(torsion_weight * J.J() for J in J_coil_torsions)
            self.drescoil += self.torsion_weight * self.stellarator_group[0].reduce_coefficient_derivatives([J.dJ_by_dcoefficients() for J in J_coil_torsions])
        else:
            self.res6 = 0 

        if self.sobolev_weight > self.ignore_tol:
            if self.num_stellarators != 1:
                raise NotImplementedError
            self.res7 = sum(self.sobolev_weight * J.J() for J in self.J_sobolev_weights) #We're not using this anyway. 
            self.dresma += self.sobolev_weight * self.J_sobolev_weights[-1].dJ_by_dcoefficients() #Probably not fine, but we're not using. 
            if not self.freezeCoils:
                self.drescoil += self.sobolev_weight * self.stellarator.reduce_coefficient_derivatives([J.dJ_by_dcoefficients() for J in self.J_sobolev_weights[:-1]]) #Probably not fine, but we're not using.
        else:
            self.res7 = 0

        if self.arclength_weight > self.ignore_tol and not self.freezeCoils:
            self.res8 = sum(self.arclength_weight * J.J() for J in self.J_arclength_weights)
            self.drescoil += self.arclength_weight * self.stellarator_group[0].reduce_coefficient_derivatives([J.dJ_by_dcoefficients() for J in self.J_arclength_weights])
        else:
            self.res8 = 0

        if self.distance_weight > self.ignore_tol and not self.freezeCoils:
            self.res9 = self.distance_weight * self.J_distance.J()
            self.drescoil += self.distance_weight * self.stellarator_group[0].reduce_coefficient_derivatives(self.J_distance.dJ_by_dcoefficients())
        else:
            self.res9 = 0

        iteration = len(self.xiterates)
        if self.qfm_weight > self.ignore_tol:
            if (self.xopt_rld is not None) and (not self.initial_qfm_opt):
                self.qfm_group = [QfmSurface(self.mmax, self.nmax, self.nfp, self.stellarator_group[i], self.ntheta, self.nphi, self.qfm_volume) for i in self.stellList]
                fopts = [self.qfm_weight*self.qfm_group[i].qfm_metric(paramsInit=self.xopt_rld[i],outdir=self.outdir,stellID=i,ftol_abs=self.ftol_abs,ftol_rel=self.ftol_rel,xtol_abs=self.xtol_abs,xtol_rel=self.xtol_rel,package=self.package,method=self.method) for i in self.stellList]
                self.res10 = sum(fopts)
                info("QFM surface(s) reloaded from previous run.")
                self.initial_qfm_opt = True
            if not self.initial_qfm_opt:  
                runs = 1
                success = False
                while runs < self.qfm_max_tries:
                    self.qfm_group = [QfmSurface(self.mmax, self.nmax, self.nfp, self.stellarator_group[i], self.ntheta, self.nphi, self.qfm_volume) for i in self.stellList]
                    
                    # Initialize parameters - circular cross section torus
                    paramsInitR = np.zeros((self.qfm_group[0].mnmax))
                    paramsInitZ = np.zeros((self.qfm_group[0].mnmax))
                    
                    approx_plasma_minor_radius = 1/np.pi*np.sqrt(self.qfm_volume/2/self.major_radius) # Minor radius of a torus
                    paramsInitR[(self.qfm_group[0].xm==1)*(self.qfm_group[0].xn==0)] = approx_plasma_minor_radius
                    paramsInitZ[(self.qfm_group[0].xm==1)*(self.qfm_group[0].xn==0)] = -1*approx_plasma_minor_radius
                    
                    paramsInit = np.hstack((paramsInitR[1::],paramsInitZ))

                    info('Beginning QFM surface optimization - attempt %d.'%runs)
                    try:
                        fopts = [self.qfm_weight*self.qfm_group[i].qfm_metric(paramsInit=paramsInit,outdir=self.outdir,stellID=i,ftol_abs=self.ftol_abs,ftol_rel=self.ftol_rel,xtol_abs=self.xtol_abs,xtol_rel=self.xtol_rel,package=self.package,method=self.method) for i in self.stellList]
                        success = True
                        self.res10 = sum(fopts)
                        break
                    except RuntimeError:
                        info('Optimization for given volume failed.')
                        self.qfm_volume = self.qfm_volume/2
                        runs += 1 
                if not success:
                    info('QFM surface not found for at least one stellarator!')
                    quit()
                info(f"Final QFM surface volume for each stellarator: {self.qfm_volume:.6e}")
                np.savetxt(str(pl.Path(self.outdir).joinpath('qfm_volume.txt')),[self.qfm_volume]) #Overwrite old file if this portion of the code runs. 
                self.initial_qfm_opt = True
            else:
                self.res10 = sum([self.qfm_weight*self.qfm_group[i].qfm_metric(outdir=self.outdir,stellID=i,ftol_abs=self.ftol_abs,ftol_rel=self.ftol_rel,xtol_abs=self.xtol_abs,xtol_rel=self.xtol_rel,package=self.package,method=self.method) for i in self.stellList])            
            if not self.freezeCoils:
                self.drescoil += np.sum(([self.qfm_weight*self.qfm_group[i].d_qfm_metric_d_coil_coeffs() for i in self.stellList]),axis=0)
            self.drescurrent += np.concatenate(([self.current_fak*self.qfm_weight*self.qfm_group[i].d_qfm_metric_d_coil_currents() for i in self.stellList])) 
        else:
            self.res10 = 0

        if self.tikhonov_weight > self.ignore_tol:
            if self.num_stellarators != 1:
                raise NotImplementedError
            self.res_tikhonov_weight = self.tikhonov_weight * np.sum((x-self.x0)**2) #Not using
            dres_tikhonov_weight = self.tikhonov_weight * 2. * (x-self.x0)#Not using
            self.dresetabar += dres_tikhonov_weight[0:1] #Not using
            self.drescurrent += dres_tikhonov_weight[self.current_dof_idxs[0]:self.current_dof_idxs[1]] #Not using
            self.dresma += dres_tikhonov_weight[self.ma_dof_idxs[0]:self.ma_dof_idxs[1]] #Not using
            if not self.freezeCoils:
                self.drescoil += dres_tikhonov_weight[self.coil_dof_idxs[0]:self.coil_dof_idxs[1]] #Not using
        else:
            self.res_tikhonov_weight = 0

        self.stochastic_qs_objective.set_magnetic_axis(self.ma_group[0].gamma) #Not using

        Jsamples = self.stochastic_qs_objective.J_samples() #Not using
        assert len(Jsamples) == self.ninsamples #Not using
        self.QSvsBS_perturbed.append(Jsamples) #Not using

        if self.quasisym_weight > self.ignore_tol:
            self.res1_det        = np.sum([0.5 * self.quasisym_weight * (J_BSvsQS[i].J_L2() + J_BSvsQS[i].J_H1()) for i in self.stellList])
            self.dresetabar_det  = np.concatenate(([0.5 * self.quasisym_weight * (J_BSvsQS[i].dJ_L2_by_detabar() + J_BSvsQS[i].dJ_H1_by_detabar()) for i in self.stellList]))
            self.drescurrent_det = np.concatenate(([0.5 * self.quasisym_weight * self.current_fak * (
                self.stellarator_group[i].reduce_current_derivatives(J_BSvsQS[i].dJ_L2_by_dcoilcurrents()) + self.stellarator_group[i].reduce_current_derivatives(J_BSvsQS[i].dJ_H1_by_dcoilcurrents())
            ) for i in self.stellList]))
            self.dresma_det      = np.concatenate(([0.5 * self.quasisym_weight * (J_BSvsQS[i].dJ_L2_by_dmagneticaxiscoefficients() + J_BSvsQS[i].dJ_H1_by_dmagneticaxiscoefficients()) for i in self.stellList]))
            if not self.freezeCoils:
                self.drescoil_det    = np.sum(([0.5 * self.quasisym_weight * (self.stellarator_group[i].reduce_coefficient_derivatives(J_BSvsQS[i].dJ_L2_by_dcoilcoefficients()) \
                    + self.stellarator_group[i].reduce_coefficient_derivatives(J_BSvsQS[i].dJ_H1_by_dcoilcoefficients())) for i in self.stellList]),axis=0)
        else:
            self.res1_det        = 0
            self.dresetabar_det  = 0
            self.drescurrent_det = 0
            self.dresma_det      = 0
            if not self.freezeCoils:
                self.drescoil_det = 0

        if self.mode == "deterministic":
            self.res1         = self.res1_det
            self.dresetabar  += self.dresetabar_det
            self.drescurrent += self.drescurrent_det
            self.dresma      += self.dresma_det 
            if not self.freezeCoils:
                self.drescoil    += self.drescoil_det
        else:
            if self.freezeCoils or self.tanMap:
                raise NotImplementedError
            self.dresetabar_det  += self.dresetabar
            self.dresma_det      += self.dresma
            self.drescoil_det    += self.drescoil
            self.drescurrent_det += self.drescurrent
            if self.mode == "stochastic":
                n = self.ninsamples
                self.res1         = sum(Jsamples)/n
                self.res1_det     = self.res1
                self.drescoil    += sum(self.stochastic_qs_objective.dJ_by_dcoilcoefficients_samples())/n
                self.drescurrent += self.current_fak * sum(self.stochastic_qs_objective.dJ_by_dcoilcurrents_samples())/n
                self.dresetabar  += sum(self.stochastic_qs_objective.dJ_by_detabar_samples())/n
                self.dresma      += sum(self.stochastic_qs_objective.dJ_by_dmagneticaxiscoefficients_samples())/n
            elif self.mode == "cvar":
                if self.freezeCoils:
                    raise NotImplementedError
                t = x[-1]
                self.res1         = self.cvar.J(t, Jsamples)
                self.res1_det     = self.res1
                self.drescoil    += self.cvar.dJ_dx(t, Jsamples, self.stochastic_qs_objective.dJ_by_dcoilcoefficients_samples())
                self.drescurrent += self.current_fak * self.cvar.dJ_dx(t, Jsamples, self.stochastic_qs_objective.dJ_by_dcoilcurrents_samples())
                self.dresetabar  += self.cvar.dJ_dx(t, Jsamples, self.stochastic_qs_objective.dJ_by_detabar_samples())
                self.dresma      += self.cvar.dJ_dx(t, Jsamples, self.stochastic_qs_objective.dJ_by_dmagneticaxiscoefficients_samples())
                self.drescvart   = self.cvar.dJ_dt(t, Jsamples)
            else:
                raise NotImplementedError
       
        self.Jvals_individual.append([self.res1, self.res2, self.res3, self.res4, self.res5, self.res6, self.res7, self.res8, self.res9, self.res10])
        self.res = sum(self.Jvals_individual[-1])
        self.perturbed_vals = [self.res - self.res1 + r for r in self.QSvsBS_perturbed[-1]] #Not using 

        if self.mode in ["deterministic", "stochastic"]:
            dreslist = [self.dresetabar]
            dreslist.append(self.drescurrent)
            dreslist.append(self.dresma)
            if not self.freezeCoils:
                dreslist.append(self.drescoil)
            self.dres = np.concatenate(tuple(dreslist))
        elif self.mode == "cvar":
            if self.freezeCoils or self.tanMap:
                raise NotImplementedError
            self.dres = np.concatenate((
                self.dresetabar, self.dresma,
                self.drescurrent, self.drescoil,
                self.drescvart
            ))
        else:
            raise NotImplementedError

        if self.mode == "stochastic":
            self.dres_det = np.concatenate((
                self.dresetabar_det, self.dresma_det,
                self.drescurrent_det, self.drescoil_det
            ))

    def compute_out_of_sample(self):
        if self.stochastic_qs_objective_out_of_sample is None:
            self.stochastic_qs_objective_out_of_sample = StochasticQuasiSymmetryObjective(self.stellarator, self.sampler, self.noutsamples, self.qsf, 9999+self.seed)

        self.stochastic_qs_objective_out_of_sample.set_magnetic_axis(self.ma.gamma)
        Jsamples = np.array(self.stochastic_qs_objective_out_of_sample.J_samples())
        return Jsamples, Jsamples + sum(self.Jvals_individual[-1][1:])

    def callback(self, x, verbose=True):
        assert np.allclose(self.x, x)
        self.Jvals.append(self.res)
        norm = np.linalg.norm
        normlist = [norm(self.dres)]
        normlist.append(norm(self.dresetabar))
        normlist.append(norm(self.drescurrent))
        normlist.append(norm(self.dresma))
        if not self.freezeCoils:
            normlist.append(norm(self.drescoil))
        self.dJvals.append(tuple(normlist))
        
        if self.ninsamples > 0:
            self.Jvals_quantiles.append((np.quantile(self.perturbed_vals, 0.1), np.mean(self.perturbed_vals), np.quantile(self.perturbed_vals, 0.9)))
        self.Jvals_no_noise.append(self.res - self.res1 + 0.5 * (self.J_BSvsQS[0].J_L2() + self.J_BSvsQS[0].J_H1())) #Not correct, but doesn't seem to be used for anything. 
        self.xiterates.append(x.copy())
        self.Jvals_perturbed.append(self.perturbed_vals) #Ignore

        iteration = len(self.xiterates)-1
        info("################################################################################")
        info(f"Iteration {iteration}")
        norm = np.linalg.norm
        info(f"Objective value:         {self.res:.6e}")
        info(f"Objective values:        {self.res1:.6e}, {self.res2:.6e}, {self.res3:.6e}, {self.res4:.6e}, {self.res5:.6e}, {self.res6:.6e}, {self.res7:.6e}, {self.res8:.6e}, {self.res9:.6e}, {self.res10:.6e}")
        if self.ninsamples > 0: #Ignore this whole section
            info(f"VaR(.1), Mean, VaR(.9):  {np.quantile(self.perturbed_vals, 0.1):.6e}, {np.mean(self.perturbed_vals):.6e}, {np.quantile(self.perturbed_vals, 0.9):.6e}")
            cvar90 = np.mean(list(v for v in self.perturbed_vals if v >= np.quantile(self.perturbed_vals, 0.9)))
            cvar95 = np.mean(list(v for v in self.perturbed_vals if v >= np.quantile(self.perturbed_vals, 0.95)))
            info(f"CVaR(.9), CVaR(.95), Max:{cvar90:.6e}, {cvar95:.6e}, {max(self.perturbed_vals):.6e}")
        if not self.freezeCoils:
            info(f"Objective gradients:     {norm(self.dresetabar):.6e}, {norm(self.drescurrent):.6e}, {norm(self.dresma):.6e}, {norm(self.drescoil):.6e}")
        else:
            info(f"Objective gradients:     {norm(self.dresetabar):.6e}, {norm(self.drescurrent):.6e}, {norm(self.dresma):.6e}")
        
        max_curvature  = max(np.max(c.kappa) for c in self.stellarator_group[0]._base_coils)
        mean_curvature = np.mean([np.mean(c.kappa) for c in self.stellarator_group[0]._base_coils])
        max_torsion    = max(np.max(np.abs(c.torsion)) for c in self.stellarator_group[0]._base_coils)
        mean_torsion   = np.mean([np.mean(np.abs(c.torsion)) for c in self.stellarator_group[0]._base_coils])
        info(f"Curvature Max: {max_curvature:.3e}; Mean: {mean_curvature:.3e}")
        info(f"Torsion   Max: {max_torsion:.3e}; Mean: {mean_torsion:.3e}")
        comm = MPI.COMM_WORLD
        if ((iteration in list(range(6))) or iteration % self.image_freq == 0) and comm.rank == 0:
            self.plot('iteration-%04i.png' % iteration, iteration=iteration)
            if self.qfm_weight > self.ignore_tol:
                self.qfmPlot('qfmSurface',iteration)
        if iteration % 250 == 0 and self.noutsamples > 0: #Ignore this section
            oos_vals = self.compute_out_of_sample()[1]
            self.out_of_sample_values.append(oos_vals)
            info("Out of sample")
            info(f"VaR(.1), Mean, VaR(.9):  {np.quantile(oos_vals, 0.1):.6e}, {np.mean(oos_vals):.6e}, {np.quantile(oos_vals, 0.9):.6e}")
            info(f"CVaR(.9), CVaR(.95), Max:{np.mean(list(v for v in oos_vals if v >= np.quantile(oos_vals, 0.9))):.6e}, {np.mean(list(v for v in oos_vals if v >= np.quantile(oos_vals, 0.95))):.6e}, {max(oos_vals):.6e}")

    def qfmPlot(self, title, iteration):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for i in self.stellList:
            xopt = np.loadtxt(str(pl.Path(self.outdir).joinpath('xopt_{:}.txt'.format(i))))
            R,Z = self.qfm_group[i].position(xopt)
            plt.figure()
            plt.plot(R[0,:],Z[0,:])
            plt.xlabel(r'$R$ (m)')
            plt.ylabel(r'$Z$ (m)')
            plt.savefig(str(pl.Path(self.outdir).joinpath(title+'_%i-%04i.png'%(i,iteration))),bbox_inches='tight')
            plt.close()

    def plot(self, filename, iteration=0):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.cm as cm
        import plotly.graph_objects as go
        Checkpoint(self,iteration=iteration)
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        numCoils = len(self.stellarator_group[0].coils)
        uniqueCoils = len(self.stellarator_group[0]._base_coils)
        colors = list(map(tuple, np.delete(cm.rainbow(np.linspace(0,1,uniqueCoils+1)),-1,1)))
        for i in range(0, numCoils): 
            ax = self.stellarator_group[0].coils[i].plot(ax=ax, show=False, color=colors[i%uniqueCoils]) 
        self.ma_group[0].plot(ax=ax, show=False, closed_loop=False, color=colors[uniqueCoils])
        ax.view_init(elev=90., azim=0)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-1, 1)
        ax = fig.add_subplot(1, 2, 2, projection="3d")
        for i in range(0, numCoils): 
            ax = self.stellarator_group[0].coils[i].plot(ax=ax, show=False, color=colors[i%uniqueCoils]) 
        self.ma_group[0].plot(ax=ax, show=False, closed_loop=False, color=colors[uniqueCoils])
        ax.view_init(elev=0., azim=0)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-1, 1)
        plt.savefig(self.outdir + filename, dpi=300)
        plt.close()
       
        # Plotly (3D plotting) code is below this

        def ColorString(color):
            colorstring = 'rgb('
            for num in color: # Plotly wants to be fed RGB data in a string format
                colorstring += '{:.25f}'.format(num*255) # Multiply by 255 since Plotly does not used normalized RGB values
                if num != color[-1]:
                    colorstring += ','
            colorstring += ')'
            return colorstring

        linewidth = 5
        magnification = 1.1
        opacity = 1 # 0 gives transparent background, 1 gives colored background
        backcolor = 1 # 0 gives black background, 1 gives white background

        fig = go.Figure(layout=go.Layout(scene=dict(aspectmode='data'),showlegend=False,paper_bgcolor='rgba({:},{:},{:},{:})'.format(str(backcolor*255),str(backcolor*255),str(backcolor*255),str(opacity))))

        for i in range(0, numCoils): 
            gamma = self.stellarator_group[0].coils[i].gamma
            gamma = np.concatenate((gamma, [gamma[0,:]]))
            colorstring = ColorString(colors[i%uniqueCoils])
            fig.add_trace(go.Scatter3d(x=gamma[:, 0],y=gamma[:, 1],z=gamma[:, 2],mode='lines',line=dict(color=colorstring,width=linewidth)))
        
        gamma = self.ma_group[0].gamma
        theta = 2*np.pi/self.ma_group[0].nfp
        rotmat = np.asarray([
            [cos(theta), -sin(theta), 0],
            [sin(theta), cos(theta), 0],
            [0, 0, 1]]).T
        gamma0 = gamma.copy()
        for i in range(1, self.ma_group[0].nfp):
            gamma0 = gamma0 @ rotmat
            gamma = np.vstack((gamma, gamma0))
        colorstring = ColorString(colors[uniqueCoils])
        fig.add_trace(go.Scatter3d(x=gamma[:, 0],y=gamma[:, 1],z=gamma[:, 2],mode='lines',line=dict(color=colorstring,width=linewidth)))
       
        fig.update_layout(scene = dict(
            xaxis = dict(title='',showgrid=False,showticklabels=False,showbackground=False,visible=False),
            yaxis = dict(title='',showgrid=False,showticklabels=False,showbackground=False,visible=False),
            zaxis = dict(title='',showgrid=False,showticklabels=False,showbackground=False,visible=False)))
       
        eye_angled = np.array([1.25, 1.25, 1.25])/magnification
        fig.update_layout(scene_camera = dict(eye = dict(x=eye_angled[0],y=eye_angled[1],z=eye_angled[2])))
        fig.write_image(str(pl.Path(self.outdir).joinpath('plotly_angled_'+filename)))

        eye_side = np.array([0,np.linalg.norm(eye_angled),0])
        fig.update_layout(scene_camera = dict(eye = dict(x=eye_side[0],y=eye_side[1],z=eye_side[2])))
        fig.write_image(str(pl.Path(self.outdir).joinpath('plotly_side_'+filename)))

        eye_top = np.array([0,0,np.linalg.norm(eye_angled)])
        fig.update_layout(scene_camera = dict(eye = dict(x=eye_top[0],y=eye_top[1],z=eye_top[2])))
        fig.write_image(str(pl.Path(self.outdir).joinpath('plotly_top_'+filename)))

class SimpleNearAxisQuasiSymmetryObjective():

    def __init__(self, stellarator, ma, iota_target, eta_bar=-2.25,
                 coil_length_target=None, magnetic_axis_length_target=None,
                 curvature_weight=0., torsion_weight=0., tikhonov_weight=0., arclength_weight=0., sobolev_weight=0.,
                 minimum_distance=0.04, distance_weight=0.,
                 outdir="output/"
                 ):
        self.stellarator = stellarator
        self.ma = ma
        bs = BiotSavart(stellarator.coils, stellarator.currents)
        self.biotsavart = bs
        self.biotsavart.set_points(self.ma.gamma)
        qsf = QuasiSymmetricField(eta_bar, ma)
        self.qsf = qsf
        sigma = qsf.sigma
        iota = qsf.iota

        self.J_BSvsQS          = BiotSavartQuasiSymmetricFieldDifference(qsf, bs)
        coils = stellarator._base_coils
        self.J_coil_lengths    = [CurveLength(coil) for coil in coils]
        self.J_axis_length     = CurveLength(ma)
        if coil_length_target is not None:
            self.coil_length_targets = [coil_length_target for coil in coils]
        else:
            self.coil_length_targets = [J.J() for J in self.J_coil_lengths]
        self.magnetic_axis_length_target = magnetic_axis_length_target or self.J_axis_length.J()

        self.J_coil_curvatures = [CurveCurvature(coil, length) for (coil, length) in zip(coils, self.coil_length_targets)]
        self.J_coil_torsions   = [CurveTorsion(coil, p=4) for coil in coils]
        self.J_sobolev_weights = [SobolevTikhonov(coil, weights=[1., .1, .1, .1]) for coil in coils] + [SobolevTikhonov(ma, weights=[1., .1, .1, .1])]
        self.J_arclength_weights = [UniformArclength(coil, length) for (coil, length) in zip(coils, self.coil_length_targets)]
        self.J_distance = MinimumDistance(stellarator.coils, minimum_distance)

        self.iota_target                 = iota_target
        self.curvature_weight             = curvature_weight
        self.torsion_weight               = torsion_weight
        self.num_ma_dofs = len(ma.get_dofs())
        self.current_fak = 1./(4 * pi * 1e-7)

        self.ma_dof_idxs = (1, 1+self.num_ma_dofs)
        self.current_dof_idxs = (self.ma_dof_idxs[1], self.ma_dof_idxs[1] + len(stellarator.get_currents()))
        self.coil_dof_idxs = (self.current_dof_idxs[1], self.current_dof_idxs[1] + len(stellarator.get_dofs()))

        self.x0 = np.concatenate(([qsf.eta_bar], self.ma.get_dofs(), self.stellarator.get_currents()/self.current_fak, self.stellarator.get_dofs()))
        self.x = self.x0.copy()
        self.sobolev_weight = sobolev_weight
        self.tikhonov_weight = tikhonov_weight
        self.arclength_weight = arclength_weight
        self.distance_weight = distance_weight

        self.xiterates = []
        self.Jvals_individual = []
        self.Jvals = []
        self.dJvals = []
        self.outdir = outdir

    def set_dofs(self, x):
        x_etabar = x[0]
        x_ma = x[self.ma_dof_idxs[0]:self.ma_dof_idxs[1]]
        x_current = x[self.current_dof_idxs[0]:self.current_dof_idxs[1]]
        x_coil = x[self.coil_dof_idxs[0]:self.coil_dof_idxs[1]]
        self.t = x[-1]

        self.qsf.eta_bar = x_etabar
        self.ma.set_dofs(x_ma)
        self.biotsavart.set_points(self.ma.gamma)
        self.stellarator.set_currents(self.current_fak * x_current)
        self.stellarator.set_dofs(x_coil)

        self.biotsavart.clear_cached_properties()
        self.qsf.clear_cached_properties()

    def update(self, x, compute_derivative=True):
        self.x[:] = x
        J_BSvsQS          = self.J_BSvsQS
        J_coil_lengths    = self.J_coil_lengths
        J_axis_length     = self.J_axis_length
        J_coil_curvatures = self.J_coil_curvatures
        J_coil_torsions   = self.J_coil_torsions

        iota_target                 = self.iota_target
        magnetic_axis_length_target = self.magnetic_axis_length_target
        curvature_weight             = self.curvature_weight
        torsion_weight               = self.torsion_weight
        qsf = self.qsf

        self.set_dofs(x)

        self.dresetabar  = np.zeros(1)
        self.dresma      = np.zeros(self.ma_dof_idxs[1]-self.ma_dof_idxs[0])
        self.drescurrent = np.zeros(self.current_dof_idxs[1]-self.current_dof_idxs[0])
        self.drescoil    = np.zeros(self.coil_dof_idxs[1]-self.coil_dof_idxs[0])


        """ Objective values """

        self.res1        = 0.5 * J_BSvsQS.J_L2() + 0.5 * J_BSvsQS.J_H1()
        if compute_derivative:
            self.dresetabar  += 0.5 * J_BSvsQS.dJ_L2_by_detabar() + 0.5 * J_BSvsQS.dJ_H1_by_detabar()
            self.dresma      += 0.5 * J_BSvsQS.dJ_L2_by_dmagneticaxiscoefficients() + 0.5 * J_BSvsQS.dJ_H1_by_dmagneticaxiscoefficients()
            self.drescoil    += 0.5 * self.stellarator.reduce_coefficient_derivatives(J_BSvsQS.dJ_L2_by_dcoilcoefficients()) \
                + 0.5 * self.stellarator.reduce_coefficient_derivatives(J_BSvsQS.dJ_H1_by_dcoilcoefficients())
            self.drescurrent += 0.5 * self.current_fak * (
                self.stellarator.reduce_current_derivatives(J_BSvsQS.dJ_L2_by_dcoilcurrents()) + self.stellarator.reduce_current_derivatives(J_BSvsQS.dJ_H1_by_dcoilcurrents())
            )

        self.res2      = 0.5 * sum( (1/l)**2 * (J2.J() - l)**2 for (J2, l) in zip(J_coil_lengths, self.coil_length_targets))
        if compute_derivative:
            self.drescoil += self.stellarator.reduce_coefficient_derivatives([
                (1/l)**2 * (J_coil_lengths[i].J()-l) * J_coil_lengths[i].dJ_by_dcoefficients() for (i, l) in zip(list(range(len(J_coil_lengths))), self.coil_length_targets)])

        self.res3    = 0.5 * (1/magnetic_axis_length_target)**2 * (J_axis_length.J() - magnetic_axis_length_target)**2
        if compute_derivative:
            self.dresma += (1/magnetic_axis_length_target)**2 * (J_axis_length.J()-magnetic_axis_length_target) * J_axis_length.dJ_by_dcoefficients()

        self.res4        = 0.5 * (1/iota_target**2) * (qsf.iota-iota_target)**2
        if compute_derivative:
            self.dresetabar += (1/iota_target**2) * (qsf.iota - iota_target) * qsf.diota_by_detabar[:,0]
            self.dresma     += (1/iota_target**2) * (qsf.iota - iota_target) * qsf.diota_by_dcoeffs[:, 0]

        if curvature_weight > 0:
            self.res5      = sum(curvature_weight * J.J() for J in J_coil_curvatures)
            if compute_derivative:
                self.drescoil += self.curvature_weight * self.stellarator.reduce_coefficient_derivatives([J.dJ_by_dcoefficients() for J in J_coil_curvatures])
        else:
            self.res5 = 0
        if torsion_weight > 0:
            self.res6      = sum(torsion_weight * J.J() for J in J_coil_torsions)
            if compute_derivative:
                self.drescoil += self.torsion_weight * self.stellarator.reduce_coefficient_derivatives([J.dJ_by_dcoefficients() for J in J_coil_torsions])
        else:
            self.res6 = 0

        if self.sobolev_weight > 0:
            self.res7 = sum(self.sobolev_weight * J.J() for J in self.J_sobolev_weights)
            if compute_derivative:
                self.drescoil += self.sobolev_weight * self.stellarator.reduce_coefficient_derivatives([J.dJ_by_dcoefficients() for J in self.J_sobolev_weights[:-1]])
                self.dresma += self.sobolev_weight * self.J_sobolev_weights[-1].dJ_by_dcoefficients()
        else:
            self.res7 = 0

        if self.arclength_weight > 0:
            self.res8 = sum(self.arclength_weight * J.J() for J in self.J_arclength_weights)
            if compute_derivative:
                self.drescoil += self.arclength_weight * self.stellarator.reduce_coefficient_derivatives([J.dJ_by_dcoefficients() for J in self.J_arclength_weights])
        else:
            self.res8 = 0

        if self.distance_weight > 0:
            self.res9 = self.distance_weight * self.J_distance.J()
            if compute_derivative:
                self.drescoil += self.distance_weight * self.stellarator.reduce_coefficient_derivatives(self.J_distance.dJ_by_dcoefficients())
        else:
            self.res9 = 0

        if self.tikhonov_weight > 0:
            self.res_tikhonov_weight = self.tikhonov_weight * np.sum((x-self.x0)**2)
            if compute_derivative:
                dres_tikhonov_weight = self.tikhonov_weight * 2. * (x-self.x0)
                self.dresetabar += dres_tikhonov_weight[0:1]
                self.dresma += dres_tikhonov_weight[self.ma_dof_idxs[0]:self.ma_dof_idxs[1]]
                self.drescurrent += dres_tikhonov_weight[self.current_dof_idxs[0]:self.current_dof_idxs[1]]
                self.drescoil += dres_tikhonov_weight[self.coil_dof_idxs[0]:self.coil_dof_idxs[1]]
        else:
            self.res_tikhonov_weight = 0

        # self.Jvals_individual.append([])
        Jvals_individual = [self.res1, self.res2, self.res3, self.res4, self.res5, self.res6, self.res7, self.res8, self.res9, self.res_tikhonov_weight]
        self.res = sum(Jvals_individual)

        if compute_derivative:
            self.dres = np.concatenate((
                self.dresetabar, self.dresma,
                self.drescurrent, self.drescoil
            ))

    def clear_history(self):
        self.xiterates = []
        self.Jvals_individual = []
        self.Jvals = []
        self.dJvals = []

    def callback(self, x, verbose=True):
        self.update(x)# assert np.allclose(self.x, x)
        self.Jvals.append(self.res)
        norm = np.linalg.norm
        self.dJvals.append((
            norm(self.dres), norm(self.dresetabar), norm(self.dresma), norm(self.drescurrent), norm(self.drescoil)
        ))
        self.xiterates.append(x.copy())

        iteration = len(self.xiterates)-1
        info("################################################################################")
        info(f"Iteration {iteration}")
        norm = np.linalg.norm
        info(f"Objective value:         {self.res:.6e}")
        # info(f"Objective values:        {self.res1:.6e}, {self.res2:.6e}, {self.res3:.6e}, {self.res4:.6e}, {self.res5:.6e}, {self.res6:.6e}, {self.res7:.6e}, {self.res8:.6e}, {self.res9:.6e}, {self.res_tikhonov_weight:.6e}")
        info(f"Objective gradients:     {norm(self.dresetabar):.6e}, {norm(self.dresma):.6e}, {norm(self.drescurrent):.6e}, {norm(self.drescoil):.6e}")

        max_curvature  = max(np.max(c.kappa) for c in self.stellarator._base_coils)
        mean_curvature = np.mean([np.mean(c.kappa) for c in self.stellarator._base_coils])
        max_torsion    = max(np.max(np.abs(c.torsion)) for c in self.stellarator._base_coils)
        mean_torsion   = np.mean([np.mean(np.abs(c.torsion)) for c in self.stellarator._base_coils])
        info(f"Curvature Max: {max_curvature:.3e}; Mean: {mean_curvature:.3e}")
        info(f"Torsion   Max: {max_torsion:.3e}; Mean: {mean_torsion:.3e}")

    def plot(self, backend='plotly'):
        if backend == 'matplotlib':
            import matplotlib
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection="3d")
            for i in range(0, len(self.stellarator.coils)):
                ax = self.stellarator.coils[i].plot(ax=ax, show=False, color=["b", "g", "r", "c", "m", "y"][i%len(self.stellarator._base_coils)])
            self.ma.plot(ax=ax, show=False, closed_loop=False)
            ax.view_init(elev=90., azim=0)
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_zlim(-1, 1)
            plt.show()
        elif backend == 'plotly':
            stellarator = self.stellarator
            coils = stellarator.coils
            ma = self.ma
            gamma = coils[0].gamma
            N = gamma.shape[0]
            l = len(stellarator.coils)
            data = np.zeros((l*(N+1), 4))
            labels = [None for i in range(l*(N+1))]
            for i in range(l):
                data[(i*(N+1)):((i+1)*(N+1)-1),:-1] = stellarator.coils[i].gamma
                data[((i+1)*(N+1)-1),:-1] = stellarator.coils[i].gamma[0, :]
                data[(i*(N+1)):((i+1)*(N+1)),-1] = i
                for j in range(i*(N+1), (i+1)*(N+1)):
                    labels[j] = 'Coil %i ' % stellarator.map[i]
            N = ma.gamma.shape[0]
            ma_ = np.zeros((ma.nfp*N+1, 4))
            ma0 = ma.gamma.copy()
            theta = 2*np.pi/ma.nfp
            rotmat = np.asarray([
                [cos(theta), -sin(theta), 0],
                [sin(theta), cos(theta), 0],
                [0, 0, 1]]).T

            for i in range(ma.nfp):
                ma_[(i*N):(((i+1)*N)), :-1] = ma0
                ma0 = ma0 @ rotmat
            ma_[-1, :-1] = ma.gamma[0,:]
            ma_[:, -1] = -1
            data = np.vstack((data, ma_))
            for i in range(ma_.shape[0]):
                labels.append('Magnetic Axis')
            import plotly.express as px
            fig = px.line_3d(x=data[:,0], y=data[:,1], z=data[:,2],
                             color=labels, line_group=data[:,3].astype(np.int))
            fig.show()
        else:
            raise NotImplementedError('backend must be either matplotlib or plotly')

def plot_stellarator(stellarator, axis=None, extra_data=None):
    coils = stellarator.coils
    gamma = coils[0].gamma
    N = gamma.shape[0]
    l = len(stellarator.coils)
    data = np.zeros((l*(N+1), 3))
    labels = [None for i in range(l*(N+1))]
    groups = [None for i in range(l*(N+1))]
    for i in range(l):
        data[(i*(N+1)):((i+1)*(N+1)-1), :] = stellarator.coils[i].gamma
        data[((i+1)*(N+1)-1), :] = stellarator.coils[i].gamma[0, :]
        for j in range(i*(N+1), (i+1)*(N+1)):
            labels[j] = 'Coil %i ' % stellarator.map[i]
            groups[j] = i+1

    if axis is not None:
        N = axis.gamma.shape[0]
        ma_ = np.zeros((axis.nfp*N+1, 3))
        ma0 = axis.gamma.copy()
        theta = 2*np.pi/axis.nfp
        rotmat = np.asarray([
            [cos(theta), -sin(theta), 0],
            [sin(theta), cos(theta), 0],
            [0, 0, 1]]).T

        for i in range(axis.nfp):
            ma_[(i*N):(((i+1)*N)), :] = ma0
            ma0 = ma0 @ rotmat
        ma_[-1, :] = axis.gamma[0, :]
        data = np.vstack((data, ma_))
        for i in range(ma_.shape[0]):
            labels.append('Magnetic Axis')
            groups.append(0)

    if extra_data is not None:
        for i, extra in enumerate(extra_data):
            labels += ['Extra %i' % i ] * extra.shape[0]
            groups += [-1-i] * extra.shape[0]
            data = np.vstack((data, extra)) 
    import plotly.express as px
    fig = px.line_3d(x=data[:,0], y=data[:,1], z=data[:,2],
                     color=labels, line_group=groups)
    fig.show()
