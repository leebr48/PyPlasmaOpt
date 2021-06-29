import numpy as np
import scipy.integrate
import scipy.interpolate
from pyplasmaopt.biotsavart import BiotSavart
from pyplasmaopt.logging import info
import warnings #FIXME? 

class TangentMap():
    def __init__(self, stellarator, magnetic_axis=None, rtol=1e-10, atol=1e-10,
                constrained=True,bvp_tol=1e-8,tol=1e-8,max_nodes=50000,
                verbose=2,nphi_guess=1000,nphi_integral=1000,
                maxiter=20,axis_bvp=False,adjoint_axis_fd=True,
                adjoint_axis_bvp=True,method='LSODA',
                min_step=1e-10,max_step=1,check_adjoint=False):
        """
        stellarator: instance of CoilCollection representing modular coils
        magnetic_axis: instance of StellaratorSymmetricCylindricalFourierCurve
            representing magnetic axis
        rtol (double): relative tolerance for IVP
        atol (double): absolute tolerance for IVP
        constrained (bool): if true, "true" magnetic axis is computed rather than
            using magnetic_axis
        bvp_tol (double): tolerance for BVP 
        tol (double): tolerance for Newton solve
        max_nodes (int): maximum nodes for BVP
        verbose (int): verbosity for BVP and Newton solves
        nphi_guess (int): number of grid points for guess of axis solutions
        nphi_integral (int): number of grid points for integration along the axis
        maxiter (int): maximum number of Newton iterations for axis solve
        axis_bvp (bool): if True, scipy.integrate.solve_bvp is used to solve for 
            axis. If False, Newton method is used. 
        adjoint_axis_bvp (bool): if True, scipy.integrate.solve_bvp is used to 
            solve for adjoint axis. If False, Newton method is used.             
        method (str): algorithm to use for scipy.integrate.solve_ivp
        """
        self.stellarator = stellarator
        self.biotsavart = BiotSavart(stellarator.coils, stellarator.currents)
        self.magnetic_axis = magnetic_axis
        self.rtol = rtol
        self.atol = atol
        self.bvp_tol = bvp_tol
        self.tol = tol
        self.max_nodes = max_nodes
        self.constrained = constrained
        self.verbose = verbose
        self.nphi_guess = nphi_guess
        self.nphi_integral = nphi_integral
        self.maxiter = maxiter
        self.axis_bvp = axis_bvp
        self.adjoint_axis_bvp = adjoint_axis_bvp
        self.adjoint_axis_fd = adjoint_axis_fd 
        self.method = method 
        self.min_step = min_step
        self.max_step = max_step
        self.check_adjoint = check_adjoint
        # Polynomial solutions for current solutions
        self.axis_poly = None
        self.tangent_poly = None
        self.adjoint_axis_poly = None
        self.adjoint_tangent_poly = None

    def update_solutions(self,derivatives=True):
        """
        Computes solutions for the magnetic axis, tangent map, and corresponding
            adjoint solutions
            
        Inputs:
            derivatives (bool): If True, adjoint solutions required for 
                derivatives are computed.
        """
        phi = np.linspace(0,2*np.pi,self.nphi_guess,endpoint=True)
        phi_reverse = np.linspace(2*np.pi,0,self.nphi_guess,endpoint=True)
        if (self.constrained):
            sol, self.axis_poly = self.compute_axis(phi)
            sol, self.tangent_poly = self.compute_tangent(phi,self.axis_poly)
            if derivatives:
                sol, self.adjoint_tangent_poly = self.compute_adjoint_tangent(phi_reverse,
                                                                     self.axis_poly)
                sol, self.adjoint_axis_poly = self.compute_adjoint_axis(phi,
                         self.axis_poly,self.tangent_poly,self.adjoint_tangent_poly,quantity='iota')
                sol, self.res_adjoint_axis_poly = self.compute_adjoint_axis(phi,
                         self.axis_poly,self.tangent_poly,self.adjoint_tangent_poly,quantity='res')
                
        else:
            sol, self.axis_poly = self.compute_axis(phi)
            sol, self.tangent_poly = self.compute_tangent(phi)
            if derivatives:
                sol, self.adjoint_tangent_poly = self.compute_adjoint_tangent(phi_reverse)
                sol, self.res_adjoint_axis_poly = self.compute_adjoint_axis(phi,
                                                 self.axis_poly,quantity='res')

    def reset_solutions(self):
        """
        Reset solutions
        """
        self.axis_poly = None
        self.tangent_poly = None
        self.adjoint_axis_poly = None
        self.adjoint_tangent_poly = None

    def compute_iota(self):
        """
        Compute rotational transform from tangent map.
        Outputs:
            iota (double): value of rotational transform.
        """
        phi = np.array([2*np.pi])
        if self.tangent_poly is None:
            self.update_solutions()
        M = self.tangent_poly(phi)
        detM = M[0]*M[3] - M[1]*M[2]
        np.testing.assert_allclose(detM,1,rtol=1e-2)
        trM = M[0] + M[3]
        if (np.abs(trM/2)>1):
            raise RuntimeError('Incorrect value of trM.')
        else:
            return -1*np.arccos(trM/2)/(2*np.pi) # The -1 is for a PPO sign convention

    def compute_tangent(self,phi,axis_poly=None,adjoint=False):
        """
        Compute tangent map by solving initial value problem.
        Inputs:
            phi (1d array): 1d array for evaluation of tangent map
            axis_poly: polynomial solution for axis
            adjoint (bool): if True, Jacobian matrix for adjoint axis integration
                is computed
        Outputs:
            y (2d array (4,len(phi))): flattened tangent map on grid of
                toroidal angle
        """
        if axis_poly is not None:
            args = (adjoint,axis_poly)
        else:
            args = (adjoint,)

        y0 = np.array([1,0,0,1])
        t_span = (0,2*np.pi)
        warnings.filterwarnings("error",category=UserWarning) #FIXME?
        try:
            out = scipy.integrate.solve_ivp(self.rhs_fun,t_span,y0,
                                vectorized=False,rtol=self.rtol,atol=self.atol,
                                            t_eval=phi,args=args,dense_output=True,
                                           method=self.method,min_step=self.min_step,
                                           max_step=self.max_step)
        except ValueError:
            raise RuntimeError('solve_ivp failed due to a SciPy bug')
        except UserWarning: #FIXME?
            raise RuntimeError('solve_ivp failed due to convergence issues')
        warnings.resetwarnings() #FIXME?

        if (out.status==0):
            return out.y, out.sol
        else:
            raise RuntimeError('Error ocurred in integration of tangent map.')

    def compute_m(self,phi,axis=None):
        """
        Computes the matrix that appears on the rhs of the tangent map ODE,
            e.g. M'(phi) = m(phi), for given phi.
        Inputs:
            phi (double): toroidal angle for evaluation
            axis (2d array (2,npoints)): R and Z for current axis state
        Outputs:
            m (1d array (4)): matrix appearing on rhs of tangent map ODE
        """
        if (axis is not None):
            if (np.ndim(axis)>1):
                gamma = np.zeros((len(axis[0,:]),3))
            else:
                gamma = np.zeros((1,3))
            gamma[...,0] = axis[0,...]*np.cos(phi)
            gamma[...,1] = axis[0,...]*np.sin(phi)
            gamma[...,2] = axis[1,...]
            self.biotsavart.compute(gamma)
            X = gamma[...,0]
            Y = gamma[...,1]
            Z = gamma[...,2]
        else:
            points = phi/(2*np.pi)
            if (np.ndim(points)==0):
                points = np.array([points])
            self.magnetic_axis.points = points
            self.magnetic_axis.update()
            self.biotsavart.compute(self.magnetic_axis.gamma)
            X = self.magnetic_axis.gamma[...,0]
            Y = self.magnetic_axis.gamma[...,1]
            Z = self.magnetic_axis.gamma[...,2]

        B = self.biotsavart.B
        BX = B[...,0]
        BY = B[...,1]
        BZ = B[...,2]
        gradB = self.biotsavart.dB_by_dX
        dBXdX = gradB[...,0,0]
        dBXdY = gradB[...,1,0]
        dBXdZ = gradB[...,2,0]
        dBYdX = gradB[...,0,1]
        dBYdY = gradB[...,1,1]
        dBYdZ = gradB[...,2,1]
        dBZdX = gradB[...,0,2]
        dBZdY = gradB[...,1,2]
        dBZdZ = gradB[...,2,2]
        R = np.sqrt(X**2 + Y**2)
        BR =  X*BX/R + Y*BY/R
        BP = -Y*BX/R + X*BY/R
        dBRdR = (X**2*dBXdX + X*Y * (dBYdX + dBXdY) + Y**2 * dBYdY)/(R**2)
        dBPdR = (X*Y * (dBYdY-dBXdX) + X**2 * dBYdX - Y**2 * dBXdY)/(R**2)
        dBZdR =  dBZdX*X/R + dBZdY*Y/R
        dBRdZ =  X*dBXdZ/R + Y*dBYdZ/R
        dBPdZ = -Y*dBXdZ/R + X*dBYdZ/R
        if (np.ndim(phi)==0):
            m = np.zeros((4,1))
        else:
            m = np.zeros((4,len(phi)))
        m[0,...] = BR/BP + R*(dBRdR/BP - BR*dBPdR/BP**2)
        m[1,...] = R*(dBRdZ/BP - BR*dBPdZ/BP**2)
        m[2,...] = BZ/BP + R*(dBZdR/BP - BZ*dBPdR/BP**2)
        m[3,...] = R*(dBZdZ/BP - BZ*dBPdZ/BP**2)
        return np.squeeze(m)

    def rhs_fun(self,phi,M,adjoint=False,axis_poly=None):
        """
        Computes the RHS of the tangent map ode, e.g. M'(phi) = rhs
            for given phi and M
        Inputs:
            phi (double): toroidal angle for evaluation
            M (1d array (4)): current value of tangent map
            adjoint (bool): if True, rhs is computed for Jacobian of adjoint
                axis integration
            axis_poly (instance of scipy.interpolate.PPoly cubic spline): polynomial
                representing magnetic axis solution
        Outputs:
            rhs (1d array (4)): rhs of ODE
        """
        if (axis_poly is not None):
            m = self.compute_m(phi,axis_poly(phi))
        else:
            m = self.compute_m(phi)
        assert(np.ndim(phi)==0)

        if adjoint:
            out = -np.squeeze(np.array([m[0]*M[0] + m[2]*M[2], m[0]*M[1] + m[2]*M[3],
                                   m[1]*M[0] + m[3]*M[2], m[1]*M[1] + m[3]*M[3]]))
        else:
            out = np.squeeze(np.array([m[0]*M[0] + m[1]*M[2], m[0]*M[1] + m[1]*M[3],
                           m[2]*M[0] + m[3]*M[2], m[2]*M[1] + m[3]*M[3]]))
        return out

    def d_iota_dmagneticaxiscoeffs(self):
        """
        Compute derivative of iota wrt axis coefficients
        Outputs:
            d_iota (1d array (ncoeffs)): derivative of iota wrt axis coefficients
        """
        phi, dphi = np.linspace(2*np.pi,0,self.nphi_integral,endpoint=False,
                                retstep=True)
        # Update solutions if necessary
        if (self.tangent_poly is None or self.adjoint_tangent_poly is None):
            self.update_solutions()

        d_m = self.compute_d_m_d_magneticaxiscoeffs(phi)
        lam = self.adjoint_tangent_poly(phi)
        M = self.tangent_poly(phi)

        iota = self.compute_iota()
        fac = -1/(4*np.pi*np.sin(2*np.pi*iota))
        lambda_dot_d_m_times_M = \
              lam[0,:,None]*(d_m[0,...]*M[0,...,None] + d_m[1,...]*M[2,...,None]) \
            + lam[1,:,None]*(d_m[0,...]*M[1,...,None] + d_m[1,...]*M[3,...,None]) \
            + lam[2,:,None]*(d_m[2,...]*M[0,...,None] + d_m[3,...]*M[2,...,None]) \
            + lam[3,:,None]*(d_m[2,...]*M[1,...,None] + d_m[3,...]*M[3,...,None])
        d_iota = -fac*np.sum(lambda_dot_d_m_times_M,axis=(0))*dphi

        return d_iota

    def d_iota_dcoilcurrents(self):
        """
        Compute derivative of iota wrt coil currents.
        Outputs:
            d_iota (list of 1d arrays (ncurrents)): derivative of iota wrt
                coil currents
        """
        phi, dphi = np.linspace(2*np.pi,0,self.nphi_integral,endpoint=False,
                                retstep=True)

        if (self.tangent_poly is None):
            self.update_solutions()
        M = self.tangent_poly(phi)
        lam = self.adjoint_tangent_poly(phi)
        d_m_by_dcoilcurrents = self.compute_d_m_dcoilcurrents(phi)
        if self.constrained:
            mu = self.adjoint_axis_poly(phi)
            d_V_by_dcoilcurrents = self.compute_d_V_dcoilcurrents(phi,self.axis_poly)

        iota = self.compute_iota()
        fac = -1/(4*np.pi*np.sin(2*np.pi*iota))
        d_iota_dcoilcurrents = []
        for i in range(len(d_m_by_dcoilcurrents)):
            d_m = d_m_by_dcoilcurrents[i]
            lambda_dot_d_m_times_M = \
                  lam[0,:]*(d_m[0,...]*M[0,...] + d_m[1,...]*M[2,...]) \
                + lam[1,:]*(d_m[0,...]*M[1,...] + d_m[1,...]*M[3,...]) \
                + lam[2,:]*(d_m[2,...]*M[0,...] + d_m[3,...]*M[2,...]) \
                + lam[3,:]*(d_m[2,...]*M[1,...] + d_m[3,...]*M[3,...])
            d_iota = -fac*np.sum(lambda_dot_d_m_times_M)*dphi
            if self.constrained:
                d_V = d_V_by_dcoilcurrents[i]
                mu_dot_d_V = mu[0,:]*d_V[0,...] + mu[1,:]*d_V[1,:]
                d_iota += -np.sum(mu_dot_d_V)*dphi

            d_iota_dcoilcurrents.append(np.squeeze(d_iota))
        d_iota_dcoilcurrents = \
            self.stellarator.reduce_current_derivatives([ires for ires in
                                                         d_iota_dcoilcurrents])

        return d_iota_dcoilcurrents

    def d_iota_dcoilcoeffs(self):
        """
        Compute derivative of iota wrt coil coeffs.
        Outputs:
            d_iota (list of 1d arrays (ncoeffs)): derivatives of iota wrt
                coil coefficients
        """
        phi,dphi = np.linspace(2*np.pi,0,self.nphi_integral,endpoint=False,
                               retstep=True)
        if (self.tangent_poly is None):
            self.update_solutions()
            
        iota = self.compute_iota()
        fac = -1/(4*np.pi*np.sin(2*np.pi*iota))
            
        M = self.tangent_poly(phi)
        lam = self.adjoint_tangent_poly(phi)
        d_m_by_dcoilcoeffs = self.compute_d_m_dcoilcoeffs(phi)
        if self.constrained:
            mu = self.adjoint_axis_poly(phi)
            d_V_by_dcoilcoeffs = self.compute_d_V_dcoilcoeffs(phi,self.axis_poly)

        d_iota_dcoilcoeffs = []
        
        for i in range(len(self.stellarator.coils)):
            d_m = d_m_by_dcoilcoeffs[i]
            lambda_dot_d_m_times_M = \
                  lam[0,:,None]*(d_m[0,...]*M[0,...,None] + d_m[1,...]*M[2,...,None]) \
                + lam[1,:,None]*(d_m[0,...]*M[1,...,None] + d_m[1,...]*M[3,...,None]) \
                + lam[2,:,None]*(d_m[2,...]*M[0,...,None] + d_m[3,...]*M[2,...,None]) \
                + lam[3,:,None]*(d_m[2,...]*M[1,...,None] + d_m[3,...]*M[3,...,None])
            d_iota = -fac*np.sum(lambda_dot_d_m_times_M,axis=(0))*dphi
            if self.constrained:
                d_V = d_V_by_dcoilcoeffs[i]
                mu_dot_d_V = mu[0,...,None]*d_V[0,...] + mu[1,...,None]*d_V[1,...]
                d_iota += -np.sum(mu_dot_d_V,axis=(0))*dphi

            d_iota_dcoilcoeffs.append(d_iota)
        d_iota_dcoilcoeffs = self.stellarator.reduce_coefficient_derivatives([ires for ires in d_iota_dcoilcoeffs])

        return d_iota_dcoilcoeffs

    def compute_adjoint_tangent(self,phi,axis_poly=None):
        """
        For biotsavart and magnetic_axis objects, compute adjoint tangent map
            by solving initial value probme.
        Inputs:
            phi (1d array): toroidal angle for evaluation of adjoint variable
        """
        t_span = (2*np.pi,0)
        y0 = np.array([1,0,0,1])
        if self.constrained:
            args = (axis_poly,)
        else:
            args = ()
        out = scipy.integrate.solve_ivp(self.adjoint_rhs_fun,t_span,y0,
                                vectorized=False,rtol=self.rtol,atol=self.atol,
                                       t_eval=phi,args=args,dense_output=True,
                                       method=self.method,min_step=self.min_step,
                                       max_step=self.max_step)
        if (out.status==0):
            return out.y, out.sol
        else:
            raise RuntimeError('Error ocurred in integration of adjoint tangent map.')

    def adjoint_rhs_fun(self,phi,M,axis_poly=None):
        """
        Computes the RHS of the adjoint tangent map ODE, e.g. lambda'(phi) = rhs
            for given phi and lambda
        Inputs:
            phi (double): toroidal angle for evaluation
            lambda (1d array (4)): current value of adjoint tangent map
        Outputs:
            rhs (1d array (4)): rhs of ode
        """
        if axis_poly is not None:
            m = self.compute_m(phi,axis_poly(phi))
        else:
            m = self.compute_m(phi)

        return -np.squeeze(np.array([m[0]*M[0] + m[2]*M[2], m[0]*M[1] + m[2]*M[3],
                                     m[1]*M[0] + m[3]*M[2], m[1]*M[1] + m[3]*M[3]]))

    def compute_d_m_dcoilcoeffs(self,phi):
        """
        Computes the derivative of matrix that appears on the rhs of the tangent map ode,
        e.g. M'(phi) = m(phi) rhs, with respect to coil coeffs for given phi.
        Inputs:
            phi (1d array): toroidal angles for evaluation
        Outputs:
            d_m_dcoilcoeffs (list (ncoils) of 3d array (npoints,4,ncoeffs)): derivative
                of matrix appearing on rhs on ode wrt coil coeffs
        """
        if self.constrained:
            axis = self.axis_poly(phi)
            gamma = np.zeros((len(phi),3))
            gamma[...,0] = axis[0,...]*np.cos(phi)
            gamma[...,1] = axis[0,...]*np.sin(phi)
            gamma[...,2] = axis[1,...]
            self.biotsavart.compute(gamma)
            self.biotsavart.compute_by_dcoilcoeff(gamma)
            X = gamma[...,0]
            Y = gamma[...,1]
            Z = gamma[...,2]
        else:
            points = phi/(2*np.pi)
            self.magnetic_axis.points = np.asarray(points)
            self.magnetic_axis.update()
            self.biotsavart.compute(self.magnetic_axis.gamma)
            self.biotsavart.compute_by_dcoilcoeff(self.magnetic_axis.gamma)
            X = self.magnetic_axis.gamma[...,0]
            Y = self.magnetic_axis.gamma[...,1]
            Z = self.magnetic_axis.gamma[...,2]

        B = self.biotsavart.B
        BX = B[...,0]
        BY = B[...,1]
        BZ = B[...,2]

        gradB = self.biotsavart.dB_by_dX
        dBXdX = gradB[...,0,0]
        dBXdY = gradB[...,1,0]
        dBXdZ = gradB[...,2,0]
        dBYdX = gradB[...,0,1]
        dBYdY = gradB[...,1,1]
        dBYdZ = gradB[...,2,1]
        dBZdX = gradB[...,0,2]
        dBZdY = gradB[...,1,2]
        dBZdZ = gradB[...,2,2]

#         X = self.biotsavart.points[:,0]
#         Y = self.biotsavart.points[:,1]
#         Z = self.biotsavart.points[:,2]
        R = np.sqrt(X**2 + Y**2)
        BR =  X*BX/R + Y*BY/R
        BP = -Y*BX/R + X*BY/R
        dBRdR = (X**2*dBXdX + X*Y * (dBYdX + dBXdY) + Y**2 * dBYdY)/(R**2)
        dBPdR = (X*Y * (dBYdY-dBXdX) + X**2 * dBYdX - Y**2 * dBXdY)/(R**2)
        dBZdR =  dBZdX*X/R + dBZdY*Y/R
        dBRdZ =  dBXdZ*X/R + dBYdZ*Y/R
        dBPdZ = -dBXdZ*Y/R + dBYdZ*X/R

        R = R[:,None]
        BR = BR[:,None]
        BZ = BZ[:,None]
        BP = BP[:,None]
        X = X[:,None]
        Y = Y[:,None]
        dBRdR = dBRdR[:,None]
        dBPdR = dBPdR[:,None]
        dBZdR = dBZdR[:,None]
        dBRdZ = dBRdZ[:,None]
        dBPdZ = dBPdZ[:,None]
        dBZdZ = dBZdZ[:,None]

        # Shape: (ncoils,npoints,nparams,3)
        dB_by_dcoilcoeffs = self.biotsavart.dB_by_dcoilcoeffs
        dgradB_by_dcoilcoeffs = self.biotsavart.d2B_by_dXdcoilcoeffs

        d_m_by_dcoilcoeffs = []
        for i in range(len(dB_by_dcoilcoeffs)):
            d_B = dB_by_dcoilcoeffs[i]
            d_BX = d_B[...,0]
            d_BY = d_B[...,1]
            d_BZ = d_B[...,2]

            d_gradB = dgradB_by_dcoilcoeffs[i]
            d_dBXdX = d_gradB[...,0,0]
            d_dBXdY = d_gradB[...,1,0]
            d_dBXdZ = d_gradB[...,2,0]
            d_dBYdX = d_gradB[...,0,1]
            d_dBYdY = d_gradB[...,1,1]
            d_dBYdZ = d_gradB[...,2,1]
            d_dBZdX = d_gradB[...,0,2]
            d_dBZdY = d_gradB[...,1,2]
            d_dBZdZ = d_gradB[...,2,2]

            d_BR =  X * d_BX/R + Y * d_BY/R
            d_dBRdR = (X**2*d_dBXdX
                    + X*Y * (d_dBYdX + d_dBXdY)
                    + Y**2 * d_dBYdY)/(R**2)
            d_BP = -Y * d_BX/R + X * d_BY/R
            d_dBPdR = (X*Y * (d_dBYdY-d_dBXdX)
                    + X**2 * d_dBYdX
                    - Y**2 * d_dBXdY)/(R**2)
            d_dBZdR =  d_dBZdX*X/R + d_dBZdY*Y/R
            d_dBRdZ =  d_dBXdZ*X/R + d_dBYdZ*Y/R
            d_dBPdZ = -d_dBXdZ*Y/R + d_dBYdZ*X/R

            d_m = np.zeros((4,np.shape(d_BR)[0],np.shape(d_BR)[1]))
#             d_m[...,0] = BR/BP + R*(dBRdR/BP - BR*dBPdR/BP**2)
            d_m[0,...] = d_BR/BP - BR*d_BP/(BP*BP) \
                + R*(d_dBRdR/BP - dBRdR*d_BP/BP**2 - d_BR*dBPdR/BP**2
                    - BR*d_dBPdR/BP**2 + 2*BR*dBPdR*d_BP/(BP**3))
#             d_m[...,1] = R*(dBRdZ/BP - BR*dBPdZ/BP**2)
            d_m[1,...] = R*(d_dBRdZ/BP - dBRdZ*d_BP/BP**2 - d_BR*dBPdZ/BP**2
                    - BR*d_dBPdZ/BP**2 + 2*BR*dBPdZ*d_BP/BP**3)
#             d_m[...,2] = BZ/BP + R*(dBZdR/BP - BZ*dBPdR/BP**2)
            d_m[2,...] = d_BZ/BP - BZ*d_BP/BP**2 \
                + R*(d_dBZdR/BP - dBZdR*d_BP/BP**2 - d_BZ*dBPdR/BP**2
                    - BZ*d_dBPdR/BP**2 + 2*BZ*dBPdR*d_BP/(BP**3))
#             d_m[...,3] = R*(dBZdZ/BP - BZ*dBPdZ/BP**2)
            d_m[3,...] = R*(d_dBZdZ/BP - dBZdZ*d_BP/BP**2
                    - d_BZ*dBPdZ/BP**2 - BZ*d_dBPdZ/BP**2 + 2*BZ*dBPdZ*d_BP/BP**3)
            d_m_by_dcoilcoeffs.append(d_m)

        return d_m_by_dcoilcoeffs

    def compute_d_m_dcoilcurrents(self,phi):
        """
        Computes the derivative of  matrix that appears on the rhs of the tangent map ode,
            e.g. M'(phi) = m(phi) rhs, with respect to coil coeffs for given phi.
        Inputs:
            phi (1d array): toroidal angles for evaluation
        Outputs:
            d_m_dcoilcurrents (list (ncoils) of 2d array (4,npoints)): derivative
                of matrix appearing on rhs on ode wrt coil currents
        """
        if self.constrained:
            axis = self.axis_poly(phi)
            gamma = np.zeros((len(phi),3))
            gamma[...,0] = axis[0,...]*np.cos(phi)
            gamma[...,1] = axis[0,...]*np.sin(phi)
            gamma[...,2] = axis[1,...]
            self.biotsavart.compute(gamma)
            X = gamma[...,0]
            Y = gamma[...,1]
            Z = gamma[...,2]
        else:
            points = phi/(2*np.pi)
            self.magnetic_axis.points = np.asarray(points)
            self.magnetic_axis.update()
            self.biotsavart.compute(self.magnetic_axis.gamma)
            X = self.magnetic_axis.gamma[...,0]
            Y = self.magnetic_axis.gamma[...,1]
            Z = self.magnetic_axis.gamma[...,2]
            
        B = self.biotsavart.B
        BX = B[...,0]
        BY = B[...,1]
        BZ = B[...,2]

        gradB = self.biotsavart.dB_by_dX
        dBXdX = gradB[...,0,0]
        dBXdY = gradB[...,1,0]
        dBXdZ = gradB[...,2,0]
        dBYdX = gradB[...,0,1]
        dBYdY = gradB[...,1,1]
        dBYdZ = gradB[...,2,1]
        dBZdX = gradB[...,0,2]
        dBZdY = gradB[...,1,2]
        dBZdZ = gradB[...,2,2]

#         X = self.biotsavart.points[:,0]
#         Y = self.biotsavart.points[:,1]
#         Z = self.biotsavart.points[:,2]
        R = np.sqrt(X**2 + Y**2)
        BR =  X*BX/R + Y*BY/R
        BP = -Y*BX/R + X*BY/R
        dBRdR = (X**2*dBXdX + X*Y * (dBYdX + dBXdY) + Y**2 * dBYdY)/(R**2)
        dBPdR = (X*Y * (dBYdY-dBXdX) + X**2 * dBYdX - Y**2 * dBXdY)/(R**2)
        dBZdR =  dBZdX*X/R + dBZdY*Y/R
        dBRdZ =  dBXdZ*X/R + dBYdZ*Y/R
        dBPdZ = -dBXdZ*Y/R + dBYdZ*X/R

        # Shape: (ncoils,npoints,3)
        dB_by_dcoilcurrents = self.biotsavart.dB_by_dcoilcurrents
        dgradB_by_dcoilcurrents = self.biotsavart.d2B_by_dXdcoilcurrents

        d_m_by_dcoilcurrents = []
        for i in range(len(dB_by_dcoilcurrents)):
            d_B = dB_by_dcoilcurrents[i]
            d_BX = d_B[...,0]
            d_BY = d_B[...,1]
            d_BZ = d_B[...,2]

            d_gradB = dgradB_by_dcoilcurrents[i]
            d_dBXdX = d_gradB[...,0,0]
            d_dBXdY = d_gradB[...,1,0]
            d_dBXdZ = d_gradB[...,2,0]
            d_dBYdX = d_gradB[...,0,1]
            d_dBYdY = d_gradB[...,1,1]
            d_dBYdZ = d_gradB[...,2,1]
            d_dBZdX = d_gradB[...,0,2]
            d_dBZdY = d_gradB[...,1,2]
            d_dBZdZ = d_gradB[...,2,2]

            d_BR =  (X * d_BX + Y * d_BY)/R
            d_dBRdR = (X**2*d_dBXdX
                    + X*Y * (d_dBYdX + d_dBXdY)
                    + Y**2 * d_dBYdY)/(R**2)
            d_BP = (-Y * d_BX + X * d_BY)/R
            d_dBPdR = (X*Y * (d_dBYdY-d_dBXdX)
                    + X**2 * d_dBYdX
                    - Y**2 * d_dBXdY)/(R**2)
            d_dBZdR =  d_dBZdX*X/R + d_dBZdY*Y/R
            d_dBRdZ =  d_dBXdZ*X/R + d_dBYdZ*Y/R
            d_dBPdZ = -d_dBXdZ*Y/R + d_dBYdZ*X/R

            d_m = np.zeros((4,np.shape(d_BR)[0]))
#             d_m[...,0] = BR/BP + R*(dBRdR/BP - BR*dBPdR/BP**2)
            d_m[0,...] = d_BR/BP - BR*d_BP/BP**2 \
                + R*(d_dBRdR/BP - dBRdR*d_BP/BP**2 - d_BR*dBPdR/BP**2
                    - BR*d_dBPdR/BP**2 + 2*BR*dBPdR*d_BP/BP**3)
#             d_m[...,1] = R*(dBRdZ/BP - BR*dBPdZ/BP**2)
            d_m[1,...] = R*(d_dBRdZ/BP - dBRdZ*d_BP/BP**2 - d_BR*dBPdZ/BP**2
                    - BR*d_dBPdZ/BP**2 + 2*BR*dBPdZ*d_BP/BP**3)
#             d_m[...,2] = BZ/BP + R*(dBZdR/BP - BZ*dBPdR/BP**2)
            d_m[2,...] = d_BZ/BP - BZ*d_BP/BP**2 \
                + R*(d_dBZdR/BP - dBZdR*d_BP/BP**2 - d_BZ*dBPdR/BP**2
                    - BZ*d_dBPdR/BP**2 + 2*BZ*dBPdR*d_BP/(BP**3))
#             d_m[...,3] = R*(dBZdZ/BP - BZ*dBPdZ/BP**2)
            d_m[3,...] = R*(d_dBZdZ/BP - dBZdZ*d_BP/BP**2
                    - d_BZ*dBPdZ/BP**2 - BZ*d_dBPdZ/BP**2 + 2*BZ*dBPdZ*d_BP/BP**3)
            d_m_by_dcoilcurrents.append(d_m)

        return d_m_by_dcoilcurrents

    def compute_d_m_d_magneticaxiscoeffs(self,phi):
        """
        Computes the derivative of  matrix that appears on the rhs of the tangent map ode,
            e.g. M'(phi) = m(phi) M(phi), with respect to axis coefficients for given phi.
        Inputs:
            phi (1d array): toroidal angles for evaluation
        Outputs:
            d_m_dcoilcoeffs (3d array (npoints,ncoeffs,4)): derivative
                of matrix appearing on rhs on ode wrt axis coeffs
        """
        dmdR, dmdZ = self.compute_grad_m(phi)
        
        points = phi/(2*np.pi)
        self.magnetic_axis.points = np.asarray(points)
        self.magnetic_axis.update()
        self.biotsavart.compute(self.magnetic_axis.gamma)
        
        X = self.magnetic_axis.gamma[...,0]
        Y = self.magnetic_axis.gamma[...,1]
        Z = self.magnetic_axis.gamma[...,2]
        R = np.sqrt(X**2 + Y**2)
        
        dgamma_by_dcoeff  = self.magnetic_axis.dgamma_by_dcoeff

        d_X = dgamma_by_dcoeff[...,0]
        d_Y = dgamma_by_dcoeff[...,1]
        d_R = (d_X*X[:,None] + d_Y*Y[:,None])/R[:,None]
        d_Z = dgamma_by_dcoeff[...,2]
        return dmdR[:,:,None]*d_R[None,:,:] + dmdZ[:,:,None]*d_Z[None,:,:]

    def compute_grad_m(self,phi,axis_poly=None):
        """
        Computes the derivative of  matrix that appears on the rhs of the
            tangent map ode, e.g. M'(phi) = m(phi) M(phi), with respect, to
            cylindrical R and Z.
        Inputs:
            phi (1d array): toroidal angles for evaluation
            axis_poly : polynomial representation of axis solution
        Outputs:
            d_m_d_R (1d array (len(phi))): derivative of m wrt to R
            d_m_d_Z (1d array (len(phi))): derivative of m wrt to Z
        """
        if axis_poly is not None:
            axis = axis_poly(phi)
            if (np.ndim(phi) > 0):
                gamma = np.zeros((len(phi),3))
            else:
                gamma = np.zeros((1,3))
            gamma[...,0] = axis[0,...]*np.cos(phi)
            gamma[...,1] = axis[0,...]*np.sin(phi)
            gamma[...,2] = axis[1,...]
            self.biotsavart.compute(gamma)
            X = gamma[...,0]
            Y = gamma[...,1]
            Z = gamma[...,2]
        else:
            points = phi/(2*np.pi)
            self.magnetic_axis.points = np.asarray(points)
            self.magnetic_axis.update()
            self.biotsavart.compute(self.magnetic_axis.gamma)
            X = self.magnetic_axis.gamma[...,0]
            Y = self.magnetic_axis.gamma[...,1]
            Z = self.magnetic_axis.gamma[...,2]

        B = self.biotsavart.B
        BX = B[...,0]
        BY = B[...,1]
        BZ = B[...,2]

        gradB = self.biotsavart.dB_by_dX
        dBXdX = gradB[...,0,0]
        dBXdY = gradB[...,1,0]
        dBXdZ = gradB[...,2,0]
        dBYdX = gradB[...,0,1]
        dBYdY = gradB[...,1,1]
        dBYdZ = gradB[...,2,1]
        dBZdX = gradB[...,0,2]
        dBZdY = gradB[...,1,2]
        dBZdZ = gradB[...,2,2]
        
        # Shape: ((len(points), 3, 3, 3))
        d2Bbs_by_dXdX = self.biotsavart.d2B_by_dXdX
        d2BXdX2 = d2Bbs_by_dXdX[...,0,0,0]
        d2BYdX2 = d2Bbs_by_dXdX[...,0,0,1]
        d2BZdX2 = d2Bbs_by_dXdX[...,0,0,2]
        d2BXdY2 = d2Bbs_by_dXdX[...,1,1,0]
        d2BYdY2 = d2Bbs_by_dXdX[...,1,1,1]
        d2BZdY2 = d2Bbs_by_dXdX[...,1,1,2]
        d2BXdZ2 = d2Bbs_by_dXdX[...,2,2,0]
        d2BYdZ2 = d2Bbs_by_dXdX[...,2,2,1]
        d2BZdZ2 = d2Bbs_by_dXdX[...,2,2,2]
        d2BXdXdY = d2Bbs_by_dXdX[...,0,1,0]
        d2BYdXdY = d2Bbs_by_dXdX[...,0,1,1]
        d2BZdXdY = d2Bbs_by_dXdX[...,0,1,2]
        d2BXdXdZ = d2Bbs_by_dXdX[...,0,2,0]
        d2BYdXdZ = d2Bbs_by_dXdX[...,0,2,1]
        d2BZdXdZ = d2Bbs_by_dXdX[...,0,2,2]
        d2BXdYdZ = d2Bbs_by_dXdX[...,1,2,0]
        d2BYdYdZ = d2Bbs_by_dXdX[...,1,2,1]
        d2BZdYdZ = d2Bbs_by_dXdX[...,1,2,2]

        R = np.sqrt(X**2 + Y**2)
        dRdX = X/R
        dRdY = Y/R
        d2RdX2 = 1/R - dRdX*dRdX/R
        d2RdY2 = 1/R - dRdY*dRdY/R
        d2RdXdY = -dRdX*dRdY/R
        
        BR = (X*BX + Y*BY)/R
        BP = (-Y*BX + X*BY)/R
        
        """
        BR = (X*BX + Y*BY)/R
        """
        dBRdZ = (X*dBXdZ + Y*dBYdZ)/R
        dBRdY = (X*dBXdY + BY + Y * dBYdY)/R - dRdY*BR/R
        dBRdX = (BX + X*dBXdX + Y*dBYdX)/R - dRdX*BR/R
        
        """
        BP = (-Y*BX + X*BY)/R
        """
        dBPdX = (-Y*dBXdX + BY + X*dBYdX)/R - dRdX*BP/R
        dBPdY = (-BX -Y*dBXdY + X*dBYdY)/R - dRdY*BP/R
        dBPdZ = (-Y*dBXdZ + X*dBYdZ)/R 

        """
        dBRdZ = (X*dBXdZ + Y*dBYdZ)/R
        """
        
        d2BRdZ2 = (X*d2BXdZ2 + Y*d2BYdZ2)/R
        
        """
        dBRdY = (X*dBXdY + BY + Y * dBYdY - dRdY*BR)/R
        """
        
        d2BRdY2 = (X*d2BXdY2 + 2*dBYdY + Y*d2BYdY2
                  - d2RdY2*BR - dRdY*dBRdY)/R \
                - dRdY*dBRdY/R
        d2BRdYdZ = (X*d2BXdYdZ + dBYdZ + Y*d2BYdYdZ - dRdY*dBRdZ)/R \
        
        """
        dBRdX = (BX + X*dBXdX + Y*dBYdX - dRdX*BR)/R 
        """

        d2BRdX2 = (2*dBXdX + X*d2BXdX2 + Y*d2BYdX2 - d2RdX2*BR - dRdX*dBRdX)/R \
                - dRdX*dBRdX/R
        d2BRdXdY = (dBXdY + X*d2BXdXdY + dBYdX + Y*d2BYdXdY - d2RdXdY*BR - dRdX*dBRdY)/R \
                - dRdY*dBRdX/R
        d2BRdXdZ = (dBXdZ + X*d2BXdXdZ + Y*d2BYdXdZ - dRdX*dBRdZ)/R 
        
        """
        dBPdX = (-Y*dBXdX + BY + X*dBYdX - dRdX*BP)/R
        """
                
        d2BPdX2 = (-Y*d2BXdX2 + dBYdX + dBYdX + X*d2BYdX2 - d2RdX2*BP - dRdX*dBPdX)/R \
                - dRdX*dBPdX/R
        d2BPdXdY = (-dBXdX - Y*d2BXdXdY + dBYdY + X*d2BYdXdY- d2RdXdY*BP - dRdX*dBPdY)/R \
                - dRdY*dBPdX/R
        d2BPdXdZ = (-Y*d2BXdXdZ + dBYdZ + X*d2BYdXdZ - dRdX*dBPdZ)/R 
        
        """
        dBPdY = (-BX - Y*dBXdY + X*dBYdY - dRdY*BP)/R
        """
        
        d2BPdY2 = (-2*dBXdY - Y*d2BXdY2 + X*d2BYdY2 - d2RdY2*BP - dRdY*dBPdY)/R \
                - dRdY*dBPdY/R
        d2BPdYdZ = (-dBXdZ -Y*d2BXdYdZ + X*d2BYdYdZ - dRdY*dBPdZ)/R 
        
        """
        dBPdZ = (-Y*dBXdZ + X*dBYdZ)/R 
        """
                
        d2BPdZ2 = (-Y*d2BXdZ2 + X*d2BYdZ2)/R

        """
        dBRdR
        """

        dBRdR = (dBRdX*X + dBRdY*Y)/R
        
        dBRdR_dX = (d2BRdX2*X + dBRdX + d2BRdXdY*Y)/R \
            - dBRdR*dRdX/R
        dBRdR_dY = (d2BRdXdY*X + d2BRdY2*Y + dBRdY)/R \
            - dBRdR*dRdY/R
        d2BRdRdZ = (d2BRdXdZ*X + d2BRdYdZ*Y)/R
        
        d2BRdR2 = (dBRdR_dX*X + dBRdR_dY*Y)/R
        
        """
        dBPdR
        """
        
        dBPdR = (dBPdX*X + dBPdY*Y)/R
        
        dBPdR_dX = (d2BPdX2*X + dBPdX + d2BPdXdY*Y)/R \
            - dBPdR*dRdX/R
        dBPdR_dY = (d2BPdXdY*X + d2BPdY2*Y + dBPdY)/R \
            - dBPdR*dRdY/R
        d2BPdRdZ = (d2BPdXdZ*X + d2BPdYdZ*Y)/R
        d2BPdR2 = (dBPdR_dX*X + dBPdR_dY*Y)/R
        
        """
        dBZdR
        """
        
        dBZdR = (dBZdX*X + dBZdY*Y)/R
        
        dBZdR_dX = (d2BZdX2*X + dBZdX + d2BZdXdY*Y)/R \
            - dBZdR*dRdX/R 
        dBZdR_dY = (d2BZdXdY*X + d2BZdY2*Y + dBZdY)/R \
            - dBZdR*dRdY/R
        d2BZdRdZ = (d2BZdXdZ*X + d2BZdYdZ*Y)/R
        d2BZdR2 = (dBZdR_dX*X + dBZdR_dY*Y)/R
        
        """
        dBRdZ
        """
        
        dBRdZ = (dBXdZ*X + dBYdZ*Y)/R
        
        dBRdZ_dX = (d2BXdXdZ*X + dBXdZ + d2BYdXdZ*Y)/R \
            - dBRdZ*dRdX/R
        dBRdZ_dY = (d2BXdYdZ*X + d2BYdYdZ*Y + dBYdZ)/R \
            - dBRdZ*dRdY/R
        d2BRdZ2  = (d2BXdZ2*X + d2BYdZ2*Y)/R
        d2BRdZdR = (dBRdZ_dX*X + dBRdZ_dY*Y)/R
                
        """
        dBPdZ
        """
        
        dBPdZ = (-dBXdZ*Y + dBYdZ*X)/R

        dBPdZ_dX = (-d2BXdXdZ*Y + d2BYdXdZ*X + dBYdZ)/R \
            - dBPdZ*dRdX/R
        dBPdZ_dY = (-d2BXdYdZ*Y - dBXdZ + d2BYdYdZ*X)/R \
            - dBPdZ*dRdY/R
        d2BPdZ2 = (-d2BXdZ2*Y + d2BYdZ2*X)/R
        d2BPdZdR = (dBPdZ_dX*X + dBPdZ_dY*Y)/R
                
        dmdR = np.zeros((4,len(R)))
        dmdZ = np.zeros((4,len(Z)))
        
        """
        m[...,0] = BR/BP + R*(dBRdR/BP - BR*dBPdR/BP**2)
        """

        dmdR[0,...] = dBRdR/BP - BR*dBPdR/(BP**2) + (dBRdR/BP - BR*dBPdR/BP**2) \
            + R*(d2BRdR2/BP - dBRdR*dBPdR/BP**2 - dBRdR*dBPdR/BP**2 
                - BR*d2BPdR2/BP**2 + 2*BR*dBPdR*dBPdR/BP**3)
        dmdZ[0,...] = dBRdZ/BP - BR*dBPdZ/(BP**2) \
            + R*(d2BRdRdZ/BP - dBRdR*dBPdZ/BP**2 - dBRdZ*dBPdR/BP**2
                - BR*d2BPdRdZ/BP**2 + 2*BR*dBPdR*dBPdZ/BP**3)
        
        """
        m[...,1] = R*(dBRdZ/BP - BR*dBPdZ/BP**2)
        """
        
        dmdR[1,...] = (dBRdZ/BP - BR*dBPdZ/BP**2) \
            + R*(d2BRdRdZ/BP - dBRdZ*dBPdR/BP**2 - dBRdR*dBPdZ/BP**2
                - BR*d2BPdRdZ/BP**2 + 2*BR*dBPdZ*dBPdR/BP**3)
        dmdZ[1,...] = R*(d2BRdZ2/BP - dBRdZ*dBPdZ/BP**2 
                        - dBRdZ*dBPdZ/BP**2 - BR*d2BPdZ2/BP**2 
                        + 2*BR*dBPdZ*dBPdZ/BP**3)
        
        """
        m[...,2] = BZ/BP + R*(dBZdR/BP - BZ*dBPdR/BP**2)
        """
        dmdR[2,...] = dBZdR/BP - BZ*dBPdR/BP**2 + (dBZdR/BP - BZ*dBPdR/BP**2) \
            + R*(d2BZdR2/BP - dBZdR*dBPdR/BP**2 - dBZdR*dBPdR/BP**2
                - BZ*d2BPdR2/BP**2 + 2*BZ*dBPdR*dBPdR/BP**3)
        dmdZ[2,...] = dBZdZ/BP - BZ*dBPdZ/BP**2 \
            + R*(d2BZdRdZ/BP - dBZdR*dBPdZ/BP**2 - dBZdZ*dBPdR/BP**2
                - BZ*d2BPdRdZ/BP**2 + 2*BZ*dBPdR*dBPdZ/BP**3)
        
        """
        m[...,3] = R*(dBZdZ/BP - BZ*dBPdZ/BP**2)
        """
        dmdR[3,...] = (dBZdZ/BP - BZ*dBPdZ/BP**2) \
            + R*(d2BZdRdZ/BP - dBZdZ*dBPdR/BP**2 - dBZdR*dBPdZ/BP**2
                - BZ*d2BPdRdZ/BP**2 + 2*BZ*dBPdZ*dBPdR/BP**3)
        dmdZ[3,...] = R*(d2BZdZ2/BP - dBZdZ*dBPdZ/BP**2 - dBZdZ*dBPdZ/BP**2
                        - BZ*d2BPdZ2/BP**2 + 2*BZ*dBPdZ*dBPdZ/BP**3)

        return dmdR, dmdZ

    def res_axis(self):
        """
        Computes the residual between parameterization axis and "true" magnetic
            axis
        Outputs:
            res_axis (double): residual between parameterization axis
                and true axis
        """
        phi, dphi = np.linspace(0,2*np.pi,self.nphi_integral,endpoint=False,
                                retstep=True)
        if self.axis_poly is None:
            self.update_solutions()
        axis = self.axis_poly(phi)

        self.magnetic_axis.points = np.asarray(phi/(2*np.pi))
        self.magnetic_axis.update()
        Rma = np.sqrt(self.magnetic_axis.gamma[:,0]**2 + self.magnetic_axis.gamma[:,1]**2)
        Zma = self.magnetic_axis.gamma[:,2]
        return 0.5*np.sum((axis[0,:]-Rma)**2 + (axis[1,:]-Zma)**2)*dphi

    def d_res_axis_d_magneticaxiscoeffs(self):
        """
        Compute derivative of res_axis wrt axis coefficients
        Outputs:
            d_res_axis_d_magneticaxiscoeffs (1d array (ncoeffs)): derivative of
                residual between parameterization axis and true axis wrt axis
                coeffs
        """
        phi, dphi = np.linspace(0,2*np.pi,self.nphi_integral,endpoint=False,retstep=True)

        if self.axis_poly is None:
            self.update_solutions()
        axis = self.axis_poly(phi)

        self.magnetic_axis.points = np.asarray(phi/(2*np.pi))
        self.magnetic_axis.update()

        Rma = np.sqrt(self.magnetic_axis.gamma[:,0]**2 + self.magnetic_axis.gamma[:,1]**2)
        Zma = self.magnetic_axis.gamma[:,2]
        d_Rma = (self.magnetic_axis.dgamma_by_dcoeff[...,0]*self.magnetic_axis.gamma[:,0,None] +
                 self.magnetic_axis.dgamma_by_dcoeff[...,1]*self.magnetic_axis.gamma[:,1,None]) \
            / Rma[:,None]
        d_Zma = self.magnetic_axis.dgamma_by_dcoeff[...,2]

        return np.sum((Rma[:,None]-axis[0,:,None])*d_Rma
                    + (Zma[:,None]-axis[1,:,None])*d_Zma,axis=0)*dphi

    def d_res_axis_d_coil_currents(self):
        """
        Compute derivative of res_axis wrt coil currents
        Outputs:
            d_res_axis_d_coil_currents (list of doubles): derivatives of
                residual between parameterization axis and true axis wrt coil
                currents
        """
        phi,dphi = np.linspace(0,2*np.pi,self.nphi_integral,endpoint=False,retstep=True)

        if (self.axis_poly is None or self.adjoint_axis_poly is None):
            self.update_solutions()
        axis = self.axis_poly(phi)
        mu = self.res_adjoint_axis_poly(phi)
        
        d_V_by_dcoilcurrents = self.compute_d_V_dcoilcurrents(phi,self.axis_poly)
        
        d_res_axis_dcoilcurrents = []
        for i in range(len(d_V_by_dcoilcurrents)):
            d_V = d_V_by_dcoilcurrents[i]
            mu_dot_d_V = mu[0,:]*d_V[0,...] + mu[1,:]*d_V[1,:]
            d_res_axis = - np.sum(mu_dot_d_V)*dphi
            d_res_axis_dcoilcurrents.append(d_res_axis)
        d_res_axis_dcoilcurrents = \
            self.stellarator.reduce_current_derivatives([ires for ires in d_res_axis_dcoilcurrents])

        return d_res_axis_dcoilcurrents

    def d_res_axis_d_coil_coeffs(self):
        """
        Compute derivative of res_axis wrt coil coefficients
        Outputs:
            d_res_axis_d_coil_coeffs (list of 1d arrays (ncoeffs)): derivatives of
                residual between parameterization axis and true axis wrt coil
                coeffs
        """
        phi,dphi = np.linspace(0,2*np.pi,self.nphi_integral,endpoint=False,retstep=True)

        if (self.axis_poly is None or self.adjoint_axis_poly is None):
            self.update_solutions()
        axis = self.axis_poly(phi)
        mu = self.res_adjoint_axis_poly(phi)

        d_V_by_dcoilcoeffs = self.compute_d_V_dcoilcoeffs(phi,self.axis_poly)
        d_res_axis_dcoilcoeffs = []
        for i in range(len(d_V_by_dcoilcoeffs)):
            d_V = d_V_by_dcoilcoeffs[i]
            mu_dot_d_V = mu[0,...,None]*d_V[0,...] + mu[1,...,None]*d_V[1,...]
            d_res_axis = - np.sum(mu_dot_d_V,axis=0)*dphi
            d_res_axis_dcoilcoeffs.append(d_res_axis)
        d_res_axis_dcoilcoeffs = self.stellarator.reduce_coefficient_derivatives([ires for ires in d_res_axis_dcoilcoeffs])
        return d_res_axis_dcoilcoeffs

    def compute_adjoint_axis(self,phi,axis_poly,tangent_poly=None,
                             adjoint_tangent_poly=None,quantity='iota'):
        """
        Computes adjoint variable required for computing derivative of
            axis_res metric
        Inputs:
            phi (1d array): toroidal angle for evaluation of adjoint variable
            axis_poly (instance of scipy.interpolate.PPoly cubic spline): polyomial
                representing magnetic axis solution
            tangent_poly (instance of scipy.interpolate.PPoly cubic spline): polyomial
                representing tangent map solution
            adjoint_tangent_poly (instance of scipy.interpolate.PPoly cubic spline): polyomial
                representing adjoint variable for tangent map
        """            
        # Generate initial guess using linear method
        if self.adjoint_axis_fd:
            if (self.constrained):
                fun = lambda x,y : self.rhs_fun_adjoint(x,y,axis_poly,tangent_poly,
                                        adjoint_tangent_poly,adjoint_fd=True,
                                                        quantity=quantity)
            else:
                fun = lambda x,y : self.rhs_fun_adjoint(x,y,axis_poly,
                                        adjoint_fd=True,quantity=quantity)
            
            nphi = len(phi)-1
            dphi = phi[1]-phi[0]
            
            n1= np.floor((nphi-1)/2).astype(int)
            n2= np.ceil((nphi-1)/2).astype(int)
            kk = np.array(range(1,nphi))
            if (np.remainder(nphi,2)==0):
                topc = 1/np.tan(np.array(range(1,n2+1))*dphi/2)
                col1 = np.concatenate((np.array([0]),0.5*((-1)**(kk[0:n2]))*topc,-0.5*((-1)**(kk[n2:nphi]))*np.flipud(topc[0:n1])))
            else:
                topc = 1/np.sin(np.array(range(1,n2+1))*dphi/2)
                col1 = np.concatenate((np.array([0]),0.5*((-1)**(kk[0:n2]))*topc, 0.5*((-1)**(kk[n2:nphi]))*np.flipud(topc[0:n1])))

            row1 = -col1
            DM = scipy.linalg.toeplitz(col1,row1)
            matrix = np.zeros((2*nphi,2*nphi))
            matrix[0:nphi,0:nphi] = DM
            matrix[nphi::,nphi::] = DM
            
            rhs = np.zeros((2*nphi,))
            V = fun(phi,None)
            rhs[0:nphi] = V[0,0:nphi]
            rhs[nphi::] = V[1,0:nphi]
            m = self.compute_m(phi,axis_poly(phi))
            for i in range(nphi):
                matrix[i,i] += m[0,i]
                matrix[i,nphi+i-1] += m[2,i]
                matrix[nphi+i-1,i] += m[1,i]
                matrix[nphi+i-1,nphi+i-1] += m[3,i]

            sol = np.linalg.solve(matrix,rhs)
            y0 = np.zeros((2,nphi+1))
            y0[0,0:nphi] = sol[0:nphi]
            y0[0,-1] = y0[0,0]
            y0[1,0:nphi] = sol[nphi:2*nphi]
            y0[1,-1] = y0[1,0]
        else:
            if self.adjoint_axis_poly is not None:
                y0 = self.adjoint_axis_poly(phi)
            else:
                y0 = axis_poly(phi)        
                
        if (self.constrained):
            fun = lambda x,y : self.rhs_fun_adjoint(x,y,axis_poly,tangent_poly,
                                        adjoint_tangent_poly,quantity=quantity)
        else:
            fun = lambda x,y : self.rhs_fun_adjoint(x,y,axis_poly,
                                                    quantity=quantity)
            
        if self.adjoint_axis_bvp:
            fun_jac = lambda x, y : self.jac_adjoint(x,y,axis_poly)
            out = scipy.integrate.solve_bvp(fun=fun,
                                            bc=self.bc_fun_axis,
                                            x=phi,y=y0,fun_jac=fun_jac,
                                            bc_jac=self.bc_jac,verbose=self.verbose,
                                            tol=self.bvp_tol,max_nodes=self.max_nodes)

            if (self.check_adjoint):
                # Now check solution
                out_check = scipy.integrate.solve_ivp(fun,(0,2*np.pi),out.sol(0),
                            vectorized=False,rtol=self.rtol,atol=self.atol,
                                        t_eval=phi,dense_output=True,
                                        method=self.method,
                                        min_step=self.min_step,max_step=self.max_step)
                yend = out_check.sol(2*np.pi)
                if self.verbose:
                    info(f'Residual in adjoint axis: {np.linalg.norm(out.sol(0)-yend)}')
            
            if (out.status==0):
                # Evaluate polynomial on grid
                return out.sol(phi), out.sol
            else:
                raise RuntimeError('Error ocurred in integration of axis.') 
        else:
            t_span = (0,2*np.pi)
            niter = 0
            y0 = y0[:,0]
            for niter in range(self.maxiter):
                out = scipy.integrate.solve_ivp(fun,t_span,y0,
                            vectorized=False,rtol=self.rtol,atol=self.atol,
                                        t_eval=phi,dense_output=True,
                                        method=self.method,min_step=self.min_step,
                                        max_step=self.max_step)
                yend = out.sol(2*np.pi)
                if (self.verbose):
                    info(f'Norm: {np.linalg.norm(yend-y0)}')
                if (np.abs(yend[0]-y0[0]) < self.tol and np.abs(yend[1]-y0[1]) < self.tol):
                    if (self.verbose):
                        info(f'yend: {yend}')
                        info(f'y0: {y0}')
                        info('Newton iteration converged')
                    return out.y, out.sol
                tangent_sol, tangent_poly = self.compute_tangent(np.array([2*np.pi]),axis_poly,adjoint=True)
                U = np.zeros((2,2))
                U[0,0] = tangent_sol[0]
                U[0,1] = tangent_sol[1]
                U[1,0] = tangent_sol[2]
                U[1,1] = tangent_sol[3]
                mat = np.eye(2) - U
                step = np.linalg.solve(mat,yend-y0)
                y0 += step
            raise RuntimeError('Exceeded maxiter in compute_adjoint_axis.')            

    def compute_axis(self,phi):
        """
        For biotsavart and magnetic_axis objects, compute rotational transform
            from tangent map by solving initial value problem.
        Inputs:
            phi (1d array): 1d array for evaluation of tangent map
        Outputs:
            y (2d array (2,len(phi))): axis on grid of toroidal angle
        """    
        if self.axis_bvp:
            if (self.axis_poly is not None):
                y0 = self.axis_poly(phi)
            else:
                self.magnetic_axis.points = np.asarray(phi/(2*np.pi))
                self.magnetic_axis.update()
                axis = self.magnetic_axis.gamma
                y0 = np.zeros((2,len(phi)))
                y0[0,:] = np.sqrt(axis[:,0]**2 + axis[:,1]**2)
                y0[1,:] = axis[:,2]
            out = scipy.integrate.solve_bvp(fun=self.rhs_fun_axis,
                                        bc=self.bc_fun_axis,
                                        x=phi,y=y0,
                                        fun_jac=self.compute_jac,
                                        bc_jac=self.bc_jac,verbose=self.verbose,
                                        tol=self.bvp_tol,max_nodes=self.max_nodes)
            if (out.status==0):
                # Evaluate polynomial on grid
                return out.sol(phi), out.sol
            else:
                raise RuntimeError('Error ocurred in integration of axis.')
        else:
            if (self.axis_poly is not None):
                y0 = self.axis_poly(0)
            else:
                self.magnetic_axis.points = np.asarray([0])
                self.magnetic_axis.update()
                axis = self.magnetic_axis.gamma
                y0 = np.zeros((2,))
                y0[0] = np.sqrt(axis[...,0]**2 + axis[...,1]**2)
                y0[1] = axis[...,2]
            t_span = (0,2*np.pi)
            niter = 0
            for niter in range(self.maxiter):
                out = scipy.integrate.solve_ivp(self.rhs_fun_axis,t_span,y0,
                            vectorized=False,rtol=self.rtol,atol=self.atol,
                                        t_eval=phi,dense_output=True,
                                        method=self.method,
                                        min_step=self.min_step,max_step=self.max_step)
                yend = out.sol(2*np.pi)
                if self.verbose:
                    info(f'Norm: {np.linalg.norm(yend-y0)}')
                if (np.abs(yend[0]-y0[0]) < self.tol and np.abs(yend[1]-y0[1]) < self.tol):
                    if self.verbose:
                        info(f'yend: {yend}')
                        info(f'y0: {y0}')
                        info('Newton iteration converged')
                    return out.y, out.sol
                tangent_sol, tangent_poly = self.compute_tangent(np.array([2*np.pi]),out.sol)
                U = np.zeros((2,2))
                U[0,0] = tangent_sol[0]
                U[0,1] = tangent_sol[1]
                U[1,0] = tangent_sol[2]
                U[1,1] = tangent_sol[3]
                mat = np.eye(2) - U
                step = np.linalg.solve(mat,yend-y0)
                y0 += step
            
            raise RuntimeError('Exceeded maxiter in compute_axis.')

    def rhs_fun_axis(self,phi,axis):
        """
        Computes rhs of magnetic field line flow ode, i.e.
            r'(phi) = V(phi)
        Inputs:
            phi (1d array): toroidal angle for evaluation of rhs
            axis (2d array (2,len(phi))): R and Z for evaluation of rhs
        Outputs:
            V (2d array (2,len(phi))): R and Z components of rhs
        """
        if np.ndim(phi)>0:
            gamma = np.zeros((len(phi),3))
        else:
            gamma = np.zeros((1,3))
            
        gamma[...,0] = axis[0,...]*np.cos(phi)
        gamma[...,1] = axis[0,...]*np.sin(phi)
        gamma[...,2] = axis[1,...]
        X = gamma[...,0]
        Y = gamma[...,1]
        Z = gamma[...,2]

        self.biotsavart.compute(gamma)

        B = self.biotsavart.B
        BX = B[...,0]
        BY = B[...,1]
        BZ = B[...,2]
        R = np.sqrt(X**2 + Y**2)
        BR =  X*BX/R + Y*BY/R
        BP = -Y*BX/R + X*BY/R
        if (np.ndim(phi)>0):
            V = np.zeros((2,len(phi)))
        else:
            V = np.zeros((2,))
        V[0,...] = R*BR/BP
        V[1,...] = R*BZ/BP
        return V

    def rhs_fun_adjoint(self,phi,eta=None,axis_poly=None,tangent_poly=None,
                        adjoint_poly=None,adjoint_fd=False,quantity='iota'):
        """
        Compute rhs of adjoint problem for res_axis metric, i.e.
            mu'(\phi) = V(phi)
        Inputs:
            phi (1d array): toroidal angle for evaluation of rhs
            eta (2d array (2,len(phi))): mu_R and mu_Z for evaluation of rhs
            axis_poly (instance of scipy.interpolate.PPoly cubic spline): polynomial
                representing magnetic axis solution
            tangent_poly (instance of scipy.interpolate.PPoly cubic spline): polynomial
                representing tangent map solution
            adjoint_poly (instance of scipy.interpolate.PPoly cubic spline): polynomial
                representing "lambda" adjoint solution
        Outputs:
            V (2d array (2,len(phi))): R and Z components of rhs
        """
        if (np.ndim(phi)>0):
            V = np.zeros((2,len(phi)))
        else:
            V = np.zeros((2))
            
        if (adjoint_fd==False):
            if (self.constrained):
                m = self.compute_m(phi,axis_poly(phi))
            else:
                m = self.compute_m(phi)
            V[0,...] += -m[0,...]*eta[0,...] - m[2,...]*eta[1,...]
            V[1,...] += -m[3,...]*eta[1,...] - m[1,...]*eta[0,...]
            
        if (quantity=='iota'):
            if self.constrained:
                dmdR, dmdZ = self.compute_grad_m(phi,axis_poly)
            else:
                dmdR, dmdZ = self.compute_grad_m(phi)
            M = tangent_poly(phi)
            lam = adjoint_poly(phi)
            lambda_dot_dmdR_times_M = \
                  lam[0,...]*(dmdR[0,...]*M[0,...] + dmdR[1,...]*M[2,...]) \
                + lam[1,...]*(dmdR[0,...]*M[1,...] + dmdR[1,...]*M[3,...]) \
                + lam[2,...]*(dmdR[2,...]*M[0,...] + dmdR[3,...]*M[2,...]) \
                + lam[3,...]*(dmdR[2,...]*M[1,...] + dmdR[3,...]*M[3,...])
            lambda_dot_dmdZ_times_M = \
                  lam[0,...]*(dmdZ[0,...]*M[0,...] + dmdZ[1,...]*M[2,...]) \
                + lam[1,...]*(dmdZ[0,...]*M[1,...] + dmdZ[1,...]*M[3,...]) \
                + lam[2,...]*(dmdZ[2,...]*M[0,...] + dmdZ[3,...]*M[2,...]) \
                + lam[3,...]*(dmdZ[2,...]*M[1,...] + dmdZ[3,...]*M[3,...])
            iota = self.compute_iota()
            fac = -1/(4*np.pi*np.sin(2*np.pi*iota))
            V[0,...] += - np.squeeze(fac*lambda_dot_dmdR_times_M)
            V[1,...] += - np.squeeze(fac*lambda_dot_dmdZ_times_M)
        else:
            axis = axis_poly(phi)
            if np.ndim(phi) == 0:
                self.magnetic_axis.points = np.array([phi/(2*np.pi)])
            else:
                self.magnetic_axis.points = phi/(2*np.pi)
            self.magnetic_axis.update()
            gamma_ma = self.magnetic_axis.gamma
            R_ma = np.sqrt(gamma_ma[...,0]**2 + gamma_ma[...,1]**2)
            Z_ma = gamma_ma[...,2]

            V[0,...] += np.squeeze(axis[0,...] - R_ma)
            V[1,...] += np.squeeze(axis[1,...] - Z_ma)           
            
        return V

    def jac_adjoint(self,phi,y,axis_poly):
        """
        Computes jacobian of rhs of adjoint equation (rhs_fun_adjoint)
        Inputs:
            phi (1d array): toroidal angle for evaluation
            y (2d array (2,len(phi))): mu_R and mu_Z for evaluation
        Outputs:
            jac (3d array (2,2,len(phi))): jacobian matrix on phi grid
        """
        m = self.compute_m(phi,axis_poly(phi))
        jac = np.zeros((2,2,len(phi)))
        jac[0,0,:] = m[0,:]
        jac[1,0,:] = m[1,:]
        jac[0,1,:] = m[2,:]
        jac[1,1,:] = m[3,:]
        return -jac

    def compute_jac(self,phi,y):
        """
        Computes jacobian matrix for magnetic axis bvp (compute_axis)
        Inputs:
            phi (1d array): toroidal angle for evaluation
            y (2d array (2,len(phi))): R and Z for evaluation
        Outputs:
            jac (3d array (2,2,len(phi))): jacobian matrix on phi grid
        """
        m = self.compute_m(phi,y)
        jac = np.zeros((2,2,len(phi)))
        jac[0,0,...] = m[0,...]
        jac[0,1,...] = m[1,...]
        jac[1,0,...] = m[2,...]
        jac[1,1,...] = m[3,...]
        return jac

    def bc_jac(self,ya,yb):
        """
        Jacobian for boundary condition function for magnetic axis bvp
            (bc_fun_axis)
        Inputs:
            ya (1d array(2)): axis solution at phi = 0
            yb (1d array(2)): axis solution at phi = 2\pi
        Outputs:
            jac (2d array(2,2), 2d array(2,2)): jacobian for boundary condition
                at 0 and 2\pi
        """
        return np.eye(2), -np.eye(2)

    def bc_fun_axis(self,axisa,axisb):
        """
        Boundary condition function for magnetic axis bvp (compute_axis)
        Inputs:
            axisa (1d array(2)): Magnetic axis solution at phi = 0
            axisb (1d array(2)): Magnetic axis solution at phi = 2*pi
        """
        return axisa - axisb

    def compute_d_V_dcoilcurrents(self,phi,axis_poly):
        """
        Computes the derivative of the rhs of the
            axis ode, e.g. r'(phi) = V(phi), with respect to coil coeffs for
            given phi.
        Inputs:
            phi (1d array): toroidal angles for evaluation
            axis_poly: polynomial representation of axis solution
        Outputs:
            d_V_dcoilcurrents (list (ncoils) of 2d array (2,npoints)): derivative
                of V wrt coil currents
        """
        axis = axis_poly(phi)
        gamma = np.zeros((len(phi),3))
        gamma[:,0] = axis[0,:]*np.cos(phi)
        gamma[:,1] = axis[0,:]*np.sin(phi)
        gamma[:,2] = axis[1,:]
        self.biotsavart.compute(gamma)
        
        X = gamma[:,0]
        Y = gamma[:,1]
        Z = gamma[:,2]

        B = self.biotsavart.B
        BX = B[...,0]
        BY = B[...,1]
        BZ = B[...,2]

#         X = self.biotsavart.points[:,0]
#         Y = self.biotsavart.points[:,1]
#         Z = self.biotsavart.points[:,2]
        R = np.sqrt(X**2 + Y**2)
        BR =  X*BX/R + Y*BY/R
        BP = -Y*BX/R + X*BY/R

        # Shape: (ncoils,npoints,3)
        dB_by_dcoilcurrents = self.biotsavart.dB_by_dcoilcurrents

        d_V_by_dcoilcurrents = []
        for i in range(len(dB_by_dcoilcurrents)):
            d_B = dB_by_dcoilcurrents[i]
            d_BX = d_B[...,0]
            d_BY = d_B[...,1]
            d_BZ = d_B[...,2]

            d_BR =  (X * d_BX + Y * d_BY)/R
            d_BP =  (-Y * d_BX + X * d_BY)/R

            d_V = np.zeros((2,np.shape(d_BR)[0]))
            d_V[0,...] = R * (d_BR/BP - BR*d_BP/BP**2)
            d_V[1,...] = R * (d_BZ/BP - BZ*d_BP/BP**2)
            d_V_by_dcoilcurrents.append(d_V)

        return d_V_by_dcoilcurrents

    def compute_d_V_dcoilcoeffs(self,phi,axis_poly):
        """
        Computes the derivative of the rhs of the
            axis ode, e.g. r'(phi) = V(phi), with respect to coil coeffs for
            given phi.
        Inputs:
            phi (1d array): toroidal angles for evaluation
            axis_poly : polynomial representation of axis solution
        Outputs:
            d_V_dcoilcoeffs (list (ncoils) of 3d array (2,npoints,ncoeffs)):
                derivative of V wrt coil coeffs
        """
        axis = axis_poly(phi)
        gamma = np.zeros((len(phi),3))
        gamma[:,0] = axis[0,:]*np.cos(phi)
        gamma[:,1] = axis[0,:]*np.sin(phi)
        gamma[:,2] = axis[1,:]
        X = gamma[:,0]
        Y = gamma[:,1]
        Z = gamma[:,2]
        self.biotsavart.compute(gamma)
        self.biotsavart.compute_by_dcoilcoeff(gamma)

        B = self.biotsavart.B
        BX = B[...,0]
        BY = B[...,1]
        BZ = B[...,2]

#         X = self.biotsavart.points[:,0]
#         Y = self.biotsavart.points[:,1]
#         Z = self.biotsavart.points[:,2]
        R = np.sqrt(X**2 + Y**2)
        BR =  X*BX/R + Y*BY/R
        BP = -Y*BX/R + X*BY/R

        # Shape: (ncoils,npoints,nparams,3)
        dB_by_dcoilcoeffs = self.biotsavart.dB_by_dcoilcoeffs

        d_V_by_dcoilcoeffs = []
        for i in range(len(dB_by_dcoilcoeffs)):
            d_B = dB_by_dcoilcoeffs[i]
            d_BX = d_B[...,0]
            d_BY = d_B[...,1]
            d_BZ = d_B[...,2]

            d_BR =  (X[:,None] * d_BX + Y[:,None] * d_BY)/R[:,None]
            d_BP = (-Y[:,None] * d_BX + X[:,None] * d_BY)/R[:,None]

            d_V = np.zeros((2,np.shape(d_BR)[0],np.shape(d_BR)[1]))
            d_V[0,...] = R[:,None] * (d_BR/BP[:,None] - BR[:,None]*d_BP/BP[:,None]**2)
            d_V[1,...] = R[:,None] * (d_BZ/BP[:,None] - BZ[:,None]*d_BP/BP[:,None]**2)
            d_V_by_dcoilcoeffs.append(d_V)

        return d_V_by_dcoilcoeffs

    def ft_RZ(self,nfp=3,Nt=6,nphi=10000,adjoint=False):
        '''
        Calculates the Fourier coefficients for the R and Z
        coordinates of the magnetic axis.
        
        Inputs:
        nfp (int): the number of field periods in the device 
            (NOTE: we assume stellarator symmetry holds)
        Nt (int): number of harmonics to compute - Z will 
            have (Nt) harmonics, and R will have (Nt+1)
        nphi (1D array): number of toroidal angle values on 
            which R and Z are evaluated

        Outputs:
        Rcoeffs (1D array): Fourier coefficients for R
        Zcoeffs (1D array): Fourier coefficients for Z
        '''

        P = 2*np.pi/nfp # Period

        phi = np.linspace(0, P, num=nphi, endpoint=False)
        diff_phi = np.repeat(P,nphi)/nphi
        
        if not adjoint:
            R,Z = self.axis_poly(phi)
        else: 
            R,Z = self.adjoint_axis_poly(phi)
        
        Rcoeffs = np.zeros(Nt+1)
        Zcoeffs = np.zeros(Nt)
       
        # These are just standard Fourier transform formulas

        Rcoeffs[0] = 1/P * np.einsum('i,i->',R,diff_phi)

        for k in range(1,Nt+1):
            Rcoeffs[k] = 2/P * np.einsum('i,i->',R*np.cos(2*np.pi/P*k*phi),diff_phi)
            Zcoeffs[k-1] = 2/P * np.einsum('i,i->',Z*np.sin(2*np.pi/P*k*phi),diff_phi)

        return (Rcoeffs, Zcoeffs)
