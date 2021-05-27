import numpy as np
from math import pi

class BiotSavartQuasiSymmetricFieldDifference():

    def __init__(self, quasi_symmetric_field, biotsavart):
        self.quasi_symmetric_field = quasi_symmetric_field
        self.biotsavart = biotsavart

    def J_L2(self):
        Bbs        = self.biotsavart.B
        Bqs        = self.quasi_symmetric_field.B
        arc_length = self.quasi_symmetric_field.magnetic_axis.incremental_arclength[:, 0]
        return np.sum(arc_length[:, None] * (Bbs-Bqs)**2)/len(arc_length)

    def dJ_L2_by_dcoilcoefficients(self):
        Bbs                = self.biotsavart.B
        Bqs                = self.quasi_symmetric_field.B
        dBbs_by_dcoilcoeff = self.biotsavart.dB_by_dcoilcoeffs
        arc_length = self.quasi_symmetric_field.magnetic_axis.incremental_arclength[:, 0]
        res = []
        temp = (Bbs-Bqs) * arc_length[:, None]
        for dB in dBbs_by_dcoilcoeff:
            res.append(np.einsum('ij,ikj->k', temp, dB) * 2 / len(arc_length))
        return res

    def dJ_L2_by_dcoilcurrents(self):
        Bbs                   = self.biotsavart.B
        dBbs_by_dcoilcurrents = self.biotsavart.dB_by_dcoilcurrents
        Bqs                   = self.quasi_symmetric_field.B
        arc_length = self.quasi_symmetric_field.magnetic_axis.incremental_arclength[:, 0]
        res = []
        temp = (Bbs-Bqs) * arc_length[:, None]
        for dB in dBbs_by_dcoilcurrents:
            res.append(np.einsum('ij,ij', temp, dB) * 2 / len(arc_length))
        return res

    def dJ_L2_by_dmagneticaxiscoefficients(self):
        dgamma_by_dphi        = self.quasi_symmetric_field.magnetic_axis.dgamma_by_dphi[:,0,:]
        dgamma_by_dcoeff      = self.quasi_symmetric_field.magnetic_axis.dgamma_by_dcoeff
        d2gamma_by_dphidcoeff = self.quasi_symmetric_field.magnetic_axis.d2gamma_by_dphidcoeff[:, 0, :, :]
        arc_length = self.quasi_symmetric_field.magnetic_axis.incremental_arclength[:, 0]

        Bbs        = self.biotsavart.B
        dBbs_by_dX = self.biotsavart.dB_by_dX
        Bqs        = self.quasi_symmetric_field.B
        dBqs_by_dcoeffs = self.quasi_symmetric_field.dB_by_dcoeffs

        num_coeff = dgamma_by_dcoeff.shape[1]
        res = 2*np.einsum('ij,ikj,imk,i->m', (Bbs-Bqs), dBbs_by_dX, dgamma_by_dcoeff, arc_length)
        res -= 2*np.einsum('ij,imj,i->m', (Bbs-Bqs), dBqs_by_dcoeffs, arc_length)
        res += np.einsum('i,i,imj,ij->m', (1/arc_length), np.sum((Bbs-Bqs)**2, axis=1), d2gamma_by_dphidcoeff, dgamma_by_dphi)
        res *= 1/arc_length.shape[0]
        return res

    def dJ_L2_by_detabar(self):
        Bbs             = self.biotsavart.B
        arc_length      = self.quasi_symmetric_field.magnetic_axis.incremental_arclength[:, 0]
        Bqs             = self.quasi_symmetric_field.B
        dBqs_by_detabar = self.quasi_symmetric_field.dB_by_detabar
        res = np.zeros((1, ))
        res[0] -= np.sum(2*(Bbs-Bqs)*dBqs_by_detabar[:, 0, :] * arc_length[:, None])
        res *= 1/arc_length.shape[0]
        return res

    def J_H1(self):
        dBbs_by_dX = self.biotsavart.dB_by_dX
        arc_length = self.quasi_symmetric_field.magnetic_axis.incremental_arclength[:, 0]
        dBqs_by_dX = self.quasi_symmetric_field.dB_by_dX
        return np.sum(arc_length[:, None, None] * (dBbs_by_dX-dBqs_by_dX)**2)/len(arc_length)

    def dJ_H1_by_dcoilcoefficients(self):
        dBbs_by_dX            = self.biotsavart.dB_by_dX
        d2Bbs_by_dXdcoilcoeff = self.biotsavart.d2B_by_dXdcoilcoeffs
        dBqs_by_dX            = self.quasi_symmetric_field.dB_by_dX
        arc_length = self.quasi_symmetric_field.magnetic_axis.incremental_arclength[:, 0]
        res = []
        temp = (arc_length[:, None, None])*(dBbs_by_dX-dBqs_by_dX)
        for dB in d2Bbs_by_dXdcoilcoeff:
            res.append(np.einsum('ijk,iljk->l', temp, dB) * 2 / len(arc_length))
        return res

    def dJ_H1_by_dcoilcurrents(self):
        dBbs_by_dX               = self.biotsavart.dB_by_dX
        d2Bbs_by_dXdcoilcurrents = self.biotsavart.d2B_by_dXdcoilcurrents
        dBqs_by_dX               = self.quasi_symmetric_field.dB_by_dX
        arc_length = self.quasi_symmetric_field.magnetic_axis.incremental_arclength[:, 0]
        res = []
        for dB in d2Bbs_by_dXdcoilcurrents:
            res.append(np.einsum('ijk,ijk,i', dBbs_by_dX-dBqs_by_dX, dB, arc_length) * 2 / len(arc_length))
        return res

    def dJ_H1_by_dmagneticaxiscoefficients(self):

        gamma                 = self.quasi_symmetric_field.magnetic_axis.gamma
        dgamma_by_dphi        = self.quasi_symmetric_field.magnetic_axis.dgamma_by_dphi[:,0,:]
        dgamma_by_dcoeff      = self.quasi_symmetric_field.magnetic_axis.dgamma_by_dcoeff
        d2gamma_by_dphidcoeff = self.quasi_symmetric_field.magnetic_axis.d2gamma_by_dphidcoeff[:, 0, :, :]
        d2Bqs_by_dcoeffsdX    = self.quasi_symmetric_field.d2B_by_dcoeffsdX
        arc_length = self.quasi_symmetric_field.magnetic_axis.incremental_arclength[:, 0]

        dBbs_by_dX    = self.biotsavart.dB_by_dX
        d2Bbs_by_dXdX = self.biotsavart.d2B_by_dXdX
        dBqs_by_dX    = self.quasi_symmetric_field.dB_by_dX

        num_coeff = dgamma_by_dcoeff.shape[1]
        res = 2 * np.einsum('ijk,ijlk,iml,i->m',(dBbs_by_dX-dBqs_by_dX), d2Bbs_by_dXdX, dgamma_by_dcoeff, arc_length)
        res -= 2*np.einsum('ijk,imjk,i->m', (dBbs_by_dX-dBqs_by_dX), d2Bqs_by_dcoeffsdX, arc_length)
        res += np.einsum('i,i,iml,il->m', (1/arc_length), np.sum(np.sum((dBbs_by_dX-dBqs_by_dX)**2, axis=1), axis=1), d2gamma_by_dphidcoeff, dgamma_by_dphi)
        res *= 1/gamma.shape[0]
        return res

    def dJ_H1_by_detabar(self):
        dBbs_by_dX         = self.biotsavart.dB_by_dX
        dBqs_by_dX         = self.quasi_symmetric_field.dB_by_dX
        d2Bqs_by_detabardX = self.quasi_symmetric_field.d2B_by_detabardX
        arc_length = self.quasi_symmetric_field.magnetic_axis.incremental_arclength[:, 0]

        res = np.zeros((1, ))
        res[0] -= np.sum(2*(dBbs_by_dX-dBqs_by_dX)*d2Bqs_by_detabardX[:, 0, :, :] * arc_length[:, None, None])
        res *= 1/arc_length.shape[0]
        return res

class BiotSavartQuasiSymmetricFieldDifferenceRenormalized():

    def __init__(self, quasi_symmetric_field, biotsavart):
        self.quasi_symmetric_field = quasi_symmetric_field
        self.biotsavart = biotsavart
        
    # NOTE: J, K, L, and M standing alone below refer to chain rule terms, not the magnetic axis, the L2 norm, etc. 

    # Building blocks
    def Bbs(self):
        return self.biotsavart.B # ( (number of points on axis), (Bx,By,Bz) )

    def Bqs(self):
        return self.quasi_symmetric_field.B # ( (number of points on axis), (Bx,By,Bz) )

    def arc_length(self):
        return self.quasi_symmetric_field.magnetic_axis.incremental_arclength[:,0] #Vector with TOTAL axis length as each element.

    def diff_arc_length(self):
        return self.arc_length()/self.arc_length().shape[0] #This makes each element the size of each integration interval (standard Riemann sum technique).

    def dgamma_by_dphi(self):
        return self.quasi_symmetric_field.magnetic_axis.dgamma_by_dphi[:,0,:] # ( (number of points on axis), (drx,dry,drz) )

    def dgamma_by_dphi_norm(self):
        return np.einsum('ij->i',self.dgamma_by_dphi()**2) # This factor is used a lot below. ( (number of points on axis) )

    def dgamma_by_dcoeff(self):
        return self.quasi_symmetric_field.magnetic_axis.dgamma_by_dcoeff # ( (number of points on axis), (number of magnetic axis coefficients), (rx,ry,rz) )

    def d2gamma_by_dphidcoeff(self):
        return self.quasi_symmetric_field.magnetic_axis.d2gamma_by_dphidcoeff[:, 0, :, :] # ( (number of points on axis), (number of magnetic axis coefficients), (drx,dry,drz) )

    def dBbs_by_dX(self):
        return self.biotsavart.dB_by_dX # ( (number of points on axis), (dx,dy,dz), (Bx,By,Bz) )

    def num_coeff(self):
        return self.dgamma_by_dcoeff().shape[1] # Number of magnetic axis coefficients?

    def dBqs_by_dX(self):
        return self.quasi_symmetric_field.dB_by_dX # ( (number of points on axis), (dx,dy,dz), (Bx,By,Bz) )

    def L2_obj_numerator(self):
        return np.einsum('aj,a->',(self.Bqs()-self.Bbs())**2,self.diff_arc_length()) # $J$ in your notes

    def L2_norm_factor(self):
        return np.einsum('aj,a->',self.Bbs()**2,self.diff_arc_length()) # $K$ in your notes

    def H1_obj_numerator(self):
        return np.einsum('ajk,a->',(self.dBqs_by_dX()-self.dBbs_by_dX())**2,self.diff_arc_length()) # $L$ in your notes

    def H1_norm_factor(self):
        return np.einsum('ajk,a->',self.dBbs_by_dX()**2,self.diff_arc_length()) # $M$ in your notes

    # Actual calculations
    def J_L2(self):
        return self.L2_obj_numerator()/self.L2_norm_factor()

    def dJ_L2_by_dcoilcoefficients(self):
        dBbs_by_dcoilcoeff = self.biotsavart.dB_by_dcoilcoeffs # ( (number of coils), (number of points on axis), (number of coil coefficients), (Bx,By,Bz) )
        temp_J = (self.Bbs()-self.Bqs()) * self.diff_arc_length()[:, None] #dJ/dc_i [from notes] bits that are not in the derivative.
        temp_K = self.Bbs() * self.diff_arc_length()[:, None] #dK/dc_i [from notes] bits that are not in the derivative. 
        res = []
        for dB in dBbs_by_dcoilcoeff: #For each coil...
            dJ_dci = 2 * np.einsum('ij,ikj->k',temp_J,dB) #i sums over points on axis [integral], j sums over Bx,By,Bz (and derivatives) [dot product]
            dK_dci = 2 * np.einsum('ij,ikj->k',temp_K,dB) #i sums over points on axis [integral], j sums over Bx,By,Bz (and derivatives) [dot product]
            chain_ruled = (self.L2_norm_factor()*dJ_dci-self.L2_obj_numerator()*dK_dci)/self.L2_norm_factor()**2
            res.append(chain_ruled)
        return res

    def dJ_L2_by_dcoilcurrents(self):
        dBbs_by_dcoilcurrents = self.biotsavart.dB_by_dcoilcurrents #( (number of coils), (number of points on axis), (Bx,By,Bz) )
        temp_J = (self.Bbs()-self.Bqs()) * self.diff_arc_length()[:, None] #dJ/dc_i [from notes] bits that are not in the derivative.
        temp_K = self.Bbs() * self.diff_arc_length()[:, None] #dK/dc_i [from notes] bits that are not in the derivative.
        res = []
        for dB in dBbs_by_dcoilcurrents: #For each coil...
            dJ_dci = 2 * np.einsum('ij,ij',temp_J,dB) #i sums over points on axis [integral], j sums over Bx,By,Bz (and derivatives) [dot product]
            dK_dci = 2 * np.einsum('ij,ij',temp_K,dB) #i sums over points on axis [integral], j sums over Bx,By,Bz (and derivatives) [dot product]
            chain_ruled = (self.L2_norm_factor()*dJ_dci-self.L2_obj_numerator()*dK_dci)/self.L2_norm_factor()**2
            res.append(chain_ruled)  
        return res

    def dJ_L2_by_dmagneticaxiscoefficients(self):
        dBqs_by_dcoeffs = self.quasi_symmetric_field.dB_by_dcoeffs # ( (number of points on axis), (number of magnetic axis coefficients), (Bx,By,Bz) )
        
        # dJ/dmi bits
        term1_part1 = 2 * np.einsum('ij,ikj,i->k',self.Bqs()-self.Bbs(),dBqs_by_dcoeffs,self.diff_arc_length()) # check
        term1_part2 = - 2 * np.einsum('ij,ikj,imk,i->m',(self.Bqs()-self.Bbs()),self.dBbs_by_dX(),self.dgamma_by_dcoeff(),self.diff_arc_length())
        term1 = term1_part1 + term1_part2

        term2 = np.einsum('i,i,imj,ij->m', 1/self.arc_length(), np.sum((self.Bqs()-self.Bbs())**2, axis=1), self.d2gamma_by_dphidcoeff(), self.dgamma_by_dphi())/self.arc_length().shape[0]
        dJ_dmi = term1 + term2

        # dK/dmi bits
        term1 = 2 * np.einsum('ij,ikj,imk,i->m',self.Bbs(),self.dBbs_by_dX(),self.dgamma_by_dcoeff(),self.diff_arc_length())
        term2 = np.einsum('i,i,imj,ij->m', 1/self.arc_length(), np.sum(self.Bbs()**2, axis=1), self.d2gamma_by_dphidcoeff(), self.dgamma_by_dphi()) /self.arc_length().shape[0]       
        
        dK_dmi = term1 + term2

        # Chain rule
        chain_ruled = (self.L2_norm_factor()*dJ_dmi-self.L2_obj_numerator()*dK_dmi)/self.L2_norm_factor()**2

        return chain_ruled

    def dJ_L2_by_detabar(self):
        dBqs_by_detabar = self.quasi_symmetric_field.dB_by_detabar # ( (number of points on axis), (1), (Bx,By,Bz) )
        
        # dJ/dmi bits
        dJ_dmi = 2 * np.einsum('ij,ikj,i->k',self.Bqs()-self.Bbs(),dBqs_by_detabar,self.diff_arc_length())

        # dK/dmi bits
        ## These become zero in this case

        # Chain rule
        chain_ruled = dJ_dmi/self.L2_norm_factor()

        return chain_ruled

    def J_H1(self):
        return self.H1_obj_numerator()/self.H1_norm_factor()

    def dJ_H1_by_dcoilcoefficients(self):
        d2Bbs_by_dXdcoilcoeff = self.biotsavart.d2B_by_dXdcoilcoeffs # ( (number of coils), (number of points on axis), (number of coil coefficients), (dx,dy,dz), (Bx,By,Bz) )
        temp_L = (self.dBbs_by_dX()-self.dBqs_by_dX())*self.diff_arc_length()[:, None, None]
        temp_M = self.dBbs_by_dX()*self.diff_arc_length()[:, None, None]
        res = []
        for dB in d2Bbs_by_dXdcoilcoeff: # For each coil...
            dL_dci = 2 * np.einsum('ijk,iljk->l', temp_L, dB) # Double sum/contraction and integral.
            dM_dci = 2 * np.einsum('ijk,iljk->l', temp_M, dB) # Double sum/contraction and integral. 
            chain_ruled = (self.H1_norm_factor()*dL_dci-self.H1_obj_numerator()*dM_dci)/self.H1_norm_factor()**2
            res.append(chain_ruled)
        return res

    def dJ_H1_by_dcoilcurrents(self):
        d2Bbs_by_dXdcoilcurrents = self.biotsavart.d2B_by_dXdcoilcurrents # ( (number of coils), (number of points on axis), (dx,dy,dz), (Bx,By,Bz) )
        res = []
        for dB in d2Bbs_by_dXdcoilcurrents: # For each coil... 
            dL_dci = 2 * np.einsum('ijk,ijk,i', self.dBbs_by_dX()-self.dBqs_by_dX(), dB, self.diff_arc_length())
            dM_dci = 2 * np.einsum('ijk,ijk,i', self.dBbs_by_dX(), dB, self.diff_arc_length())
            chain_ruled = (self.H1_norm_factor()*dL_dci-self.H1_obj_numerator()*dM_dci)/self.H1_norm_factor()**2 
            res.append(chain_ruled)
        return res

    def dJ_H1_by_dmagneticaxiscoefficients(self):
        d2Bqs_by_dcoeffsdX    = self.quasi_symmetric_field.d2B_by_dcoeffsdX # ( (number of points axis), (number of MA coeffs), (dx,dy,dz), (Bx,By,Bz) )
        d2Bbs_by_dXdX         = self.biotsavart.d2B_by_dXdX # ( (number of points on axis), (dx,dy,dz), (dx,dy,dz), (Bx, By, Bz) )

        # dL/dmi bits
        term1_part1 = 2 * np.einsum('ajk,aijk,a->i',self.dBqs_by_dX()-self.dBbs_by_dX(),d2Bqs_by_dcoeffsdX,self.diff_arc_length())
        term1_part2_innersum = np.einsum('ain,anjk->aijk',self.dgamma_by_dcoeff(),d2Bbs_by_dXdX)
        term1_part2 = -2 * np.einsum('ajk,aijk,a->i',self.dBqs_by_dX()-self.dBbs_by_dX(),term1_part2_innersum,self.diff_arc_length())
        term1 = term1_part1 + term1_part2

        term2_sum = np.einsum('ikj,ij->ik',self.d2gamma_by_dphidcoeff(),self.dgamma_by_dphi())
        term2_numerator = np.einsum('ajk->a',(self.dBqs_by_dX()-self.dBbs_by_dX())**2)
        term2 = np.einsum('a,a,ai,a->i',term2_numerator,1/self.dgamma_by_dphi_norm(),term2_sum,self.diff_arc_length())

        dL_dmi = term1 + term2

        # dM/dmi bits

        term1_innersum = np.einsum('ain,anjk->aijk',self.dgamma_by_dcoeff(),d2Bbs_by_dXdX)
        term1 = 2 * np.einsum('ajk,aijk,a->i',self.dBbs_by_dX(),term1_innersum,self.diff_arc_length())

        term2_sum = np.einsum('ikj,ij->ik',self.d2gamma_by_dphidcoeff(),self.dgamma_by_dphi())
        term2_numerator = np.einsum('ajk->a',self.dBbs_by_dX()**2)
        term2 = np.einsum('a,a,ai,a->i',term2_numerator,1/self.dgamma_by_dphi_norm(),term2_sum,self.diff_arc_length())

        dM_dmi = term1 + term2

        # Chain rule
        
        chain_ruled = (self.H1_norm_factor()*dL_dmi-self.H1_obj_numerator()*dM_dmi)/self.H1_norm_factor()**2

        return chain_ruled

    def dJ_H1_by_detabar(self):
        d2Bqs_by_detabardX = self.quasi_symmetric_field.d2B_by_detabardX # ( (number of points on axis), (1), (dx,dy,dz), (Bx,By,Bz) )
        
        #self.dBqs_by_dX ( (number of points on axis), (dx,dy,dz), (Bx,By,Bz) )
        
        # These expressions are much simpler since the magnetic axis does not depend on etabar. 

        dL_dmi = 2 * np.einsum('ajk,adjk,a->d',self.dBqs_by_dX()-self.dBbs_by_dX(),d2Bqs_by_detabardX,self.diff_arc_length())

        chain_ruled = dL_dmi/self.H1_norm_factor()

        return chain_ruled

class SquaredMagneticFieldNormOnCurve(object):

    r"""
    This objective calculates
        J = \int_{curve} |B(s)|^2 ds
    given a curve and a Biot Savart kernel.
    """

    def __init__(self, curve, biotsavart):
        self.curve = curve
        self.biotsavart = biotsavart

    def J(self):
        quadrature_points = self.curve.gamma
        arc_length = np.linalg.norm(self.curve.dgamma_by_dphi[:,0,:], axis=1)
        B = self.biotsavart.compute(quadrature_points).B
        return np.sum(arc_length[:, None] * (B**2))/quadrature_points.shape[0]

    def dJ_by_dcoilcoefficients(self):
        """
        Calculate the derivatives with respect to the coil coefficients.
        """

        quadrature_points = self.curve.gamma
        arc_length = np.linalg.norm(self.curve.dgamma_by_dphi[:,0,:], axis=1)

        B = self.biotsavart.compute(quadrature_points).B
        dB_by_dcoilcoeff = self.biotsavart.compute_by_dcoilcoeff(quadrature_points).dB_by_dcoilcoeffs
        res = []
        for dB in dB_by_dcoilcoeff:
            res.append(np.einsum('ij,ikj,i->k', B, dB, arc_length) * 2 / quadrature_points.shape[0])
        return res

    def dJ_by_dcurvecoefficients(self):
        """
        Calculate the derivatives with respect to the coefficients describing
        the shape of the curve that we are integrating the magnetic field over.
        """

        gamma                 = self.curve.gamma
        dgamma_by_dphi        = self.curve.dgamma_by_dphi[:,0,:]
        dgamma_by_dcoeff      = self.curve.dgamma_by_dcoeff
        d2gamma_by_dphidcoeff = self.curve.d2gamma_by_dphidcoeff[:, 0, :, :]

        arc_length = np.linalg.norm(dgamma_by_dphi, axis=1)
        self.biotsavart.compute(gamma)
        B        = self.biotsavart.B
        dB_by_dX = self.biotsavart.dB_by_dX

        num_coeff = dgamma_by_dcoeff.shape[1]
        res = np.zeros((num_coeff, ))
        for i in range(num_coeff):
            for k1 in range(3):
                for k2 in range(3):
                    res[i] += 2 * np.sum(B[:, k1] * dB_by_dX[:, k1, k2] * dgamma_by_dcoeff[:, i, k2] * arc_length)
            res[i] += np.sum((1/arc_length) * np.sum(B**2, axis=1) * np.sum(d2gamma_by_dphidcoeff[:, i, :] * dgamma_by_dphi, axis=1))
        res *= 1/gamma.shape[0]
        return res

class SquaredMagneticFieldGradientNormOnCurve(object):

    r"""
    This objective calculates
        J = \int_{curve} |∇B(s)|^2 ds
    given a curve and a Biot Savart kernel.
    """

    def __init__(self, curve, biotsavart):
        self.curve = curve
        self.biotsavart = biotsavart

    def J(self):
        quadrature_points = self.curve.gamma
        arc_length = np.linalg.norm(self.curve.dgamma_by_dphi[:,0,:], axis=1)
        dB_by_dX = self.biotsavart.compute(quadrature_points).dB_by_dX
        return np.sum(arc_length * (np.sum(np.sum(dB_by_dX**2, axis=1), axis=1)))/quadrature_points.shape[0]

    def dJ_by_dcoilcoefficients(self):
        """
        Calculate the derivatives with respect to the coil coefficients.
        """

        quadrature_points = self.curve.gamma
        arc_length = np.linalg.norm(self.curve.dgamma_by_dphi[:,0,:], axis=1)

        dB_by_dX = self.biotsavart.compute(quadrature_points).dB_by_dX
        d2B_by_dXdcoilcoeff = self.biotsavart.compute_by_dcoilcoeff(quadrature_points).d2B_by_dXdcoilcoeffs
        res = []
        for dB in d2B_by_dXdcoilcoeff:
            res.append(np.einsum('ijk,iljk,i->l', dB_by_dX, dB, arc_length) * 2 / quadrature_points.shape[0])
        return res

    def dJ_by_dcurvecoefficients(self):
        """
        Calculate the derivatives with respect to the coefficients describing
        the shape of the curve that we are integrating the gradient of the
        magnetic field over.
        """

        gamma                 = self.curve.gamma
        dgamma_by_dphi        = self.curve.dgamma_by_dphi[:,0,:]
        dgamma_by_dcoeff      = self.curve.dgamma_by_dcoeff
        d2gamma_by_dphidcoeff = self.curve.d2gamma_by_dphidcoeff[:, 0, :, :]

        arc_length = np.linalg.norm(dgamma_by_dphi, axis=1)
        self.biotsavart.compute(gamma)
        dB_by_dX = self.biotsavart.dB_by_dX
        d2B_by_dXdX = self.biotsavart.d2B_by_dXdX

        num_coeff = dgamma_by_dcoeff.shape[1]
        res = np.zeros((num_coeff, ))
        for i in range(num_coeff):
            for k1 in range(3):
                for k2 in range(3):
                    for k3 in range(3):
                        res[i] += 2.0 * np.sum(dB_by_dX[:, k1, k2] * d2B_by_dXdX[:, k1, k2, k3] * dgamma_by_dcoeff[:, i, k3] * arc_length)
            res[i] += np.sum((1/arc_length) * np.sum(np.sum(dB_by_dX**2, axis=1), axis=1) * np.sum(d2gamma_by_dphidcoeff[:, i, :] * dgamma_by_dphi, axis=1))
        res *= 1/gamma.shape[0]
        return res


class CurveLength():

    r"""
    J = \int_{curve} 1 ds
    """

    def __init__(self, curve):
        self.curve = curve

    def J(self):
        arc_length = np.linalg.norm(self.curve.dgamma_by_dphi[:,0,:], axis=1)
        return np.mean(arc_length)

    def dJ_by_dcoefficients(self):
        dgamma_by_dphi        = self.curve.dgamma_by_dphi[:,0,:]
        d2gamma_by_dphidcoeff = self.curve.d2gamma_by_dphidcoeff[:, 0, :, :]
        num_coeff = d2gamma_by_dphidcoeff.shape[1]
        res = np.zeros((num_coeff, ))
        arc_length = np.linalg.norm(dgamma_by_dphi, axis=1)
        for i in range(num_coeff):
            res[i] = np.mean((1/arc_length) * np.sum(d2gamma_by_dphidcoeff[:, i, :] * dgamma_by_dphi, axis=1))
        return res


class CurveCurvature():

    r"""
    J = \int_{curve} \kappa ds
    """

    def __init__(self, curve, desired_length=None):
        self.curve = curve
        if desired_length is None:
            self.desired_kappa = 0
        else:
            radius = desired_length/(2*pi)
            self.desired_kappa = 1/radius

    def J(self):
        arc_length = np.linalg.norm(self.curve.dgamma_by_dphi[:,0,:], axis=1)
        kappa = self.curve.kappa[:, 0]
        return np.mean(np.maximum(kappa-self.desired_kappa, 0)**4 * arc_length)

    def dJ_by_dcoefficients(self):
        kappa                 = self.curve.kappa[:,0]
        dkappa_by_dcoeff      = self.curve.dkappa_by_dcoeff[:,:,0]
        dgamma_by_dphi        = self.curve.dgamma_by_dphi[:,0,:]
        d2gamma_by_dphidcoeff = self.curve.d2gamma_by_dphidcoeff[:, 0, :, :]
        arc_length            = np.linalg.norm(dgamma_by_dphi, axis=1)

        num_coeff = d2gamma_by_dphidcoeff.shape[1]
        res       = np.zeros((num_coeff, ))
        for i in range(num_coeff):
            res[i]  = np.mean((np.maximum(kappa-self.desired_kappa, 0)**4/arc_length) * np.sum(d2gamma_by_dphidcoeff[:, i, :] * dgamma_by_dphi, axis  = 1))
            res[i] += np.mean(4*(np.maximum(kappa-self.desired_kappa, 0))**3 * dkappa_by_dcoeff[:,i] * arc_length)
        return res


class CurveTorsion():

    r"""
    J = \int_{curve} \tau^p ds
    """

    def __init__(self, curve, p=2):
        self.curve = curve
        self.p = p

    def J(self):
        arc_length = np.linalg.norm(self.curve.dgamma_by_dphi[:,0,:], axis=1)
        torsion    = self.curve.torsion[:, 0]
        return np.mean(torsion**self.p * arc_length)

    def dJ_by_dcoefficients(self):
        torsion               = self.curve.torsion[:,0]
        dtorsion_by_dcoeff    = self.curve.dtorsion_by_dcoeff[:,:,0]
        dgamma_by_dphi        = self.curve.dgamma_by_dphi[:,0,:]
        d2gamma_by_dphidcoeff = self.curve.d2gamma_by_dphidcoeff[:, 0, :, :]
        arc_length            = np.linalg.norm(dgamma_by_dphi, axis=1)

        num_coeff = d2gamma_by_dphidcoeff.shape[1]
        res       = np.zeros((num_coeff, ))
        for i in range(num_coeff):
            res[i]  = np.mean((torsion**self.p/arc_length) * np.sum(d2gamma_by_dphidcoeff[:, i, :] * dgamma_by_dphi, axis  = 1))
            res[i] += np.mean(self.p*torsion**(self.p-1) * dtorsion_by_dcoeff[:,i] * arc_length)
        return res


class SobolevTikhonov():

    def __init__(self, curve, weights=[1., 1., 0., 0.]):
        self.curve = curve
        if not len(weights) == 4:
            raise ValueError(
                "You should pass 4 weights: for the L^2, H^1, H^2 and H^3 norm.")
        self.weights = weights
        self.initial_curve = (curve.gamma.copy(), curve.dgamma_by_dphi.copy(
        ), curve.d2gamma_by_dphidphi.copy(), curve.d3gamma_by_dphidphidphi)

    def J(self):
        res = 0
        curve = self.curve
        num_points = curve.gamma.shape[0]
        weights = self.weights
        if weights[0] > 0:
            res += weights[0] * \
                np.sum((curve.gamma-self.initial_curve[0])**2)/num_points
        if weights[1] > 0:
            res += weights[1] * np.sum((curve.dgamma_by_dphi -
                                        self.initial_curve[1])**2)/num_points
        if weights[2] > 0:
            res += weights[2] * np.sum((curve.d2gamma_by_dphidphi -
                                        self.initial_curve[2])**2)/num_points
        if weights[3] > 0:
            res += weights[3] * np.sum((curve.d3gamma_by_dphidphidphi -
                                        self.initial_curve[3])**2)/num_points
        return res

    def dJ_by_dcoefficients(self):
        curve = self.curve
        num_coeff = curve.dgamma_by_dcoeff.shape[1]
        num_points = curve.gamma.shape[0]
        res = np.zeros((num_coeff, ))
        weights = self.weights
        if weights[0] > 0:
            for i in range(num_coeff):
                res[i] += weights[0] * np.sum(
                    2*(curve.gamma-self.initial_curve[0])*curve.dgamma_by_dcoeff[:, i, :])/num_points
        if weights[1] > 0:
            for i in range(num_coeff):
                res[i] += weights[1] * np.sum(2*(curve.dgamma_by_dphi-self.initial_curve[1])
                                              * curve.d2gamma_by_dphidcoeff[:, :, i, :])/num_points
        if weights[2] > 0:
            for i in range(num_coeff):
                res[i] += weights[2] * np.sum(2*(curve.d2gamma_by_dphidphi-self.initial_curve[2])
                                              * curve.d3gamma_by_dphidphidcoeff[:, :, :, i, :])/num_points
        if weights[3] > 0:
            for i in range(num_coeff):
                res[i] += weights[3] * np.sum(2*(curve.d3gamma_by_dphidphidphi-self.initial_curve[3])
                                              * curve.d4gamma_by_dphidphidphidcoeff[:, :, :, :, i, :])/num_points
        return res


class UniformArclength():

    def __init__(self, curve, desired_length):
        self.curve = curve
        self.desired_arclength = desired_length

    def J(self):
        num_points = self.curve.gamma.shape[0]
        return np.sum((self.curve.incremental_arclength-self.desired_arclength)**2)/num_points

    def dJ_by_dcoefficients(self):
        num_points = self.curve.gamma.shape[0]
        num_coeff = self.curve.dgamma_by_dcoeff.shape[1]
        res = np.zeros((num_coeff, ))
        for i in range(num_coeff):
            res[i] = np.sum(
                2 * (self.curve.incremental_arclength-self.desired_arclength)
                * self.curve.dincremental_arclength_by_dcoeff[:, i, :]
            )/num_points
        return res


class MinimumDistance():

    def __init__(self, curves, minimum_distance):
        self.curves = curves
        self.minimum_distance = minimum_distance

    def min_dist(self):
        res = 1e10
        for i in range(len(self.curves)):
            gamma1 = self.curves[i].gamma
            for j in range(i):
                gamma2 = self.curves[j].gamma
                dists = np.sqrt(np.sum((gamma1[:, None, :] - gamma2[None, :, :])**2, axis=2))
                res = min(res, np.min(dists))
        return res

    def J(self):
        from scipy.spatial.distance import cdist
        res = 0
        for i in range(len(self.curves)):
            gamma1 = self.curves[i].gamma
            l1 = self.curves[i].incremental_arclength[:, 0, None]
            for j in range(i):
                gamma2 = self.curves[j].gamma
                l2 = self.curves[j].incremental_arclength[None, :, 0]
                dists = np.sqrt(np.sum((gamma1[:, None, :] - gamma2[None, :, :])**2, axis=2))
                alen = l1 * l2
                res += np.sum(alen * np.maximum(self.minimum_distance-dists, 0)**2)/(gamma1.shape[0]*gamma2.shape[0])
        return res

    def dJ_by_dcoefficients(self):
        res = []
        for i in range(len(self.curves)):
            gamma1 = self.curves[i].gamma
            dgamma1 = self.curves[i].dgamma_by_dcoeff
            numcoeff1 = self.curves[i].dgamma_by_dcoeff.shape[1]
            l1 = self.curves[i].incremental_arclength[:, 0, None]
            dl1 = self.curves[i].dincremental_arclength_by_dcoeff[:, :, 0, None]
            res.append(np.zeros((numcoeff1, )))
            for j in range(i):
                gamma2 = self.curves[j].gamma
                dgamma2 = self.curves[j].dgamma_by_dcoeff
                l2 = self.curves[j].incremental_arclength[None, :, 0]
                dl2 = self.curves[j].dincremental_arclength_by_dcoeff[None, :, :, 0]
                numcoeff2 = self.curves[j].dgamma_by_dcoeff.shape[1]
                diffs = gamma1[:, None, :] - gamma2[None, :, :]

                dists = np.sqrt(np.sum(diffs**2, axis=2))
                npmax = np.maximum(self.minimum_distance - dists, 0)
                if np.sum(npmax) < 1e-15:
                    continue

                l1l2npmax = l1*l2*npmax
                for ii in range(numcoeff1):
                    res[i][ii] += np.sum(dl1[:, ii, :] * l2 * npmax**2)/(gamma1.shape[0]*gamma2.shape[0])
                    res[i][ii] += np.sum(-2 * l1l2npmax * np.sum(dgamma1[:, ii, :][:, None, :] * diffs, axis=2)/dists)/(gamma1.shape[0]*gamma2.shape[0])
                for jj in range(numcoeff2):
                    res[j][jj] += np.sum(l1 * dl2[:, :, jj] * npmax**2)/(gamma1.shape[0]*gamma2.shape[0])
                    res[j][jj] -= np.sum(-2 * l1l2npmax * np.sum(dgamma2[:, jj, :][None, :, :] * diffs, axis=2)/dists)/(gamma1.shape[0]*gamma2.shape[0])
        return res
