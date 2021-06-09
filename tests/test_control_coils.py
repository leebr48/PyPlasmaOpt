import pytest
import numpy as np
from pyplasmaopt.curve import ControlCoil
from pyplasmaopt.finite_differences import finite_difference_derivative

rtol = 1e-5
atol = 1e-7
npoints = 100

def random_param_picker():
    pi = np.pi

    r = np.random.uniform(low=0.2,high=0.5)
    R0 = np.random.uniform(low=r+0.2,high=r+2)
    phi = np.random.uniform(low=pi/4,high=3*pi/4)
    zc = np.random.uniform(low=-1,high=1)
    a = np.random.uniform(low=pi/4,high=3*pi/4)
    e = np.random.uniform(low=pi/4,high=3*pi/4)

    return np.array([R0,phi,zc,a,e,r])

def test_dgamma_by_dcoeff(npoints=npoints,rtol=rtol,atol=atol):
    points = np.linspace(0,1,npoints,endpoint=False)

    CC = ControlCoil(points)

    params = random_param_picker()
    
    def evaluate(x):
        CC.set_dofs(x)
        return CC.gamma
    
    CC.set_dofs(params)
    analytic = CC.dgamma_by_dcoeff
    fd = finite_difference_derivative(params, evaluate, epsilon=1e-7, method='forward')
    fd = np.swapaxes(fd,0,1)

    assert(np.allclose(analytic,fd,rtol=rtol,atol=atol)) 

def test_dgamma_by_dphi(rtol=rtol,atol=atol):
    points = np.array([0.5])

    CC = ControlCoil(points)

    dofs = random_param_picker()
    CC.set_dofs(dofs)
    
    def evaluate(x):
        CC2 = ControlCoil(x)
        CC2.set_dofs(dofs)
        return CC2.gamma
    
    analytic = CC.dgamma_by_dphi
    fd = finite_difference_derivative(points, evaluate, epsilon=1e-8, method='forward') #Slightly different step size than the rest
    fd = np.swapaxes(fd,0,1)
    
    assert(np.allclose(analytic,fd,rtol=rtol,atol=atol)) 

def test_d2gamma_by_dphidcoeff_1(npoints=npoints,rtol=rtol,atol=atol):
    points = np.linspace(0,1,npoints,endpoint=False)

    CC = ControlCoil(points)

    params = random_param_picker()
    
    def evaluate(x):
        CC.set_dofs(x)
        return CC.dgamma_by_dphi
    
    CC.set_dofs(params)
    analytic = CC.d2gamma_by_dphidcoeff
    fd = finite_difference_derivative(params, evaluate, epsilon=1e-7, method='centered')
    fd = np.swapaxes(fd,0,1)
    fd = np.swapaxes(fd,1,2)

    assert(np.allclose(analytic,fd,rtol=rtol,atol=atol)) 

def test_d2gamma_by_dphidcoeff_2(rtol=rtol,atol=atol): # This is just a permutation of the derivative above. 
    points = np.array([0.5])

    CC = ControlCoil(points)

    dofs = random_param_picker()
    CC.set_dofs(dofs)
    
    def evaluate(x):
        CC2 = ControlCoil(x)
        CC2.set_dofs(dofs)
        return CC2.dgamma_by_dcoeff
    
    analytic = CC.d2gamma_by_dphidcoeff
    fd = finite_difference_derivative(points, evaluate, epsilon=1e-7, method='centered')

    assert(np.allclose(analytic,fd,rtol=rtol,atol=atol)) 

def test_d2gamma_by_dphidphi(rtol=rtol,atol=atol):
    points = np.array([0.5])

    CC = ControlCoil(points)

    dofs = random_param_picker()
    CC.set_dofs(dofs)
    
    def evaluate(x):
        CC2 = ControlCoil(x)
        CC2.set_dofs(dofs)
        return CC2.dgamma_by_dphi
    
    analytic = CC.d2gamma_by_dphidphi
    fd = finite_difference_derivative(points, evaluate, epsilon=1e-7, method='centered')

    assert(np.allclose(analytic,fd,rtol=rtol,atol=atol)) 

def test_d3gamma_by_dphidphidcoeff_1(npoints=npoints,rtol=rtol,atol=atol):
    points = np.linspace(0,1,npoints,endpoint=False)

    CC = ControlCoil(points)

    params = random_param_picker()
    
    def evaluate(x):
        CC.set_dofs(x)
        return CC.d2gamma_by_dphidphi
    
    CC.set_dofs(params)
    analytic = CC.d3gamma_by_dphidphidcoeff
    fd = finite_difference_derivative(params, evaluate, epsilon=1e-7, method='centered')
    fd = np.swapaxes(fd,0,1)
    fd = np.swapaxes(fd,1,2)
    fd = np.swapaxes(fd,2,3)

    assert(np.allclose(analytic,fd,rtol=rtol,atol=atol))

def test_d3gamma_by_dphidphidcoeff_2(rtol=rtol,atol=atol): # This is just a permutation of the derivative above. 
    points = np.array([0.5])

    CC = ControlCoil(points)

    dofs = random_param_picker()
    CC.set_dofs(dofs)
    
    def evaluate(x):
        CC2 = ControlCoil(x)
        CC2.set_dofs(dofs)
        return CC2.d2gamma_by_dphidcoeff
    
    analytic = CC.d3gamma_by_dphidphidcoeff
    fd = finite_difference_derivative(points, evaluate, epsilon=1e-7, method='centered')

    assert(np.allclose(analytic,fd,rtol=rtol,atol=atol)) 

def test_d3gamma_by_dphidphidphi(rtol=rtol,atol=atol):
    points = np.array([0.5])

    CC = ControlCoil(points)

    dofs = random_param_picker()
    CC.set_dofs(dofs)
    
    def evaluate(x):
        CC2 = ControlCoil(x)
        CC2.set_dofs(dofs)
        return CC2.d2gamma_by_dphidphi
    
    analytic = CC.d3gamma_by_dphidphidphi
    fd = finite_difference_derivative(points, evaluate, epsilon=1e-7, method='centered')

    assert(np.allclose(analytic,fd,rtol=rtol,atol=atol)) 

def test_d4gamma_by_dphidphidphidcoeff_1(npoints=npoints,rtol=rtol,atol=atol):
    points = np.linspace(0,1,npoints,endpoint=False)

    CC = ControlCoil(points)

    params = random_param_picker()
    
    def evaluate(x):
        CC.set_dofs(x)
        return CC.d3gamma_by_dphidphidphi
    
    CC.set_dofs(params)
    analytic = CC.d4gamma_by_dphidphidphidcoeff
    fd = finite_difference_derivative(params, evaluate, epsilon=1e-7, method='centered')
    fd = np.swapaxes(fd,0,1)
    fd = np.swapaxes(fd,1,2)
    fd = np.swapaxes(fd,2,3)
    fd = np.swapaxes(fd,3,4)

    assert(np.allclose(analytic,fd,rtol=rtol,atol=atol))

def test_d4gamma_by_dphidphidphidcoeff_2(rtol=rtol,atol=atol): # This is just a permutation of the derivative above. 
    points = np.array([0.5])

    CC = ControlCoil(points)

    dofs = random_param_picker()
    CC.set_dofs(dofs)
    
    def evaluate(x):
        CC2 = ControlCoil(x)
        CC2.set_dofs(dofs)
        return CC2.d3gamma_by_dphidphidcoeff
    
    analytic = CC.d4gamma_by_dphidphidphidcoeff
    fd = finite_difference_derivative(points, evaluate, epsilon=1e-7, method='centered')

    assert(np.allclose(analytic,fd,rtol=rtol,atol=atol)) 
