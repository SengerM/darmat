import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from .common_functions import *

class dSPDC:
	def __init__(self, lambda_pump, crystal_l, n_pump, n_signal, m_dark_photon, alpha):
		if n_pump < 0 or n_signal < 0:
			raise ValueError('Negative refractive index received! I do not support this...')
		if alpha < 0 or alpha > 1:
			raise ValueError('The value of "alpha" must be between 0 and 1 (by definition of alpha).')
		if crystal_l < 0:
			raise ValueError('The value of "crystal_l" must be positive.')
		if lambda_pump < 0:
			raise ValueError('The value of "lambda_pump" must be positive.')
		if m_dark_photon < 0:
			raise ValueError('The mass of the dark photon must be >= 0.')
		
		self.lambda_pump = lambda_pump # In meters.
		self.crystal_l = crystal_l # Nonlinear medium length in meters.
		self.n_pump = n_pump
		self.n_signal = n_signal
		self.m_dark_photon = m_dark_photon
		self.alpha = alpha # Ratio of signal frequency to pump frequency, i.e. omega_signal/omega_pump.
		
		self.omega_pump = 2*np.pi*const.c/lambda_pump
		
		radicando = (1-alpha)**2 - m_dark_photon**2*const.c**4/self.omega_pump**2/const.hbar**2
		self.Xi = radicando**.5 if radicando >= 0 else float('nan')
		self.a = alpha*n_signal/self.Xi
		
		self.independent_theta_name = theta_name(self.a).get('independent')
		self.dependent_theta_name = theta_name(self.a).get('dependent')
		self.q_zeros, self.independent_theta_zeros, _ = zeros_of_W_in_branch_1(self.lambda_pump, self.crystal_l, self.n_pump, self.n_signal, self.alpha, self.Xi)
	
	def W_as_function_of_independent_theta(self, independent_theta_vals = None, branch = 'branch_1'):
		if independent_theta_vals is None:
			minimum_distance_between_zeros = min(np.diff(self.independent_theta_zeros))
			theta_step = minimum_distance_between_zeros/20
			independent_theta_vals = np.linspace(0, np.pi, int(np.pi/theta_step))
		W, independent_theta_name = W_in_branch_as_function_of_independent_theta(
											self.lambda_pump, 
											self.crystal_l, 
											self.n_pump, 
											self.n_signal, 
											self.alpha, 
											self.Xi, 
											independent_theta_vals, 
											branch)
		return independent_theta_vals, W
	
	def W_as_function_of_dependent_theta(self, dependent_theta_vals = None, branch = 'branch_1'):
		if dependent_theta_vals is None:
			minimum_distance_between_zeros = min(np.diff(self.independent_theta_zeros))
			theta_step = minimum_distance_between_zeros/20
			dependent_theta_vals = np.linspace(0, np.pi, int(np.pi/theta_step))
		W, dependent_theta_name = W_in_branch_as_function_of_dependent_theta(
											self.lambda_pump, 
											self.crystal_l, 
											self.n_pump, 
											self.n_signal, 
											self.alpha, 
											self.Xi, 
											dependent_theta_vals, 
											branch)
		W = np.array(W)
		W[np.isnan(W)] = 0
		return dependent_theta_vals, W
	
	def observed_W(self, theta = None):
		# The parameter "theta" is the detector angle.
		theta, W1_indep = self.W_as_function_of_independent_theta(independent_theta_vals = theta, branch = 'branch_1')
		theta, W2_indep = self.W_as_function_of_independent_theta(independent_theta_vals = theta, branch = 'branch_2')
		theta, W1_dep = self.W_as_function_of_dependent_theta(dependent_theta_vals = theta, branch = 'branch_1')
		theta, W2_dep = self.W_as_function_of_dependent_theta(dependent_theta_vals = theta, branch = 'branch_2')
		if self.independent_theta_name == 'theta_s':
			W_photons = W1_indep + W2_indep
			W_dark_photons = W1_dep + W2_dep
		else:
			W_photons = W1_dep + W2_dep
			W_dark_photons = W1_indep + W2_indep
		return theta, W_photons, W_dark_photons, W1_indep, W2_indep, W1_dep, W2_dep
	
	def f_factor(self, theta = None):
		theta_vals, f_dSPDC, _, _, _, _, _ = self.observed_W(theta)
		if theta is None:
			return theta_vals, f_dSPDC
		else:
			return f_dSPDC
	
	def plot_W_in_thetas_space(self, theta_s=None, theta_i=None):
		if theta_s is None:
			theta_s = np.linspace(0,180/180*np.pi,999)
		if theta_i is None:
			theta_i = np.linspace(0,180/180*np.pi,999)
		return plot_W_in_thetas_space(self.lambda_pump, self.crystal_l, self.n_pump, self.n_signal, self.alpha, self.Xi, theta_s, theta_i)

class PhaseMatchingFactor:
	# This was implemented for the paper on 2.oct.2020.
	def __init__(self, lambda_pump: float, crystal_l: float, n_pump: float, n_signal: float, m: float):
		if n_pump < 0 or n_signal < 0:
			raise ValueError('Negative refractive index received! I do not support this...')
		if crystal_l < 0:
			raise ValueError('The value of "crystal_l" must be positive.')
		if lambda_pump < 0:
			raise ValueError('The value of "lambda_pump" must be positive.')
		if m < 0:
			raise ValueError('The mass of the dark photon must be >= 0.')
		
		self.lambda_pump = lambda_pump
		self.crystal_l = crystal_l
		self.n_pump = n_pump
		self.n_signal = n_signal
		self.m = m
	
	@property
	def lambda_pump(self):
		return self._lambda_pump
	@lambda_pump.setter
	def lambda_pump(self, x):
		self._lambda_pump = x
	
	@property
	def crystal_l(self):
		return self._crystal_l
	@crystal_l.setter
	def crystal_l(self, x):
		self._crystal_l = x
	
	@property
	def n_pump(self):
		return self._n_pump
	@n_pump.setter
	def n_pump(self, x):
		self._n_pump = x
	
	@property
	def n_signal(self):
		return self._n_signal
	@n_signal.setter
	def n_signal(self, x):
		self._n_signal = x
	
	@property
	def m(self):
		return self._m
	@m.setter
	def m(self, x):
		self._m = x
	
	def prob_density(self, alpha, theta, phi):
		l = self.crystal_l
		n_s = self.n_signal
		n_p = self.n_pump
		λp = self.lambda_pump
		m = self.m
		wp = 2*np.pi/λp
		Xi = (1-alpha)**2-m**2/wp**2
		Xi[Xi<0] = float('NaN')
		Xi = Xi**.5
		radicando_mistico = Xi**2-alpha**2*n_s**2*np.sin(theta)**2
		radicando_mistico[radicando_mistico<0] = float('NaN')
		sinc1 = sinc(np.pi*l/λp*(alpha*n_s*np.cos(theta)+radicando_mistico**.5-n_p))
		sinc2 = sinc(np.pi*l/λp*(alpha*n_s*np.cos(theta)-radicando_mistico**.5-n_p))
		return (2*np.pi)**3*alpha**2*(1-alpha)*n_s**3*1**2*np.sin(theta)/λp/radicando_mistico**.5*(sinc1**2 + sinc2**2)
