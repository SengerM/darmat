import numpy as np
import scipy.constants as const
from .common_functions import sinc

class SPDC:
	def __init__(self, lambda_pump, crystal_l, n_pump, n_signal, n_idler):
		if n_pump < 0 or n_signal < 0 or n_idler < 0:
			raise ValueError('Negative refractive index received! I do not support this...')
		if crystal_l < 0:
			raise ValueError('The value of "crystal_l" must be positive.')
		if lambda_pump < 0:
			raise ValueError('The value of "lambda_pump" must be positive.')
		
		self.lambda_pump = lambda_pump # In meters.
		self.crystal_l = crystal_l # Nonlinear medium length in meters.
		self.n_pump = n_pump
		self.n_signal = n_signal
		self.n_idler = n_idler
		self.omega_p = 2*np.pi*const.c/lambda_pump
	
	def f(self, theta_d = None, phi_d = None, lambda_d = None):
		alpha_d = self.lambda_pump/lambda_d
		Xi = self.n_idler*(1-alpha_d)
		theta_i_branch_1 = np.arcsin(alpha_d*self.n_signal/Xi*np.sin(theta_d))
		theta_i_branch_2 = np.pi - theta_i_branch_1
		sinc_factor = sinc(np.pi*self.crystal_l/self.lambda_pump*(alpha_d*self.n_signal*np.cos(theta_d) + Xi*np.cos(theta_i_branch_1) - self.n_pump))**2
		sinc_factor += sinc(np.pi*self.crystal_l/self.lambda_pump*(alpha_d*self.n_signal*np.cos(theta_d) + Xi*np.cos(theta_i_branch_2) - self.n_pump))**2
		sinc_factor[np.isnan(sinc_factor)] = 0
		return 2*sinc_factor

class dSPDC:
	def __init__(self, lambda_pump, crystal_l, n_pump, n_signal, m_dark_photon):
		if n_pump < 0 or n_signal < 0:
			raise ValueError('Negative refractive index received! I do not support this...')
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
		self.omega_p = 2*np.pi*const.c/lambda_pump
	
	def f(self, theta_d = None, phi_d = None, lambda_d = None):
		alpha_d = self.lambda_pump/lambda_d
		Xi = ((1-alpha_d)**2 - self.m_dark_photon**2*const.c**4/const.hbar**2/self.omega_p**2)**.5
		theta_i_branch_1 = np.arcsin(alpha_d*self.n_signal/Xi*np.sin(theta_d))
		theta_i_branch_2 = np.pi - theta_i_branch_1
		sinc_factor = sinc(np.pi*self.crystal_l/self.lambda_pump*(alpha_d*self.n_signal*np.cos(theta_d) + Xi*np.cos(theta_i_branch_1) - self.n_pump))**2
		sinc_factor += sinc(np.pi*self.crystal_l/self.lambda_pump*(alpha_d*self.n_signal*np.cos(theta_d) + Xi*np.cos(theta_i_branch_2) - self.n_pump))**2
		sinc_factor[np.isnan(sinc_factor)] = 0
		return sinc_factor
