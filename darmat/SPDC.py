import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from .crystal import Crystal
import numbers
from .emission_events import Photon
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from .common_functions import *

class OldSPDC:
	def __init__(self, lambda_pump, crystal_l, n_pump, n_signal, n_idler, alpha):
		if n_pump < 0 or n_signal < 0 or n_idler < 0:
			raise ValueError('Negative refractive index received! I do not support this...')
		if alpha < 0 or alpha > 1:
			raise ValueError('The value of "alpha" must be between 0 and 1 (by definition of alpha).')
		if crystal_l < 0:
			raise ValueError('The value of "crystal_l" must be positive.')
		if lambda_pump < 0:
			raise ValueError('The value of "lambda_pump" must be positive.')
		
		self.lambda_pump = lambda_pump # In meters.
		self.crystal_l = crystal_l # Nonlinear medium length in meters.
		self.n_pump = n_pump
		self.n_signal = n_signal
		self.n_idler = n_idler
		self.alpha = alpha # Ratio of signal frequency to pump frequency, i.e. omega_signal/omega_pump.
		
		self.omega_pump = 2*np.pi*const.c/lambda_pump
		
		self.Xi = n_idler*(1-alpha)
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
		total_W = W1_indep + W2_indep + W1_dep + W2_dep
		return theta, total_W, W1_indep, W2_indep, W1_dep, W2_dep
	
	def f_factor(self, theta = None):
		theta_vals, f_SPDC, _, _, _, _ = self.observed_W(theta)
		if theta is None:
			return theta_vals, f_SPDC
		else:
			return f_SPDC
	
	def plot_W_in_thetas_space(self, theta_s=None, theta_i=None):
		if theta_s is None:
			theta_s = np.linspace(0,180/180*np.pi,999)
		if theta_i is None:
			theta_i = np.linspace(0,180/180*np.pi,999)
		return plot_W_in_thetas_space(self.lambda_pump, self.crystal_l, self.n_pump, self.n_signal, self.alpha, self.Xi, theta_s, theta_i)

class SPDC:
	def __init__(self, lambda_pump, crystal):
		if lambda_pump < 0:
			raise ValueError('The value of "lambda_pump" must be positive.')
		if not isinstance(crystal, Crystal):
			raise ValueError('"crystal" must be an instance of the Crystal class')
		
		self.photon_pump = Photon(theta=0, phi=0, wavelength=lambda_pump)
		self.crystal = crystal
		
	def single_photon_intensity(self, theta_d = None, phi_d = None, lambda_d = None):
		if isinstance(theta_d, numbers.Number) and isinstance(phi_d, numbers.Number) and isinstance(lambda_d, numbers.Number):
			alpha_d = self.photon_pump.get('wavelength')/lambda_d
			detector_compatible_SPDC_events = SPDC_events_seen_by_single_photon_detector(
				theta_d = theta_d, 
				phi_d = phi_d, 
				omega_d = self.photon_pump.get('omega')*alpha_d, 
				omega_p = self.photon_pump.get('omega'),
				crystal = self.crystal
			)
			intensities = []
			for event in detector_compatible_SPDC_events:
				intensities.append(
					self.crystal.SPDC_intensity(event)
				)
			return np.nansum(intensities)
		elif all([hasattr(param, '__iter__') for param in [theta_d, phi_d, lambda_d]]): # if all the parameters are lists
			if len(theta_d) == len(phi_d) == len(lambda_d):
				intensities = []
				for t,p,l in zip(theta_d,phi_d,lambda_d):
					intensities.append(self.single_photon_intensity(t,p,l))
				return np.array(intensities)
		elif hasattr(theta_d, '__iter__') and isinstance(phi_d, numbers.Number) and isinstance(lambda_d, numbers.Number):
			return self.single_photon_intensity(theta_d, [phi_d]*len(theta_d), [lambda_d]*len(theta_d))
		elif isinstance(theta_d, numbers.Number) and isinstance(phi_d, numbers.Number) and hasattr(lambda_d, '__iter__'):
			return self.single_photon_intensity([theta_d]*len(lambda_d), [phi_d]*len(lambda_d), lambda_d)
		elif isinstance(theta_d, numbers.Number) and hasattr(phi_d, '__iter__') and isinstance(lambda_d, numbers.Number):
			return self.single_photon_intensity([theta_d]*len(phi_d), phi_d, [lambda_d]*len(phi_d))
		else:
			raise NotImplementedError('The combination of parameters you gave me is not yet implemented, sorry')
