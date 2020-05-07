import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from .common_functions import W_in_branch_as_function_of_independent_theta, W_in_branch_as_function_of_dependent_theta, Xi, plot_W_in_thetas_space, polarization_Upsilon, SPDC_events_seen_by_single_photon_detector, phase_matching_sinc
from .crystal import Crystal
import numbers

class SPDC:
	def __init__(self, lambda_pump, crystal, theta_s_dipole = None, phi_s_dipole = None, theta_i_dipole = None, phi_i_dipole = None):
		if lambda_pump < 0:
			raise ValueError('The value of "lambda_pump" must be positive.')
		if not isinstance(crystal, Crystal):
			raise ValueError('"crystal" must be an instance of the Crystal class')
		
		self.lambda_pump = lambda_pump # In meters.
		self.crystal = crystal
		self.omega_pump = 2*np.pi*const.c/lambda_pump
		self.theta_s_dipole = theta_s_dipole, 
		self.phi_s_dipole = phi_s_dipole, 
		self.theta_i_dipole = theta_i_dipole, 
		self.phi_i_dipole = phi_i_dipole
		
	def W_as_function_of_independent_theta(self, independent_theta_vals, branch, alpha):
		W, independent_theta_name = W_in_branch_as_function_of_independent_theta(
			lambda_pump = self.lambda_pump,
			crystal_l = self.crystal.crystal_length,
			n_pump = self.crystal.n(wavelength = self.lambda_pump),
			n_signal = self.crystal.n(wavelength = self.lambda_pump/alpha), 
			alpha = alpha, 
			Xi = Xi(n_idler = self.crystal.n(wavelength = self.lambda_pump/(1-alpha)), alpha = alpha),
			independent_theta_vals = independent_theta_vals, 
			branch = branch
		)
		return independent_theta_vals, W
	
	def W_as_function_of_dependent_theta(self, dependent_theta_vals, branch, alpha):
		W, dependent_theta_name = W_in_branch_as_function_of_dependent_theta(
			lambda_pump = self.lambda_pump,
			crystal_l = self.crystal.crystal_length,
			n_pump = self.crystal.n(wavelength = self.lambda_pump),
			n_signal = self.crystal.n(wavelength = self.lambda_pump/alpha), 
			alpha = alpha, 
			Xi = Xi(n_idler = self.crystal.n(wavelength = self.lambda_pump/(1-alpha)), alpha = alpha),
			dependent_theta_vals = dependent_theta_vals, 
			branch = branch
		)
		W = np.array(W)
		W[np.isnan(W)] = 0
		return dependent_theta_vals, W
	
	def observed_W(self, theta, alpha):
		# The parameter "theta" is the detector angle.
		theta, W1_indep = self.W_as_function_of_independent_theta(independent_theta_vals = theta, branch = 'branch_1', alpha = alpha)
		theta, W2_indep = self.W_as_function_of_independent_theta(independent_theta_vals = theta, branch = 'branch_2', alpha = alpha)
		theta, W1_dep = self.W_as_function_of_dependent_theta(dependent_theta_vals = theta, branch = 'branch_1', alpha = alpha)
		theta, W2_dep = self.W_as_function_of_dependent_theta(dependent_theta_vals = theta, branch = 'branch_2', alpha = alpha)
		total_W = W1_indep + W2_indep + W1_dep + W2_dep
		return theta, total_W, W1_indep, W2_indep, W1_dep, W2_dep
	
	def f_factor(self, theta, alpha):
		theta_vals, f_SPDC, _, _, _, _ = self.observed_W(theta, alpha)
		return f_SPDC
	
	def plot_W_in_thetas_space(self, alpha, theta_s=None, theta_i=None):
		if theta_s is None:
			theta_s = np.linspace(0,180/180*np.pi,999)
		if theta_i is None:
			theta_i = np.linspace(0,180/180*np.pi,999)
		return plot_W_in_thetas_space(
			lambda_pump = self.lambda_pump, 
			crystal_l = self.crystal.crystal_length, 
			n_pump = self.crystal.n(wavelength = self.lambda_pump),
			n_signal = self.crystal.n(wavelength = self.lambda_pump/alpha), 
			alpha = alpha, 
			Xi = Xi(n_idler = self.crystal.n(wavelength = self.lambda_pump/(1-alpha)), alpha = alpha),
			theta_s = theta_s, 
			theta_i = theta_i
		)
	
	def single_photon_intensity(self, theta_d = None, phi_d = None, lambda_d = None):
		if isinstance(theta_d, numbers.Number) and isinstance(phi_d, numbers.Number) and isinstance(lambda_d, numbers.Number):
			alpha_d = self.lambda_pump/lambda_d
			_Xi = Xi(
				n_idler = self.crystal.n(wavelength = self.lambda_pump/(1-alpha_d)), 
				alpha = alpha_d
			)
			SPDC_events = SPDC_events_seen_by_single_photon_detector(
				theta_d = theta_d, 
				phi_d = phi_d, 
				omega_d = self.omega_pump*alpha_d, 
				omega_p = self.omega_pump,
				crystal = self.crystal
			)
			intensities = []
			for event in SPDC_events:
				intensities.append(
					phase_matching_sinc(
						theta_s = event.get('theta_s'), 
						theta_i = event.get('theta_i'), 
						alpha = event.get('alpha'), 
						crystal_l = self.crystal.crystal_length, 
						lambda_p = self.lambda_pump, 
						n_pump = self.crystal.n(wavelength = self.lambda_pump), 
						n_signal = self.crystal.n(wavelength = self.lambda_pump/event.get('alpha')), 
						Xi = Xi(
							n_idler = self.crystal.n(wavelength = event.get('lambda_i')),
							alpha = event.get('alpha')
						)
					)
				)
			return np.nansum(intensities)
		elif all([hasattr(param, '__iter__') for param in [theta_d, phi_d, lambda_d]]): # if all the parameters are lists
			if len(theta_d) == len(phi_d) == len(lambda_d):
				intensities = []
				for t,p,l in zip(theta_d,phi_d,lambda_d):
					intensities.append(self.single_photon_intensity(t,p,l))
				return intensities
		else:
			raise NotImplementedError('The combination of parameters you gave me is not yet implemented, sorry')
