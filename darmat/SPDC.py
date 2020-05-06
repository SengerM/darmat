import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from .common_functions import W_in_branch_as_function_of_independent_theta, W_in_branch_as_function_of_dependent_theta, Xi, plot_W_in_thetas_space
from .crystal import Crystal

class SPDC:
	def __init__(self, lambda_pump, crystal):
		if lambda_pump < 0:
			raise ValueError('The value of "lambda_pump" must be positive.')
		if not isinstance(crystal, Crystal):
			raise ValueError('"crystal" must be an instance of the Crystal class')
		
		self.lambda_pump = lambda_pump # In meters.
		self.crystal = crystal
		self.omega_pump = 2*np.pi*const.c/lambda_pump
		
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

