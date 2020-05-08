import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import matplotlib.colors as colors
import numbers
import warnings
from .emission_events import SPDCEvent, Photon
from . import crystal as crystal_pkg

def sinc(x):
	return np.sinc(x/np.pi)

def theta_name(a):
	if a < 0:
		raise ValueError('The value of "a" cannot be less than 0.')
	if a < 1:
		return {'independent': 'theta_s', 'dependent': 'theta_i'}
	if a > 1:
		return {'independent': 'theta_i', 'dependent': 'theta_s'}
	if a == 1:
		raise ValueError('"a = 1" not implemented yet.')

def zeros_of_W_in_branch_1(lambda_pump, crystal_l, n_pump, n_signal, alpha, Xi):
	a = alpha*n_signal/Xi
	# First find an approximate range for the "q" values
	theta_test = np.linspace(0, np.pi, 999)
	if a <= 1:
		q_range = crystal_l/lambda_pump*(n_pump - alpha*n_signal*np.cos(theta_test) - (Xi**2 - alpha**2*n_signal**2*np.sin(theta_test)**2)**.5)
	if a > 1:
		q_range = crystal_l/lambda_pump*(n_pump - (alpha**2*n_signal**2 - Xi**2*np.sin(theta_test)**2)**.5 - Xi*np.cos(theta_test))
	if a == 1:
		raise ValueError('"a = 1" is not implemented')
	independent_theta_zeros = []
	q_zeros = []
	for q in range(int(np.floor(min(q_range))), int(np.ceil(max(q_range)))):
		if q == 0:
			continue
		if a <= 1:
			cosenando = ((n_pump - lambda_pump/crystal_l*q)**2 - Xi**2 + n_signal**2*alpha**2)/2/n_signal/alpha/(n_pump - lambda_pump/crystal_l*q)
		if a > 1:
			cosenando = ((n_pump - lambda_pump/crystal_l*q)**2 + Xi**2 - n_signal**2*alpha**2)/2/Xi/(n_pump - lambda_pump/crystal_l*q)
		if cosenando < -1 or cosenando > 1:
			continue
		q_zeros.append(q)
		independent_theta_zeros.append(np.arccos(cosenando))
	return q_zeros, independent_theta_zeros, theta_name(a).get('independent')

# W functions ↓ --------------------------------------------------------

def W_branch_1_a_less_than_one(lambda_pump, crystal_l, n_pump, n_signal, alpha, Xi, theta_s):
	a = n_signal*alpha/Xi
	if a > 1:
		raise ValueError('"a = n_signal*alpha/Xi" is greater than 1.')
	return sinc(np.pi*crystal_l/lambda_pump*(n_pump - Xi*(a*np.cos(theta_s) + (1-a**2*np.sin(theta_s)**2)**.5)))**2

def W_branch_2_a_less_than_one(lambda_pump, crystal_l, n_pump, n_signal, alpha, Xi, theta_s):
	a = n_signal*alpha/Xi
	if a > 1:
		raise ValueError('"a = n_signal*alpha/Xi" is greater than 1.')
	return sinc(np.pi*crystal_l/lambda_pump*(n_pump - Xi*(a*np.cos(theta_s) - (1-a**2*np.sin(theta_s)**2)**.5)))**2

def W_branch_1_a_greater_than_one(lambda_pump, crystal_l, n_pump, n_signal, alpha, Xi, theta_i):
	a = n_signal*alpha/Xi
	if a < 1:
		raise ValueError('"a = n_signal*alpha/Xi" is less than 1.')
	return sinc(np.pi*crystal_l/lambda_pump*(n_pump - Xi*(np.cos(theta_i) + (a**2 - np.sin(theta_i)**2)**.5)))**2

def W_branch_2_a_greater_than_one(lambda_pump, crystal_l, n_pump, n_signal, alpha, Xi, theta_i):
	a = n_signal*alpha/Xi
	if a < 1:
		raise ValueError('"a = n_signal*alpha/Xi" is less than 1.')
	return sinc(np.pi*crystal_l/lambda_pump*(n_pump - Xi*(np.cos(theta_i) - (a**2 - np.sin(theta_i)**2)**.5)))**2

def W_in_branch_as_function_of_independent_theta(lambda_pump, crystal_l, n_pump, n_signal, alpha, Xi, independent_theta_vals, branch='branch_1'):
	a = alpha*n_signal/Xi
	if branch not in ['branch_1', 'branch_2']:
		raise ValueError('"branch" must be one of ' + str(['branch_1', 'branch_2']))
	if a < 1:
		if branch == 'branch_1':
			W = W_branch_1_a_less_than_one(lambda_pump, crystal_l, n_pump, n_signal, alpha, Xi, theta_s=independent_theta_vals)
		if branch == 'branch_2':
			W = W_branch_2_a_less_than_one(lambda_pump, crystal_l, n_pump, n_signal, alpha, Xi, theta_s=independent_theta_vals)
	if a > 1:
		if branch == 'branch_1':
			W = W_branch_1_a_greater_than_one(lambda_pump, crystal_l, n_pump, n_signal, alpha, Xi, theta_i=independent_theta_vals)
		if branch == 'branch_2':
			W = W_branch_2_a_greater_than_one(lambda_pump, crystal_l, n_pump, n_signal, alpha, Xi, theta_i=independent_theta_vals)
	return W, theta_name(a).get('independent')

def W_in_branch_as_function_of_dependent_theta(lambda_pump, crystal_l, n_pump, n_signal, alpha, Xi, dependent_theta_vals = None, branch = 'branch_1'):
	a = alpha*n_signal/Xi
	dependent_theta_vals = np.array(dependent_theta_vals)
	if branch not in ['branch_1', 'branch_2']:
		raise ValueError('"branch" must be one of ' + str(['branch_1', 'branch_2']))
	if branch == 'branch_1':
		independent_theta_vals, independent_theta_name = independent_theta_from_dependent_theta_in_branch_1(a, dependent_theta_vals)
	if branch == 'branch_2':
		independent_theta_vals, independent_theta_name = independent_theta_from_dependent_theta_in_branch_2(a, dependent_theta_vals)
	W_first_half, _ = W_in_branch_as_function_of_independent_theta(
								lambda_pump, 
								crystal_l, 
								n_pump, 
								n_signal, 
								alpha, 
								Xi, 
								independent_theta_vals[0],
								branch)
	W_second_half, _ = W_in_branch_as_function_of_independent_theta(
								lambda_pump, 
								crystal_l, 
								n_pump, 
								n_signal, 
								alpha, 
								Xi, 
								independent_theta_vals[1],
								branch)
	if isinstance(W_first_half, numbers.Number):
		W_first_half = [W_first_half]
	if isinstance(W_second_half, numbers.Number):
		W_second_half = [W_second_half]
	W = [w1+w2 for w1,w2 in zip(W_first_half,W_second_half)]
	return W, theta_name(a).get('dependent')

def W_in_thetas_space(lambda_pump, crystal_l, n_pump, n_signal, alpha, Xi, theta_s, theta_i):
	return sinc(np.pi*crystal_l/lambda_pump*(n_pump - alpha*n_signal*np.cos(theta_s) - Xi*np.cos(theta_i)))**2

def plot_W_in_thetas_space(lambda_pump, crystal_l, n_pump, n_signal, alpha, Xi, theta_s, theta_i):
	ts, ti = np.meshgrid(theta_s, theta_i)
	fig, ax = plt.subplots()
	ax.set_xlabel(r'$\theta _s$ (deg)')
	ax.set_ylabel(r'$\theta _i$ (deg)')
	Z = W_in_thetas_space(
								  lambda_pump = lambda_pump, 
								  crystal_l = crystal_l, 
								  n_pump = n_pump, 
								  n_signal = n_signal, 
								  alpha = alpha, 
								  Xi = Xi, 
								  theta_s = ts, 
								  theta_i = ti
								)
	cs = ax.pcolormesh(
			   ts*180/np.pi,
			   ti*180/np.pi,
			   Z, 
			   cmap = 'Blues_r',
			   norm = colors.LogNorm(vmin=Z.min(), vmax=Z.max()),
			   # ~ vmin = 0,
			   # ~ vmax = 1,
			   rasterized = True
			 )
	cbar = fig.colorbar(cs)
	for branch in ['branch_1', 'branch_2']:
		dependent_theta_vals, dependent_theta_name = dependent_theta_from_independent_theta(
																a = n_signal*alpha/Xi, 
																independent_theta = np.linspace(0,np.pi), 
																branch = branch)
		if dependent_theta_name == 'theta_i':
			ax.plot(np.linspace(0,np.pi)*180/np.pi, dependent_theta_vals*180/np.pi, color = (0,0,0), linestyle = '--')
		if dependent_theta_name == 'theta_s':
			ax.plot(dependent_theta_vals*180/np.pi, np.linspace(0,np.pi)*180/np.pi, color = (0,0,0), linestyle = '--')
	return fig

# W functions ↑ --------------------------------------------------------

# Theta relating functions ↓ -------------------------------------------

def dependent_theta_from_independent_theta_in_branch_1(a, independent_theta_vals):
	# The parameter "a" is defined as "alpha*n_signal/Xi"
	if a < 0:
		raise ValueError('Values of "a" less than 0 are not valid.')
	if a < 1:
		dependent_theta_vals = np.arcsin(a*np.sin(independent_theta_vals))
	if a > 1:
		dependent_theta_vals = np.arcsin(a**-1*np.sin(independent_theta_vals))
	if a == 1:
		dependent_theta_vals = independent_theta_vals
	return dependent_theta_vals, theta_name(a).get('dependent')

def dependent_theta_from_independent_theta_in_branch_2(a, independent_theta_vals):
	# The parameter "a" is defined as "alpha*n_signal/Xi"
	if a < 0:
		raise ValueError('Values of "a" less than 0 are not valid.')
	if a < 1:
		dependent_theta_vals = np.pi - np.arcsin(a*np.sin(independent_theta_vals))
	if a > 1:
		dependent_theta_vals = np.pi - np.arcsin(a**-1*np.sin(independent_theta_vals))
	if a == 1:
		dependent_theta_vals = independent_theta_vals
	return dependent_theta_vals, theta_name(a).get('dependent')

def independent_theta_from_dependent_theta_in_branch_1(a, dependent_theta_vals):
	if a < 0:
		raise ValueError('Values of "a" less than 0 are not valid.')
	if a < 1:
		cutoff_angle = np.arcsin(a)
		dependent_theta_vals[dependent_theta_vals > cutoff_angle] = float('nan')
		independent_theta_vals = [np.arcsin(a**-1*np.sin(dependent_theta_vals)), np.pi - np.arcsin(a**-1*np.sin(dependent_theta_vals))]
	if a > 1:
		cutoff_angle = np.arcsin(a**-1)
		dependent_theta_vals[dependent_theta_vals > cutoff_angle] = float('nan')
		independent_theta_vals = [np.arcsin(a*np.sin(dependent_theta_vals)), np.pi - np.arcsin(a*np.sin(dependent_theta_vals))]
	if a == 1:
		independent_theta_vals = [dependent_theta_vals, [float('nan')]*len(dependent_theta_vals)]
	return independent_theta_vals, theta_name(a).get('independent')

def independent_theta_from_dependent_theta_in_branch_2(a, dependent_theta_vals):
	if a < 0:
		raise ValueError('Values of "a" less than 0 are not valid.')
	if a < 1:
		cutoff_angle = np.pi - np.arcsin(a)
		dependent_theta_vals[dependent_theta_vals < cutoff_angle] = float('nan')
		independent_theta_vals = [np.arcsin(a**-1*np.sin(dependent_theta_vals)), np.pi - np.arcsin(a**-1*np.sin(dependent_theta_vals))]
	if a > 1:
		cutoff_angle = np.pi - np.arcsin(a**-1)
		dependent_theta_vals[dependent_theta_vals < cutoff_angle] = float('nan')
		independent_theta_vals = [np.arcsin(a*np.sin(dependent_theta_vals)), np.pi - np.arcsin(a*np.sin(dependent_theta_vals))]
	if a == 1:
		independent_theta_vals = [dependent_theta_vals, [float('nan')]*len(dependent_theta_vals)]
	return independent_theta_vals, theta_name(a).get('independent')

def independent_theta_from_dependent_theta(a, dependent_theta, branch):
	if branch not in ['branch_1', 'branch_2']:
		raise ValueError('"branch" must be one of ' + str(['branch_1', 'branch_2']))
	if branch == 'branch_1':
		return independent_theta_from_dependent_theta_in_branch_1(a, dependent_theta)
	if branch == 'branch_2':
		return independent_theta_from_dependent_theta_in_branch_2(a, dependent_theta)

def dependent_theta_from_independent_theta(a, independent_theta, branch):
	if branch not in ['branch_1', 'branch_2']:
		raise ValueError('"branch" must be one of ' + str(['branch_1', 'branch_2']))
	if branch == 'branch_1':
		return dependent_theta_from_independent_theta_in_branch_1(a, independent_theta)
	if branch == 'branch_2':
		return dependent_theta_from_independent_theta_in_branch_2(a, independent_theta)

# Theta relating functions ↑ -------------------------------------------

def Xi(n_idler = None, alpha = None, m_dark_photon = None, omega_p = None):
	if  n_idler != None and alpha != None and m_dark_photon == None and omega_p == None: # This means that we are in the SPDC case.
		return n_idler*(1-alpha)
	elif m_dark_photon != None and omega_p != None and alpha != None and n_idler == None: # This means that we are in the dSPDC case.
		return ((1-alpha)**2 - m_dark_photon**2*const.c**4/const.hbar**2/omega_p**2)
	else:
		raise ValueError('Cannot understand what you want from the given parameters')

def phase_matching_sinc(theta_s, theta_i, alpha, crystal_l, lambda_p, n_pump, n_signal, Xi):
	return sinc(np.pi*crystal_l/lambda_p*(n_pump - alpha*n_signal*np.cos(theta_s) - Xi*np.cos(theta_i)))**2

def polarization_Upsilon(theta_s, theta_i, phi_s, phi_i, theta_s_dipole, phi_s_dipole, theta_i_dipole, phi_i_dipole, m_dark_photon = None, omega_i = None):
	# This is the function I defined in my notes on May 5 2020.
	
	def polar2cartessian(theta, phi):
		return np.array((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)))
	
	ks = polar2cartessian(theta_s, phi_s)
	ki = polar2cartessian(theta_i, phi_i)
	ds = polar2cartessian(theta_s_dipole, phi_s_dipole)
	di = polar2cartessian(theta_i_dipole, phi_i_dipole)
	
	if m_dark_photon == None and omega_i == None: # We are in the SPDC situation
		return np.sin(np.arccos(np.dot(ks,ds)))**2*np.sin(np.arccos(np.dot(ki,di)))**2
	elif m_dark_photon != None and omega_i != None: # We are in the dSPDC situation
		return np.sin(np.arccos(np.dot(ks,ds)))**2*((m_dark_photon*const.c**2/omega_i/const.hbar)**2*np.cos(np.arccos(np.dot(ki,di)))**2 + (m_dark_photon*const.c**2/omega_i/const.hbar)**4*np.sin(np.arccos(np.dot(ki,di)))**2)
	else:
		raise ValueError('Cannot understand what you want from the given parameters')

def SPDC_events_seen_by_single_photon_detector(theta_d, phi_d, omega_d, omega_p, crystal):
	# See my notes on May 6 2020.
	if not isinstance(theta_d, numbers.Number) or not isinstance(phi_d, numbers.Number) or not isinstance(omega_d, numbers.Number) or not isinstance(omega_p, numbers.Number) or not isinstance(crystal, crystal_pkg.Crystal):
		raise ValueError('This function does not operate through arrays, only single numbers')
	with warnings.catch_warnings(): # https://stackoverflow.com/questions/29347987/why-cant-i-suppress-numpy-warnings
		warnings.filterwarnings('ignore', r'invalid value encountered in arcsin')
		alpha_d = omega_d/omega_p
		lambda_p = 2*np.pi*const.c/omega_p
		detected_photon = Photon(
			theta = theta_d, 
			phi = phi_d, 
			omega = omega_d
		)
		undetected_photon = Photon(
			theta = 'Not yet implemented for unisotropic materials', 
			phi = 'Not yet implemented for unisotropic materials', 
			omega = omega_p - omega_d
		)
		pump_photon = Photon(
			theta = 0,
			phi = 0,
			omega = omega_p
		)
		events = [
			SPDCEvent( # Signal detection, forward-forward.
				theta_s = theta_d, 
				phi_s = phi_d, 
				omega_s = omega_p*alpha_d, 
				theta_i = np.arcsin(alpha_d*crystal.n(detected_photon)/Xi(n_idler=crystal.n(undetected_photon), alpha=alpha_d)*np.sin(theta_d)), 
				omega_p = omega_p,
				photon_pump = pump_photon
			),
			SPDCEvent( # Idler detection, forward-forward.
				theta_s = np.arcsin(Xi(n_idler=crystal.n(detected_photon), alpha=1-alpha_d)/(1-alpha_d)/crystal.n(undetected_photon)*np.sin(theta_d)), 
				phi_i = phi_d, 
				omega_s = omega_p - omega_d, 
				theta_i = theta_d, 
				omega_p = omega_p,
				photon_pump = pump_photon
			),
			SPDCEvent( # Signal detection, forward-backward.
				theta_s = theta_d, 
				phi_s = phi_d, 
				omega_s = omega_p*alpha_d, 
				theta_i = np.pi - np.arcsin(alpha_d*crystal.n(detected_photon)/Xi(n_idler=crystal.n(undetected_photon), alpha=alpha_d)*np.sin(theta_d)), 
				omega_p = omega_p,
				photon_pump = pump_photon
			),
			SPDCEvent( # Idler detection, forward-backward.
				theta_s = np.pi - np.arcsin(Xi(n_idler=crystal.n(detected_photon), alpha=1-alpha_d)/(1-alpha_d)/crystal.n(undetected_photon)*np.sin(theta_d)), 
				phi_i = phi_d, 
				omega_s = omega_p*(1-alpha_d), 
				theta_i = theta_d, 
				omega_p = omega_p,
				photon_pump = pump_photon
			),
		]
	return events

########################################################################

if __name__ == '__main__':
	theta_d = 24*np.pi/180
	omega_p = 405e-9
	omega_d = omega_p/2
	alpha_d = omega_d/omega_p
	phi_d = 0
	a = .5
	
	fig, ax = plt.subplots()
	
	for branch in ['branch_1', 'branch_2']:
		dependent_theta_vals, dependent_theta_name = dependent_theta_from_independent_theta(
																a = a, 
																independent_theta = np.linspace(0,np.pi), 
																branch = branch)
		if dependent_theta_name == 'theta_i':
			ax.plot(np.linspace(0,np.pi)*180/np.pi, dependent_theta_vals*180/np.pi, color = (0,0,0), linestyle = '--')
		if dependent_theta_name == 'theta_s':
			ax.plot(dependent_theta_vals*180/np.pi, np.linspace(0,np.pi)*180/np.pi, color = (0,0,0), linestyle = '--')
	
	for event in SPDC_events_seen_by_single_photon_detector(theta_d, phi_d, omega_d, omega_p, a):
		ax.plot([event.get('theta_s')*180/np.pi], [event.get('theta_i')*180/np.pi], marker='o', color=(0,0,0))
	
	ax.plot([0,180], [theta_d*180/np.pi, theta_d*180/np.pi], linestyle='--', color=(0,0,0))
	ax.plot([theta_d*180/np.pi, theta_d*180/np.pi], [0,180], linestyle='--', color=(0,0,0))
	
	plt.show()
