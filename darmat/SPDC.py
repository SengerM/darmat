import numpy as np
import scipy.constants as const
from .darmat_rand import sample_with_boxes

def sinc(x):
	return np.sinc(x/np.pi)

def SPDC_intensity_profile(theta, lambda_pump, l, n_pump, n_signal, n_idler, alpha, amplitude=1):
	return sinc(np.pi*l/lambda_pump*
					(
					n_pump - alpha*n_signal*np.cos(theta) 
					- n_idler*(
								(1-alpha)**2 - alpha**2*n_signal**2/n_idler**2*np.sin(theta)**2
							  )**.5
					)
				  )**2

def theta_phasematch_SPDC(lambda_pump, l, n_pump, n_signal, n_idler, alpha):
	costheta = (n_pump**2 + alpha**2*n_signal**2 - n_idler**2*(1-alpha)**2)/2/alpha/n_pump/n_signal
	return np.arccos(costheta) if costheta >= -1 and costheta <= 1 else float('nan')

def theta_cutoff_SPDC(n_signal, n_idler, alpha):
	arg = (1-alpha)/alpha*n_idler/n_signal
	return np.arcsin(arg) if arg <= 1 else float('nan') 

def SPDC_zeros(lambda_pump, l, n_pump, n_signal, n_idler, alpha, q_try = range(-300,300)):
	theta_zeros = []
	q_zeros = []
	for q in q_try:
		if q == 0:
			continue
		radicando = 1 - ((lambda_pump/l*q - n_pump)**2 + n_signal**2*alpha**2 - n_idler**2*(1-alpha)**2)**2/(4*n_signal**2*alpha**2*(lambda_pump/l*q - n_pump)**2)
		if radicando < 0:
			continue
		xq = np.sqrt(radicando)
		if xq < -1 or xq > 1 or np.isnan(xq):
			continue
		theta_zeros.append(np.arcsin(xq))
		q_zeros.append(q)
		if len(theta_zeros) == 1:
			continue
		if theta_zeros[-1] < theta_zeros[-2]: # This means that we are no longer finding zeros
			theta_zeros = theta_zeros[:-1]
			q_zeros = q_zeros[:-1]
			break
	return q_zeros, theta_zeros

class SPDC:
	def __init__(self, lambda_pump, crystal_l, n_pump, n_signal, n_idler, alpha):
		if n_pump < 0 or n_signal < 0 or n_idler < 0:
			raise ValueError('Negative refractive index received! I do not support this...')
		if alpha < 0 or alpha > 1:
			raise ValueError('The value of "alpha" must be between 0 and 1 (by definition of alpha).')
		if crystal_l < 0:
			raise ValueError('The value of "crystal_l" must be positive.')
		if lambda_pump < 0:
			raise ValueError('The value of "lambda_pump" must be positive.')
		
		self.lambda_pump = lambda_pump
		self.crystal_l = crystal_l # Nonlinear medium length.
		self.n_pump = n_pump
		self.n_signal = n_signal
		self.n_idler = n_idler
		self.alpha = alpha # Ratio of signal frequency to pump frequency, i.e. omega_signal/omega_pump.
		
		self.theta_signal_phasematch = theta_phasematch_SPDC(self.lambda_pump, self.crystal_l, self.n_pump, self.n_signal, self.n_idler, self.alpha)
		self.theta_signal_cutoff = theta_cutoff_SPDC(self.n_signal, self.n_idler, self.alpha)
		
		self._theta_signal_zeros = []
		self._q_signal_zeros = []
	
	def theta_signal_zeros(self, q_try = range(-300,300)):
		if self._theta_signal_zeros == [] or self._q_signal_zeros == []:
			q, theta = SPDC_zeros(self.lambda_pump, self.crystal_l, self.n_pump, self.n_signal, self.n_idler, self.alpha, q_try)
			self._theta_signal_zeros = theta
			self._q_signal_zeros = q
		return self._q_signal_zeros, self._theta_signal_zeros
	
	def signal_intensity(self, theta_signal = None, amplitude=1):
		if theta_signal is None:
			max_theta = self.theta_signal_cutoff if not np.isnan(self.theta_signal_cutoff) else np.pi/2
			q, theta_q = self.theta_signal_zeros()
			if len(theta_q) >= 2:
				step_theta = (np.diff(np.array(theta_q))).min()/20
			else:
				step_theta = max_theta/100
			theta_signal = np.linspace(0,max_theta,int(max_theta/step_theta))
		return theta_signal, SPDC_intensity_profile(theta_signal, self.lambda_pump, self.crystal_l, self.n_pump, self.n_signal, self.n_idler, self.alpha, amplitude)
	
	def idler_intensity(self, theta_idler=None, amplitude=1):
		if theta_idler is None:
			theta_idler,_ = self.signal_intensity()
		theta_idler[theta_idler<0] = float('nan')
		theta_idler[theta_idler>np.pi/2] = float('nan')
		_,intensity = self.signal_intensity(theta_signal = np.arcsin((1-self.alpha)*self.n_idler/self.alpha/self.n_signal*np.sin(theta_idler)), amplitude = amplitude)
		return theta_idler, intensity
	
	def signal_samples(self, n_samples=1):
		_, theta_zeros = self.theta_signal_zeros()
		theta_signal_samples = sample_with_boxes(
								  f = lambda x: self.signal_intensity(x)[1],
								  xi = np.array([0] + [theta_zeros[k] for k in range(len(theta_zeros)-1)]), 
								  xf = np.array([theta_zeros[k] for k in range(len(theta_zeros))]), 
								  y = np.array(
												[self.signal_intensity(np.linspace(0,theta_zeros[0]))[1].max()*1.1] + 
												[
													self.signal_intensity(np.linspace(theta_zeros[k],theta_zeros[k+1]))[1].max()*1.1
													for k in range(len(theta_zeros)-1)
												]
											  ), 
								  N = n_samples
								)
		phi_signal_samples = np.random.rand(n_samples)*2*np.pi
		return phi_signal_samples, theta_signal_samples
	
	def theta_idler(self, theta_signal):
		return np.arcsin(self.alpha/(1-self.alpha)*self.n_signal/self.n_idler*np.sin(theta_signal))

########################################################################
########################################################################
########################################################################

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

# W functions ↑ --------------------------------------------------------

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
	independent_theta_vals, independent_theta_name = independent_theta_from_dependent_theta_in_branch_1(a, dependent_theta_vals)
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
	W = [w1+w2 for w1,w2 in zip(W_first_half,W_second_half)]
	return W, theta_name(a).get('dependent')

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

class new_SPDC:
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
			independent_theta_vals = np.linspace(0, np.pi, int(np.pi/theta_step))
			dependent_theta_vals, dependent_theta_name = dependent_theta_from_independent_theta(self.a, independent_theta_vals, branch)
		W, dependent_theta_name = W_in_branch_as_function_of_dependent_theta(
											self.lambda_pump, 
											self.crystal_l, 
											self.n_pump, 
											self.n_signal, 
											self.alpha, 
											self.Xi, 
											dependent_theta_vals, 
											branch)
		return dependent_theta_vals, W
	
