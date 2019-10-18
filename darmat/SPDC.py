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

def zeros_SPDC_dSPDC(lambda_pump, crystal_l, n_pump, n_signal, alpha, Xi):
	a = alpha*n_signal/Xi
	# First find an approximate range for the "q" values
	theta_test = np.linspace(0, np.pi, 999)
	if a < 1:
		q_range = crystal_l/lambda_pump*(n_pump - alpha*n_signal*np.cos(theta_test) - (Xi**2 - alpha**2*n_signal**2*np.sin(theta_test)**2)**.5)
	if a > 1:
		q_range = crystal_l/lambda_pump*(n_pump - (alpha**2*n_signal**2 - Xi**2*np.sin(theta_test)**2)**.5 - Xi*np.cos(theta_test))
	if a == 1:
		raise ValueError('"a = 1" is not yet implemented')
	independent_theta_zeros = []
	q_zeros = []
	for q in range(int(np.floor(min(q_range))), int(np.ceil(max(q_range)))):
		if q == 0:
			continue
		if a < 1:
			cosenando = ((n_pump - lambda_pump/crystal_l*q)**2 - Xi**2 + n_signal**2*alpha**2)/2/n_signal/alpha/(n_pump - lambda_pump/crystal_l*q)
		if a > 1:
			cosenando = ((n_pump - lambda_pump/crystal_l*q)**2 + Xi**2 - n_signal**2*alpha**2)/2/Xi/(n_pump - lambda_pump/crystal_l*q)
		if cosenando < -1 or cosenando > 1:
			continue
		q_zeros.append(q)
		independent_theta_zeros.append(np.arccos(cosenando))
	return q_zeros, independent_theta_zeros, ('theta_s' if a < 1 else 'theta_i')

def intensity_in_branch_1(lambda_pump, crystal_l, n_pump, n_signal, alpha, Xi, independent_theta_vals = None):
	if independent_theta_vals == None:
		_, independent_theta_zeros, who_is_independent = zeros_SPDC_dSPDC(lambda_pump, crystal_l, n_pump, n_signal, alpha, Xi)
		minimum_distance_between_zeros = min(np.diff(independent_theta_zeros))
		theta_step = minimum_distance_between_zeros/20
		independent_theta_vals = np.linspace(0, np.pi, int(np.pi/theta_step))
	independent_theta_vals = np.array(independent_theta_vals)
	a = alpha*n_signal/Xi
	if a < 1:
		W = sinc(np.pi*crystal_l/lambda_pump*(n_pump - alpha*n_signal*np.cos(independent_theta_vals) - (Xi**2 - alpha**2*n_signal**2*np.sin(independent_theta_vals)**2)**.5))**2
		independent_theta_name = 'theta_s'
	if a > 1:
		W =sinc(np.pi*crystal_l/lambda_pump*(n_pump - (alpha**2*n_signal**2 - Xi**2*np.sin(independent_theta_vals)**2)**.5 - Xi*np.cos(independent_theta_vals)))**2
		independent_theta_name = 'theta_i'
	return independent_theta_vals, W, independent_theta_name

# ~ def dependent_variable_in_branch_1(a, independent_angle):
	# ~ # The parameter "a" is defined as "alpha*n_signal/Xi"
	# ~ if a < 0:
		# ~ raise ValueError('Values of "a" less than 0 are not valid.')
	# ~ if a < 1:
		# ~ theta_dependent

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
		
		self.independent_theta_name = 'theta_s' if self.a < 1 else 'theta_i'
		self.q_zeros, self.independent_theta_zeros, _ = zeros_SPDC_dSPDC(self.lambda_pump, self.crystal_l, self.n_pump, self.n_signal, self.alpha, self.Xi)
	
	def intensity_in_branch_1(self, theta = None):
		theta, W, independent_theta_name = intensity_in_branch_1(self.lambda_pump, self.crystal_l, self.n_pump, self.n_signal, self.alpha, self.Xi, theta)
		return theta, W, independent_theta_name
		
