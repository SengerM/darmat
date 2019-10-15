import numpy as np
import scipy.constants as const
from .darmat_rand import sample_with_boxes

def theta_phasematch_dSPDC(lambda_pump, n_pump, n_signal, alpha, m):
	omega_pump = 2*np.pi*const.c/lambda_pump
	costheta = (n_pump**2 + alpha**2*n_signal**2 - (1-alpha)**2 + m**2*const.c**4/omega_pump**2/const.hbar**2)/2/alpha/n_pump/n_signal
	return np.arccos(costheta) if costheta >= -1 and costheta <= 1 else float('nan')

def theta_cutoff_dSPDC(lambda_pump, l, n_pump, n_signal, alpha, m):
	omega_pump = 2*np.pi*const.c/lambda_pump
	arg = ((1-alpha)**2 - m**2*const.c**4/omega_pump**2/const.hbar**2)**.5/alpha/n_signal
	return np.arcsin(arg) if arg <= 1 else float('nan')

def dSPDC_intensity_profile(theta, lambda_pump, l, n_pump, n_signal, alpha, m):
	omega_pump = 2*np.pi*const.c/lambda_pump
	return np.sinc(np.pi*l/lambda_pump*(n_pump - alpha*n_signal*np.cos(theta)
				   - ((1-alpha)**2 - m**2*const.c**4/const.hbar**2/omega_pump**2
				   - alpha**2*n_signal**2*np.sin(theta)**2)**.5))**2

def dSPDC_zeros(lambda_pump, l, n_pump, n_signal, alpha, m, q_try = range(-300,300)):
	omega_pump = 2*np.pi*const.c/lambda_pump
	theta_zeros = []
	q_zeros = []
	for q in q_try:
		if q == 0:
			continue
		xq = np.sqrt(
			   1 - ((lambda_pump/l*q/np.pi - n_pump)**2 + n_signal**2*alpha**2 - 
			   (1-alpha)**2 + m**2*const.c**4/omega_pump**2/const.hbar**2)**2 /
			   (4*n_signal**2*alpha**2*(lambda_pump/l*q/np.pi - n_pump)**2)
			  )
		if xq < -1 or xq > 1 or np.isnan(xq):
			continue
		theta_zeros.append(np.arcsin(xq))
		q_zeros.append(q)
		if len(theta_zeros) == 1:
			continue
		if theta_zeros[-1] < theta_zeros[-2]: # This means that we are no longer finding zeros
			theta_zeros = theta_zeros[:-2]
			q_zeros = q_zeros[:-2]
			break
	return q_zeros, theta_zeros

class dSPDC:
	def __init__(self, lambda_pump, crystal_l, n_pump, n_signal, alpha, dark_photon_mass):
		if n_pump < 0 or n_signal < 0:
			raise ValueError('Negative refractive index received! I do not support this...')
		if crystal_l < 0:
			raise ValueError('The value of "crystal_l" must be positive.')
		if lambda_pump < 0:
			raise ValueError('The value of "lambda_pump" must be positive.')
		if dark_photon_mass < 0:
			raise ValueError('The mass of the dark photon must be possitive!')
		if alpha < 0 or alpha > 1 - const.c**2*dark_photon_mass/const.hbar/(2*np.pi*const.c/lambda_pump):
			raise ValueError('The value of "alpha" must be between 0 and "1 - c**2*m/hbar/omega_pump" (by definition of alpha).')
		
		self.lambda_pump = lambda_pump
		self.crystal_l = crystal_l # Nonlinear medium length.
		self.n_pump = n_pump
		self.n_signal = n_signal
		self.alpha = alpha # Ratio of signal frequency to pump frequency, i.e. omega_signal/omega_pump.
		self.dark_photon_mass = dark_photon_mass
		
		self.theta_signal_phasematch = theta_phasematch_dSPDC(self.lambda_pump, self.n_pump, self.n_signal, self.alpha, self.dark_photon_mass)
		self.theta_signal_cutoff = theta_cutoff_dSPDC(self.lambda_pump, self.crystal_l, self.n_pump, self.n_signal, self.alpha, self.dark_photon_mass)
		
		self._theta_signal_zeros = []
		self._q_signal_zeros = []
	
	def theta_signal_zeros(self, q_try = range(-300,300)):
		if self._theta_signal_zeros == [] or self._q_signal_zeros == []:
			q, theta = dSPDC_zeros(self.lambda_pump, self.crystal_l, self.n_pump, self.n_signal, self.alpha, self.dark_photon_mass, q_try)
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
		return theta_signal, dSPDC_intensity_profile(theta_signal, self.lambda_pump, self.crystal_l, self.n_pump, self.n_signal, self.alpha, self.dark_photon_mass)
	
	def signal_samples(self, n_samples=1):
		q, theta = self.theta_signal_zeros()
		theta_signal_samples = sample_with_boxes(
								  f = lambda x: self.signal_intensity(x)[1],
								  xi = np.array([theta[k] for k in range(len(theta)-1)]), 
								  xf = np.array([theta[k+1] for k in range(len(theta)-1)]), 
								  y = np.array(
												[
													self.signal_intensity(np.linspace(theta[k],theta[k+1]))[1].max()*1.1
													for k in range(len(theta)-1)
												]
											  ), 
								  N = n_samples
								)
		phi_signal_samples = np.random.rand(n_samples)*2*np.pi
		return phi_signal_samples, theta_signal_samples
	
	def theta_idler(self, theta_signal):
		omega_pump = 2*np.pi*const.c/self.lambda_pump
		return np.arcsin(self.alpha*self.n_signal/np.sqrt((1-self.alpha)**2 - self.dark_photon_mass**2*const.c**4/omega_pump**2/const.hbar**2)*np.sin(theta_signal))
	
