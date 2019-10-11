import numpy as np
import scipy.constants as const
from .darmat_rand import sample_with_boxes

def theta_phasematch_dSPDC(lambda_pump, n_pump, n_signal, alpha, m):
	omega_pump = 2*np.pi*const.c/lambda_pump
	costheta = (n_pump**2 + alpha**2*n_signal**2 - (1-alpha)**2 + m**2*const.c**4/omega_pump**2/const.hbar**2)/2/alpha/n_pump/n_signal
	return np.arccos(costheta) if costheta >= -1 and costheta <= 1 else float('nan')

def theta_cutoff_dSPDC(lambda_pump, l, n_pump, n_signal, alpha, m):
	omega_pump = 2*np.pi*const.c/lambda_pump
	return np.arcsin(((1-alpha)**2 - m**2*const.c**4/omega_pump**2/const.hbar**2)**.5/alpha/n_signal)

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
	
	def intensity(self, theta_signal = None, amplitude=1):
		if theta_signal is None:
			max_theta = self.theta_signal_cutoff if not np.isnan(self.theta_signal_cutoff) else np.pi
			q, theta_q = self.theta_signal_zeros()
			if len(theta_q) >= 2:
				step_theta = (np.diff(np.array(theta_q))).min()/20
			else:
				step_theta = max_theta/100
			theta_signal = np.linspace(0,max_theta,int(max_theta/step_theta))
		return theta_signal, dSPDC_intensity_profile(theta_signal, self.lambda_pump, self.crystal_l, self.n_pump, self.n_signal, self.alpha, self.dark_photon_mass)
	
	def samples(self, n_samples=1):
		q, theta = self.theta_signal_zeros()
		return sample_with_boxes(
								  f = lambda x: self.intensity(x)[1],
								  xi = np.array([theta[k] for k in range(len(theta)-1)]), 
								  xf = np.array([theta[k+1] for k in range(len(theta)-1)]), 
								  y = np.array(
												[
													self.intensity(np.linspace(theta[k],theta[k+1]))[1].max()*1.1
													for k in range(len(theta)-1)
												]
											  ), 
								  N = n_samples
								)


########################################################################

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	
	dspdc = dSPDC(
					lambda_pump = 1000e-9, 
					crystal_l = .007e-3, 
					n_pump = 1.5, 
					n_signal = 2.5, 
					alpha = .5, 
					dark_photon_mass = .8e-37
				)
	
	fig, ax = plt.subplots()
	
	q_zeros, theta_zeros = dspdc.theta_signal_zeros()
	for k,t in enumerate(theta_zeros):
		ax.plot([t*180/np.pi]*2, [0,1], color = (.8,.8,1))
		ax.text(t*180/np.pi, 1, str(q_zeros[k]), color = (.8,.8,1))
	
	ax.plot(
			dspdc.theta_signal_phasematch*180/np.pi*np.array([1,1]),
			[0, 1],
			color = (0,.8,0),
			label = 'Phase matching angle'
		   )
	
	ax.plot(
			[180/np.pi*dspdc.theta_signal_cutoff]*2,
			[0,1],
			linestyle = '--',
			color = (0,0,0),
			label = 'Cutoff angle'
		   )
	
	theta, intensity = dspdc.intensity()
	ax.plot(theta*180/np.pi,
			intensity,
			label = 'SPDC'
			)
	
	ax.set_xlabel('Signal angle (degrees)')
	ax.set_ylabel(r'$\propto W_{12}$')
	ax.legend()
	fig.suptitle('New technology')
	
	fig, ax = plt.subplots()
	fig.suptitle('Samples')
	ax.set_xlabel(r'$\theta_s$')
	ax.hist(
			np.array(dspdc.samples(9999))*180/np.pi,
			bins = 'auto',
		   )
	ax.set_yscale('log')
	
	plt.show()

	
	plt.show()
