import numpy as np
import scipy.constants as const

def SPDC_intensity_profile(theta, lambda_pump, l, n_pump, n_signal, n_idler, alpha, amplitude=1):
	return np.sinc(np.pi*l/lambda_pump*
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

def theta_phasematch_dSPDC(lambda_pump, n_pump, n_signal, alpha, m):
	omega_pump = 2*np.pi*const.c/lambda_pump
	costheta = (n_pump**2 + alpha**2*n_signal**2 - (1-alpha)**2 + m**2*const.c**4/omega_pump**2/const.hbar**2)/2/alpha/n_pump/n_signal
	return np.arccos(costheta) if costheta >= -1 and costheta <= 1 else float('nan')

def theta_cutoff_SPDC(n_signal, n_idler, alpha):
	return np.arcsin((1-alpha)/alpha*n_signal/n_idler)

def theta_cutoff_dSPDC(lambda_pump, l, n_pump, n_signal, alpha, m):
	omega_pump = 2*np.pi*const.c/lambda_pump
	return np.arcsin(((1-alpha)**2 - m**2*const.c**4/omega_pump**2/const.hbar**2)**.5/alpha/n_signal)

def dSPDC_intensity_profile(theta, lambda_pump, l, n_pump, n_signal, alpha, m):
	omega_pump = 2*np.pi*const.c/lambda_pump
	return np.sinc(np.pi*l/lambda_pump*(n_pump - alpha*n_signal*np.cos(theta)
				   - ((1-alpha)**2 - m**2*const.c**4/const.hbar**2/omega_pump**2
				   - alpha**2*n_signal**2*np.sin(theta)**2)**.5))**2

def SPDC_zeros(lambda_pump, l, n_pump, n_signal, n_idler, alpha, q_try = range(-300,300)):
	theta_zeros = []
	q_zeros = []
	for q in q_try:
		if q == 0:
			continue
		xq = np.sqrt(
			   1 - ((lambda_pump/l*q/np.pi - n_pump)**2 + n_signal**2*alpha**2 - 
			   n_idler**2*(1-alpha)**2)**2 /
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

class SPDC:
	
	def __init__(self, lambda_pump, crystal_l, n_pump, n_signal, n_idler, alpha):
		self.lambda_pump = lambda_pump
		self.crystal_l = crystal_l # Nonlinear medium length.
		self.n_pump = n_pump
		self.n_signal = n_signal
		self.n_idler = n_idler
		self.alpha = alpha # Ratio of signal frequency to pump frequency, i.e. omega_signal/omega_pump.
		
		self.theta_phasematch = theta_phasematch_SPDC(self.lambda_pump, self.crystal_l, self.n_pump, self.n_signal, self.n_idler, self.alpha)
		self.theta_signal_cutoff = theta_cutoff_SPDC(self.n_signal, self.n_idler, self.alpha)
		
		self._zeros_theta = []
		self._zeros_q = []
	
	def zeros(self, q_try = range(-300,300)):
		if self._zeros_theta == [] or self._zeros_q == []:
			q, theta = SPDC_zeros(self.lambda_pump, self.crystal_l, self.n_pump, self.n_signal, self.n_idler, self.alpha, q_try)
			self._zeros_theta = theta
			self._zeros_q = q
		return self._zeros_q, self._zeros_theta
	
	
########################################################################

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	
	LAMBDA_PUMP = 1000e-9
	L = .007e-3
	D = 10e-2
	N_PUMP = 1.5
	N_SIGNAL = 2.5
	N_IDLER = N_SIGNAL
	ALPHA = .5
	DARK_PHOTON_MASS = .8e-37
	
	theta = np.linspace(0, np.pi/2, 99999)

	omega_pump = 2*np.pi*const.c/LAMBDA_PUMP
	print('Cutoff angle dSPDC = ' + str(180/np.pi*theta_cutoff_dSPDC(LAMBDA_PUMP, L, N_PUMP, N_SIGNAL, ALPHA, DARK_PHOTON_MASS)))
	print('Cutoff angle SPDC = ' + str(180/np.pi*theta_cutoff_SPDC(N_SIGNAL, N_IDLER, ALPHA)))
	
	print('theta phasematch dSPDC = ' + str(theta_phasematch_dSPDC(LAMBDA_PUMP, N_PUMP, N_SIGNAL, ALPHA, DARK_PHOTON_MASS)*180/np.pi))
	
	fig, ax = plt.subplots()
	
	q_zeros, theta_zeros = SPDC_zeros(LAMBDA_PUMP, L, N_PUMP, N_SIGNAL, N_IDLER, ALPHA)
	for k,t in enumerate(theta_zeros):
		ax.plot([t*180/np.pi]*2, [0,1], color = (.8,.8,1))
		ax.text(t*180/np.pi, 1, str(q_zeros[k]), color = (.8,.8,1))
	
	q_zeros, theta_zeros = dSPDC_zeros(LAMBDA_PUMP, L, N_PUMP, N_SIGNAL, ALPHA, DARK_PHOTON_MASS)
	for k,t in enumerate(theta_zeros):
		ax.plot([t*180/np.pi]*2, [0,1], color = (1,.8,.8))
		ax.text(t*180/np.pi, .8, str(q_zeros[k]), color = (1,.8,.8))
	
	ax.plot(
			theta_phasematch_SPDC(LAMBDA_PUMP, L, N_PUMP, N_SIGNAL, N_IDLER, ALPHA)*180/np.pi*np.array([1,1]),
			[0, 1],
			color = (0,.8,0),
			label = 'Phase matching angle'
		   )
	ax.plot(
			theta_phasematch_dSPDC(LAMBDA_PUMP, N_PUMP, N_SIGNAL, ALPHA, DARK_PHOTON_MASS)*180/np.pi*np.array([1,1]),
			[0, 1],
			color = (0,.8,0)
		   )
	ax.plot(
			[180/np.pi*theta_cutoff_SPDC(N_SIGNAL, N_IDLER, ALPHA)]*2,
			[0,1],
			linestyle = '--',
			color = (0,0,0),
			label = 'Cutoff angle'
		   )
	ax.plot(
			[180/np.pi*theta_cutoff_dSPDC(LAMBDA_PUMP, L, N_PUMP, N_SIGNAL, ALPHA, DARK_PHOTON_MASS)]*2,
			[0,1],
			linestyle = '--',
			color = (0,0,0)
		   )
	
	ax.plot(theta*180/np.pi,
			SPDC_intensity_profile(
							  theta,
							  lambda_pump = LAMBDA_PUMP,
							  l = L,
							  n_pump = N_PUMP,
							  n_signal = N_SIGNAL,
							  n_idler = N_IDLER,
							  alpha = ALPHA
							),
			label = 'SPDC'
			)
	ax.plot(theta*180/np.pi,
			dSPDC_intensity_profile(
							  theta,
							  lambda_pump = LAMBDA_PUMP,
							  l = L,
							  n_pump = N_PUMP,
							  n_signal = N_SIGNAL,
							  alpha = ALPHA,
							  m = DARK_PHOTON_MASS
							),
			label = 'dark SPDC'
			)
	
	ax.set_xlabel('Signal angle (degrees)')
	ax.set_ylabel(r'$\propto W_{12}$')
	ax.legend()
	fig.suptitle('Old technology')
	
	############
	############
	############
	
	spdc = SPDC(
					lambda_pump = LAMBDA_PUMP, 
					crystal_l = L, 
					n_pump = N_PUMP, 
					n_signal = N_SIGNAL, 
					n_idler = N_IDLER, 
					alpha = ALPHA
				)
	
	fig, ax = plt.subplots()
	
	q_zeros, theta_zeros = spdc.zeros()
	for k,t in enumerate(theta_zeros):
		ax.plot([t*180/np.pi]*2, [0,1], color = (.8,.8,1))
		ax.text(t*180/np.pi, 1, str(q_zeros[k]), color = (.8,.8,1))
	
	ax.set_xlabel('Signal angle (degrees)')
	ax.set_ylabel(r'$\propto W_{12}$')
	ax.legend()
	fig.suptitle('New technology')
	
	plt.show()
