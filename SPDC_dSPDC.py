import numpy as np
import scipy.constants as const

def SPDC_intensity_profile(r, lambda_pump, l, d, n_pump, n_signal, n_idler, alpha, amplitude=1):
	return np.sinc(np.pi*l/lambda_pump*(n_pump - alpha*n_signal*d/(d**2+r**2)**.5 
					- n_idler*((1-alpha)**2 - alpha**2*n_signal**2/n_idler**2*r**2/(d**2+r**2))**.5))**2

def dSPDC_intensity_profile(r, lambda_pump, l, d, n_pump, n_signal, alpha, m):
	omega_pump = 2*np.pi*const.c/lambda_pump
	return np.sinc(np.pi*l/lambda_pump*(n_pump - alpha*n_signal*d/(d**2+r**2)**.5
				   - ((1-alpha)**2 - m**2*const.c**4/const.hbar**2/omega_pump**2
				   - alpha**2*n_signal**2*r**2/(d**2+r**2))**.5))**2

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	
	LAMBDA_PUMP = 1000e-9
	L = .007e-3
	D = 10e-2
	N_PUMP = 1.5
	N_SIGNAL = 3
	N_IDLER = N_SIGNAL
	ALPHA = .5
	DARK_PHOTON_MASS = 0
	
	r = np.linspace(0, .3, 999999)

	omega_pump = 2*np.pi*const.c/LAMBDA_PUMP
	print('Cutoff angle dSPDC = ' + str(
			180/np.pi*np.arcsin(((1-ALPHA)**2 - DARK_PHOTON_MASS**2*const.c**2/omega_pump**2/const.hbar**2)**.5/ALPHA/N_SIGNAL)
		))
	print('Cutoff angle SPDC = ' + str(180/np.pi*np.arcsin((1-ALPHA)/ALPHA*N_SIGNAL/N_IDLER)))

	fig, ax = plt.subplots()
	ax.plot(np.arcsin(r/(D**2 + r**2)**.5)*180/np.pi,
			SPDC_intensity_profile(
							  r,
							  lambda_pump = LAMBDA_PUMP,
							  l = L,
							  d = D,
							  n_pump = N_PUMP,
							  n_signal = N_SIGNAL,
							  n_idler = N_IDLER,
							  alpha = ALPHA
							),
			label = 'SPDC'
			)
	ax.plot(np.arcsin(r/(D**2 + r**2)**.5)*180/np.pi,
			dSPDC_intensity_profile(
							  r,
							  lambda_pump = LAMBDA_PUMP,
							  l = L,
							  d = D,
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
	
	plt.show()
