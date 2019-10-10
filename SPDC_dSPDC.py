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

def theta_cutoff_SPDC(n_signal, n_idler, alpha):
	return np.arcsin((1-alpha)/alpha*n_signal/n_idler)

def theta_cutoff_dSPDC(lambda_pump, l, n_pump, n_signal, alpha, m):
	omega_pump = 2*np.pi*const.c/lambda_pump
	return np.arcsin(((1-alpha)**2 - m**2*const.c**4/omega_pump**2/const.hbar**2)**.5/alpha/n_signal)

def theta_phasematch_dSPDC(lambda_pump, l, n_pump, n_signal, alpha, m, omega_pump):
	costheta = (n_pump**2 + alpha**2*n_signal**2 - (1-alpha)**2 - m**2*const.c*4/omega_pump**2/const.hbar**2)/2/alpha/n_pump/n_signal
	return np.arccos(costheta) if costheta >= -1 and costheta <= 1 else float('nan')

def dSPDC_intensity_profile(theta, lambda_pump, l, n_pump, n_signal, alpha, m):
	omega_pump = 2*np.pi*const.c/lambda_pump
	return np.sinc(np.pi*l/lambda_pump*(n_pump - alpha*n_signal*np.cos(theta)
				   - ((1-alpha)**2 - m**2*const.c**4/const.hbar**2/omega_pump**2
				   - alpha**2*n_signal**2*np.sin(theta)**2)**.5))**2

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	
	LAMBDA_PUMP = 1000e-9
	L = .007e-3
	D = 10e-2
	N_PUMP = 1.5
	N_SIGNAL = 3
	N_IDLER = N_SIGNAL
	ALPHA = .5
	DARK_PHOTON_MASS = 1.1e-36
	
	theta = np.linspace(0, np.pi/2, 99999)

	omega_pump = 2*np.pi*const.c/LAMBDA_PUMP
	print('Cutoff angle dSPDC = ' + str(180/np.pi*theta_cutoff_dSPDC(LAMBDA_PUMP, L, N_PUMP, N_SIGNAL, ALPHA, DARK_PHOTON_MASS)))
	print('Cutoff angle SPDC = ' + str(180/np.pi*theta_cutoff_SPDC(N_SIGNAL, N_IDLER, ALPHA)))
	
	print('theta phasematch dSPDC = ' + str(theta_phasematch_dSPDC(LAMBDA_PUMP, L, N_PUMP, N_SIGNAL, ALPHA, DARK_PHOTON_MASS, omega_pump)*180/np.pi))
	
	fig, ax = plt.subplots()
	theta_q_last = float('nan')
	# ~ for q in range(-300,300):
		# ~ if q == 0:
			# ~ continue
		# ~ xq = np.sqrt(
			   # ~ 1 - ((LAMBDA_PUMP/L*q/np.pi - N_PUMP)**2 + N_SIGNAL**2*ALPHA**2 - 
			   # ~ N_IDLER**2*(1-ALPHA)**2)**2 /
			   # ~ (4*N_SIGNAL**2*ALPHA**2*(LAMBDA_PUMP/L*q/np.pi - N_PUMP)**2)
			  # ~ )
		# ~ theta_q = np.arcsin(xq)
		# ~ if theta_q < theta_q_last:
			# ~ break
		# ~ theta_q_last = theta_q
		# ~ ax.plot(
				# ~ theta_q*180/np.pi*np.array([1,1]),
				# ~ [0, 1],
				# ~ color = (.7,.7,1)
			   # ~ )
	
	# ~ theta_q_last = float('nan')
	# ~ for q in range(-200,100):
		# ~ xq = np.sqrt(
			   # ~ 1 - ((LAMBDA_PUMP/L*q/np.pi - N_PUMP)**2 + N_SIGNAL**2*ALPHA**2 - 
			   # ~ (1-ALPHA)**2 + DARK_PHOTON_MASS**2*const.c**4/omega_pump**2/const.hbar**2)**2 /
			   # ~ (4*N_SIGNAL**2*ALPHA**2*(LAMBDA_PUMP/L*q/np.pi - N_PUMP)**2)
			  # ~ )
		# ~ theta_q = np.arcsin(xq)
		# ~ if theta_q < theta_q_last:
			# ~ break
		# ~ theta_q_last = theta_q
		# ~ ax.plot(
				# ~ theta_q*180/np.pi*np.array([1,1]),
				# ~ [0, 1],
				# ~ color = (1,.7,.7)
			   # ~ )
	
	ax.plot(
			theta_phasematch_SPDC(LAMBDA_PUMP, L, N_PUMP, N_SIGNAL, N_IDLER, ALPHA)*180/np.pi*np.array([1,1]),
			[0, 1],
			color = (0,.8,0),
			label = 'Phase matching angle'
		   )
	ax.plot(
			theta_phasematch_dSPDC(LAMBDA_PUMP, L, N_PUMP, N_SIGNAL, ALPHA, DARK_PHOTON_MASS, omega_pump)*180/np.pi*np.array([1,1]),
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
	
	plt.show()
