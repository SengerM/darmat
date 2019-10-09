import numpy as np
from SPDC_dSPDC import SPDC_intensity_profile, dSPDC_intensity_profile

class SPDC:
	
	def __init__(self, lambda_pump, crystal_l, n_pump, n_signal, n_idler, alpha):
		slef.lambda_pump = lambda_pump
		self.crystal_l = crystal_l
		self.n_pump = n_pump
		self.n_signal = n_signal
		self.n_idler = n_idler
		self.alpha = alpha
		
		self.theta_phasematch = (n_pump**2 + alpha**2*n_signal**2 - n_idler**2*(1-alpha)**2)/2/alpha/n_pump/n_signal
		self.theta_phasematch = np.arccos(self.theta_phasematch) if self.theta_phasematch >= -1 and self.theta_phasematch <= 1 else float('nan')
		self.theta_signal_cutoff = np.arcsin((1-alpha)/alpha*n_signal/n_idler)

def _sample_with_boxes(f, xi, xf, y, N = 1):
	i_choices = [i for i in range(len(xi))]
	p_i = (xf - xi)*y/((xf - xi)*y).sum()
	samples = []
	while True:
		i = np.random.choice(i_choices, p = p_i)
		X = np.random.uniform(xi[i], xf[i])
		Y = np.random.uniform(0, y[i])
		if Y < f(X): 
			samples.append(X)
		if len(samples) == N:
			return samples

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	
	def f(x):
		if x < 0 or x > 1:
			return 0
		else:
			return -x+1
	
	fig, ax = plt.subplots()
	x_axis = np.linspace(-2,2,999)
	ax.plot(x_axis, [f(x) for x in x_axis])
	
	samples = _sample_with_boxes(f, 
								xi = np.array([0, .5]), 
								xf = np.array([.5, 1]),
								y = np.array([f(0), f(.5)]),
								N = 9999
								)
	fig, ax = plt.subplots()
	ax.hist(samples)
	
	
	
	plt.show()
