import numpy as np
from SPDC_dSPDC import SPDC_intensity_profile, dSPDC_intensity_profile

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
