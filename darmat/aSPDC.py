import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from .common_functions import sinc

class aSPDC:
	def __init__(self, g: float, pump_power: float, lambda_pump: float, crystal_l: float, n_pump: float, n_signal: float, m_axion: float, e_pump: tuple):
		if n_pump < 0 or n_signal < 0:
			raise ValueError('Negative refractive index received! I do not support this...')
		if crystal_l < 0:
			raise ValueError('The value of "crystal_l" must be positive.')
		if lambda_pump < 0:
			raise ValueError('The value of "lambda_pump" must be positive.')
		if m_axion < 0:
			raise ValueError('The mass of the dark photon must be >= 0.')
		
		self.g = g
		self.pump_power = pump_power
		self.lambda_pump = lambda_pump
		self.crystal_l = crystal_l
		self.n_pump = n_pump
		self.n_signal = n_signal
		self.m_axion = m_axion
		self.e_pump = e_pump # Pump polarization, no need to be normalized since it will be internally normalized.
	
	@property
	def g(self):
		return self._g
	@g.setter
	def g(self, x):
		self._g = x
	
	@property
	def pump_power(self):
		return self._g
	@pump_power.setter
	def pump_power(self, x):
		self._pump_power = x
	
	@property
	def lambda_pump(self):
		return self._lambda_pump
	@lambda_pump.setter
	def lambda_pump(self, x):
		self._lambda_pump = x
	
	@property
	def crystal_l(self):
		return self._crystal_l
	@crystal_l.setter
	def crystal_l(self, x):
		self._crystal_l = x
	
	@property
	def n_pump(self):
		return self._n_pump
	@n_pump.setter
	def n_pump(self, x):
		self._n_pump = x
	
	@property
	def n_signal(self):
		return self._n_signal
	@n_signal.setter
	def n_signal(self, x):
		self._n_signal = x
	
	@property
	def m_axion(self):
		return self._m_axion
	@m_axion.setter
	def m_axion(self, x):
		self._m_axion = x
	
	@property
	def e_pump(self):
		return self._e_pump
	@e_pump.setter
	def e_pump(self, x):
		if len(x) != 2:
			raise TypeError(f'The "e_pump" must be a tuple, list, array, etc, of the form (ex,ey) specifying the polarization of the pump photon. Received {x}')
		x = np.array(x)
		self._e_pump = x/np.linalg.norm(x)
	
	def _vector_term(self, alpha, theta, phi):
		return 1 # Not implemented yet.
		
	
	def prob_density(self, alpha, theta, phi):
		g = self.g
		l = self.crystal_l
		P = self.pump_power
		ns = self.n_signal
		n_p = self.n_pump
		c = const.c
		hbar = const.hbar
		λp = self.lambda_pump
		Q = self._vector_term(alpha, theta, phi)
		m = self.m_axion
		wp = 2*np.pi*c/λp
		Xi = (1-alpha)**2-m**2*c**2/wp**2/hbar**2
		Xi[Xi<0] = float('NaN')
		Xi = Xi**.5
		radicando_mistico = Xi**2-alpha**2*ns**2*np.sin(theta)**2
		radicando_mistico[radicando_mistico<1e-3] = float('NaN')
		sinc1 = sinc(np.pi*l/λp*(alpha*ns*np.cos(theta)+radicando_mistico**.5-n_p))
		sinc2 = sinc(np.pi*l/λp*(alpha*ns*np.cos(theta)-radicando_mistico**.5-n_p))
		return g**2*l**2*P**2/16/np.pi/c**3/hbar*ns**3/n_p**3*Q*alpha**4*np.sin(theta)/(radicando_mistico)**.5*(l**2/λp**2*sinc1**2 + l**2/λp**2*sinc2**2)
		# ~ return 1/radicando_mistico**.5
