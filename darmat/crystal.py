import inspect
from types import LambdaType, FunctionType

class Crystal:
	def __init__(self, crystal_length):
		self.crystal_length = crystal_length
	
	def n(self, **kwargs):
		# Each type of crystal must overload this method.
		# For example: n(wavelength) or n(wavelength, direction)
		raise NotImplemented

class IsotropicCrystal(Crystal):
	def __init__(self, crystal_length, refractive_index):
		if isinstance(refractive_index, FunctionType) != True or inspect.getargspec(refractive_index).args != ['wavelength']:
			raise ValueError('Refractive index must be a function with signature "n = n(wavelength)"')
		self.refractive_index = refractive_index
		super().__init__(crystal_length)
	
	def n(self, **kwargs):
		if 'wavelength' not in kwargs:
			raise ValueError('wavelength is a required argument')
		return self.refractive_index(wavelength = kwargs.get('wavelength'))

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	import numpy as np
	
	isocrystal = IsotropicCrystal(refractive_index = lambda wavelength: 1.4*(wavelength/500e-9)**2, crystal_length = 1e-3)
	print('Crystal length = ' + str(isocrystal.crystal_length))
	wavelength = np.linspace(400e-9, 800e-9)
	plt.plot(wavelength*1e9, isocrystal.n(wavelength=wavelength))
	plt.show()
