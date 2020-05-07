import inspect
from types import LambdaType, FunctionType
from .emission_events import Photon

class Crystal:
	def __init__(self, crystal_length):
		self.crystal_length = crystal_length
	
	def n(self, photon: Photon):
		# Each type of crystal must overload this method.
		raise NotImplemented

class IsotropicCrystal(Crystal):
	def __init__(self, crystal_length, refractive_index):
		if isinstance(refractive_index, FunctionType) != True or inspect.getargspec(refractive_index).args != ['wavelength']:
			raise ValueError('Refractive index must be a function with signature "n = n(wavelength)"')
		self.refractive_index = refractive_index
		super().__init__(crystal_length)
	
	def n(self, photon: Photon):
		return self.refractive_index(wavelength = photon.get('wavelength'))
