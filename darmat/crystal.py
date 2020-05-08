import inspect
from types import LambdaType, FunctionType
from .emission_events import Photon, SPDCEvent
from . import common_functions

class Crystal:
	def __init__(self, crystal_length):
		self.crystal_length = crystal_length
	
	def n(self, photon: Photon):
		# Each type of crystal must overload this method.
		raise NotImplementedError()
	
	def SPDC_intensity(event: SPDCEvent):
		# Each type of crystal must overload this method.
		raise NotImplementedError()

class IsotropicCrystal(Crystal):
	def __init__(self, crystal_length, refractive_index):
		if isinstance(refractive_index, FunctionType) != True or inspect.getargspec(refractive_index).args != ['wavelength']:
			raise ValueError('Refractive index must be a function with signature "n = n(wavelength)"')
		self.refractive_index = refractive_index
		super().__init__(crystal_length)
	
	def n(self, photon: Photon):
		return self.refractive_index(wavelength = photon.get('wavelength'))
	
	def SPDC_intensity(self, event: SPDCEvent):
		return common_functions.phase_matching_sinc(
			theta_s = event.get('theta_s'), 
			theta_i = event.get('theta_i'), 
			alpha = event.get('alpha'), 
			crystal_l = self.crystal_length, 
			lambda_p = event.get('lambda_p'), 
			n_pump = self.n(event.photon_pump), 
			n_signal = self.n(event.photon_signal), 
			Xi = common_functions.Xi(
				n_idler = self.n(event.photon_idler),
				alpha = event.get('alpha')
			)
		)

class UniaxialCrystal4Noobs(Crystal):
	def __init__(self, crystal_length, pump_refractive_index, signal_idler_refractive_index):
		if isinstance(pump_refractive_index, FunctionType) != True or inspect.getargspec(pump_refractive_index).args != ['wavelength'] or isinstance(signal_idler_refractive_index, FunctionType) != True or inspect.getargspec(signal_idler_refractive_index).args != ['wavelength']:
			raise ValueError('Refractive index must be a function with signature "n = n(wavelength)"')
		self.pump_refractive_index = pump_refractive_index
		self.signal_idler_refractive_index = signal_idler_refractive_index
		super().__init__(crystal_length)
	
	def n(self, photon: Photon):
		if photon.get('beam_name') == 'signal' or photon.get('beam_name') == 'idler':
			return self.signal_idler_refractive_index(wavelength = photon.get('wavelength'))
		else:
			return self.pump_refractive_index(wavelength = photon.get('wavelength'))
	
	def SPDC_intensity(self, event: SPDCEvent):
		return common_functions.phase_matching_sinc(
			theta_s = event.get('theta_s'), 
			theta_i = event.get('theta_i'), 
			alpha = event.get('alpha'), 
			crystal_l = self.crystal_length, 
			lambda_p = event.get('lambda_p'), 
			n_pump = self.n(event.photon_pump), 
			n_signal = self.n(event.photon_signal), 
			Xi = common_functions.Xi(
				n_idler = self.n(event.photon_idler),
				alpha = event.get('alpha')
			)
		)

