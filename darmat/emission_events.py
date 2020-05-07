import numpy as np
import scipy.constants as const

class Photon:
	def __init__(self, theta, phi, omega=None, wavelength=None):
		if omega == wavelength == None:
			raise ValueError('You must specify either omega or the wavelength')
		self.vars = {
			'theta': theta, 
			'phi': phi, 
			'omega': omega if wavelength == None else 2*const.pi*const.c/wavelength
		}
	
	def get(self, what):
		if what == 'lambda' or what == 'wavelength':
			return 2*const.pi*const.c/self.get('omega')
		return self.vars.get(what)
		
	def __str__(self):
		return 'Photon: theta=' + str(self.get('theta')*180/np.pi) + ' deg, phi=' + str(self.get('phi')*180/np.pi) + ' deg, lambda=' + str(const.c/self.get('omega')*2*const.pi*1e9) + ' nm'

class SPDCEvent:
	def __init__(self, theta_s, omega_s, theta_i, phi_s=None, phi_i=None, omega_i=None, omega_p=None):
		if omega_i == omega_p == None:
			raise ValueError('Must specify at least one of omega_i or omega_p')
		if phi_s == phi_i == None:
			raise ValueError('Must specify at least one of phi_s or phi_i')
		if phi_s != None and phi_i != None:
			raise ValueError('Must specify only one of phi_s or phi_i')
		self.photon_signal = Photon(
			theta = theta_s, 
			phi = phi_s if phi_i == None else phi_i + np.pi, 
			omega = omega_s
		)
		self.photon_idler = Photon(
			theta = theta_i, 
			phi = phi_i if phi_s == None else phi_s + np.pi, 
			omega = omega_i if omega_p==None else omega_p-omega_s
		)
	
	def get(self, what):
		if what[-1] == 's':
			return self.photon_signal.get(what[:-2])
		elif what[-1] == 'i':
			return self.photon_idler.get(what[:-2])
		elif what == 'alpha':
			ws = self.photon_signal.get('omega')
			wi = self.photon_idler.get('omega')
			return ws/(ws+wi)
		else:
			return None
	
	def __str__(self):
		return 'SPDCEvent:\nSignal: ' + str(self.photon_signal) + '\nIdler: ' + str(self.photon_idler)
