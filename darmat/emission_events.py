import numpy as np

class PhotonEmissionEvent:
	def __init__(self, theta, phi, omega):
		self.vars = {'theta': theta, 'phi': phi, 'omega': omega}
	
	def get(self, what):
		return self.vars.get(what)

class SPDCEvent:
	def __init__(self, theta_s, omega_s, theta_i, phi_s=None, phi_i=None, omega_i=None, omega_p=None):
		if omega_i == omega_p == None:
			raise ValueError('Must specify at least one of omega_i or omega_p')
		if phi_s == phi_i == None:
			raise ValueError('Must specify at least one of phi_s or phi_i')
		if phi_s != None and phi_i != None:
			raise ValueError('Must specify only one of phi_s or phi_i')
		self.signal = PhotonEmissionEvent(
			theta = theta_s, 
			phi = phi_s if phi_i == None else phi_i + np.pi, 
			omega = omega_s
		)
		self.idler = PhotonEmissionEvent(
			theta = theta_i, 
			phi = phi_i if phi_s == None else phi_s + np.pi, 
			omega = omega_i if omega_p==None else omega_p-omega_s
		)
	
	def get(self, what):
		if what[-1] == 's':
			return self.signal.get(what[:-2])
		elif what[-1] == 'i':
			return self.idler.get(what[:-2])
		elif what == 'alpha':
			ws = self.signal.get('omega')
			wi = self.idler.get('omega')
			return ws/(ws+wi)
		else:
			return None
