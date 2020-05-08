import numpy as np

class DDDVector:
	def __init__(self, cartessian_components=None, spherical_components=None):
		if cartessian_components != None and spherical_components == None:
			if not (isinstance(cartessian_components, tuple) and len(cartessian_components) == 3):
				raise ValueError('"cartessian_components" must be a tuple:' + str(cartessian_components))
			self._x = cartessian_components[0]
			self._y = cartessian_components[1]
			self._z = cartessian_components[2]
		elif spherical_components != None and cartessian_components == None:
			if not (isinstance(spherical_components, tuple) and len(spherical_components) == 3):
				raise ValueError('"cartessian_components" must be a tuple:' + str(spherical_components))
			self._x = spherical_components[0]*np.sin(spherical_components[2])*np.cos(spherical_components[1])
			self._y = spherical_components[0]*np.sin(spherical_components[2])*np.sin(spherical_components[1])
			self._z = spherical_components[0]*np.cos(spherical_components[2])
		else:
			raise ValueError('You must specify only one set of components')
	
	@property
	def theta(self):
		return np.arctan2((self._x**2+self._y**2)**.5, self._z)
	
	@property
	def phi(self):
		return np.arctan2(self._y, self._x)
