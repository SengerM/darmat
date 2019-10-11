# darmat

DarMat experiment Python package.

## Instalation

```
pip install git+https://github.com/SengerM/darmat
```

## Usage

Below there is an example of usage.

```Python
from darmat.SPDC import SPDC
from darmat.dSPDC import dSPDC
import matplotlib.pyplot as plt
import numpy as np

LAMBDA_PUMP = 1000e-9
CRYSTAL_L = .007e-3
N_PUMP = 1.5
N_SIGNAL = 2.5
N_IDLER = N_SIGNAL
ALPHA = .5
DARK_PHOTON_MASS = .8e-37

spdc = SPDC(
				lambda_pump = LAMBDA_PUMP, 
				crystal_l = CRYSTAL_L, 
				n_pump = N_PUMP, 
				n_signal = N_SIGNAL, 
				n_idler = N_IDLER, 
				alpha = ALPHA
			)

fig, ax = plt.subplots()
fig.suptitle('SPDC intensity')
ax.set_xlabel('Signal angle (degrees)')
ax.set_ylabel(r'$\propto W_{12}$')

q_zeros, theta_zeros = spdc.theta_signal_zeros()
for k,t in enumerate(theta_zeros):
	ax.plot([t*180/np.pi]*2, [0,1], color = (.8,.8,1))
	ax.text(t*180/np.pi, 1, str(q_zeros[k]), color = (.8,.8,1))

ax.plot(
		spdc.theta_signal_phasematch*180/np.pi*np.array([1,1]),
		[0, 1],
		color = (0,.8,0),
		label = 'Phase matching angle'
	   )

ax.plot(
		[180/np.pi*spdc.theta_signal_cutoff]*2,
		[0,1],
		linestyle = '--',
		color = (0,0,0),
		label = 'Cutoff angle'
	   )

theta, intensity = spdc.intensity()
ax.plot(theta*180/np.pi,
		intensity,
		label = 'SPDC'
		)

ax.legend()

fig, ax = plt.subplots()
fig.suptitle('SPDC samples')
ax.set_xlabel(r'$\theta_s$')
ax.hist(
		np.array(spdc.samples(9999))*180/np.pi,
		bins = 'auto',
	   )
ax.set_yscale('log')

########################################################################

dspdc = dSPDC(
				lambda_pump = LAMBDA_PUMP, 
				crystal_l = CRYSTAL_L, 
				n_pump = N_PUMP, 
				n_signal = N_SIGNAL, 
				alpha = ALPHA, 
				dark_photon_mass = DARK_PHOTON_MASS
			)

fig, ax = plt.subplots()
fig.suptitle('Dark SPDC intensity')
ax.set_xlabel('Signal angle (degrees)')
ax.set_ylabel(r'$\propto W_{12}$')

q_zeros, theta_zeros = dspdc.theta_signal_zeros()
for k,t in enumerate(theta_zeros):
	ax.plot([t*180/np.pi]*2, [0,1], color = (.8,.8,1))
	ax.text(t*180/np.pi, 1, str(q_zeros[k]), color = (.8,.8,1))

ax.plot(
		dspdc.theta_signal_phasematch*180/np.pi*np.array([1,1]),
		[0, 1],
		color = (0,.8,0),
		label = 'Phase matching angle'
	   )

ax.plot(
		[180/np.pi*dspdc.theta_signal_cutoff]*2,
		[0,1],
		linestyle = '--',
		color = (0,0,0),
		label = 'Cutoff angle'
	   )

theta, intensity = dspdc.intensity()
ax.plot(theta*180/np.pi,
		intensity,
		label = 'SPDC'
		)

ax.legend()

fig, ax = plt.subplots()
fig.suptitle('Dark SPDC samples')
ax.set_xlabel(r'$\theta_s$')
ax.hist(
		np.array(dspdc.samples(9999))*180/np.pi,
		bins = 'auto',
	   )
ax.set_yscale('log')
	
plt.show()
```
