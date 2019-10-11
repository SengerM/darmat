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
import matplotlib.pyplot as plt
import numpy as np

spdc = SPDC(
				lambda_pump = 1000e-9, 
				crystal_l = .007e-3, 
				n_pump = 1.5, 
				n_signal = 2.5, 
				n_idler = 2.5, 
				alpha = .6
			)

fig, ax = plt.subplots()

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

ax.set_xlabel('Signal angle (degrees)')
ax.set_ylabel(r'$\propto W_{12}$')
ax.legend()
fig.suptitle('New technology')

fig, ax = plt.subplots()
fig.suptitle('Samples')
ax.set_xlabel(r'$\theta_s$')
ax.hist(
		np.array(spdc.samples(9999))*180/np.pi,
		bins = 'auto',
	   )
ax.set_yscale('log')

plt.show()
```
