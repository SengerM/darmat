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

LAMBDA_PUMP = 405e-9
CRYSTAL_L = .3e-3
N_SIGNAL = 1.67
N_IDLER = N_SIGNAL
N_PUMP = 1.55
ALPHA = .82
DARK_PHOTON_MASS = 0#.8e-37

THETA_VALS = np.linspace(0/180*np.pi,180/180*np.pi,1e6)

spdc = SPDC(
				lambda_pump = LAMBDA_PUMP, 
				crystal_l = CRYSTAL_L, 
				n_pump = N_PUMP, 
				n_signal = N_SIGNAL, 
				n_idler = N_IDLER, 
				alpha = ALPHA
			)

dspdc = dSPDC(
				lambda_pump = LAMBDA_PUMP, 
				crystal_l = CRYSTAL_L, 
				n_pump = N_PUMP, 
				n_signal = N_SIGNAL, 
				m_dark_photon = DARK_PHOTON_MASS, 
				alpha = ALPHA
			)

fig, ax = plt.subplots()
fig.suptitle('Total photons detected')
ax.set_xlabel('Detector angle (degrees)')
theta, total_W, W1_indep, W2_indep, W1_dep, W2_dep = spdc.observed_W(theta = THETA_VALS)
ax.plot(theta*180/np.pi, total_W, label = 'Photons SPDC')
theta, W_photons, W_dark_photons, W1_indep, W2_indep, W1_dep, W2_dep = dspdc.observed_W(theta = THETA_VALS)
ax.plot(theta*180/np.pi, W_photons, label = 'Photons dSPDC')
ax.set_ylabel(r'$\propto W_{12}$')
ax.legend()
ax.set_yscale('log')
ax.grid(which='both')

spdc.plot_W_in_thetas_space().suptitle('SPDC')
dspdc.plot_W_in_thetas_space().suptitle('dSPDC')

plt.show()

```
