from qutip import *
import numpy as np
from Atoms import *
from Driving import *
import matplotlib.pyplot as plt

N = 1

omega = 2 * np.pi * 16 * 10 ** 3  # MHz
#omega = 2 * np.pi * 21 * 10 ** (-20)  # MHz

#omega = 0  # MHz

#Omega_R = 2 * np.pi * 25.7 * 10 ** 0  # MHz

Omega_R = 2 * np.pi * 23.7 * 10 ** 0  # MHz

gamma = 2 * np.pi * 15.0  # MHz

J = 0 * 2 * np.pi * 25.7 * 10 ** 0  # MHz

averages = 1500

sampling_rate = 2 * np.pi * 64 * 10 ** 3  # MHz
endtime = 0.2
timesteps = int(endtime * sampling_rate)
data = np.loadtxt('Forward3MHzcsv.txt')
timesteps = 2 * len(data)


bath = 'Forward3MHzcsv.txt'
#bath = 'Markovian'

gamma1 = 0  # MHz




################### PHASE WALKS ############################################ 33333333333333333333333333333333
data = np.loadtxt('Forward3MHzcsv.txt')
timesteps = len(data)
endtime = 6
pertubation_length = endtime / 1
perturb_times = np.linspace(0, pertubation_length, timesteps)

# Total time.
T = perturb_times[-1]
# Number of steps.
Nsteps = int(len(perturb_times ) /100)
# Time step size
dt = T / Nsteps
# Create an empty array to store the realizations.
x = np.empty((averages, Nsteps + 1))
# Initial values of x.
x[:, 0] = 0

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

for gamma in [3, 5, 15, 30]:

    phase_noise = brownian(x[:, 0], Nsteps, dt /15, np.sqrt(gamma), out=x[:, 1:])

    # phase_noise = davies_harte(perturb_times[-1], len(perturb_times), 0.1, np.sqrt(gamma))
    # print(phase_noise[0:len(perturb_times)])

    t = np.linspace(0.0, Nsteps * dt, Nsteps)

    for k in range(0, int(averages / 100)):
        if gamma == 4:
            ax[0, 1].plot(t, phase_noise[k], color='#85bb65', linewidth=0.1)
            # ax[0, 1].plot(t, phase_noise, color='#85bb65', linewidth=0.1)
        elif gamma == 14:
            ax[0, 1].plot(t, phase_noise[k], color='#CC7722', linewidth=0.1)
            # ax[0, 1].plot(t, phase_noise, color='#CC7722', linewidth=0.1)
        elif gamma == 31:
            ax[0, 1].plot(t, phase_noise[k], color='#800020', linewidth=0.1)
            # ax[0, 1].plot(t, phase_noise, color='#800020', linewidth=0.1)

    if gamma == 30:
        # ax[0, 1].plot(t, np.mean(phase_noise, axis=0), color='red', linestyle='', linewidth=1.0, marker="o", markersize="0.01")
        ax[0, 1].plot(t, np.sqrt(np.var(phase_noise, axis=0)), color='#800020', linestyle='',
                      linewidth=1.0,
                      label='$\gamma = 2\Omega_R$', marker="s", markersize="4", markerfacecolor='none', markeredgecolor='#800020')
        ax[0, 1].plot(t, np.sqrt(gamma * t / 15), color='#800020', linestyle='--', linewidth=1.0)
        # ,label='$\sqrt{30 MHz  t}$')
        # ax[0, 1].plot(t, -np.sqrt(gamma * t), color='#800020', linestyle='--', linewidth=1.0)

    if gamma == 3:
        # ax[0, 1].plot(t, np.mean(phase_noise, axis=0), color='green', linestyle='', linewidth=1.0, marker="o", markersize="0.01")

        ax[0, 1].plot(t, np.sqrt(np.var(phase_noise, axis=0)), color='#025669', linestyle='',
                      linewidth=1.0,
                      label='$\gamma = \Omega_R/5$' ,marker="^", markersize="4", markerfacecolor='none', markeredgecolor = '#025669')
        ax[0, 1].plot(t, np.sqrt(gamma * t / 15), color='#025669', linestyle='--', linewidth=1.0)
        # , label='Expected Standard Deviation = $\sqrt{3 MHz t}$')
        # ax[0, 1].plot(t, -np.sqrt(gamma * t), color='#85bb65', linestyle='--', linewidth=1.0)

    if gamma == 4:
        # ax[0, 1].plot(t, np.mean(phase_noise, axis=0), color='green', linestyle='', linewidth=1.0, marker="o", markersize="0.01")

        ax[0, 1].plot(t, -np.sqrt(np.var(phase_noise, axis=0)), color='#800080', linestyle='-',
                      linewidth=1.0,
                      label='$\gamma = \Omega_R/3$ MHz')
        ax[0, 1].plot(t, -np.sqrt(gamma * t / 15), color='#800080', linestyle='--', linewidth=1.0)
        # , label='Expected Standard Deviation = $\sqrt{3 MHz t}$')
        # ax[0, 1].plot(t, -np.sqrt(gamma * t), color='#85bb65', linestyle='--', linewidth=1.0)

    if gamma == 14:
        # ax[0, 1].plot(t, np.mean(phase_noise, axis=0), color='orange', linestyle='', linewidth=1.0, marker="o", markersize="0.01")

        ax[0, 1].plot(t, -np.sqrt(np.var(phase_noise, axis=0)), color='#CC7722', linestyle='-',
                      linewidth=1.0,
                      label='$\gamma = 10$ MHz')
        ax[0, 1].plot(t, -np.sqrt(gamma * t / 15), color='#CC7722', linestyle='--', linewidth=1.0)
        # ,label='$\sqrt{10 MHz t}$')
        # ax[0, 1].plot(t, -np.sqrt(gamma * t), color='#CC7722', linestyle='--', linewidth=1.0)



data = np.loadtxt('10MHz_gamma.txt')
timesteps = 2 * len(data)
endtime = 6
pertubation_length = endtime / 1

times = np.linspace(0, endtime, timesteps)

data_reversed = -data[::-1]

data = np.cumsum(data)

data_reversed = np.cumsum(data_reversed) + data[-1] + 180

data = np.append(data / 180, data_reversed / 180)

ax[0, 1].errorbar(times, data, label="Phase drift",
                  linewidth="0.4",
                  color='#85bb65')

# ax[0, 1].set_ylim([-1.4 * np.sqrt(30 * T), 1.4 * np.sqrt(30 * T)])
###ax[0, 1].set_xlim([0, 0.1])
ax[0, 1].set_xlabel(r'Time [$1/\Omega_R$]', fontsize=16)
ax[0, 1].set_ylabel(r'Phase [$\pi$]', fontsize=16)
# ax[0, 1].set_xlabel('Time [a.u.]', fontsize=16)
# ax[0, 1].set_ylabel('Apmlitude [a.u.]', fontsize=16)
ax[0, 1].legend(loc="upper left", fontsize=12)
plt.show()
################### END OF PHASE WALKS ############################################ 3333333333333333333333










