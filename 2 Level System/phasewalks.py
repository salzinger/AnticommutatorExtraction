from qutip import *
import numpy as np
from Atoms import *
from Driving import *
import matplotlib.pyplot as plt

plt.rcParams.update({
  "text.usetex": True,
    "font.family":"sans-serif",
})



#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#plt.rc('font',**{'family':'serif','serif':['Latin Modern Roman']})

#plt.rc('figure', figsize=(11.69, 8.27))

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




fig, ax = plt.subplots(1, 1, figsize=(11.69, 11.69))
#fig, ax = plt.subplots(2, 2, figsize=(11.69, 8.27))

#fig.set_dpi(250.0)

for gamma in [3, 5, 15, 30]:

    data = np.loadtxt('Forward3MHzcsv.txt')
    timesteps = len(data)
    endtime = 6
    pertubation_length = endtime / 1
    perturb_times = np.linspace(0, pertubation_length, timesteps)

    # Total time.
    T = perturb_times[-1]
    # Number of steps.
    Nsteps = int(len(perturb_times) / 100)
    # Time step size
    dt = T / Nsteps
    # Create an empty array to store the realizations.
    x = np.empty((averages, Nsteps + 1))
    # Initial values of x.
    x[:, 0] = 0

    phase_noise = brownian(x[:, 0], Nsteps, dt /15, np.sqrt(gamma), out=x[:, 1:])




    t = np.linspace(0.0, Nsteps * dt, Nsteps)

    data = np.loadtxt('10MHz_gamma.txt')
    timesteps = 2 * len(data)
    endtime = 6
    pertubation_length = endtime / 1

    perturb_times = np.linspace(0, pertubation_length, timesteps)

    # Total time.
    T = perturb_times[-1]
    # Number of steps.
    Nsteps = int(len(perturb_times))
    # Time step size
    dt = T / Nsteps
    # Create an empty array to store the realizations.
    x = np.empty((averages, Nsteps + 1))
    # Initial values of x.
    x[:, 0] = 0

    phase_noise_plot = brownian(x[:, 0], Nsteps, dt / 15, np.sqrt(gamma), out=x[:, 1:])

    times = np.linspace(0.0, Nsteps * dt, Nsteps)

    # phase_noise = davies_harte(perturb_times[-1], len(perturb_times), 0.1, np.sqrt(gamma))
    # print(phase_noise[0:len(perturb_times)])



    for k in range(0, int(averages / averages)):

        if gamma == 4:
            ax.plot(times, phase_noise_plot[k], color='#85bb65', linewidth=0.1)
            # ax[0, 1].plot(t, phase_noise, color='#85bb65', linewidth=0.1)
        elif gamma == 14:
            ax.plot(times, phase_noise_plot[k], color='#CC7722', linewidth=0.1)
            # ax[0, 1].plot(t, phase_noise, color='#CC7722', linewidth=0.1)
        elif gamma == 31:
            ax.plot(times, phase_noise_plot[k], color='#800020', linewidth=0.1, label="Single trajectory $\gamma=2\Omega_R$")
            # ax[0, 1].plot(t, phase_noise, color='#800020', linewidth=0.1)

    if gamma == 30:
        # ax[0, 1].plot(t, np.mean(phase_noise, axis=0), color='red', linestyle='', linewidth=1.0, marker="o", markersize="0.01")
        ax.plot(t, np.sqrt(np.var(phase_noise, axis=0)), color='#800020', linestyle='',
                      linewidth=1.0,
                      label='$\gamma = 2\Omega_R$', marker="s", markersize="8", markerfacecolor='none', markeredgecolor='#800020')
        ax.plot(t, np.sqrt(gamma * t / 15), color='#800020', linestyle='', linewidth=1.0)

        #ax[0, 1].plot(t, -np.sqrt(np.var(phase_noise, axis=0)), color='#800020', linestyle='-',
        #              linewidth=1.0,
        #              label='$\gamma = \Omega_R/3$ MHz')

        ax.plot(t, -np.sqrt(gamma * t / 15), color='#800020', linestyle='', linewidth=5.0)
        ax.fill_between(t, np.sqrt(gamma * t / 15), np.sqrt(3 * t / 15), color='#800020', alpha=0.2)
        ax.fill_between(t, -np.sqrt(gamma * t / 15), -np.sqrt(3 * t / 15), color='#800020', alpha=0.2)
        ax.plot(times, phase_noise_plot[k], color='#800020', linewidth=0.1, label="Single trajectory $\gamma=2\Omega_R$")
        # ,label='$\sqrt{30 MHz  t}$')
        # ax[0, 1].plot(t, -np.sqrt(gamma * t), color='#800020', linestyle='--', linewidth=1.0)

    if gamma == 3:
        # ax[0, 1].plot(t, np.mean(phase_noise, axis=0), color='green', linestyle='', linewidth=1.0, marker="o", markersize="0.01")

        ax.plot(t, np.sqrt(np.var(phase_noise, axis=0)), color='#025669', linestyle='',
                      linewidth=1.0,
                      label='$\gamma = \Omega_R/5$' ,marker="^", markersize="8", markerfacecolor='none', markeredgecolor = '#025669')
        ax.plot(t, np.sqrt(gamma * t / 15), color='#025669', linestyle='', linewidth=1.0)

        #ax[0, 1].plot(t, -np.sqrt(np.var(phase_noise, axis=0)), color='#025669', linestyle='-',
        #              linewidth=1.0,
        #              label='$\gamma = 10$ MHz')
        ax.plot(t, -np.sqrt(gamma * t / 15), color='#025669', linestyle='', linewidth=1.0)


        ax.fill_between(t, np.sqrt(3 * t / 15), -np.sqrt(3 * t / 15), color='#025669', alpha=0.2)
        # , label='Expected Standard Deviation = $\sqrt{3 MHz t}$')
        # ax[0, 1].plot(t, -np.sqrt(gamma * t), color='#85bb65', linestyle='--', linewidth=1.0)

    if gamma == 4:
        # ax[0, 1].plot(t, np.mean(phase_noise, axis=0), color='green', linestyle='', linewidth=1.0, marker="o", markersize="0.01")

        ax.plot(t, -np.sqrt(np.var(phase_noise, axis=0)), color='#800080', linestyle='-',
                      linewidth=1.0,
                      label='$\gamma = \Omega_R/3$ MHz')
        ax.plot(t, -np.sqrt(gamma * t / 15), color='#800080', linestyle='--', linewidth=1.0)
        # , label='Expected Standard Deviation = $\sqrt{3 MHz t}$')
        # ax[0, 1].plot(t, -np.sqrt(gamma * t), color='#85bb65', linestyle='--', linewidth=1.0)

    if gamma == 14:
        # ax[0, 1].plot(t, np.mean(phase_noise, axis=0), color='orange', linestyle='', linewidth=1.0, marker="o", markersize="0.01")

        ax.plot(t, -np.sqrt(np.var(phase_noise, axis=0)), color='#CC7722', linestyle='-',
                      linewidth=1.0,
                      label='$\gamma = 10$ MHz')
        ax.plot(t, -np.sqrt(gamma * t / 15), color='#CC7722', linestyle='--', linewidth=1.0)
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

#ax[0, 1].errorbar(times, data, label="Drift with reversal",
#                  linewidth="0.4",
#                  color='#85bb65')

# ax[0, 1].set_ylim([-1.4 * np.sqrt(30 * T), 1.4 * np.sqrt(30 * T)])
###ax[0, 1].set_xlim([0, 0.1])
ax.set_xlabel(r'Time [$ 2 \pi /\Omega_R$]', fontsize=22)
ax.set_ylabel(r'$\Phi_B(t) [\pi]$', fontsize=22)
ax.tick_params(axis="both", labelsize=16)
# ax[0, 1].set_xlabel('Time [a.u.]', fontsize=16)
# ax[0, 1].set_ylabel('Apmlitude [a.u.]', fontsize=16)
ax.legend(loc="lower left", fontsize=16, frameon=False)
ax.set_xlim([0, 3])
ax.set_ylim([-3, 3])

plt.savefig("phasewalks.pdf")
plt.show()





data = np.loadtxt('10MHz_gamma.txt')
# print(len(data))
#data_reversed = -data[::-1]

# there
data = np.cumsum(data)/180

data = data[40:56]*30

data[0]=0

datalarge = []

for d in range(0, len(data)):
    i=0
    while i < 100:
        datalarge.append(data[d])
        i += 1
    datalarge.append(data[d])



fig, ax = plt.subplots(2, 2, figsize=(11.69, 8.27))

ax[1, 0].plot(np.linspace(0, 4, len(datalarge)), -0.53*np.cos(datalarge+2*np.pi*np.linspace(0, 4, len(datalarge))), color='#85bb65', linewidth="2")


ax[1, 0].set_xlabel(r'Time [$ 2 \pi /\omega_{MW}$]', fontsize=12)
ax[1, 0].set_ylabel(r'$E(t) [V/m]$', fontsize=12)
ax[1, 0].tick_params(axis="both", labelsize=12)

ax[1, 0].set_yticks(ticks=np.array([-0.5, 0., 0.5]))
# ax[0, 1].set_xlabel('Time [a.u.]', fontsize=16)
# ax[0, 1].set_ylabel('Apmlitude [a.u.]', fontsize=16)
#ax[1, 0].legend(loc="lower left", fontsize=16)


ax[0, 1].set_yticks(ticks=np.array([-3, -2, -1,  0., 1, 2, 3]))

ax[0, 1].set_xlim([0, 3])
ax[0, 1].set_ylim([-3, 3])
#plt.savefig("SingleWalk1.pdf")
#plt.savefig("test.png")
#plt.show()
################### END OF PHASE WALKS ############################################ 3333333333333333333333


data = np.loadtxt('10MHz_gamma.txt')
timesteps = 2 * len(data)
endtime = 0.4
pertubation_length = endtime / 1
# t1 = np.linspace(0, endtime, timesteps)
# t2 = np.linspace(0, endtime, timesteps)
perturb_times = np.linspace(0, pertubation_length, timesteps)
fs = timesteps / endtime

bath = '10MHz_gamma.txt'

omega = 800

S = noisy_func(gamma, perturb_times, omega, bath)

data = np.cumsum(data)/180

plt.errorbar(np.linspace(0, 3, len(data)), data, label="Phase drift",
                  linewidth="0.4",
                  color='#85bb65')
plt.ylabel(r'$\Phi_B(t) [\pi]$', fontsize=12)
plt.xlabel(r'Time $[2 \pi /\Omega_R]$', fontsize=12)
plt.tick_params(axis="both", labelsize=12)

plt.yticks(ticks=np.array([-0.5, 0., 0.5, 1, 1.5]))
plt.savefig("walks.pdf")
plt.show()


#plt.plot(perturb_times, S, color='#CC7722', linestyle='-', linewidth=1.0)
#plt.show()






