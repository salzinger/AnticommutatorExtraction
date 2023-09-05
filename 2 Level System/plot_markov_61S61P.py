import numpy as np

from Atoms import *
from Driving import *
import matplotlib.pyplot as plt

plt.rcParams.update({
  "text.usetex": 0,
    "font.family":"sans-serif",
})


ax = plt.subplot(212)



N = 1

#omega = 2 * np.pi * 21 * 10 ** 3  # MHz
#Omega_R = 2 * np.pi * 25.7 * 10 ** 0  # MHz

#omega = 2 * np.pi * 21 * 10 ** (-20)  # MHz

omega = 0  # MHz
Omega_R = 2 * np.pi * 1 * 10 ** 0  # MHz



#Omega_R = 2 * np.pi * 12.3  # MHz

#gamma = 2 * np.pi * 15.0  # MHz

J = 0 * 10 ** 0  # MHz
#bath = 'Forward3MHzcsv.txt'
#data = np.loadtxt('Forward3MHzcsv.txt')
#timesteps = 2 * len(data)
#endtime = 0.2
#pertubation_length = endtime / 1

#perturb_times = np.linspace(0, pertubation_length, timesteps)

Exps = [MagnetizationX(N), MagnetizationZ(N), MagnetizationY(N), sigmaz(0, N), sigmaz(N - 1, N), upup(0, N),
        sigmap(0, N), sigmam(0, N), downdown(0, N)]

opts = Options(store_states=True, store_final_state=True)  # , nsteps=50000)

#vars = np.array([np.zeros(6400), np.zeros(6400), np.zeros(6400)])
#means = np.array([np.zeros(6400), np.zeros(6400), np.zeros(6400)])
#n=0
timesteps = 100
endtime = 3
pertubation_length = endtime / 1
perturb_times = np.linspace(0, endtime, timesteps)
init_state = productstateZ(0, 0, N)

'''
for gamma in [3, 10, 30]:


            bath = "markovian"

            init_state = productstateZ(0, 0, N)

            data = np.loadtxt('Forward3MHzcsv.txt')
            timesteps = 100
            endtime = 0.2
            pertubation_length = endtime / 1
            perturb_times = np.linspace(0, endtime, len(data))

            print(len(perturb_times))

            S1 = Cubic_Spline(perturb_times[0], perturb_times[-1],
                              noisy_func(gamma, perturb_times, omega, bath))
            S2 = Cubic_Spline(perturb_times[0], perturb_times[-1],
                              np.conj(noisy_func(gamma, perturb_times, omega, bath)))

            #print('H0...')
            #print(H0(omega, J, N))
            #print('H1...')
            #print(H1(Omega_R, N))
            #print('H2...')
            #print(H2(Omega_R, N))

            result2 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S1], [H2(Omega_R, N), S2]], init_state,
                              perturb_times, e_ops=Exps, options=opts)
            concmean = []
            # for t in range(0, timesteps):
            # concmean.append(concurrence(result2.states[t]))

            # opts = Options(store_states=True, store_final_state=True, rhs_reuse=True)
            states2 = np.array(result2.states[timesteps - 1])
            expect2 = np.array(result2.expect[:])
            expect1 = np.array([result2.expect[:]])
            #print(expect2)
            ancilla_overlap = []
            #Smean = np.zeros_like(perturb_times) + 1j * np.zeros_like(perturb_times)
            #Pmean = 0
            i = 1
            while i < int(1):  # averages + int(2 * gamma):
                print(i)
                i += 1

                S1 = Cubic_Spline(perturb_times[0], perturb_times[-1],
                                  noisy_func(gamma, perturb_times, omega, bath))
                S2 = Cubic_Spline(perturb_times[0], perturb_times[-1],
                                  np.conj(noisy_func(gamma, perturb_times, omega, bath)))
                # S = Cubic_Spline(perturb_times[0], perturb_times[-1],
                # data / 0.4)

                result2 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S1], [H2(Omega_R, N), S2]], init_state,
                                  perturb_times, e_ops=Exps, options=opts)

                #qsave(result2, "i =  %.2f, gamma =  %.2f" % (i,gamma))

                states2 += np.array(result2.states[timesteps - 1])
                expect2 += np.array(result2.expect[:])
                expect1 = np.append(expect1, [np.array(result2.expect[:])], axis=0)
                #print(expect2)
                #print(np.var(expect1, axis=0)[1])

                #Smean += np.abs(
                #    np.fft.fft(Omega_R * noisy_func(gamma, perturb_times, omega, bath)) ** 2)  # /2/np.pi/timesteps
                #Pmean += np.abs(np.sum(Omega_R * noisy_func(gamma, perturb_times, omega, bath) ** 2))  # /timesteps
                # for t in range(0, timesteps):
                #    concmean[t] += concurrence(result2.states[t])

            # noisy_data2 = noisy_func(gamma, perturb_times, omega, bath)
            # S2 = Cubic_Spline(perturb_times[0], perturb_times[-1], noisy_data2)

            #states2 = states2 / i
            expect2 = expect2 / i
            #print(expect2)
            #print(np.var(expect2))
            #Smean = Smean / i
            #Pmean = Pmean / i
            #concmean = np.array(concmean) / i
            vars[n] = np.var(expect1, axis=0)[1]
            means[n] = np.mean(expect1, axis=0)[1]

            n += 1
            #print(means)
            #print(vars)

            # print(Qobj(states2))
            # print((expect2[5]+expect2[8]).mean())
            #density_matrix = Qobj([[expect2[5][timesteps - 1], expect2[6][timesteps - 1]],
            #                       [expect2[7][timesteps - 1], expect2[8][timesteps - 1]]])
'''
################### MASTER EQUATION ############################################ 444444444444444444
#fig, ax = plt.subplots(2, 2, figsize=(10, 10))
#fig, ax = plt.subplots(1, 1, figsize=(11.69*2, 11.69))

with open('counts_0.txt') as f:
    linesm0 = f.readlines()
with open('counts_3.txt') as f:
    linesm3 = f.readlines()
with open('counts_5.txt') as f:
    linesm5 = f.readlines()
with open('counts_15.txt') as f:
    linesm10 = f.readlines()
with open('counts_30.txt') as f:
    linesm30 = f.readlines()

with open('counts_0_e.txt') as f:
    linesm0e = f.readlines()
with open('counts_3_e.txt') as f:
    linesm3e = f.readlines()
with open('counts_5_e.txt') as f:
    linesm5e = f.readlines()
with open('counts_15_e.txt') as f:
    linesm15e = f.readlines()
with open('counts_30_e.txt') as f:
    linesm30e = f.readlines()

# print(linesmf)
# print(linesm)
x0 = []
y0 = []
y0e = []
y3 = []
y3e = []
y5 = []
y5e = []
y15 = []
y15e = []
y30 = []
y30e = []

'''
3.953125
0.378901455848414
52.449438202247194
2.007774167644925
'''

de = 3.953125
de_e = 0.38

Ntot = 52.449438202247194
Ntot_e = 2


for element in range(1, 32):
    x0.append(float(linesm0[element][0:5])*15)

    y0.append((float(linesm0[element][8:18])-de)/(Ntot-de)-0.5)
    y3.append((float(linesm3[element][8:18])-de)/(Ntot-de)-0.5)
    y5.append((float(linesm5[element][8:18])-de)/(Ntot-de)-0.5)
    y15.append((float(linesm10[element][8:18])-de)/(Ntot-de)-0.5)
    y30.append((float(linesm30[element][8:18])-de)/(Ntot-de)-0.5)

    y0e.append(float(linesm0e[element][8:18])/(Ntot-de))
    y3e.append(float(linesm3e[element][8:18])/(Ntot-de))
    y5e.append(float(linesm5e[element][8:18])/(Ntot-de))
    y15e.append(float(linesm15e[element][8:18])/(Ntot-de))
    y30e.append(float(linesm30e[element][8:18])/(Ntot-de))

print(len(x0))
print(len(y0))
print(y3)

#t1 = np.linspace(0, endtime, int(timesteps))

#Omega_R = 2 * np.pi * 12.3 #MHz

S = Cubic_Spline(perturb_times[0], perturb_times[-1], func(perturb_times, omega))

result_m0 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]],
                    init_state,
                    perturb_times, [np.sqrt(0.25/15) * sigmaz(0, N), np.sqrt(0.25/15) * sigmaz(0, N)], Exps,
                    options=opts)

m0 = np.array(result_m0.expect[:])

result_m3 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]],
                    init_state,
                    perturb_times, [np.sqrt(3/15) * sigmaz(0, N), np.sqrt(3/15) * sigmaz(0, N)], Exps,
                    options=opts)

result_m5 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]],
                    init_state,
                    perturb_times, [np.sqrt(5/15) * sigmaz(0, N), np.sqrt(5/15) * sigmaz(0, N)], Exps,
                    options=opts)

result_m15 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]],
                     init_state,
                     perturb_times, [np.sqrt(15/15) * sigmaz(0, N), np.sqrt(15/15) * sigmaz(0, N)], Exps,
                     options=opts)

result_m30 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]],
                     init_state,
                     perturb_times, [sqrt(30/15) * sigmaz(0, N), sqrt(30/15) * sigmaz(0, N)], Exps,
                     options=opts)



ax.errorbar(x0, y0, y0e, marker="o", color='black', label='$\gamma = \Omega_R/60$', linestyle='',markersize="2")

ax.plot(perturb_times, np.real(m0[1]), color='black', linestyle='-')

#ax[1, 1].plot(perturb_times, np.ones_like(perturb_times)*0.5, color='grey', linestyle='--')
#ax[1, 1].plot(perturb_times, -np.ones_like(perturb_times)*0.5, color='grey', linestyle='--')
#ax[1, 1].plot(perturb_times, -np.ones_like(perturb_times)*0.0, color='grey', linestyle='--')

#ax[1, 1].plot(perturb_times, np.real(expect2[1]), color='#008b8b', label="Time Dependant")
#ax[1, 1].plot(perturb_times, means[0], color='g',
#             label="fbm 3", marker="s", markersize="0.2", linestyle="")
#ax[1, 1].plot(perturb_times, means[1], color='yellow',
#             label="fbm 10", marker="s", markersize="0.2", linestyle="")
#ax[1, 1].plot(perturb_times, means[2], color='r',
              #label="fbm 30", marker="s", markersize="0.2", linestyle="")
#ax[1, 1].plot(perturb_times, np.mean(expect1, axis=0)[1] + np.var(expect1, axis=0)[1], color='b',
#              label=" +std", marker="s", markersize="0.1", linestyle="")
#ax[1, 1].plot(perturb_times, np.mean(expect1, axis=0)[1] - np.var(expect1, axis=0)[1], color='r',
#              label="-std ", marker="s", markersize="0.1", linestyle="")

ax.errorbar(x0, y3, y3e, marker="^", color='#025669', label='$\gamma = \Omega_R/5$', linestyle='', markersize="2")
ax.plot(perturb_times, np.real(result_m3.expect[1]), color='#025669', linestyle='-')

ax.errorbar(x0, y5, y5e, marker="D", color='#800080', label='$\gamma = \Omega_R/3$', linestyle='', markersize="2")
ax.plot(perturb_times, np.real(result_m5.expect[1]), color='#800080', linestyle='-')
#ax[1, 1].fill_between(perturb_times,  np.real(result_m3.expect[1])+vars[0],
           #           np.real(result_m3.expect[1])-vars[0], alpha=0.2, color='b')

ax.errorbar(x0, y15, y15e, marker="v", color='#CC7722', label='$\gamma = \Omega_R$', linestyle='', markersize="2")
ax.plot(perturb_times, np.real(result_m15.expect[1]), color='#CC7722', linestyle='-')
#ax[1, 1].fill_between(perturb_times,  np.real(result_m10.expect[1])+vars[1],
#                      np.real(result_m10.expect[1])-vars[1], alpha=0.2, color='#CC7722')

ax.errorbar(x0, y30, y30e, marker="s", color='#800020', label='$\gamma = 2\Omega_R$', linestyle='', markersize="2")
ax.plot(perturb_times, np.real(result_m30.expect[1]), color='#800020', linestyle='-')
#ax[1, 1].fill_between(perturb_times,  np.real(result_m30.expect[1])+vars[2],
#                      np.real(result_m30.expect[1])-vars[2], alpha=0.2, color="#800020")

#ax[1, 1].set_ylim([-0.596, 0.596])
ax.set_xlabel('Time [$2 \pi /\Omega_R$]', fontsize=14)
ax.set_ylabel(r'Spin $\langle \hat{s}_z\rangle$', fontsize=14)
ax.legend(loc="lower center", fontsize=8, frameon=False)
ax.tick_params(axis="both", labelsize=8)
ax.set_ylim([-0.55, 0.55])

ax.set_yticks(ticks=np.array([-0.5, -0.25, 0., 0.25, 0.5]))
ax.set_xlim([0., 3])



#fig.tight_layout()
#plt.savefig("markov.pdf")
#plt.show()
#plt.savefig("gamma =  %.2f.png" % (gamma))  # and BW %.2f.pdf" % (noise_amplitude, bandwidth))






ax1=plt.subplot(222)

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




#fig, ax = plt.subplots(1, 1, figsize=(11.69, 11.69))
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
            ax1.plot(times, phase_noise_plot[k], color='#85bb65', linewidth=0.1)
            # ax[0, 1].plot(t, phase_noise, color='#85bb65', linewidth=0.1)
        elif gamma == 14:
            ax1.plot(times, phase_noise_plot[k], color='#CC7722', linewidth=0.1)
            # ax[0, 1].plot(t, phase_noise, color='#CC7722', linewidth=0.1)
        elif gamma == 31:
            ax1.plot(times, phase_noise_plot[k], color='#800020', linewidth=0.1, label="Single trajectory $\gamma=2\Omega_R$")
            # ax[0, 1].plot(t, phase_noise, color='#800020', linewidth=0.1)

    if gamma == 30:
        # ax[0, 1].plot(t, np.mean(phase_noise, axis=0), color='red', linestyle='', linewidth=1.0, marker="o", markersize="0.01")
        ax1.plot(t, np.sqrt(np.var(phase_noise, axis=0)), color='#800020', linestyle='',
                      linewidth=1.0,
                      label='$\gamma = 2\Omega_R \pi$', marker="s", markersize="3", markerfacecolor='none', markeredgecolor='#800020')
        ax1.plot(t, np.sqrt(gamma * t / 15), color='#800020', linestyle='', linewidth=1.0)

        #ax[0, 1].plot(t, -np.sqrt(np.var(phase_noise, axis=0)), color='#800020', linestyle='-',
        #              linewidth=1.0,
        #              label='$\gamma = \Omega_R/3$ MHz')

        ax1.plot(t, -np.sqrt(gamma * t / 15), color='#800020', linestyle='', linewidth=5.0)
        ax1.fill_between(t, np.sqrt(gamma * t / 15), np.sqrt(3 * t / 15), color='#800020', alpha=0.2)
        ax1.fill_between(t, -np.sqrt(gamma * t / 15), -np.sqrt(3 * t / 15), color='#800020', alpha=0.2)
        #ax1.plot(times, phase_noise_plot[k], color='#800020', linewidth=0.1, label="Single trajectory $\gamma=2\Omega_R$")
        np.save("singlenoise8.npy", phase_noise_plot[k])
        noiseplot=np.load("singlenoise8.npy")
        ax1.plot(times, noiseplot, color='#800020', linewidth=0.1,
                 label="Single trajectory $\gamma=2\Omega_R$")
        # ,label='$\sqrt{30 MHz  t}$')
        # ax[0, 1].plot(t, -np.sqrt(gamma * t), color='#800020', linestyle='--', linewidth=1.0)

    if gamma == 3:
        # ax[0, 1].plot(t, np.mean(phase_noise, axis=0), color='green', linestyle='', linewidth=1.0, marker="o", markersize="0.01")

        ax1.plot(t, np.sqrt(np.var(phase_noise, axis=0)), color='#025669', linestyle='',
                      linewidth=1.0,
                      label='$\gamma = \Omega_R/5$' ,marker="^", markersize="3", markerfacecolor='none', markeredgecolor = '#025669')
        ax1.plot(t, np.sqrt(gamma * t / 15), color='#025669', linestyle='', linewidth=1.0)

        #ax[0, 1].plot(t, -np.sqrt(np.var(phase_noise, axis=0)), color='#025669', linestyle='-',
        #              linewidth=1.0,
        #              label='$\gamma = 10$ MHz')
        ax1.plot(t, -np.sqrt(gamma * t / 15), color='#025669', linestyle='', linewidth=1.0)


        ax1.fill_between(t, np.sqrt(3 * t / 15), -np.sqrt(3 * t / 15), color='#025669', alpha=0.2)
        # , label='Expected Standard Deviation = $\sqrt{3 MHz t}$')
        # ax[0, 1].plot(t, -np.sqrt(gamma * t), color='#85bb65', linestyle='--', linewidth=1.0)

    if gamma == 4:
        # ax[0, 1].plot(t, np.mean(phase_noise, axis=0), color='green', linestyle='', linewidth=1.0, marker="o", markersize="0.01")

        ax1.plot(t, -np.sqrt(np.var(phase_noise, axis=0)), color='#800080', linestyle='-',
                      linewidth=1.0,
                      label='$\gamma = \Omega_R/3$ MHz')
        ax1.plot(t, -np.sqrt(gamma * t / 15), color='#800080', linestyle='--', linewidth=1.0)
        # , label='Expected Standard Deviation = $\sqrt{3 MHz t}$')
        # ax[0, 1].plot(t, -np.sqrt(gamma * t), color='#85bb65', linestyle='--', linewidth=1.0)

    if gamma == 14:
        # ax[0, 1].plot(t, np.mean(phase_noise, axis=0), color='orange', linestyle='', linewidth=1.0, marker="o", markersize="0.01")

        ax1.plot(t, -np.sqrt(np.var(phase_noise, axis=0)), color='#CC7722', linestyle='-',
                      linewidth=1.0,
                      label='$\gamma = 10$ MHz')
        ax1.plot(t, -np.sqrt(gamma * t / 15), color='#CC7722', linestyle='--', linewidth=1.0)
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
ax1.set_xlabel(r'Time [$ 2 \pi /\Omega_R$]', fontsize=14)
ax1.set_ylabel(r'$\Phi_B(t)$', fontsize=14)
ax1.tick_params(axis="both", labelsize=8)
# ax[0, 1].set_xlabel('Time [a.u.]', fontsize=16)
# ax[0, 1].set_ylabel('Apmlitude [a.u.]', fontsize=16)
ax1.legend(loc="lower left", fontsize=8, frameon=False)
ax1.set_xlim([0, 3])
ax1.set_ylim([-3, 3])




ax2=plt.subplot(221)

factor = 3

# samples = 64 * 10 ** (factor+2) #min 8 * 10 ** 6 or 2*10**7 for good results
omega = 2 * np.pi * 20 * 10 ** factor  # MHz
sample_time = np.pi / 2 * 1  # * 10 **(1-factor)
samples = int(sample_time * 64 * 10 ** (factor + 2))



gamma3 = np.load('gamma3.npy')

gamma5 = np.load('gamma5.npy')

gamma15 = np.load('gamma15.npy')

gamma30 = np.load('gamma30.npy')



lorentzsamples=np.linspace(19990, 20010, 1000)

ax2.plot(-gamma3[0],2*gamma3[1], linestyle='',
                          marker='^', markersize='3', label=r"$\gamma=\Omega_R/5$", color='#025669', markerfacecolor='none', markeredgecolor = '#025669')
ax2.plot(lorentzsamples, lorentzian(lorentzsamples, 1, omega / (2 * np.pi), 3)/sample_time, linestyle='-',
              linewidth=1,
              color='#025669')

ax2.plot(-gamma5[0], 2*gamma5[1], linestyle='',
              marker='D', markersize='3', label=r"$\gamma=\Omega_R/3$", markerfacecolor='none', markeredgecolor = '#800080')
ax2.plot(lorentzsamples, lorentzian(lorentzsamples, 1, omega / (2 * np.pi), 5)/sample_time, linestyle='-',
              linewidth=1,
              color='#800080')

ax2.plot(-gamma15[0], 2*gamma15[1], linestyle='',
              marker='v', markersize='3', label=r"$\gamma=\Omega_R$", markerfacecolor='none', markeredgecolor = "#CC7722")
ax2.plot(lorentzsamples, lorentzian(lorentzsamples, 1, omega / (2 * np.pi), 15)/sample_time, linestyle='-',
               linewidth=1,
              color="#CC7722")

ax2.plot(-gamma30[0], 2*gamma30[1], linestyle='',
              marker='s', markersize='3', label=r"$\gamma=2\Omega_R$", markerfacecolor='none', markeredgecolor = "#800020")
ax2.plot(lorentzsamples, lorentzian(lorentzsamples, 1, omega / (2 * np.pi), 30)/sample_time, linestyle='-',
              linewidth=1,
              color="#800020")


# ax[0, 0].plot(f1, Pxx_den1, linestyle='',
#              marker='o', markersize='4', linewidth=0.55, label="psd_no_noise", color="#008b8b")
# ax[0, 0].plot(f, np.ones_like(f) * np.max(Pxx_den) / 2, linestyle='-',
#              marker='o', markersize='0', linewidth=0.55, label="half_psd", color="b")

# ax[0, 0].plot(f, np.ones_like(f) * np.max(lorentzian(f, 0.5, omega / (2 * np.pi), 3)) / 2,
#              linestyle='-',
#              marker='o', markersize='0', linewidth=0.55, label="half_lorentz", color="orange")
# ax[0, 0].plot(rydfft[0]*10**6, np.abs(rydfft[1])**2,
#              linestyle='-',
#              marker='o', markersize='0', linewidth=0.55, label="rydfft", color="black")

# ax[0, 0].axvspan(20998.5, 21001.5, facecolor='g', alpha=0.5)

ax2.set_xlim([int(omega/2/np.pi)-10*10**(factor-3), int(omega/2/np.pi)+10*10**(factor-3)])
ax2.set_ylim(bottom=0)
# ax[0, 0].set_xlim([-21020, -20980])
# [0:int(len(perturb_times) / 2)]
# ax[0, 0].plot(freq[int(len(perturb_times)/2)+2000: int(len(perturb_times))-4000], lorentzian(freq, Pmean, omega/(2*np.pi),
#                                                              gamma)[int(len(perturb_times)/2)+2000: int(len(perturb_times))-4000], linestyle='',
#                                                                marker='o', markersize='2', linewidth=0.0)
# ax[0, 0].plot(freq[int(len(perturb_times)/2):int(len(perturb_times))], lorentzian(freq, Pmean, omega/(2*np.pi),
#                                                              gamma)[int(len(perturb_times)/2):int(len(perturb_times))],
#              linestyle='-',
#              marker='o', markersize='0', linewidth=1.0,
#              label="Lorentzian with FWHM gamma= %.2f MHz" % gamma)
# print("Pmean=", Pmean)
# print("sum fourier", np.sum(fourier[0:int(len(perturb_times))]))
# print(np.sum(lorentzian(freq, Pmean, omega / (2 * np.pi),
#                        gamma)[0:int(len(perturb_times))]))

# ax[0, 1].plot(perturb_times, np.real(2 * noisy_func(gamma, perturb_times, omega, bath)), linestyle='--',
#              marker='o', markersize='3',
#              linewidth=1.0)
# ax[0, 1].set_xlabel('Time [us]', fontsize=16)
# ax[0, 1].set_ylabel('Coupling Amplitude', fontsize=16)
# ax[0, 1].set_xlim([0.099, 0.101])


ax2.tick_params(axis="both", labelsize=8)

ax2.set_xlabel(r'$\Delta$ [$\Omega_R$]', fontsize=14)
ax2.set_ylabel(r'PSD / P$_{\mathrm{Carrier}}$ [1/Hz]', fontsize=14)

ax2.legend(loc="upper left", fontsize=8, frameon=0)


plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)


plt.savefig("newwalks.pdf")
plt.show()
