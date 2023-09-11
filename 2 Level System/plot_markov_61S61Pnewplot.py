import numpy as np

from Atoms import *
from Driving import *
import matplotlib.pyplot as plt

plt.rcParams.update({
  "text.usetex": 1,
    "font.family":"sans-serif",
})


#ax = plt.subplot(212)

ax = plt.subplot(222)

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

#ax.errorbar(x0, y3, y3e, marker="^", color='#025669', label='$\gamma = \Omega_R/5$', linestyle='', markersize="2")
#ax.plot(perturb_times, np.real(result_m3.expect[1]), color='#025669', linestyle='-')

ax.errorbar(x0, y5, y5e, marker="D", color='#025669', label='$\gamma = \Omega_R/3$', linestyle='', markersize="2")
ax.plot(perturb_times, np.real(result_m5.expect[1]), color='#025669', linestyle='-')
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
ax.set_xlim([0., 2])



#fig.tight_layout()
#plt.savefig("markov.pdf")
#plt.show()
#plt.savefig("gamma =  %.2f.png" % (gamma))  # and BW %.2f.pdf" % (noise_amplitude, bandwidth))






#ax1=plt.subplot(222)



ax1=plt.subplot(221)

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
        ax1.fill_between(t, np.sqrt(gamma * t / 15), np.sqrt(5 * t / 15), color='#800020', alpha=0.2)
        ax1.fill_between(t, -np.sqrt(gamma * t / 15), -np.sqrt(5 * t / 15), color='#800020', alpha=0.2)
        #ax1.plot(times, phase_noise_plot[k], color='#800020', linewidth=0.1, label="Single trajectory $\gamma=2\Omega_R$")
        np.save("singlenoise2.npy", phase_noise_plot[k])
        noiseplot=np.load("singlenoise1.npy")
        ax1.plot(times, noiseplot, color='#800020', linewidth=0.1,
                 label="Single trajectory $\gamma=2\Omega_R$")
        # ,label='$\sqrt{30 MHz  t}$')
        # ax[0, 1].plot(t, -np.sqrt(gamma * t), color='#800020', linestyle='--', linewidth=1.0)

    if gamma == 5:
        # ax[0, 1].plot(t, np.mean(phase_noise, axis=0), color='green', linestyle='', linewidth=1.0, marker="o", markersize="0.01")

        ax1.plot(t, np.sqrt(np.var(phase_noise, axis=0)), color='#025669', linestyle='',
                      linewidth=1.0,
                      label='$\gamma = \Omega_R/5$' ,marker="^", markersize="3", markerfacecolor='none', markeredgecolor = '#025669')
        ax1.plot(t, np.sqrt(gamma * t / 15), color='#025669', linestyle='', linewidth=1.0)

        #ax[0, 1].plot(t, -np.sqrt(np.var(phase_noise, axis=0)), color='#025669', linestyle='-',
        #              linewidth=1.0,
        #              label='$\gamma = 10$ MHz')
        ax1.plot(t, -np.sqrt(gamma * t / 15), color='#025669', linestyle='', linewidth=1.0)


        ax1.fill_between(t, np.sqrt(gamma * t / 15), -np.sqrt(gamma * t / 15), color='#025669', alpha=0.2)
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
ax1.set_xlim([0, 2])
ax1.set_ylim([-3, 3])






plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)


plt.savefig("newwalks.pdf")
plt.show()


import numpy as np

from Atoms import *
from Driving import *
import matplotlib.pyplot as plt

plt.rcParams.update({
  "text.usetex": 0,

})


#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('font',**{'family':'serif','serif':['Latin Modern Roman']})

plt.rc('figure', figsize=(11.69, 8.27))


N = 1

#omega = 2 * np.pi * 21 * 10 ** 3  # MHz
#omega = 2 * np.pi * 21 * 10 ** (-20)  # MHz

omega = 0  # MHz

#Omega_R = 2 * np.pi * 25.7 * 10 ** 0  # MHz

Omega_R = 2 * np.pi * 14.6 * 10 ** 0  # MHz

Omega_R = 2 * np.pi * 15 * 10 ** 0  # MHz

gamma = 2 * np.pi * 15.0  # MHz

J = 0 * 10 ** 0  # MHz
bath = '10MHz_gamma.txt'
data = np.loadtxt('10MHz_gamma.txt')
timesteps = 2 * len(data)
endtime = 6
pertubation_length = endtime / 1

perturb_times = np.linspace(0, pertubation_length, timesteps)

Exps = [MagnetizationX(N), MagnetizationZ(N), MagnetizationY(N), sigmaz(0, N), sigmaz(N - 1, N), upup(0, N),
        sigmap(0, N), sigmam(0, N), downdown(0, N)]

opts = Options(store_states=True, store_final_state=True)  # , nsteps=50000)
figure = plt.plot()
c = Bloch(figure)
#c.make_sphere()

Flist = []

for Omega_R in np.linspace(2*np.pi*1, 2*np.pi*1, 1, endpoint=1):
    #print("Omega_R: ", Omega_R)
    # print("sampling: ", sampling_rate)
    init_state = productstateZ(0, 0, N)
    print(Omega_R)
    # timesteps = int(endtime * sampling_rate)
    data = np.loadtxt('10MHz_gamma.txt')
    timesteps = 2 * len(data)
    endtime = 6
    pertubation_length = endtime / 1
    # t1 = np.linspace(0, endtime, timesteps)
    # t2 = np.linspace(0, endtime, timesteps)
    perturb_times = np.linspace(0, pertubation_length, timesteps)
    fs = timesteps / endtime
    # print(len(perturb_times))

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    for rise_time in np.linspace(700, 1000, 1, endpoint=1):

        rise_time = int(rise_time)

        print("rise_time= ", rise_time)


        for second_rise_time in np.linspace(900, 1500, 1, endpoint=1):

            g = Omega_R

            second_rise_time = int(second_rise_time)

            print("second_rise_time= ", second_rise_time)

            S1 = Cubic_Spline(perturb_times[0], perturb_times[-1],
                              noisy_func(gamma, perturb_times, omega, bath, rise_time, second_rise_time))
            S2 = Cubic_Spline(perturb_times[0], perturb_times[-1],
                              np.conj(noisy_func(gamma, perturb_times, omega, bath, rise_time, second_rise_time)))
            '''
            # S = Cubic_Spline(perturb_times[0], perturb_times[-1],
            # data / 0.4)
            #print(noisy_func(gamma, perturb_times, omega, bath))

            plt.plot(perturb_times, np.absolute(noisy_func(gamma, perturb_times, omega, bath))*2, label="Mag")
            plt.plot(perturb_times, np.angle(noisy_func(gamma, perturb_times, omega, bath)), label="Phase")
            plt.xlim(-0.001, 0.1)
            plt.ylim(-1, 2.25)
            plt.show()
            '''
            result_single = mesolve([H0(omega, J, N), [H1(Omega_R, N), S1], [H2(Omega_R, N), S2]], init_state,
                                    perturb_times, e_ops=Exps, options=opts)

            states = result_single.states

            expect_single = np.array(result_single.expect[:])
            qsave(result_single, "Omega_R =  %.2f" % Omega_R)
            # print("gamma: ", gamma)
            # print("Bandwidth", bandwidth)
            i = 1
            # random_phase = noise_amplitude * np.random.randn(perturb_times.shape[0])

            bath = '10MHz_gamma.txt'

            gamma = 0
            data = np.loadtxt('10MHz_gamma.txt')
            timesteps = 2 * len(data)
            endtime = 6
            pertubation_length = endtime / 1
            perturb_times = np.linspace(0, pertubation_length, timesteps)

            S1 = Cubic_Spline(perturb_times[0], perturb_times[-1],
                              noisy_func(gamma, perturb_times, omega, bath, rise_time, second_rise_time))
            S2 = Cubic_Spline(perturb_times[0], perturb_times[-1],
                              np.conj(noisy_func(gamma, perturb_times, omega, bath, rise_time, second_rise_time)))
            # S = Cubic_Spline(perturb_times[0], perturb_times[-1],
            #                 data[0:32000]/0.4)

            #print('H0...')
            #print(H0(omega, J, N))
            #print('H1...')
            #print(H1(Omega_R, N))
            #print('H2...')
            #print(H2(Omega_R, N))

            #result2 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S1], [H2(Omega_R, N), S2]], init_state,
            #                  perturb_times, e_ops=Exps, options=opts)
            concmean = []
            # for t in range(0, timesteps):
            # concmean.append(concurrence(result2.states[t]))

            # opts = Options(store_states=True, store_final_state=True, rhs_reuse=True)
            #states2 = np.array(result2.states[timesteps - 1])
            #expect2 = np.array(result2.expect[:])

            #################### SINGLE TRAJECTORY ######################################## 222222222222222222222222222


            # vec1 = [np.real(x1), np.real(y1), np.real(z1)]
            # vec2 = [np.real(x2), np.real(y2), np.real(z2)]
            # vec3 = [np.real(x3), np.real(y3), np.real(z3)]
            # th = np.linspace(0, 2*np.pi, 20)
            colorlist = []#np.zeros_like(data)
            data = np.cumsum(data)
            min = np.min(data)
            max = np.max(data)
            #print(max)
            for element in data:
                if element > -88.5:
                    if element > 1.14:
                        colorlist.append([1-1*element/max, 1-40/80*element/max, 1-10/80*element/max, 1])
                    elif element < -1.14:
                        colorlist.append([1-10/80*element/min, 1-40/80*element/min, 1-1*element/min, element/min/4])
                    else:
                        colorlist.append([1, 0, 0, 1])
                elif element < -91.5:
                    colorlist.append([1-10/80*element/min, 1-40/80*element/min, 1-1*element/min, 1])
                else:
                    colorlist.append([0, 0, 0, 1])

            #print(len(colorlist))
            #print(len(expect_single[0]))

            expectsx = []
            expectsy = []
            expectsz = []
            colors = []

            for n in range(0, len(data)):
                if np.mod(n, 10) == 0:
                    expectsx.append(np.real(expect_single[0][n] * 2))
                    expectsy.append(np.real(expect_single[2][n] * 2))
                    expectsz.append(np.real(expect_single[1][n] * 2))
                    colors.append(colorlist[n])

            xz = np.real(expect_single[0]*2)
            yz = np.real(expect_single[2]*2)
            zz = np.real(expect_single[1]*2)
            c.point_color = colors
            c.point_size = [5, 20, 100]
            c.add_points([expectsx, expectsy, expectsz], 'm')
            #plt.colorbar(c.fig)
            c.sphere_alpha = 0.05
            # c.add_vectors(vec1)
            # c.add_vectors(vec2)
            # c.add_vectors(vec3)


            #c.clear()

            data = np.loadtxt('10MHz_gamma.txt')
            timesteps = 2 * len(data)
            endtime = 6
            pertubation_length = endtime / 1
            perturb_times = np.linspace(0, pertubation_length, timesteps)

            with open('counts_z_full_norm.txt') as f:
                linescountsz = f.readlines()

            with open('phase_fits_full_norm.txt') as f:
                linesphase = f.readlines()


            tmw = []
            z = []
            zerror = []
            amp = []
            amperror = []
            phase = []
            phaseerror = []
            y = []
            yerror = []
            xerror = []
            x = []
            total = []
            offset = []

            Ntot = 29.46153846153846
            de = 1.8333333333333333#-1.8

            Ntot = 24
            Ntot_std = 4

            Ntot = 38.416666666666664
            Ntot_std = 6.48

            de = 1.5

            #de = 1.5
            de_std=0.5

            #Ntot = 26.4
            #Ntot_std = 4

            #de = 1.5
            #de_std=0.5

            norm = []

            for element in range(2, 53):

                norm.append(2*float(linesphase[element][50:59])-de)

            mean_norm = np.mean(norm)

            print("Norm = ", mean_norm, "std_mean = ", np.sqrt(np.var(norm))/len(norm))


            #mean_norm = (24-de + mean_norm)/2

            #mean_norm = 24 - de

            for element in range(0, 51):

                norm[element] = mean_norm

            #print(2*np.mean(norm)-de)

            #norm = 2*np.mean(norm)-de



            for element in range(2, 53):
                tmw.append(float(linescountsz[element][0:5])*15)

                z.append(float(linescountsz[element][6:15]))

                zerror.append(float(linescountsz[element][16:25]))

                amp.append(float(linesphase[element][15:29]))

                #phase.append(float(linesphase[element][29:41])*2*np.pi/360+np.pi)

                phase.append(float(linesphase[element][29:38]) * 2 * np.pi / 360 + np.pi)

                offset.append(float(linesphase[element][50:59]))

                total.append( np.sqrt(  ( float(linesphase[element][15:29]) )**2  +  ( float(linescountsz[element][6:15]))**2 )  )

            for element in range(56, 107):
                amperror.append(float(linesphase[element][9:17]))
                try:
                    phaseerror.append(float(linesphase[element][20:30])*2*np.pi/360)
                except:
                    print("ERROR NAN")
                    phaseerror.append(200 * 2 * np.pi / 360)

            '''

            norm = []

            for element in range(2, 53):

                norm.append(2*float(linesphase[element][50:59])-de)

            mean_norm = np.mean(norm)

            print("Norm = ", mean_norm, "std_mean = ", np.sqrt(np.var(norm))/len(norm))


            #mean_norm = (24-de + mean_norm)/2

            #mean_norm = 24 - de

            for element in range(0, 51):

                norm[element] = mean_norm

            #print(2*np.mean(norm)-de)

            #norm = 2*np.mean(norm)-de



            for element in range(2, 53):
                tmw.append(float(linescountsz[element][0:5])*15)

                z.append((float(linescountsz[element][6:15])-de)/norm[element-2] - 0.5)

                zerror.append(float(linescountsz[element][16:25])/norm[element-2])

                amp.append(float(linesphase[element][15:29])/norm[element-2])

                #phase.append(float(linesphase[element][29:41])*2*np.pi/360+np.pi)

                phase.append(float(linesphase[element][29:38]) * 2 * np.pi / 360 + np.pi)

                offset.append(float(linesphase[element][50:59]))

                total.append( np.sqrt(  ( float(linesphase[element][15:29]) / norm[element-2] )**2  +  ( (float(linescountsz[element][6:15])-de) / norm[element-2]  -  0.5)**2 )  )

            for element in range(56, 107):
                amperror.append(float(linesphase[element][9:17])/(norm[element-2-56]))
                try:
                    phaseerror.append(float(linesphase[element][20:30])*2*np.pi/360)
                except:
                    print("ERROR NAN")
                    phaseerror.append(200 * 2 * np.pi / 360)

######################################################################


            norm = []

            for element in range(2, 53):

                norm.append(2*float(linesphase[element][46:54])-de)

            mean_norm = np.mean(norm)

            print(mean_norm)

            for element in range(0, 51):

                norm[element] = mean_norm


            for element in range(2, 53):
                tmw.append(float(linescountsz[element][0:5])*15)

                z.append((float(linescountsz[element][6:15])-de)/norm[element-2] - 0.5)

                zerror.append(float(linescountsz[element][16:25])/norm[element-2])

                amp.append(float(linesphase[element][15:26])/norm[element-2])

                #phase.append(float(linesphase[element][29:41])*2*np.pi/360+np.pi)

                phase.append(float(linesphase[element][26:36]) * 2 * np.pi / 360 + np.pi)

                offset.append(float(linesphase[element][50:59]))

                total.append( np.sqrt(  ( float(linesphase[element][15:26]) / norm[element-2] )**2  +  ( (float(linescountsz[element][6:15])-de) / norm[element-2]  -  0.5)**2 )  )


            for element in range(56, 107):
                amperror.append(float(linesphase[element][9:17])/(norm[element-2-56]))
                try:
                    phaseerror.append(float(linesphase[element][20:30])*2*np.pi/360)
                except:
                    print("ERROR NAN")
                    phaseerror.append(200 * 2 * np.pi / 360)

            '''


            #print("phases: ", phase)

            #print("z: ", z)

            #print("amps : ", amp)

            print("total : ", np.mean(total))

            print("sqrt(var) /len (total) : ", sqrt(np.var(total))/len(total))



            #print(tmw)
            #print(z)
            #print(zerror)

            #print(amp)
            #print(total)

            #print(amperror)
            #print(phase)
            #print(phaseerror)

            data = np.loadtxt('10MHz_gamma.txt')
            #len(data)
            timesteps = 2 * len(data)
            endtime = 6
            pertubation_length = endtime / 1

            data_reversed = -data[::-1]

            data = np.cumsum(data)

            data_reversed = np.cumsum(data_reversed) + data[-1] - 180

            data = np.append(data / 180, data_reversed / 180)

            #print(len(data))

            #print(len(noisy_func(gamma, perturb_times, omega, bath)))

            ax[0, 0].errorbar(perturb_times, data, label="Phase drift",
                              linewidth="0.4",
                              color='#85bb65')

            #ax[0, 0].errorbar(np.linspace(0, perturb_times[-1], len(noisy_func(gamma, perturb_times, omega, bath, rise_time))), -np.angle(noisy_func(gamma, perturb_times, omega, bath, rise_time))/np.pi, label="Phase drift filtered",
            #                  linewidth="0.4",
            #                  color='black')

            ax[0, 0].errorbar(np.linspace(0, perturb_times[-1], len(data)), np.abs(noisy_func(gamma, perturb_times, omega, bath, rise_time, second_rise_time)), label="Amp of Phase drift filtered",
                              linewidth="0.4",
                              color='black')


            ax[1, 0].plot(perturb_times, np.real(expect_single[1]), color='#85bb65', linestyle="-")
            ax[1, 0].plot(perturb_times, np.sqrt(np.real(expect_single[0])**2+np.real(expect_single[2])**2), color='black', linestyle="-")
            ax[1, 0].errorbar(tmw, amp, amperror, color="black", label=r"$\sqrt{\langle \sigma_x \rangle^2 + \langle \sigma_y \rangle^2}/2$", markersize="4", marker="s",
                         linestyle="")
            ax[1, 0].errorbar(tmw, z, zerror, color='#85bb65', label=r"$\langle \sigma_z \rangle/2}$", markersize="5", marker="o",
                         linestyle="")
            ax[1, 0].set_ylim([-0.68, 0.68])
            ax[1, 0].legend(loc="lower center", fontsize=12)

            #print(len(tmw))
            #print(len(perturb_times) / 51)
            #discr=[]
            rho_measured = np.array([])
            rho_ideal = np.array([])

            F = []
            F1 = []
            F2 = []
            F3 = []
            F4 = []

            diz=[]
            dix=[]
            diy=[]

            for t in range(0, 51):
                #print(np.real(expect_single[1])[t*1004])
                #print(np.real(expect_single[1])[t*1004]-z[t])
                #discr.append((np.real(expect_single[1])[t*511])-z[t])

                np.append(rho_ideal, (qeye(2) + expect_single[1][t*511]*sigmaz(0, 1)*2 + expect_single[2][t*511]*sigmay(0, 1)*2 + expect_single[0][t*511]*sigmax(0, 1)*2).dag()/2)

                np.append(rho_measured, (qeye(2)*(2*total[t]) + z[t]*sigmaz(0, 1)*2 + amp[t] * np.cos(phase[t])*sigmay(0, 1)*2 + amp[t] * np.sin(phase[t])*sigmax(0, 1)*2)/2)

                F.append(np.sqrt(((qeye(2) + expect_single[1][t*511]*sigmaz(0, 1)*2 + expect_single[2][t*511]*sigmay(0, 1)*2 + expect_single[0][t*511]*sigmax(0, 1)*2).dag()/2

                                  * ((qeye(2)*(2*total[t]) + z[t]*sigmaz(0, 1)*2 + amp[t] * np.cos(phase[t])*sigmay(0, 1)*2 + amp[t] * np.sin(phase[t])*sigmax(0, 1)*2)/2).unit()).tr()))

                #F.append(np.sqrt( (  ((qeye(2) + expect_single[1][t*511]*sigmaz(0, 1)*2 + expect_single[2][t*511]*sigmay(0, 1)*2 + expect_single[0][t*511]*sigmax(0, 1)*2)/2).dag()
                #          * (qeye(2) + z[t]*sigmaz(0, 1)*2 + amp[t] * np.cos(phase[t])*sigmay(0, 1)*2 + amp[t] * np.sin(phase[t])*sigmax(0, 1)*2)/2   ).tr()))

                #F.append( (( (qeye(2) + expect_single[1][t*511]*sigmaz(0, 1)*2 + expect_single[2][t*511]*sigmay(0, 1)*2 + expect_single[0][t*511]*sigmax(0, 1)*2)/2).dag()
                #          * (qeye(2) + expect_single[1][t*511]*sigmaz(0, 1)*2 + expect_single[2][t*511]*sigmay(0, 1)*2 + expect_single[0][t*511]*sigmax(0, 1)*2)/2).tr())

                #print((qeye(2) + expect_single[1][t*511]*sigmaz(0, 1) + expect_single[2][t*511]*sigmay(0, 1) + expect_single[0][t*511]*sigmax(0, 1))/2)

                F1.append( np.sqrt((   qutip.to_choi(((qeye(2) + expect_single[1][t*511]*sigmaz(0, 1)*2 + expect_single[2][t*511]*sigmay(0, 1)*2 + expect_single[0][t*511]*sigmax(0, 1)*2))/2).dag()
                          * qutip.to_choi(  ((qeye(2)*(2*total[t]) + z[t]*sigmaz(0, 1)*2 + amp[t] * np.cos(phase[t])*sigmay(0, 1)*2 + amp[t] * np.sin(phase[t])*sigmax(0, 1)*2)/2 ).unit())   ).tr()))

                #F1.append( ( qutip.to_choi((qeye(2) + expect_single[1][t*511]*sigmaz(0, 1)*2 + expect_single[2][t*511]*sigmay(0, 1)*2 + expect_single[0][t*511]*sigmax(0, 1)*2)/2).dag()
                #          * qutip.to_choi((qeye(2) + expect_single[1][t*511]*sigmaz(0, 1)*2 + expect_single[2][t*511]*sigmay(0, 1)*2 + expect_single[0][t*511]*sigmax(0, 1)*2)/2)).tr())

                #print(states[t*511].dag()*states[t*511])

                F2.append((states[t*511].dag() * ((qeye(2)*(2*total[t]) + z[t]*sigmaz(0, 1)*2 + amp[t] * np.cos(phase[t])*sigmay(0, 1)*2 + amp[t] * np.sin(phase[t])*sigmax(0, 1)*2)/2).unit() * states[t*511] ).sqrtm())

                #F2.append( (states[t*511].dag() * (qeye(2) + expect_single[1][t*511]*sigmaz(0, 1)*2 + expect_single[2][t*511]*sigmay(0, 1)*2 + expect_single[0][t*511]*sigmax(0, 1)*2)/2)
                #          * states[t*511])

                F3.append( fidelity(   (qeye(2) + expect_single[1][t*511]*sigmaz(0, 1)*2 + expect_single[2][t*511]*sigmay(0, 1)*2 + expect_single[0][t*511]*sigmax(0, 1)*2)/2,
                                    ((qeye(2)*(2*total[t]) + z[t]*sigmaz(0, 1)*2 + amp[t] * np.cos(phase[t])*sigmay(0, 1)*2 + amp[t] * np.sin(phase[t])*sigmax(0, 1)*2)/2).unit()    ))

                F4.append( process_fidelity((qeye(2) + expect_single[1][t*511]*sigmaz(0, 1)*2 + expect_single[2][t*511]*sigmay(0, 1)*2 + expect_single[0][t*511]*sigmax(0, 1)*2)/2,
                                            ((qeye(2)*(2*total[t]) + z[t]*sigmaz(0, 1)*2 + amp[t] * np.cos(phase[t])*sigmay(0, 1)*2 + amp[t] * np.sin(phase[t])*sigmax(0, 1)*2)/2).unit()    ))



                diz.append( -(np.real(expect_single[1])[t*511] - z[t]) / zerror[t] )

                dix.append( ((np.real(expect_single[2])[t*511]) - amp[t] * np.cos(phase[t])) / (np.sqrt((amperror[t]*np.cos(phase[t]))**2 + (amp[t]*np.sin(phase[t])*phaseerror[t])**2)) )

                diy.append( ((np.real(expect_single[0])[t*511]) - amp[t] * np.sin(phase[t])) / (np.sqrt((amperror[t]*np.sin(phase[t]))**2 + (amp[t]*np.cos(phase[t])*phaseerror[t])**2)) )

            #print(np.mean(F))
            #print(np.std(F))
            #print(np.mean(F1))
            #print(np.std(F1))
            #print(np.mean(F2))
            #print(np.std(F2))
            #print(np.mean(F3))
            #print(np.std(F3))

            Fmean = np.mean(F)
            Fmin = np.min(F)
            Fend = F[-1]

            print(Fmean)
            print(Fmin)

            Flist.append([np.round((Fmean-.9887002869959799)*100, decimals=3), np.round((Fmin-.9591676085073163)*100, decimals=3), np.round((Fend-.9967099830257092)*100 , decimals=3), rise_time, second_rise_time, Omega_R/(2*np.pi)])


            #print(np.std(F4))
            #print(np.sum(diz))
            #print(np.sum(dix))
            #print(np.sum(diy))

            #ax[1, 0].errorbar(tmw, diz, label=r"$z [\sigma]$", linestyle="--", markersize="4", marker="o",
             #             color='grey')


            #ax[1, 0].errorbar(tmw, F, label=r"$Fidelity $", linestyle="--", markersize="4", marker="o",
            #              color='black')

            ax[0, 1].errorbar(tmw, F, label=r"$F =\sqrt{ Tr[\rho^\dagger_{ideal} \rho_{measured}]}$", linestyle="--", markersize="4", marker="o",
                          color='black')

           # ax[0, 1].errorbar(tmw, F2, label=r"$F =\sqrt{ \langle \Psi \vert \rho_{measured} \vert \Psi \rangle}$", linestyle="", markersize="3", marker="o",
           #              color='blue')

            #ax[1, 0].errorbar(tmw, F3, label="Fidelity Qutip", linestyle="", markersize="3", marker="^",
            #             color='grey')



            ax[0, 1].errorbar(tmw, F1, label=r"$F_{process} = Tr[\chi^\dagger_{ideal}*\chi_{measured}]$", linestyle="--", markersize="4", marker="o",
                          color='g')

            #ax[1, 0].errorbar(tmw, F4, label="Process Fidelity Qutip", linestyle="", markersize="3", marker="s",
            #             color='orange')

            #ax[1, 0].errorbar(tmw, 1 - (np.abs( np.array(total)*2-1 ))/4 , #np.sqrt( np.array(amperror)**2 + np.array(zerror)**2),
            #                    color="grey", label=r"$1-\sqrt{\langle \sigma_x \rangle^2 + \langle \sigma_y \rangle^2 + \langle \sigma_z \rangle^2}$", markersize="4", marker="s", linestyle="")


            #ax[1, 0].errorbar(tmw, dix, label=r"$y [\sigma]$", linestyle="--", markersize="4", marker="o",
            #              color='#85bb65')

            #ax[1, 0].errorbar(tmw, diy, label=r"$x [\sigma]$", linestyle="--", markersize="4", marker="o",
            #              color='black')

            #ax[1, 0].errorbar(tmw, z, zerror, label=r"$\langle \sigma_z \rangle$", linestyle="", markersize="4", marker="o",
            #              color='#85bb65')
            #ax[1, 0].plot(perturb_times, np.real(expect_single[1]), color='#85bb65', linestyle="-")

            #ax[1, 0].errorbar(tmw, amp, amperror, label=r"$\sqrt{\langle \sigma_x \rangle^2 + \langle \sigma_y \rangle^2}/2$",
            #              linestyle="",
            #              markersize="3", marker="s", color='black')

            #ax[1, 0].plot(perturb_times, np.real(np.sqrt(expect_single[2] ** 2 + expect_single[0] ** 2)), color="black",
            #              linestyle="--")

            #ax[0, 1].set_xlabel(r'Time [$\mu$s]', fontsize=14)
            ax[0, 1].set_ylabel(r'', fontsize=14)
            #ax[1, 0].set_ylim([-0.599, 0.599])
            # ax[1, 0].plot(perturb_times, np.real(expect_me[1]), label="sigma_z, ME with sqrt(gamma)*L")
            #ax[0, 1].legend(loc="lower left", fontsize=12)

            ax[1, 0].plot(perturb_times, np.ones_like(perturb_times) * 0.5, color='grey', linestyle='--')
            ax[1, 0].plot(perturb_times, -np.ones_like(perturb_times) * 0.5, color='grey', linestyle='--')
            ax[1, 0].plot(perturb_times, -np.ones_like(perturb_times) * 0.0, color='grey', linestyle='--')


            ax[0, 1].plot(perturb_times, np.ones_like(perturb_times) * 1, color='grey', linestyle='--')
            #ax[1, 0].plot(perturb_times, -np.ones_like(perturb_times) * 1, color='grey', linestyle='--')
            #ax[1, 0].plot(perturb_times, -np.ones_like(perturb_times) * 0.0, color='grey', linestyle='--')

            ax[1, 1].plot(perturb_times, np.ones_like(perturb_times) * 0.5, color='grey', linestyle='--')
            ax[1, 1].plot(perturb_times, -np.ones_like(perturb_times) * 0.5, color='grey', linestyle='--')
            ax[1, 1].plot(perturb_times, -np.ones_like(perturb_times) * 0.0, color='grey', linestyle='--')





            #ax[1, 1].errorbar(tmw, (1-np.array(F1))*10, label=r"$(1-F_{process})*10$", linestyle="--", markersize="4", marker="o",
            #              color='g')
            ax[1, 1].plot(perturb_times, -np.real(expect_single[0]), color='blue', linestyle="-")
            ax[1, 1].plot(perturb_times, -np.real(expect_single[2]), color='purple', linestyle="-")
            #ax[1, 1].plot(perturb_times, np.sqrt(np.real(expect_single[1])**2+np.real(expect_single[0])**2+np.real(expect_single[2])**2), color='grey', linestyle="--", label="")
            #ax[1, 1].errorbar(tmw, total, color="grey", markersize="4", marker="o", label=r"$\sqrt{\langle \sigma_z \rangle^2 + \langle \sigma_x \rangle^2 + \langle \sigma_y \rangle^2}$",
            #             linestyle="")
            ax[1, 1].errorbar(tmw, total, np.sqrt( np.array(amperror)**2 + np.array(zerror)**2),
                                color="grey", label=r"$\sqrt{\langle \sigma_x \rangle^2 + \langle \sigma_y \rangle^2 + \langle \sigma_z \rangle^2}/2$", markersize="4", marker="s", linestyle="")
            #ax[1, 1].errorbar(tmw, -np.array(amp * np.cos(phase)),  np.sqrt((np.array(amperror)*np.cos(np.array(phase)))**2+(np.array(amp)*np.sin(np.array(phase))*np.array(phaseerror))**2),
            #                    color='purple', label=r"$\langle \sigma_y \rangle/2}$", markersize="4", marker="o", linestyle="")
            #ax[1, 1].errorbar(tmw, -np.array(amp * np.sin(phase)),  np.sqrt((np.array(amperror)*np.sin(np.array(phase)))**2+(np.array(amp)*np.cos(np.array(phase))*np.array(phaseerror))**2),
            #                    color="blue", label=r"$\langle \sigma_x \rangle/2}$", markersize="4", marker="s", linestyle="")




            ax[1, 0].plot(perturb_times, np.ones_like(perturb_times) * 0.5, color='grey', linestyle='--')
            ax[1, 0].plot(perturb_times, -np.ones_like(perturb_times) * 0.5, color='grey', linestyle='--')
            ax[1, 0].plot(perturb_times, -np.ones_like(perturb_times) * 0.0, color='grey', linestyle='--')

            ax[1, 1].set_ylim([-0.68, 0.68])
            ax[1, 1].legend(loc="lower center", fontsize=12)

            #ax[1, 0].set_xlabel(r'Time [$1/\Omega_R$]', fontsize=16)
            #ax[1, 0].set_ylabel('', fontsize=14)
            #ax[1, 0].set_ylabel('Fidelity', fontsize=14)
            #ax[1, 1].set_xlabel(r'Time [$1/\Omega_R$]', fontsize=16)
            #ax[1, 1].set_ylabel('Magnetization', fontsize=16)

            #ax[0, 1].set_xlabel(r'Time [$1/\Omega_R$]', fontsize=16)
            #ax[1, 0].set_ylabel('Magnetization', fontsize=16)

            print(Flist)
            #plt.show()

            #plt.yticks(np.arange(0, 6, 0.25))
            #plt.xticks(np.linspace(0, 6, 12))
            #plt.axis('scaled')

            #plt.savefig("Omega_R =  %.2f.png" % (
            #    Omega_R))  # and BW %.2f.pdf" % (noise_amplitude, bandwidth))

print(Flist)

#c.render()
plt.show()

plt.rcParams.update({
  "text.usetex": 0,
    "font.family":"sans-serif",
})


fig, ax = plt.subplots(1, 1, figsize=(11.69, 8.27/4))

ax.tick_params(axis="both", labelsize=16)


ax.errorbar(perturb_times, data,
                  linewidth="0.4",
                  color='#85bb65')

#ax[0].set_xlabel(r'Time [$1/\Omega_R$]', fontsize=14)
#ax[0].set_ylabel(r'$\Phi_B(t)[\pi]$', fontsize=18)
#ax[0].legend(loc="lower center", fontsize=12)

#ax[0].set_ylim([-0.6, 0.75])
ax.set_xlim([0, 6.051])

#plt.savefig("phasewalk.pdf")
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(11.69, 8.27/2))



#ax[0].errorbar(tmw, total , np.sqrt( np.array(amperror)**2 + np.array(zerror)**2),
 #                   color="grey", label=r"$\sqrt{\langle \sigma_x \rangle^2 + \langle \sigma_y \rangle^2 + \langle \sigma_z \rangle^2}/2$", markersize="4", marker="s", linestyle="")

ax.errorbar(tmw, -np.array(amp * np.sin(phase)),  np.sqrt((np.array(amperror)*np.sin(np.array(phase)))**2+(np.array(amp)*np.cos(np.array(phase))*np.array(phaseerror))**2),
                    color='black', label=r"$\langle \hat{s}_x \rangle$", markersize="4", marker="s", linestyle="")

ax.errorbar(tmw, -np.array(amp * np.cos(phase)),  np.sqrt((np.array(amperror)*np.cos(np.array(phase)))**2+(np.array(amp)*np.sin(np.array(phase))*np.array(phaseerror))**2),
                    color='#800080', label=r"$\langle \hat{s}_y \rangle$", markersize="4", marker="o", linestyle="")

ax.errorbar(tmw, z, zerror, color='#85bb65', label=r"$\langle \hat{s}_z \rangle$", markersize="5", marker="o",
                  linestyle="")

ax.plot(perturb_times, -np.real(expect_single[0]), color='black', linestyle="-")
ax.plot(perturb_times, -np.real(expect_single[2]), color='#800080', linestyle="-")
ax.plot(perturb_times, np.real(expect_single[1]), color='#85bb65', linestyle="-")




#ax[0].errorbar(tmw, total, np.sqrt(np.array(amperror) ** 2 + np.array(zerror) ** 2),
#                  color="grey",
#                  #label=r"$\sqrt{\langle \sigma_x \rangle^2 + \langle \sigma_y \rangle^2 + \langle \sigma_z \rangle^2}/2$",
#                  label=r"Total",
#                  markersize="4", marker="s", linestyle="")
ax.tick_params(axis="both", labelsize=16)


ax.plot(perturb_times, np.ones_like(perturb_times) * 0.5, color='grey', linestyle='--')
ax.plot(perturb_times, -np.ones_like(perturb_times) * 0.5, color='grey', linestyle='--')
ax.plot(perturb_times, -np.ones_like(perturb_times) * 0.0, color='grey', linestyle='--')

ax.set_ylim([-0.75, 0.75])
ax.set_xlim([0, 6.051])
ax.set_yticks(ticks=np.array([-0.5, -0.25, 0., 0.25, 0.5]))




#ax.set_ylim([-0.68, 0.68])
#ax.set_xlabel(r'Time [$2 \pi /\Omega_R$]', fontsize=18)
ax.set_ylabel(r'Spin $\langle \hat{s}_i \rangle$', fontsize=18)
ax.legend(loc="lower center", fontsize=14)
ax.set_xlabel(r'Time [$2 \pi /\Omega_R$]', fontsize=18)

plt.savefig("thereandback.pdf")


plt.show()



fig, ax = plt.subplots(1, 1, figsize=(11.69, 8.27/4))
ax.tick_params(axis="both", labelsize=16)


ax.errorbar(tmw, F2,
                  #label=r"$F =\sqrt{ \langle \Psi \vert \rho_{measured} \vert \Psi \rangle}$",
                  linestyle="dotted", markersize="3", marker="o",
                  color='black')

#ax[2].set_xlabel(r'Time [$1/\Omega_R$]', fontsize=18)
#ax[2].set_ylabel('Fidelity', fontsize=18)
ax.set_xlim([0, 6.051])
#ax[2].legend(loc="lower left", fontsize=12)
plt.savefig("Fidelity.pdf")


plt.show()







