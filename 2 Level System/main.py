from qutip import *
import numpy as np
from Atoms import *
from Driving import *
import matplotlib.pyplot as plt
from scipy import integrate


plt.rcParams.update({
  "text.usetex": True,
})

N = 2

omega = 2 * np.pi * 0 * 21 * 10 ** 3  # MHz
#omega = 2 * np.pi * 21 * 10 ** (-20)  # MHz

#omega = 0  # MHz

#Omega_R = 2 * np.pi * 25.7 * 10 ** 0  # MHz

Omega_R = 2 * np.pi * 23.7 * 10 ** 0  # MHz

gamma = 2 * np.pi * 15.0  # MHz

J = 0 * 2 * np.pi * 25.7 * 10 ** 0  # MHz

averages = 15

sampling_rate = 2 * np.pi * 64 * 10 ** 3  # MHz
endtime = 0.2
timesteps = int(endtime * sampling_rate)
data = np.loadtxt('Forward3MHzcsv.txt')
timesteps = 2 * len(data)


bath = 'Forward3MHzcsv.txt'
#bath = 'Markovian'

gamma1 = 0  # MHz

pertubation_length = endtime / 1

t1 = np.linspace(0, endtime, timesteps)
t2 = np.linspace(0, endtime, timesteps)

perturb_times = np.linspace(0, pertubation_length, timesteps)

Exps = [MagnetizationX(N), MagnetizationZ(N), MagnetizationY(N), sigmaz(0, N), sigmaz(N - 1, N), upup(0, N),
        sigmap(0, N), sigmam(0, N), downdown(0, N)]

opts = Options(store_states=True, store_final_state=True)  # , nsteps=50000)

# print('done')
# data1 = file.read()
# print(data)
# print(len(data))

# data_reversed = -data[::-1]


# print(len(data_reversed))


# plt.clear()

# data = np.append(data, data_reversed)

# plt.plot(np.linspace(0, 0.2, int(len(data))), np.cumsum(data))
# plt.plot(np.linspace(0.1, 0.2, int(len(data))), np.cumsum(-data_reversed)+np.cumsum(data)[-1])
# plt.ylabel('Phase [°]')
# plt.xlabel('Time [us]')
# plt.legend()
# plt.show()


for o in np.linspace(0*np.pi*23, 0*np.pi*25, 1):
    #print("Omega_R: ", Omega_R)
    for s in np.logspace(1 * omega, 10 * omega, num=1, base=np.e):
        # print("sampling: ", sampling_rate)
        init_state = productstateZ(0, 0, N)
        # timesteps = int(endtime * sampling_rate)
        data = np.loadtxt('Forward3MHzcsv.txt')
        timesteps = 2 * len(data)
        endtime = 0.2
        pertubation_length = endtime / 1
        # t1 = np.linspace(0, endtime, timesteps)
        # t2 = np.linspace(0, endtime, timesteps)
        perturb_times = np.linspace(0, pertubation_length, timesteps)
        fs = timesteps / endtime
        # print(len(perturb_times))
        for g in np.logspace(np.log(0.1 * Omega_R), np.log(10 * Omega_R), num=1, base=np.e):

            S1 = Cubic_Spline(perturb_times[0], perturb_times[-1],
                              noisy_func(gamma, perturb_times, omega, bath))
            S2 = Cubic_Spline(perturb_times[0], perturb_times[-1],
                              np.conj(noisy_func(gamma, perturb_times, omega, bath)))

            # S = Cubic_Spline(perturb_times[0], perturb_times[-1],
            # data / 0.4)

            result_single = mesolve([H0(omega, J, N), [H1(Omega_R, N), S1], [H2(Omega_R, N), S2]], init_state,
                                    perturb_times, e_ops=Exps, options=opts)

            expect_single = np.array(result_single.expect[:])

            # print("gamma: ", gamma)
            # print("Bandwidth", bandwidth)
            i = 1
            # random_phase = noise_amplitude * np.random.randn(perturb_times.shape[0])

            bath = "markovian"

            gamma = 0
            data = np.loadtxt('Forward3MHzcsv.txt')
            timesteps = 1 * len(data)
            endtime = 1
            pertubation_length = endtime / 1
            perturb_times = np.linspace(0, pertubation_length, timesteps)

            S1 = Cubic_Spline(perturb_times[0], perturb_times[-1],
                              noisy_func(gamma, perturb_times, omega, bath))
            S2 = Cubic_Spline(perturb_times[0], perturb_times[-1],
                              np.conj(noisy_func(gamma, perturb_times, omega, bath)))
            # S = Cubic_Spline(perturb_times[0], perturb_times[-1],
            #                 data[0:32000]/0.4)

            #print('H0...')
            #print(H0(omega, J, N))
            #print('H1...')
            #print(H1(Omega_R, N))
            #print('H2...')
            #print(H2(Omega_R, N))

            result2 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S1], [H2(Omega_R, N), S2]], init_state,
                              perturb_times, e_ops=Exps, options=opts)
            concmean = []
            #print(result2.states[10])
            for t in range(0, timesteps):
                concmean.append(concurrence(result2.states[t]))

            plt.plot(perturb_times, concmean)
            plt.show()

            #print(concmean)

            # opts = Options(store_states=True, store_final_state=True, rhs_reuse=True)
            states2 = np.array(result2.states[timesteps - 1])
            expect2 = np.array(result2.expect[:])
            ancilla_overlap = []
            Smean = np.zeros_like(perturb_times) + 1j * np.zeros_like(perturb_times)
            Pmean = 0

            while i < 1:  # averages + int(2 * gamma):
                #print(i)
                i += 1

                S1 = Cubic_Spline(perturb_times[0], perturb_times[-1],
                                  noisy_func(gamma, perturb_times, omega, bath))
                S2 = Cubic_Spline(perturb_times[0], perturb_times[-1],
                                  np.conj(noisy_func(gamma, perturb_times, omega, bath)))
                # S = Cubic_Spline(perturb_times[0], perturb_times[-1],
                # data / 0.4)

                result2 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S1], [H2(Omega_R, N), S2]], init_state,
                                  perturb_times, e_ops=Exps, options=opts)

                states2 += np.array(result2.states[timesteps - 1])
                expect2 += np.array(result2.expect[:])

                Smean += np.abs(
                    np.fft.fft(Omega_R * noisy_func(gamma, perturb_times, omega, bath)) ** 2)  # /2/np.pi/timesteps
                Pmean += np.abs(np.sum(Omega_R * noisy_func(gamma, perturb_times, omega, bath) ** 2))  # /timesteps
                # for t in range(0, timesteps):
                #    concmean[t] += concurrence(result2.states[t])

            # noisy_data2 = noisy_func(gamma, perturb_times, omega, bath)
            # S2 = Cubic_Spline(perturb_times[0], perturb_times[-1], noisy_data2)

            states2 = states2 / i
            expect2 = expect2 / i
            Smean = Smean / i
            Pmean = Pmean / i
            concmean = np.array(concmean) / i

            # print(Qobj(states2))
            # print((expect2[5]+expect2[8]).mean())
            density_matrix = Qobj([[expect2[5][timesteps - 1], expect2[6][timesteps - 1]],
                                   [expect2[7][timesteps - 1], expect2[8][timesteps - 1]]])
            # print(density_matrix)
            # result3 = mesolve(H0(omega, J, N), Qobj(states2), t2, [], e_ops=Exps, options=opts)

            # print('Initial state ....')
            # print(productstateZ(0, 0, N))
            # print(productstateZ(0, 0, N).dag()*sigmaz(1, N)*productstateZ(0, 0, N))

            # print('Commutator:', 1j * Commutator[0][0])
            # print('AntiCommutator: ', AntiCommutator[0][0])
            # print(np.correlate(S2(perturb_times), S2(perturb_times), "valid"))
            # result_me = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]],
            #                    result_t1.states[timesteps - 1],
            #                    perturb_times, [noise_amplitude * sigmap(1, 0, N) / 10, noise_amplitude * sigmam(1, 0, N) / 10], Exps,
            #                    options=opts)

            # print(Pmean)

            #fig, ax = plt.subplots(2, 2, figsize=(10, 10))

            #timesteps = 2 * len(data)
            #endtime = 0.2
            #pertubation_length = endtime / 1
            #perturb_times = np.linspace(0, pertubation_length, timesteps)

            #ax[0, 0].plot(perturb_times, np.real(expect2[1]), color='#85bb65')
            #ax[0, 0].set_xlabel('Time [us]', fontsize=16)
            #ax[0, 0].set_ylabel('Expectation Value', fontsize=16)
            # ax[1, 0].plot(perturb_times, np.real(expect_me[1]), label="sigma_z, ME with sqrt(gamma)*L")
            #ax[0, 0].legend(loc="lower center")
            #ax[0, 0].set_ylim([-0.501, -0.499])
            #plt.show()










            #################### SPECTRA ######################################## 11111111111111111111111111111111111111111111111111111111111

            fig, ax = plt.subplots(2, 2, figsize=(10, 10))

            '''
            freq = np.fft.fftfreq(perturb_times.shape[-1], d=1 / sampling_rate)
            fourier = Smean  # np.max(Smean) #np.abs(np.fft.fft(brownian_func(gamma, perturb_times, omega, sampling_rate)))

            # max_pos = int(len(perturb_times) / 2) - np.argmax(fourier[int(len(perturb_times) / 2): len(perturb_times)])
            # print(len(perturb_times))
            # print(len(freq))
            # print(freq)
            # print(perturb_times)

            # ax[0, 0].plot(freq[0:int(len(perturb_times)/2)], fourier[0:int(len(perturb_times)/2)], linestyle='',
            #              marker='o', markersize='2', linewidth=0.0) #[0:int(len(perturb_times) / 2)]

            # ax[0, 0].plot(freq[0:int(len(perturb_times) / 2)], lorentzian(freq, Pmean, omega/(2*np.pi),
            #                                                              gamma)[0:int(len(perturb_times) / 2)],
            #              linestyle='-',
            #              marker='o', markersize='0', linewidth=1.0,
            #              label="Lorentzian with FWHM gamma= %.2f MHz" % gamma)

            # ax[0, 0].plot(freq[int(len(perturb_times)/2)+2000: int(len(perturb_times))-4000], fourier[int(len(perturb_times)/2)+2000: int(len(perturb_times))-4000], linestyle='',
            #              marker='o', markersize='2', linewidth=0.0)
            half = noisy_func(gamma, perturb_times, omega, bath)[0:int(len(perturb_times) / 2 - 10)]
            long_real = np.append(half, half[::-1])
            # long_nn = np.append(func(perturb_times, omega)[0:int(len(perturb_times) / 2 - 10)],
            #                    func(perturb_times, omega)[0:int(len(perturb_times) / 2 - 10)])

            samples = 2 * 10 ** 6
            sample_time = 2
            # for x in range(0, 4):
            #    long_real = np.append(long, noisy_func(gamma, perturb_times, omega, bath)[0:int(len(perturb_times) / 2 - 10)])
            # long_nn = np.append(long, func(perturb_times, omega)[0:int(len(perturb_times) / 2 - 10)])

            long = sqrt(2) * noisy_func(3, np.linspace(0, sample_time, samples), omega, "markovian")
            long_nn = sqrt(2) * func(np.linspace(0, sample_time, samples), omega)

            fs = samples / sample_time

            f_real, Pxx_real = signal.welch(
                sqrt(2) * long_real, len(long_real) / 0.2,
                nperseg=10 ** 7)

            f, Pxx_den = signal.welch(
                long, fs,
                nperseg=samples)
            f1, Pxx_den1 = signal.welch(
                long_nn, fs,
                nperseg=samples)

            # rydfft = get_fft(long)

            print("welch sum:", np.sum(Pxx_den))
            print("welch sum no noise:", np.sum(Pxx_den1))
            print("real sum:", np.sum(Pxx_real))
            print("lorentz sum:", np.sum(lorentzian(f, 1, omega / (2 * np.pi), 3)))
            '''

            #ax2 = plt.subplot(222)


            factor = 3

            #samples = 64 * 10 ** (factor+2) #min 8 * 10 ** 6 or 2*10**7 for good results
            omega= 2 * np.pi * 20 * 10 ** factor #MHz
            sample_time = np.pi/2 * 1  # * 10 **(1-factor)
            samples = int(sample_time * 64 * 10 ** (factor+2))

            #print(samples)
            #gamma0 = average_psd(0, omega, samples, sample_time, 1)
            gamma3 = average_psd(3 * 10**(factor-3), omega, samples, sample_time, 400) #min 40 or 400 for good results
            gamma5 = average_psd(5 * 10**(factor-3), omega, samples, sample_time, 400)
            gamma15 = average_psd(15 * 10**(factor-3), omega, samples, sample_time, 400)
            gamma30 = average_psd(30 * 10**(factor-3), omega, samples, sample_time, 400)

            #print("welch sum" , np.sum(gamma0[1]))

            #print("Lorentz sum", np.sum(lorentzian(gamma0[0], 1, omega / (2 * np.pi), 2/np.pi))/sample_time)
            '''
            long = sqrt(2) * noisy_func(0, np.linspace(0, sample_time, samples), omega, "markovian")

            # generate some data
            #x = np.arange(0., np.pi, np.pi/10)
            #y = np.sin(x)
            #y = np.random.uniform(size=300)
            y = long
            yunbiased = y - np.mean(y)
            ynorm = np.sum(yunbiased ** 2)
            #print(ynorm)
            #print(samples)
            acor = np.correlate(yunbiased, yunbiased, "same")#/samples # / ynorm
            # use only second half
            acor = acor[int(len(acor) / 2):]


            acorfft = np.fft.fft(acor)
            #acorfftfreq = np.fft.fftfreq(acor.size, sample_time/samples)



            #plt.plot(acor)
            #plt.show()

            #auto =np.array(autocross(long,long))

            auto = np.array(autocorr4(long))

            autofft = np.fft.fft(auto)

            autofftfreq = np.fft.fftfreq(auto.size, sample_time / samples)

            autofftabs = (autofft.real**2+autofft.imag**2)/samples**2

            print("Fouriersum", np.sum(autofftabs))



            #print(auto)


            # ax[0, 0].plot(f_real, Pxx_real, linestyle='-',
            #           marker='s', markersize='6', linewidth=0.55, label="PSD $\gamma=3$ MHz Exp", color="#85bb65")

            ax[1, 1].plot(np.linspace(0, len(auto), auto.size), auto, linestyle='',
                          marker='^', markersize='4', label=r"$autocorrelation$", color='#025669', markerfacecolor='black', markeredgecolor = 'black')

            ax[1, 0].plot(autofftfreq,  autofftabs, linestyle='',
                          marker='^', markersize='4', label=r"$autocorrelation$", color='#025669', markerfacecolor='black', markeredgecolor = 'black')

            ax[1, 0].plot(autofftfreq, lorentzian(autofftfreq, 1, omega / (2 * np.pi), 0.1), linestyle='-',
                          linewidth=1,
                          color='#025669')
                          
            '''

            #ax[0, 1].plot(np.linspace(0, len(acor), acor.size), acor, linestyle='',
            #              marker='^', markersize='4', label=r"$autocorrelation$", color='#025669', markerfacecolor='black', markeredgecolor = 'black')

            #ax[1, 0].plot(acorfft[0],  (acorfft[1].real**2+acorfft[1].imag**2), linestyle='',
            #              marker='^', markersize='4', label=r"$autocorrelation$", color='#025669', markerfacecolor='black', markeredgecolor = 'black')

            #ax[1, 0].plot(np.linspace(0, len(auto), auto.size), np.real(auto), linestyle='',
            #              marker='^', markersize='4', label=r"$autocorrelation$", color='#025669', markerfacecolor='black', markeredgecolor = 'black')

            #ax[0, 0].plot(-gamma0[0], 2*gamma0[1], linestyle='',
            #              marker='^', markersize='4', label=r"$\gamma=0$", color='#025669', markerfacecolor='black', markeredgecolor = 'black')
            #ax[0, 0].plot(gamma0[0], lorentzian(gamma0[0], 1, omega / (2 * np.pi), 2/np.pi)/sample_time, linestyle='-',
            #              linewidth=1,
            #              color='black')

            np.save('gamma3.npy', gamma3)
            np.save('gamma5.npy', gamma5)
            np.save('gamma15.npy', gamma15)
            np.save('gamma30.npy', gamma30)



            ax[0, 0].plot(-gamma3[0],2* gamma3[1], linestyle='',
                          marker='^', markersize='4', label=r"$\gamma=\Omega_R/5$", color='#025669', markerfacecolor='none', markeredgecolor = '#025669')
            ax[0, 0].plot(gamma3[0], lorentzian(gamma3[0], 1, omega / (2 * np.pi), 3)/sample_time, linestyle='-',
                          linewidth=1,
                          color='#025669')



            ax[0, 0].plot(-gamma5[0], 2*gamma5[1], linestyle='',
                          marker='D', markersize='4', label=r"$\gamma=\Omega_R/3$", markerfacecolor='none', markeredgecolor = '#800080')
            ax[0, 0].plot(gamma5[0], lorentzian(gamma5[0], 1, omega / (2 * np.pi), 5)/sample_time, linestyle='-',
                          linewidth=1,
                          color='#800080')

            ax[0, 0].plot(-gamma15[0], 2*gamma15[1], linestyle='',
                          marker='v', markersize='4', label=r"$\gamma=\Omega_R$", markerfacecolor='none', markeredgecolor = "#CC7722")
            ax[0, 0].plot(gamma15[0], lorentzian(gamma15[0], 1, omega / (2 * np.pi), 15)/sample_time, linestyle='-',
                           linewidth=1,
                          color="#CC7722")

            ax[0, 0].plot(-gamma30[0], 2*gamma30[1], linestyle='',
                          marker='s', markersize='4', label=r"$\gamma=2\Omega_R$", markerfacecolor='none', markeredgecolor = "#800020")
            ax[0, 0].plot(gamma30[0], lorentzian(gamma30[0], 1, omega / (2 * np.pi), 30)/sample_time, linestyle='-',
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

            ax[0, 0].set_xlim([int(omega/2/np.pi)-10*10**(factor-3), int(omega/2/np.pi)+10*10**(factor-3)])
            ax[0, 0].set_ylim(bottom=0)
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



            ax[0, 0].legend(loc="upper left", fontsize=16)

            ax[0, 0].tick_params(axis="both", labelsize=16)

            ax[0, 0].set_xlabel(r'$\omega$ [$\Omega_R$]', fontsize=26)
            ax[0, 0].set_ylabel(r'PSD / P$_{Carrier}$ [1/Hz]', fontsize=16)

            ax[0, 0].legend(loc="upper left", fontsize=12)
            #ax[0, 0].set_xticks(ticks=np.array([-3, -2, -1, 0., 1, 2, 3]))
            #ax[0, 0].set_xticks(np.linspace(-10, 10, 5))


            #################### END OF SPECTRA ######################################## 1111111111111111111111111111







            #fig, ax = plt.subplots(2, 2, figsize=(10, 10))


            #################### SINGLE TRAJECTORY ######################################## 222222222222222222222222222

            data = np.loadtxt('Forward3MHzcsv.txt')
            timesteps = 2 * len(data)
            endtime = 0.2
            pertubation_length = endtime / 1
            perturb_times = np.linspace(0, pertubation_length, timesteps)

            with open('m.txt') as f:
                linesm = f.readlines()

            with open('magn_z_forward.txt') as f:
                linesmagnzforward = f.readlines()
            with open('magn_z_forward_error.txt') as f:
                linesmagnzforwarderror = f.readlines()
            with open('magn_z_backward.txt') as f:
                linesmagnzbackward = f.readlines()
            with open('magn_z_backward_error.txt') as f:
                linesmagnzbackwarderror = f.readlines()

            with open('phase_amp_forward.txt') as f:
                    linesphase_ampforward = f.readlines()
            with open('phase_amp_forward_error.txt') as f:
                    linesphase_ampforwarderror = f.readlines()
            with open('phase_amp_backward.txt') as f:
                    linesphase_ampbackward = f.readlines()
            with open('phase_amp_backward_error.txt') as f:
                    linesphase_ampbackwarderror = f.readlines()

            with open('phase_forward.txt') as f:
                    linesphase_forward = f.readlines()
            with open('phase_forward_error.txt') as f:
                    linesphase_forwarderror = f.readlines()
            with open('phase_backward.txt') as f:
                    linesphase_backward = f.readlines()
            with open('phase_backward_error.txt') as f:
                    linesphase_backwarderror = f.readlines()

            with open('mf.txt') as f:
                linesmf = f.readlines()
            with open('mfxy.txt') as f:
                linesmfxy = f.readlines()
            with open('mxy.txt') as f:
                linesmxy = f.readlines()

            x = []
            z = []
            zerror = []
            amp = []
            amperror = []
            phase = []
            phaseerror = []
            y = []
            xy = []


            for element in range(1, 20):
                x.append(float(linesmf[element][0:5]))
                y.append(float(linesmf[element][11:18]))
                xy.append(float(linesmfxy[element][11:18]))

            for element in range(0, 19):
                z.append(float(linesmagnzforward[element][5:11]))
                zerror.append(float(linesmagnzforwarderror[element][5:11]))
                amp.append(float(linesphase_ampforward[element][5:11]))
                amperror.append(float(linesphase_ampforwarderror[element][5:11]))
                phase.append(float(linesphase_forward[element][5:11])/180)
                phaseerror.append(float(linesphase_forwarderror[element][5:11])/180)



            for element in range(1, 21):
                z.append(float(linesmagnzbackward[element][11:18]))
                zerror.append(float(linesmagnzbackwarderror[element][11:18]))
                amp.append(float(linesphase_ampbackward[element][11:18]))
                amperror.append(float(linesphase_ampbackwarderror[element][11:18]))
                phase.append(float(linesphase_backward[element][11:18])/180)
                phaseerror.append(float(linesphase_backwarderror[element][11:18])/180)

            for element in range(1, 21):
                x.append(float(linesm[element][0:5]))
                y.append(float(linesm[element][11:18]))
                xy.append(float(linesmxy[element][12:18]))


            phaseerror[0]-=0.7


            data = np.loadtxt('Forward3MHzcsv.txt')

            data_reversed = -data[::-1]

            data = np.cumsum(data)

            data_reversed = np.cumsum(data_reversed) + data[-1] + 180

            data = np.append(data/180, data_reversed/180)

            ax[1, 1].errorbar(perturb_times, data, label="Phase drift",
                              linewidth="0.4",
                              color='#85bb65')

            data = np.loadtxt('Forward3MHzcsv.txt')

            ax[1, 1].errorbar(x, phase, phaseerror, label="Phase xy-plane",
                          linestyle="",
                          markersize="5", marker="v", color='black')
            #ax[0, 0].errorbar(x, amp, amperror, label=r"$\sqrt{\langle \sigma_x \rangle^2 + \langle \sigma_y \rangle^2}$",
            #              linestyle="",
            #              markersize="5", marker="v", color='black')
            #ax[0, 0].plot(perturb_times, np.sqrt(expect_single[2] ** 2 + expect_single[0] ** 2), color="black",
            #              linestyle="--")
            #ax[1, 0].plot(x, amp*np.cos(phase), color="b", label=r"$x$", markersize="5", marker="o",
            #              linestyle="")

            #ax[0, 0].plot(perturb_times, (-np.arccos(expect_single[2] / np.sqrt((expect_single[2] ** 2) + expect_single[0] ** 2))) / (2*np.pi), color='red',
            #              label=r"$Phase$", linewidth="1")
            ax[1, 1].plot(perturb_times, np.real(2*np.arcsin(expect_single[2] / np.sqrt(expect_single[2] ** 2 + expect_single[0] ** 2+0.001)) / (np.pi) + 0.2), color='black',
                           linewidth="1")

            ax[1, 1].set_xlabel('Time [us]', fontsize=14)
            ax[1, 1].set_ylabel('Phase [$\pi$]', fontsize=14)



            for n in range(0,len(phase)):
                phase[n] = phase[n] * np.pi


            ax[1, 0].plot(perturb_times, np.real(expect_single[1]), color='#85bb65')
            #ax[1, 0].plot(perturb_times, np.real(expect_single[0]), color='blue', label=r"$x$", linewidth="1")
            #ax[1, 0].plot(perturb_times, np.real(expect_single[2]), color='grey', linewidth="1")

            #ax[1, 0].errorbar(x, amp*np.cos(phase), np.sqrt((amperror*np.cos(phase))**2+(amp*np.sin(phase)*phaseerror)**2), color="b", label=r"$x$", markersize="5", marker="o",
            #              linestyle="")

            #ax[1, 0].errorbar(x, amp*np.sin(phase), np.sqrt((amperror*np.sin(phase))**2+(amp*np.cos(phase)*phaseerror)**2), color="grey", label=r"$y$", markersize="5", marker="o",
            #              linestyle="--")

            ax[1, 0].errorbar(x, z, zerror, label=r"$\langle \sigma_z \rangle$", linestyle="", markersize="5", marker="o",
                          color='#85bb65')

            ax[1, 0].errorbar(x, amp, amperror, label=r"$\sqrt{\langle \sigma_x \rangle^2 + \langle \sigma_y \rangle^2}$",
                          linestyle="",
                          markersize="5", marker="v", color='black')

            # ax[1, 0].plot(perturb_times, np.real(expect2[0]), label="sigma_x, Time Dependent Hamiltonian")
            # ax[1, 0].plot(perturb_times, np.real(expect2[2]), label="sigma_y, Time Dependent Hamiltonian")
            #ax[1, 0].plot(perturb_times, np.sqrt(expect_single[2] ** 2 + expect_single[0] ** 2 + expect_single[1] ** 2), color="black",
            #              linestyle="--")

            ax[1, 0].plot(perturb_times, np.real(np.sqrt(expect_single[2] ** 2 + expect_single[0] ** 2)), color="black",
                          linestyle="-")
            # ax[1, 0].plot(perturb_times, concmean, label="overlap-bell-basis")
            # ax[1, 0].plot(perturb_times, np.exp(- perturb_times * gamma), color="orange", label="exp(- gamma * t)")
            # ax[1, 0].plot(perturb_times, -np.exp(- perturb_times * gamma), color="orange")
            ax[1, 0].set_xlabel('Time [$\mu$s]', fontsize=14)
            ax[1, 0].set_ylabel('Magnetization', fontsize=14)
            ax[1, 0].set_ylim([-0.596, 0.596])
            # ax[1, 0].plot(perturb_times, np.real(expect_me[1]), label="sigma_z, ME with sqrt(gamma)*L")
            ax[1, 0].legend(loc="lower center", fontsize=12)

            #################### END OF SINGLE TRAJECTORY ######################################## 2222222222222222222














            ################### PHASE WALKS ############################################ 33333333333333333333333333333333
            data = np.loadtxt('Forward3MHzcsv.txt')
            timesteps = len(data)
            endtime = 6
            pertubation_length = endtime / 1
            perturb_times = np.linspace(0, pertubation_length, timesteps)

            # Total time.
            T = perturb_times[-1]
            # Number of steps.
            Nsteps = int(len(perturb_times)/100)
            # Time step size
            dt = T / Nsteps
            # Create an empty array to store the realizations.
            x = np.empty((averages, Nsteps + 1))
            # Initial values of x.
            x[:, 0] = 0

            for gamma in [3, 5, 15, 30]:

                phase_noise = brownian(x[:, 0], Nsteps, dt/15, np.sqrt(gamma), out=x[:, 1:])

                #phase_noise = davies_harte(perturb_times[-1], len(perturb_times), 0.1, np.sqrt(gamma))
                #print(phase_noise[0:len(perturb_times)])

                t = np.linspace(0.0, Nsteps * dt, Nsteps)

                for k in range(0, int(averages / 100)):
                    if gamma == 4:
                        ax[0, 1].plot(t, phase_noise[k], color='#85bb65', linewidth=0.1)
                        #ax[0, 1].plot(t, phase_noise, color='#85bb65', linewidth=0.1)
                    elif gamma == 14:
                        ax[0, 1].plot(t, phase_noise[k], color='#CC7722', linewidth=0.1)
                        #ax[0, 1].plot(t, phase_noise, color='#CC7722', linewidth=0.1)
                    elif gamma == 31:
                        ax[0, 1].plot(t, phase_noise[k], color='#800020', linewidth=0.1)
                        #ax[0, 1].plot(t, phase_noise, color='#800020', linewidth=0.1)

                if gamma == 3:
                    # ax[0, 1].plot(t, np.mean(phase_noise, axis=0), color='green', linestyle='', linewidth=1.0, marker="o", markersize="0.01")

                    ax[0, 1].plot(t, np.sqrt(np.var(phase_noise, axis=0)), color='#025669', linestyle='',
                                  linewidth=1.0,
                                  label='$\gamma = 3$ MHz',marker="^", markersize="3")
                    ax[0, 1].plot(t, np.sqrt(gamma * t/15), color='#025669', linestyle='--', linewidth=1.0)
                    #, label='Expected Standard Deviation = $\sqrt{3 MHz t}$')
                    #ax[0, 1].plot(t, -np.sqrt(gamma * t), color='#85bb65', linestyle='--', linewidth=1.0)

                if gamma == 4:
                    # ax[0, 1].plot(t, np.mean(phase_noise, axis=0), color='green', linestyle='', linewidth=1.0, marker="o", markersize="0.01")

                    ax[0, 1].plot(t, -np.sqrt(np.var(phase_noise, axis=0)), color='#800080', linestyle='-',
                                  linewidth=1.0,
                                  label='$\gamma = 3$ MHz')
                    ax[0, 1].plot(t, -np.sqrt(gamma * t / 15), color='#800080', linestyle='--', linewidth=1.0)
                    # , label='Expected Standard Deviation = $\sqrt{3 MHz t}$')
                    # ax[0, 1].plot(t, -np.sqrt(gamma * t), color='#85bb65', linestyle='--', linewidth=1.0)

                if gamma == 14:
                    # ax[0, 1].plot(t, np.mean(phase_noise, axis=0), color='orange', linestyle='', linewidth=1.0, marker="o", markersize="0.01")

                    ax[0, 1].plot(t, -np.sqrt(np.var(phase_noise, axis=0)), color='#CC7722', linestyle='-',
                                  linewidth=1.0,
                                  label='$\gamma = 10$ MHz')
                    ax[0, 1].plot(t, -np.sqrt(gamma * t/15), color='#CC7722', linestyle='--', linewidth=1.0)
                    #,label='$\sqrt{10 MHz t}$')
                    #ax[0, 1].plot(t, -np.sqrt(gamma * t), color='#CC7722', linestyle='--', linewidth=1.0)

                if gamma == 30:
                    # ax[0, 1].plot(t, np.mean(phase_noise, axis=0), color='red', linestyle='', linewidth=1.0, marker="o", markersize="0.01")
                    ax[0, 1].plot(t, np.sqrt(np.var(phase_noise, axis=0)), color='#800020', linestyle='',
                                  linewidth=1.0,
                                  label='$\gamma = 30$ MHz',marker="s", markersize="3")
                    ax[0, 1].plot(t, np.sqrt(gamma * t/15), color='#800020', linestyle='--', linewidth=1.0)
                    # ,label='$\sqrt{30 MHz  t}$')
                    #ax[0, 1].plot(t, -np.sqrt(gamma * t), color='#800020', linestyle='--', linewidth=1.0)




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

            #ax[0, 1].set_ylim([-1.4 * np.sqrt(30 * T), 1.4 * np.sqrt(30 * T)])
            ###ax[0, 1].set_xlim([0, 0.1])
            ax[0, 1].set_xlabel('[$1/\Omega_R$]', fontsize=16)
            ax[0, 1].set_ylabel('Phase [$\pi$]', fontsize=16)
            #ax[0, 1].set_xlabel('Time [a.u.]', fontsize=16)
            #ax[0, 1].set_ylabel('Apmlitude [a.u.]', fontsize=16)
            ax[0, 1].legend(loc="upper left", fontsize=12)

            ################### END OF PHASE WALKS ############################################ 3333333333333333333333














            ################### MASTER EQUATION ############################################ 444444444444444444

            with open('m0.txt') as f:
                linesm0 = f.readlines()
            with open('m3.txt') as f:
                linesm3 = f.readlines()
            with open('m10.txt') as f:
                linesm10 = f.readlines()
            with open('m30.txt') as f:
                linesm30 = f.readlines()
            with open('m0err.txt') as f:
                linesm0e = f.readlines()
            with open('m3err.txt') as f:
                linesm3e = f.readlines()
            with open('m10err.txt') as f:
                linesm10e = f.readlines()
            with open('m30err.txt') as f:
                linesm30e = f.readlines()

            # print(linesmf)
            # print(linesm)
            x0 = []
            y0 = []
            y0e = []
            y3 = []
            y3e = []
            y10 = []
            y10e = []
            y30 = []
            y30e = []

            for element in range(1, 22):
                x0.append(float(linesm0[element][0:5]))
                y0.append(float(linesm0[element][8:18]))
                y3.append(float(linesm3[element][8:18]))
                y10.append(float(linesm10[element][8:18]))
                y30.append(float(linesm30[element][8:18]))
                y0e.append(float(linesm0e[element][8:18]))
                y3e.append(float(linesm3e[element][8:18]))
                y10e.append(float(linesm10e[element][8:18]))
                y30e.append(float(linesm30e[element][8:18]))

            t1 = np.linspace(0, endtime, int(timesteps / 10))

            Omega_R = 2 * np.pi * 12.3 #MHz
            omega=0

            S = Cubic_Spline(perturb_times[0], perturb_times[-1], func(perturb_times, omega))

            result_m0 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]],
                                init_state,
                                t1, [], Exps,
                                options=opts)

            m0 = np.array(result_m0.expect[:])

            result_m3 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]],
                                init_state,
                                t1, [np.sqrt(3) * sigmaz(0, N), np.sqrt(3) * sigmaz(0, N)], Exps,
                                options=opts)
            result_m10 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]],
                                 init_state,
                                 t1, [np.sqrt(10) * sigmaz(0, N), np.sqrt(10) * sigmaz(0, N)], Exps,
                                 options=opts)
            result_m30 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]],
                                 init_state,
                                 t1, [np.sqrt(30) * sigmaz(0, N), np.sqrt(30) * sigmaz(0, N)], Exps,
                                 options=opts)

            ax[1, 1].errorbar(x0, y0, y0e, marker="o", color='black', label='$\gamma = 0$ MHz', linestyle='')

            ax[1, 1].plot(t1, np.real(m0[1]), color='black', linestyle='--')

            #ax[1, 1].plot(perturb_times, np.real(expect2[1]), color='black', label="Time Dependant"),

            ax[1, 1].errorbar(x0, y3, y3e, marker="^", color='#85bb65', label='$\gamma = 3$ MHz', linestyle='')
            ax[1, 1].plot(t1, np.real(result_m3.expect[1]), color='#85bb65', linestyle='--')
            ax[1, 1].errorbar(x0, y10, y10e, marker="v", color='#CC7722', label='$\gamma = 10$ MHz', linestyle='')
            ax[1, 1].plot(t1, np.real(result_m10.expect[1]), color='#CC7722', linestyle='--')
            ax[1, 1].errorbar(x0, y30, y30e, marker="s", color='#800020', label='$\gamma = 30$ MHz', linestyle='')
            ax[1, 1].plot(t1, np.real(result_m30.expect[1]), color='#800020', linestyle='--')
            # ax[1, 1].plot(t, -np.sqrt(gamma * t), color='black', linestyle='--', linewidth=2.0)
            ax[1, 1].set_ylim([-0.596, 0.596])
            ax[1, 1].set_xlabel('Time [$\mu$s]', fontsize=14)
            ax[1, 1].set_ylabel('Magnetization', fontsize=14)
            ax[1, 1].legend(loc="upper center", fontsize=12)

            fig.tight_layout()

            plt.show()

            plt.savefig("Powerspectrum.pdf")

            #plt.savefig("bath" + bath + ", Omega_R =  %.2f, sampling =  %.2f,gamma = %.2f.png" % (
            #    Omega_R, sampling_rate, gamma))  # and BW %.2f.pdf" % (noise_amplitude, bandwidth))

            ################### MASTER EQUATION ############################################4444444444444444












################### COMMUTATOR / ANTICOMMUTATOR ############################################ 5555555555555555555555555


'''

opts = Options(store_states=True, store_final_state=True)

Omega_R = 2 * np.pi * 11 * 10 ** 0  # MHz

Commutatorlist = []
Anticommutatorlist = []

#t1 = np.linspace(0, 0.08, int(timesteps / 10))
#S = Cubic_Spline(t1[0], t1[-1], func(t1, omega))

sampling_rate = 10**5

perturbation = 0.067

full_time = np.linspace(0, 0.2 + perturbation, int((0.2 + perturbation)*sampling_rate))
S = Cubic_Spline(full_time[0], full_time[-1], func(full_time, omega))

t1 = np.linspace(0, perturbation, int(perturbation*sampling_rate))
t2 = np.linspace(perturbation, 0.2 + perturbation, int(0.2*sampling_rate))


#result_t1 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]],
#                    productstateZ(0, N - 1, N), full_time, [], Exps, options=opts)

#print(result_t1.expect[1])

#result_t1t2 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]],
#                      result_t1.states[len(t1) - 1], t2, [], Exps, options=opts)

#Perturb_incoherent = MagnetizationZ(N)
#Perturb_coherent = MagnetizationZ(N)
#Measure = MagnetizationZ(N)


#result_AB = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]],
                    #Perturb_incoherent * result_t1.states[len(t1) - 1], t2, [], Exps, options=opts)

#result_AB_comm = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]],
                     #    Perturb_coherent * result_t1.states[len(t1) - 1], t2, [], Exps, options=opts)

#for t in range(0, len(t2)):
#    prod_AB = result_t1t2.states[t - 1].dag() * Measure * result_AB.states[t - 1]

#    prod_AB_comm = result_t1t2.states[t - 1].dag() * Measure * result_AB_comm.states[t - 1]

 #   prod_BA = result_AB.states[t - 1].dag() * Measure * result_t1t2.states[t - 1]

 #   prod_BA_comm = result_AB_comm.states[t - 1].dag() * Measure * result_t1t2.states[t - 1]

 #   Commutator = prod_AB_comm - prod_BA_comm

 #   AntiCommutator = prod_AB + prod_BA

  #  Commutatorlist.append(Commutator[0][0][0])
  #  Anticommutatorlist.append(AntiCommutator[0][0][0])
    # print('Commutator:', 1j * Commutator[0][0])
    # print('AntiCommutator: ', AntiCommutator[0][0])

with open('56S_unperturb.txt') as f:
    linesunp = f.readlines()



with open('56S_unperturbed.txt') as f:
    linesun56S = f.readlines()
with open('56S_coherent_perturb.txt') as f:
    linescoh56S = f.readlines()
with open('56S_incoherent_perturb.txt') as f:
    linesinc56S = f.readlines()

with open('55P_unperturbed.txt') as f:
    linesun55P = f.readlines()
with open('55P_coherent_perturb.txt') as f:
    linescoh55P = f.readlines()
with open('55P_incoherent_perturb.txt') as f:
    linesinc55P = f.readlines()


with open('56P_coherent_perturb.txt') as f:
    linescoh56P = f.readlines()
with open('56P_incoherent_perturb.txt') as f:
    linesinc56P = f.readlines()

x0 = []
un56S = []
un55P = []
unp = []
un=[]

coh56S = []
coh56P = []
coh55P = []
coh=[]
comm=[]

inc56S = []
inc56P = []
inc55P = []
inc=[]
anti=[]

for element in range(1, 22):
    x0.append(float(linesun56S[element][0:5]) + 0*perturbation)

    un56S.append(float(linesun56S[element][8:18]))
    un55P.append(float(linesun55P[element][8:18]))
    un.append(float(linesun56S[element][8:18])-float(linesun55P[element][8:18]))

    #coh.append((float(linesun[element][8:18]) - float(linescoh[element][8:18]))/(40*0.0075))
    coh56S.append(float(linescoh56S[element][8:18]))
    coh56P.append(float(linescoh56P[element][8:18]))
    coh55P.append(float(linescoh55P[element][8:18]))
    coh.append(float(linescoh56S[element][8:18])-float(linescoh55P[element][8:18]))
    comm.append(float(linesun56S[element][8:18])-float(linesun55P[element][8:18])-(float(linescoh56S[element][8:18])-float(linescoh55P[element][8:18])))

   #inc.append((float(linesun[element][8:18]) - float(linesinc[element][8:18]))/(40*0.0075))
    inc56S.append(float(linesinc56S[element][8:18]))
    inc55P.append(float(linesinc55P[element][8:18]))
    inc56P.append(float(linesinc56P[element][8:18]))
    inc.append(float(linesinc56S[element][8:18])-float(linesinc55P[element][8:18]))
    anti.append(float(linesun56S[element][8:18]) - float(linesun55P[element][8:18]) - (
                float(linesinc56S[element][8:18]) - float(linesinc55P[element][8:18])))

    unp.append(float(linesunp[element][8:18]))

# freq = np.fft.fftfreq(t2.shape[-1])
# plt.plot(freq, np.real(np.fft.fft(Commutatorlist)), linestyle='--', marker='o', markersize='5', label="Commutator")
# plt.plot(freq, np.real(np.fft.fft(Anticommutatorlist)), linestyle='--', marker='o', markersize='5',
#         label="Anticommutator")
# plt.xlim(-1 / 2, 1 / 2)
# plt.legend()
# plt.show()


# plt.plot(t1, result_t1.expect[1], label="r$\langle \sigma_z(t1) \rangle$", linestyle="", marker="o", markersize="1",
#         color="b")
# plt.plot(t2, 100*np.imag(Commutatorlist), label="Im(Commutator)", color="b")
# plt.plot(t2, np.real(Anticommutatorlist), label="Re(Anticommutator)", linestyle="",marker="o",markersize="1", color="r")
# plt.plot(t2, np.imag(Anticommutatorlist), label="Im(Anticommutator)", color="r")
# plt.legend()
# plt.xlabel('$t_1$')
# plt.show()

# plt.plot(t2, np.real(Commutatorlist), label="Re(Commutator)", linestyle="",marker="o",markersize="1", color="b")

plt.plot(x0, un56S, marker="o", color='#008b8b', linestyle='', label="56S")
plt.plot(x0, un55P, marker="o", color='orange', linestyle='', label="55P")
#plt.plot(x0, unp, marker="o", color='#800020', label='Unperturbed', linestyle='')

#plt.plot(full_time, result_t1.expect[1], label=r"Unperturbed_Expect", linestyle="-", marker="o",
#         markersize="0", color="#008b8b")

plt.plot(full_time, -0.5*np.cos(Omega_R * full_time)+0.5, linestyle="-", marker="o",
         markersize="0", color="#008b8b")
plt.plot(full_time, 0.5*np.cos(Omega_R * full_time)+0.5, linestyle="-", marker="o",
         markersize="0", color="orange")


plt.plot(x0, coh56S, marker="o", color='blue', label='56S Perturbed', linestyle='--', markersize="5")
plt.plot(x0, coh55P, marker="o", color='red', label='55P Perturbed', linestyle='--', markersize="5")
#plt.plot(t2, np.cos(Omega_R * t1[-1]) - np.cos(Omega_R * (t1[-1] + t2 - perturbation)),
#         label=r"$cos(\Omega_R*t_1)-cos(\Omega_R*(t_1+t_2))$ ", linestyle="-", marker="o",
#         markersize="0", color="black")

plt.legend()
plt.xlabel('Time [us]', fontsize=16)
plt.ylabel('Population', fontsize=16)
# plt.xlim([0, .18])
plt.show()


#plt.plot(x0, inc56S, marker="o", color='#800020', label='56S incoherent Perturbed', linestyle='--', markersize="5")

plt.plot(x0, un56S, marker="o", color='#008b8b', linestyle='', label="56S")
plt.plot(x0, un55P, marker="o", color='orange', linestyle='', label="55P")
#plt.plot(x0, unp, marker="o", color='#800020', label='Unperturbed', linestyle='')

#plt.plot(full_time, result_t1.expect[1], label=r"Unperturbed_Expect", linestyle="-", marker="o",
#         markersize="0", color="#008b8b")

plt.plot(full_time, -0.5*np.cos(Omega_R * full_time)+0.5, linestyle="-", marker="o",
         markersize="0", color="#008b8b")
plt.plot(full_time, 0.5*np.cos(Omega_R * full_time)+0.5, linestyle="-", marker="o",
         markersize="0", color="orange")


plt.plot(x0, inc56S, marker="o", color='blue', label='56S Perturbed', linestyle='--', markersize="5")
plt.plot(x0, inc55P, marker="o", color='red', label='55P Perturbed', linestyle='--', markersize="5")
plt.plot(x0, inc56P, marker="o", color='purple', label='55P Perturbed', linestyle='--', markersize="5")
#plt.plot(t2, np.cos(Omega_R * t1[-1]) - np.cos(Omega_R * (t1[-1] + t2 - perturbation)),
#         label=r"$cos(\Omega_R*t_1)-cos(\Omega_R*(t_1+t_2))$ ", linestyle="-", marker="o",
#         markersize="0", color="black")

plt.legend()
plt.xlabel('Time [us]', fontsize=16)
plt.ylabel('Population', fontsize=16)
# plt.xlim([0, .18])
plt.show()
#plt.plot(t2, 3/8 + np.cos(2 * Omega_R * t1[-1]) / 8 + np.cos(Omega_R * t1[-1]) / 2
#                 + np.cos(Omega_R * (t1[-1] + t2 - perturbation)),
#         label=r"$\frac{3}{8} + \frac{cos(2 \Omega_R  t_1)}{8} + \frac{cos(\Omega_R t_1}{2} + cos(\Omega_R (t_1 + t_2))$ ", linestyle="-", marker="o",
#         markersize="0", color="r")


#plt.plot(t2, result_t1t2.expect[1],
#         label=r"Unperturbed_Expect", linestyle="-", marker="o",
#         markersize="0", color="#008b8b")
# plt.plot(x0, un-inc, color='#008b8b', linestyle='-')
#plt.plot(t2, -np.imag(Commutatorlist),
#         label=r"$- i \langle \sigma_z(t_1) \sigma_z(t_2) - \sigma_z(t_2) \sigma_z(t_1)  \rangle$ ", color="b",
#         linestyle="", marker="o", markersize="1")
#plt.plot(t2, np.real(Anticommutatorlist),
#         label=r"$\langle \sigma_z(t_1)\sigma_z(t_2) + \sigma_z(t_2)\sigma_z(t_1) \rangle$", linestyle="", marker="o",
#         markersize="1", color="r")

# plt.plot(t2, np.imag(Anticommutatorlist), label="Im(Anticommutator)", color="r")

plt.plot(x0, un, marker="o", color='#008b8b', linestyle='', label=r"Unperturbed")
#plt.plot(x0, un55P, marker="o", color='orange', linestyle='', label="55P")
#plt.plot(x0, unp, marker="o", color='#800020', label='Unperturbed', linestyle='')

#plt.plot(full_time, result_t1.expect[1], label=r"Unperturbed_Expect", linestyle="-", marker="o",
#         markersize="0", color="#008b8b")

plt.plot(full_time, -np.cos(Omega_R * full_time), linestyle="-", marker="o",
         markersize="0", color="#008b8b")
#plt.plot(full_time, 0.5*np.cos(Omega_R * full_time)+0.5, linestyle="-", marker="o",
#         markersize="0", color="orange")


plt.plot(x0, coh, marker="o", color='blue', label='Perturbed', linestyle='--', markersize="5")
#plt.plot(x0, coh55P, marker="o", color='red', label='55P Perturbed', linestyle='--', markersize="5")
#plt.plot(t2, np.cos(Omega_R * t1[-1]) - np.cos(Omega_R * (t1[-1] + t2 - perturbation)),
#         label=r"$cos(\Omega_R*t_1)-cos(\Omega_R*(t_1+t_2))$ ", linestyle="-", marker="o",
plt.legend()
plt.xlabel('Time [us]', fontsize=16)
plt.ylabel(r'$\langle \sigma_z \rangle$', fontsize=16)
# plt.xlim([0, .18])
plt.show()



plt.plot(x0, un, marker="o", color='#008b8b', linestyle='', label=r"Unperturbed")
#plt.plot(x0, un55P, marker="o", color='orange', linestyle='', label="55P")
#plt.plot(x0, unp, marker="o", color='#800020', label='Unperturbed', linestyle='')

#plt.plot(full_time, result_t1.expect[1], label=r"Unperturbed_Expect", linestyle="-", marker="o",
#         markersize="0", color="#008b8b")

plt.plot(full_time, -np.cos(Omega_R * full_time), linestyle="-", marker="o",
         markersize="0", color="#008b8b")
#plt.plot(full_time, 0.5*np.cos(Omega_R * full_time)+0.5, linestyle="-", marker="o",
#         markersize="0", color="orange")


plt.plot(x0, inc, marker="o", color='blue', label='Perturbed', linestyle='--', markersize="5")
#plt.plot(x0, coh55P, marker="o", color='red', label='55P Perturbed', linestyle='--', markersize="5")
#plt.plot(t2, np.cos(Omega_R * t1[-1]) - np.cos(Omega_R * (t1[-1] + t2 - perturbation)),
#         label=r"$cos(\Omega_R*t_1)-cos(\Omega_R*(t_1+t_2))$ ", linestyle="-", marker="o",
plt.legend()
plt.xlabel('Time [us]', fontsize=16)
plt.ylabel(r'$\langle \sigma_z \rangle$', fontsize=16)
# plt.xlim([0, .18])
plt.show()



#plt.plot(x0, un, marker="o", color='#008b8b', linestyle='', label=r"Unperturbed")
#plt.plot(x0, un55P, marker="o", color='orange', linestyle='', label="55P")
#plt.plot(x0, unp, marker="o", color='#800020', label='Unperturbed', linestyle='')

#plt.plot(full_time, result_t1.expect[1], label=r"Unperturbed_Expect", linestyle="-", marker="o",
#         markersize="0", color="#008b8b")

#plt.plot(full_time, -np.cos(Omega_R * full_time), linestyle="-", marker="o",
#         markersize="0", color="#008b8b")
#plt.plot(full_time, 0.5*np.cos(Omega_R * full_time)+0.5, linestyle="-", marker="o",
#         markersize="0", color="orange")

full_time = np.linspace(0, 0.2, int((0.2)*sampling_rate))


plt.plot(x0, comm, marker="o", color='orange', label='Coherent Perturbation', linestyle='', markersize="5")
plt.plot(full_time, -np.sin(Omega_R * full_time+0.005)*0.55, linestyle="-", marker="o",
         markersize="0", color="orange")

plt.plot(x0, anti, marker="o", color='blue', label='Incoherent Perturbation', linestyle='--', markersize="5")
plt.plot(full_time, np.cos(Omega_R * full_time+0.5)*0.25, linestyle="-", marker="o",
         markersize="0", color="blue")
#plt.plot(x0, coh55P, marker="o", color='red', label='55P Perturbed', linestyle='--', markersize="5")
#plt.plot(t2, np.cos(Omega_R * t1[-1]) - np.cos(Omega_R * (t1[-1] + t2 - perturbation)),
#         label=r"$cos(\Omega_R*t_1)-cos(\Omega_R*(t_1+t_2))$ ", linestyle="-", marker="o",
plt.legend()
plt.xlabel('Time [us]', fontsize=16)
plt.ylabel(r'$\langle \sigma_z \rangle - \langle \sigma_z \rangle_{pert}$', fontsize=16)
# plt.xlim([0, .18])
plt.show()
'''