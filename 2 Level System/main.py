from qutip import *
import numpy as np
from Atoms import *
from Driving import *
import matplotlib.pyplot as plt
import array

# def convert(s):
# The function that converts the string to float
#    s = s.strip().replace(',', '.')
#    return float(s)


data = array.array('d')  # an array of type double (float of 64 bits)

# with open("noise_gamma_3.csv", 'r') as f:
#    for l in f:
#        strnumbers = l.split('\t')
#        data.extend( (convert(s) for s in strnumbers if s!='') )
# A generator expression here.

data = np.loadtxt('Forward3MHzcsv.txt')
# print(data)


N = 1

omega = 2 * np.pi * 21 * 10 ** 3  # MHz

Omega_R = 2 * np.pi * 25.7 * 10 ** 0  # MHz

gamma = 2 * np.pi * 15.0  # MHz

J = 0  # MHz

averages = 150

sampling_rate = 2 * np.pi * 64 * 10 ** 0  # MHz
endtime = 0.2
timesteps = int(endtime * sampling_rate)
timesteps = 2 * len(data)

bath = 'Forward3MHzcsv.txt'

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


for o in np.logspace(np.log(15 * Omega_R), np.log(100 * Omega_R), num=1, base=np.e):
    # print("omega: ", omega)
    for s in np.logspace(np.log(5 * omega), np.log(10 * omega), num=1, base=np.e):
        # print("sampling: ", sampling_rate)
        init_state = productstateZ(0, 0, N)
        # timesteps = int(endtime * sampling_rate)
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

            timesteps = 1 * len(data)
            endtime = 0.1
            pertubation_length = endtime / 1
            perturb_times = np.linspace(0, pertubation_length, timesteps)

            S1 = Cubic_Spline(perturb_times[0], perturb_times[-1], noisy_func(gamma, perturb_times, omega, bath))
            S2 = Cubic_Spline(perturb_times[0], perturb_times[-1],
                              np.conj(noisy_func(gamma, perturb_times, omega, bath)))
            # S = Cubic_Spline(perturb_times[0], perturb_times[-1],
            #                 data[0:32000]/0.4)

            # print('H0...')
            # print(H0(omega, J, N))
            # print('H1...')
            # print(H1(Omega_R, N))
            # print('H2...')
            # print(H2(Omega_R, N))

            result2 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S1], [H2(Omega_R, N), S2]], init_state,
                              perturb_times, e_ops=Exps, options=opts)
            concmean = []
            # for t in range(0, timesteps):
            # concmean.append(concurrence(result2.states[t]))

            # opts = Options(store_states=True, store_final_state=True, rhs_reuse=True)
            states2 = np.array(result2.states[timesteps - 1])
            expect2 = np.array(result2.expect[:])
            ancilla_overlap = []
            Smean = np.zeros_like(perturb_times) + 1j * np.zeros_like(perturb_times)
            Pmean = 0

            while i < 20:  # averages + int(2 * gamma):
                # print(i)
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

            #################### SPECTRA ######################################## 11111111111111111111111111111111111111111111111111111111111

            fig, ax = plt.subplots(2, 2, figsize=(10, 10))

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

            gamma3 = average_psd(3, omega, samples, sample_time, 2)
            gamma10 = average_psd(10, omega, samples, sample_time, 2)
            gamma30 = average_psd(30, omega, samples, sample_time, 2)

            ax[0, 0].plot(-gamma3[0], gamma3[1], linestyle='',
                          marker='^', markersize='4', linewidth=0.55, label="PSD $\gamma=3$ MHz", color="#85bb65")

            # ax[0, 0].plot(f_real, Pxx_real, linestyle='-',
            #           marker='s', markersize='6', linewidth=0.55, label="PSD $\gamma=3$ MHz Exp", color="#85bb65")

            ax[0, 0].plot(gamma3[0], 0.5 * lorentzian(gamma3[0], 1, omega / (2 * np.pi), 3), linestyle='-',
                          marker='^', markersize='0', linewidth=0.55, label="Lorentzian $\gamma=3$ MHz",
                          color="#85bb65")
            ax[0, 0].plot(-gamma10[0], gamma10[1], linestyle='',
                          marker='v', markersize='4', linewidth=0.55, label="PSD $\gamma=10$ MHz", color="#CC7722")
            ax[0, 0].plot(gamma10[0], 0.5 * lorentzian(gamma10[0], 1, omega / (2 * np.pi), 10), linestyle='-',
                          marker='v', markersize='0', linewidth=0.55, label="Lorentzian $\gamma=10$ MHz",
                          color="#CC7722")
            ax[0, 0].plot(-gamma30[0], gamma30[1], linestyle='',
                          marker='s', markersize='4', linewidth=0.55, label="PSD $\gamma=30$ MHz", color="#800020")
            ax[0, 0].plot(gamma30[0], 0.5 * lorentzian(gamma30[0], 1, omega / (2 * np.pi), 30), linestyle='-',
                          marker='o', markersize='0', linewidth=0.55, label="Lorentzian $\gamma=30$ MHz",
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

            ax[0, 0].set_xlim([20980, 21020])
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

            ax[0, 0].legend(loc="upper right")

            ax[0, 0].set_xlabel('f [MHz]', fontsize=16)
            ax[0, 0].set_ylabel('PSD [V**2/Hz]', fontsize=16)

            #################### END OF SPECTRA ######################################## 1111111111111111111111111111














            #################### SINGLE TRAJECTORY ######################################## 222222222222222222222222222

            timesteps = 2 * len(data)
            endtime = 0.2
            pertubation_length = endtime / 1
            perturb_times = np.linspace(0, pertubation_length, timesteps)

            with open('m.txt') as f:
                linesm = f.readlines()
            with open('mf.txt') as f:
                linesmf = f.readlines()
            with open('mfxy.txt') as f:
                linesmfxy = f.readlines()
            with open('mxy.txt') as f:
                linesmxy = f.readlines()

            x = []
            y = []
            xy = []
            for element in range(1, 21):
                x.append(float(linesmf[element][0:5]))
                y.append(float(linesmf[element][11:18]))
                xy.append(float(linesmfxy[element][11:18]))
            for element in range(1, 21):
                x.append(float(linesm[element][0:5]))
                y.append(float(linesm[element][11:18]))
                xy.append(float(linesmxy[element][11:18]))

            ax[1, 0].plot(perturb_times, np.real(expect_single[1]), color='#85bb65')
            ax[1, 0].plot(x, y, label=r"$\langle \sigma_z \rangle$", linestyle="", markersize="5", marker="o",
                          color='#85bb65')
            ax[1, 0].plot(x, xy, label=r"$\sqrt{\langle \sigma_x \rangle^2 + \langle \sigma_y \rangle^2}$",
                          linestyle="",
                          markersize="5", marker="v", color='black')
            # ax[1, 0].plot(perturb_times, np.real(expect2[0]), label="sigma_x, Time Dependent Hamiltonian")
            # ax[1, 0].plot(perturb_times, np.real(expect2[2]), label="sigma_y, Time Dependent Hamiltonian")
            ax[1, 0].plot(perturb_times, np.sqrt(expect_single[2] ** 2 + expect_single[0] ** 2), color="black",
                          linestyle="--")
            # ax[1, 0].plot(perturb_times, concmean, label="overlap-bell-basis")
            # ax[1, 0].plot(perturb_times, np.exp(- perturb_times * gamma), color="orange", label="exp(- gamma * t)")
            # ax[1, 0].plot(perturb_times, -np.exp(- perturb_times * gamma), color="orange")
            ax[1, 0].set_xlabel('Time [us]', fontsize=16)
            ax[1, 0].set_ylabel('Magnetization', fontsize=16)
            # ax[1, 0].plot(perturb_times, np.real(expect_me[1]), label="sigma_z, ME with sqrt(gamma)*L")
            ax[1, 0].legend(loc="lower center")

            #################### END OF SINGLE TRAJECTORY ######################################## 2222222222222222222














            ################### PHASE WALKS ############################################ 33333333333333333333333333333333

            timesteps = len(data)
            endtime = 0.1
            pertubation_length = endtime / 1
            perturb_times = np.linspace(0, pertubation_length, timesteps)

            # Total time.
            T = perturb_times[-1]
            # Number of steps.
            Nsteps = len(perturb_times)
            # Time step size
            dt = T / Nsteps
            # Create an empty array to store the realizations.
            x = np.empty((averages, Nsteps + 1))
            # Initial values of x.
            x[:, 0] = 0

            for gamma in [3, 10, 30]:

                phase_noise = brownian(x[:, 0], Nsteps, dt, np.sqrt(gamma), out=x[:, 1:])

                t = np.linspace(0.0, Nsteps * dt, Nsteps)

                for k in range(0, int(averages / 10)):
                    if gamma == 3:
                        ax[0, 1].plot(t, phase_noise[k], color='#85bb65', linewidth=0.1)
                    elif gamma == 10:
                        ax[0, 1].plot(t, phase_noise[k], color='#CC7722', linewidth=0.00)
                    elif gamma == 30:
                        ax[0, 1].plot(t, phase_noise[k], color='black', linewidth=0.00)

                if gamma == 3:
                    # ax[0, 1].plot(t, np.mean(phase_noise, axis=0), color='green', linestyle='', linewidth=1.0, marker="o", markersize="0.01")

                    ax[0, 1].plot(t, np.sqrt(np.var(phase_noise, axis=0)), color='#85bb65', linestyle='-',
                                  linewidth=1.0,
                                  label='$\gamma = 3$ MHz')
                    ax[0, 1].plot(t, np.sqrt(gamma * t), color='#85bb65', linestyle='--', linewidth=1.0)
                    # ,label='Expected Standard Deviation = sqrt(gamma * t)')
                    ax[0, 1].plot(t, -np.sqrt(gamma * t), color='#85bb65', linestyle='--', linewidth=1.0)

                if gamma == 10:
                    # ax[0, 1].plot(t, np.mean(phase_noise, axis=0), color='orange', linestyle='', linewidth=1.0, marker="o", markersize="0.01")

                    ax[0, 1].plot(t, np.sqrt(np.var(phase_noise, axis=0)), color='#CC7722', linestyle='-',
                                  linewidth=1.0,
                                  label='$\gamma = 10$ MHz')
                    ax[0, 1].plot(t, np.sqrt(gamma * t), color='#CC7722', linestyle='--', linewidth=1.0)
                    # ,label='Expected Standard Deviation = sqrt(gamma * t)')
                    ax[0, 1].plot(t, -np.sqrt(gamma * t), color='#CC7722', linestyle='--', linewidth=1.0)

                if gamma == 30:
                    # ax[0, 1].plot(t, np.mean(phase_noise, axis=0), color='red', linestyle='', linewidth=1.0, marker="o", markersize="0.01")
                    ax[0, 1].plot(t, np.sqrt(np.var(phase_noise, axis=0)), color='#800020', linestyle='-',
                                  linewidth=1.0,
                                  label='$\gamma = 30$ MHz')
                    ax[0, 1].plot(t, np.sqrt(gamma * t), color='#800020', linestyle='--', linewidth=1.0)
                    # ,label='Expected Standard Deviation = sqrt(gamma * t)')
                    ax[0, 1].plot(t, -np.sqrt(gamma * t), color='#800020', linestyle='--', linewidth=1.0)

            ax[0, 1].set_ylim([-1.1 * np.sqrt(30 * T), 1.1 * np.sqrt(30 * T)])
            ax[0, 1].set_xlim([0, 0.1])
            ax[0, 1].set_xlabel('Time [us]', fontsize=16)
            ax[0, 1].set_ylabel('Phase [$\pi$]', fontsize=16)
            ax[0, 1].legend(loc="lower left")

            ################### END OF PHASE WALKS ############################################ 3333333333333333333333














            ################### TD VS MASTER EQUATION ############################################ 444444444444444444

            with open('m0.txt') as f:
                linesm0 = f.readlines()
            with open('m3.txt') as f:
                linesm3 = f.readlines()
            with open('m10.txt') as f:
                linesm10 = f.readlines()
            with open('m30.txt') as f:
                linesm30 = f.readlines()

            # print(linesmf)
            # print(linesm)
            x0 = []
            y0 = []
            y3 = []
            y10 = []
            y30 = []

            for element in range(1, 22):
                x0.append(float(linesm0[element][0:5]))
                y0.append(float(linesm0[element][8:18]))
                y3.append(float(linesm3[element][8:18]))
                y10.append(float(linesm10[element][8:18]))
                y30.append(float(linesm30[element][8:18]))

            t1 = np.linspace(0, endtime, int(timesteps / 10))

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

            ax[1, 1].plot(x0, y0, marker="o", color='#008b8b', label='$\gamma = 0$ MHz', linestyle='')
            ax[1, 1].plot(t1, np.real(m0[1]), color='#008b8b', linestyle='-')

            ax[1, 1].plot(perturb_times, np.real(expect2[1]), color='#85bb65', label="Time Dependant"),

            ax[1, 1].plot(x0, y3, marker="^", color='#85bb65', label='$\gamma = 3$ MHz', linestyle='')
            ax[1, 1].plot(t1, result_m3.expect[1], color='#85bb65', linestyle='-')
            ax[1, 1].plot(x0, y10, marker="v", color='#CC7722', label='$\gamma = 10$ MHz', linestyle='')
            ax[1, 1].plot(t1, result_m10.expect[1], color='#CC7722', linestyle='-')
            ax[1, 1].plot(x0, y30, marker="s", color='#800020', label='$\gamma = 30$ MHz', linestyle='')
            ax[1, 1].plot(t1, result_m30.expect[1], color='#800020', linestyle='-')
            # ax[1, 1].plot(t, -np.sqrt(gamma * t), color='black', linestyle='--', linewidth=2.0)
            # ax[1, 1].set_ylim([-1.2 * np.sqrt(gamma * T), 1.2 * np.sqrt(gamma * T)])
            ax[1, 1].set_xlabel('Time [us]', fontsize=16)
            ax[1, 1].set_ylabel('Magnetization', fontsize=16)
            ax[1, 1].legend(loc="upper center")

            fig.tight_layout()
            plt.show()
            plt.savefig("bath" + bath + ", omega =  %.2f, sampling =  %.2f,gamma = %.2f.png" % (
                omega, sampling_rate, gamma))  # and BW %.2f.pdf" % (noise_amplitude, bandwidth))

            ################### END OF TD VS MASTER EQUATION ############################################4444444444444444













################### COMMUTATOR / ANTICOMMUTATOR ############################################ 5555555555555555555555555

opts = Options(store_states=True, store_final_state=True)

Omega_R = 2 * np.pi * 10 * 10 ** 0  # MHz

Commutatorlist = []
Anticommutatorlist = []

# t1 = np.linspace(0, 0.08, int(timesteps / 10))

t1 = np.linspace(0, 0.08, int(timesteps / 10))

t2 = np.linspace(0.08, 0.2 + 0.08, int(timesteps / 10))

result_t1 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]],
                    productstateZ(0, N - 1, N), t1, [], Exps, options=opts)

result_t1t2 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]],
                      result_t1.states[len(t1) - 1], t2, [], Exps, options=opts)

Perturb_incoherent = identity(2) - MagnetizationZ(N)
Perturb_coherent = MagnetizationZ(N)
Measure = MagnetizationZ(N)

result_AB = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]],
                    Perturb_incoherent * result_t1.states[len(t1) - 1], t2, [], Exps, options=opts)

result_AB_comm = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]],
                         Perturb_coherent * result_t1.states[len(t1) - 1], t2, [], Exps, options=opts)

for t in range(0, len(t2)):
    prod_AB = result_t1t2.states[t - 1].dag() * Measure * result_AB.states[t - 1]

    prod_AB_comm = result_t1t2.states[t - 1].dag() * Measure * result_AB_comm.states[t - 1]

    prod_BA = result_AB.states[t - 1].dag() * Measure * result_t1t2.states[t - 1]

    prod_BA_comm = result_AB_comm.states[t - 1].dag() * Measure * result_t1t2.states[t - 1]

    Commutator = prod_AB_comm - prod_BA_comm

    AntiCommutator = prod_AB + prod_BA

    Commutatorlist.append(Commutator[0][0][0])
    Anticommutatorlist.append(AntiCommutator[0][0][0])
    # print('Commutator:', 1j * Commutator[0][0])
    # print('AntiCommutator: ', AntiCommutator[0][0])

with open('56S_unperturb.txt') as f:
    linesunp = f.readlines()
with open('56S_unperturbed.txt') as f:
    linesun = f.readlines()
with open('56S_coherent_perturb.txt') as f:
    linescoh = f.readlines()
with open('56S_incoherent_perturb.txt') as f:
    linesinc = f.readlines()

x0 = []
un = []
unp = []
coh = []
inc = []

for element in range(1, 22):
    x0.append(float(linesun[element][0:5]) + 0.08)
    un.append(float(linesun[element][8:18]) - 0.5)
    coh.append(float(linesun[element][8:18]) - float(linescoh[element][8:18]))
    inc.append(float(linesun[element][8:18]) - float(linesinc[element][8:18]))
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

plt.plot(x0, un, marker="o", color='#008b8b', label='Unperturbed', linestyle='')
plt.plot(x0, coh, marker="o", color='b', label='Commutator', linestyle='', markersize="5")
plt.plot(x0, inc, marker="o", color='r', label='Anti-Commutator', linestyle='', markersize="5")
plt.plot(t1, result_t1.expect[1], label=r"Unperturbed_Expect", linestyle="-", marker="o",
         markersize="0", color="#008b8b")
plt.plot(t2, np.cos(Omega_R * t1[-1]) - np.cos(Omega_R * (t1[-1] + t2)),
         label=r"$cos(\Omega_R*t_1)-cos(\Omega_R*(t_1+t_2))$ ", linestyle="-", marker="o",
         markersize="0", color="black")
plt.plot(t2, result_t1t2.expect[1],
         label=r"Unperturbed_Expect", linestyle="-", marker="o",
         markersize="0", color="#008b8b")
# plt.plot(x0, un-inc, color='#008b8b', linestyle='-')
plt.plot(t2, -np.imag(Commutatorlist),
         label=r"$- i \langle \sigma_z(t_1) \sigma_z(t_2) - \sigma_z(t_2) \sigma_z(t_1)  \rangle$ ", color="b",
         linestyle="", marker="o", markersize="1")
plt.plot(t2, np.real(Anticommutatorlist),
         label=r"$\langle \sigma_z(t_1)\sigma_z(t_2) + \sigma_z(t_2)\sigma_z(t_1) \rangle$", linestyle="", marker="o",
         markersize="1", color="r")

# plt.plot(t2, np.imag(Anticommutatorlist), label="Im(Anticommutator)", color="r")
plt.legend()
plt.xlabel('$t_2 - t_1$')
# plt.xlim([0, .18])
plt.show()
