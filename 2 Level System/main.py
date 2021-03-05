from qutip import *
import numpy as np
from Atoms import *
from Driving import *
import matplotlib.pyplot as plt

N = 1

omega = 2 * np.pi * 35 * 10 ** 0  # MHz

Omega_R = 2 * np.pi * 5.6  # MHz

J = 0  # MHz

bandwidth = 10  # MHz

sampling_rate = 2 * np.pi * 65 * 10 ** 0  # MHz
endtime = 1
timesteps = int(endtime * sampling_rate)

gamma1 = 0  # MHz

pertubation_length = endtime / 1

t1 = np.linspace(0, endtime, timesteps)
t2 = np.linspace(0, endtime, timesteps)

noise_amplitude = 0.000

perturb_times = np.linspace(0, pertubation_length, timesteps)
random_phase = noise_amplitude * np.random.randn(perturb_times.shape[0])

S1 = Cubic_Spline(perturb_times[0], perturb_times[-1], noisy_func(noise_amplitude, perturb_times, omega, bandwidth))

Exps = [MagnetizationX(N), MagnetizationZ(N), MagnetizationY(N), sigmaz(0, N), sigmaz(N - 1, N), upup(0, N),
        sigmap(0, N), sigmam(0, N), downdown(0, N)]

opts = Options(store_states=True, store_final_state=True)

for noise_amplitude in np.linspace(0, 0.05, num=6):
    print("Noise Amplitude", noise_amplitude)
    for bandwidth in np.linspace(1, 9, num=1):
        #print("Bandwidth", bandwidth)
        i = 1
        # random_phase = noise_amplitude * np.random.randn(perturb_times.shape[0])
        S = Cubic_Spline(perturb_times[0], perturb_times[-1], brownian_func(noise_amplitude, perturb_times, omega))

        # print('H0...')
        # print(H0(omega, J, N))
        # print('H1...')
        # print(H1(Omega_R, N))
        # print('H2...')
        # print(H2(Omega_R, N))

        result2 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]], productstateZ(0, 0, N),
                          perturb_times, e_ops=Exps, options=opts)

        # opts = Options(store_states=True, store_final_state=True, rhs_reuse=True)
        states2 = np.array(result2.states[timesteps - 1])
        expect2 = np.array(result2.expect[:])
        ancilla_overlap = []

        while i < 200:
            #print(i)
            i += 1
            # random_phase = noise_amplitude * np.random.randn(perturb_times.shape[0])
            S = Cubic_Spline(perturb_times[0], perturb_times[-1],
                             brownian_func(noise_amplitude, perturb_times, omega))

            result2 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]], productstateZ(0, 0, N),
                              perturb_times, e_ops=Exps, options=opts)

            states2 += np.array(result2.states[timesteps - 1])
            expect2 += np.array(result2.expect[:])

        # func2 = lambda t: 0.5j * np.exp(-1j * t * 1 * omega) - 0.5j * np.exp(1j * t * 1 * omega)
        # noisy_func2 = lambda t: func2(t + random_phase)
        noisy_data2 = brownian_func(noise_amplitude, perturb_times, omega)
        S2 = Cubic_Spline(perturb_times[0], perturb_times[-1], noisy_data2)

        states2 = states2 / i
        expect2 = expect2 / i
        # print(Qobj(states2))
        # print((expect2[5]+expect2[8]).mean())
        density_matrix = Qobj([[expect2[5][timesteps - 1], expect2[6][timesteps - 1]],
                               [expect2[7][timesteps - 1], expect2[8][timesteps - 1]]])
        # print(density_matrix)
        result3 = mesolve(H0(omega, J, N), Qobj(states2), t2, [], e_ops=Exps, options=opts)

        # print('Initial state ....')
        # print(productstateZ(0, 0, N))
        # print(productstateZ(0, 0, N).dag()*sigmaz(1, N)*productstateZ(0, 0, N))

        # print('Commutator:', 1j * Commutator[0][0])
        # print('AntiCommutator: ', AntiCommutator[0][0])
        # print(np.correlate(S2(perturb_times), S2(perturb_times), "valid"))
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        freq = np.fft.fftfreq(perturb_times.shape[-1], d=1 / sampling_rate)
        fourier = np.abs(np.fft.fft(brownian_func(noise_amplitude, perturb_times, omega)))

        max_neg = len(perturb_times) - np.argmax(fourier[0: int(len(perturb_times) / 2)])
        max_pos = int(len(perturb_times) / 2) - np.argmax(fourier[int(len(perturb_times) / 2): len(perturb_times)])

        ax[0, 0].plot(freq, fourier, linestyle='',
                      marker='o', markersize='2', linewidth=0.0)
        ax[0, 0].plot(freq, lorentzian(freq, fourier[max_pos], max_pos, 0.1), linestyle='-',
                      marker='o', markersize='0', linewidth=1.0)
        ax[0, 0].plot(freq, lorentzian(freq, fourier[max_neg], max_neg, 0.1), linestyle='-',
                      marker='o', markersize='0', linewidth=1.0)

        ax[0, 0].set_xlabel('F [MHz]')
        ax[0, 0].set_ylabel('Coupling Amplitude')

        ax[0, 1].plot(perturb_times, np.real(S2(perturb_times)), linestyle='-', marker='o', markersize='0', linewidth=1.0)
        ax[1, 1].set_xlim([0, 0.1])
        ax[0, 1].set_xlabel('Time [us]')

        ax[1, 0].plot(perturb_times, np.real(expect2[1]), label="MagnetizationZ")
        ax[1, 0].set_xlabel('Time Dependent Perturbation [us]')

        ax[1, 1].plot(perturb_times, np.real(S2(perturb_times)), linestyle='--', marker='o', markersize='3', linewidth=1.0)
        ax[1, 1].set_xlabel('Time [us]')
        ax[1, 1].set_xlim([0, 0.01])
        fig.tight_layout()
        # plt.show()
        plt.savefig("Phase noise at RMS %.2f and BW %.2f.pdf" % (noise_amplitude, bandwidth))
