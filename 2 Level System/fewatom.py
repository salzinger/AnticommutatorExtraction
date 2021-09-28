from Atoms import *
from Driving import *
import matplotlib.pyplot as plt
import array

N = 2

omega = 2 * np.pi * 20 * 10 ** (-10)  # MHz

Omega_R = 0 * np.pi * 20 * 10 ** (-1)  # MHz

gamma = 0 * np.pi * 15.0  # MHz

J = np.pi * 20 * 10 ** (-1)  # MHz

bath="markovian"

endtime = 1

timesteps= 1000

pertubation_length = endtime / 1

t1 = np.linspace(0, endtime, timesteps)
t2 = np.linspace(0, endtime, timesteps)

perturb_times = np.linspace(0, pertubation_length, timesteps)

Exps = [MagnetizationX(N), MagnetizationZ(N), MagnetizationY(N), sigmaz(0, N), sigmaz(N - 1, N), upup(0, N),
        sigmap(0, N), sigmam(0, N), downdown(0, N)]

opts = Options(store_states=True, store_final_state=True)  # , nsteps=50000)
init_state = productstateZ(0, 1, N)

pertubation_length = endtime / 1
perturb_times = np.linspace(0, pertubation_length, timesteps)

S1 = Cubic_Spline(perturb_times[0], perturb_times[-1],
                  noisy_func(gamma, perturb_times, omega, bath))
S2 = Cubic_Spline(perturb_times[0], perturb_times[-1],
                  np.conj(noisy_func(gamma, perturb_times, omega, bath)))
# S = Cubic_Spline(perturb_times[0], perturb_times[-1],
#                 data[0:32000]/0.4)

print('H0...')
print(H0(omega, J, N))
print('H1...')
print(H1(Omega_R, N))
print('H2...')
print(H2(Omega_R, N))

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

i=1

while i < 1:  # averages + int(2 * gamma):
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

fig, ax = plt.subplots(2, 2, figsize=(10, 10))


ax[0, 0].plot(perturb_times, np.real(expect2[0]), color='#85bb65',label="mag_x")
ax[0, 0].plot(perturb_times, np.real(expect2[1]), color='black',label="mag_z")
ax[0, 0].plot(perturb_times, np.real(expect2[5]), color='red', label="upup 1st_atom")
ax[0, 0].plot(perturb_times, np.real(expect2[3]), color='blue', label="sigma_z 1st_atom")
ax[0, 0].plot(perturb_times, np.real(expect2[4]), color='green', label="sigma_z 2nd_atom")
ax[0, 0].set_xlabel('Time [us]', fontsize=16)
ax[0, 0].set_ylabel('Expectation Value', fontsize=16)
# ax[1, 0].plot(perturb_times, np.real(expect_me[1]), label="sigma_z, ME with sqrt(gamma)*L")
ax[0, 0].legend(loc="lower center")
#ax[0, 0].set_ylim([-0.501, -0.499])
plt.show()
