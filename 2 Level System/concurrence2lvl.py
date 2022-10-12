from qutip import *
import numpy as np
from Atoms import *
from Driving import *
import matplotlib.pyplot as plt
from scipy import integrate

opts = Options(store_states=True, store_final_state=True)  # , nsteps=50000)

#plt.rcParams.update({
#  "text.usetex": True,
#})

N = 2

Exps = [MagnetizationX(N), MagnetizationY(N), MagnetizationZ(N), sigmaz(0, N), sigmaz(0, N), sigmaz(N - 1, N),
        upup(0, N), upup(1, N), upup(N - 2, N), upup(N - 1, N),
        sigmap(0, N), sigmam(0, N), downdown(0, N)]

Omega_R = 0 * np.pi  # MHz

#gamma = 0.01 * np.pi  # MHz

J = 1 * np.pi / N  # MHz

bath = "markovian"

endtime = 2

timesteps = 200

pertubation_length = endtime / 1

perturb_times = np.linspace(0, pertubation_length, timesteps)

gamma = 10

omega = 0

init_state = bellstate(0, 1, N)
init_state = productstateZ(0, 1, N)
#init_state = productstateX(0, 1, N)

noise1 = noisy_func(gamma, perturb_times, omega, bath)
noise2 = noisy_func(gamma, perturb_times, omega, bath)
noise3 = noisy_func(gamma, perturb_times, omega, bath)
noise4 = noisy_func(gamma, perturb_times, omega, bath)

S11 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise1)
S12 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise1))
S21 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise2)
S22 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise2))
S31 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise3)
S32 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise3))
S41 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise4)
S42 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise4))

if N==1:

    result2 = mesolve([H0(omega, J, N), [-Omega_R * sigmap(0, N), S11], [-Omega_R * sigmam(0, N), S12],
                       ], init_state,
                      perturb_times, e_ops=Exps, options=opts)

else:
    result2 = mesolve([H0(omega, J, N), [-Omega_R * sigmap(0, N), S11], [-Omega_R * sigmam(1, N), S12],
                                        [-Omega_R * sigmap(1, N), S21], [-Omega_R * sigmam(1, N), S22]#,
                                        #[-Omega_R * sigmap(1, 2, N), S31], [-Omega_R * sigmam(1, 2, N), S32],
                                        #[-Omega_R * sigmap(1, 3, N), S41], [-Omega_R * sigmam(1, 3, N), S42],
                       ], init_state,
                      perturb_times, e_ops=Exps, options=opts)

concmean = []
for t in range(0, timesteps):
    concmean.append(concurrence(result2.states[t]))

plt.plot(perturb_times, concmean)
plt.show()

print(concmean)

#concmean = []

# opts = Options(store_states=True, store_final_state=True, rhs_reuse=True)
states2 = np.array(result2.states[timesteps - 1])
expect2 = np.array(result2.expect[:])
ancilla_overlap = []
Smean = np.zeros_like(perturb_times) + 1j * np.zeros_like(perturb_times)
Pmean = 0

i = 1

while i < 20:  # averages + int(2 * gamma):
    print(i)
    i += 1
    noise1 = noisy_func(gamma, perturb_times, omega, bath)
    noise2 = noisy_func(gamma, perturb_times, omega, bath)
    noise3 = noisy_func(gamma, perturb_times, omega, bath)
    noise4 = noisy_func(gamma, perturb_times, omega, bath)

    S11 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise1)
    S12 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise1))
    S21 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise2)
    S22 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise2))
    S31 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise3)
    S32 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise3))
    S41 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise4)
    S42 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise4))

    result2 = mesolve([H0(omega, J, N), [Omega_R * sigmap(0, N), S11], [Omega_R * sigmam(0, N), S12],
                                        [Omega_R * sigmap(1, N), S21], [Omega_R * sigmam(1, N), S22]#,
                                        #[Omega_R * sigmap(1, 2, N), S31], [Omega_R * sigmam(1, 2, N), S32],
                                        #[Omega_R * sigmap(1, 3, N), S41], [Omega_R * sigmam(1, 3, N), S42],
                       ], init_state,
                      perturb_times, e_ops=Exps, options=opts)

    states2 += np.array(result2.states[timesteps - 1])
    expect2 += np.array(result2.expect[:])

    #Smean += np.abs(np.fft.fft(Omega_R * noise ** 2)) # /2/np.pi/timesteps
    #Pmean += np.abs(np.sum(Omega_R * noise ** 2)) # /timesteps
    for t in range(0, timesteps):
        concmean[t] += concurrence(result2.states[t])

# noisy_data2 = noisy_func(gamma, perturb_times, omega, bath)
# S2 = Cubic_Spline(perturb_times[0], perturb_times[-1], noisy_data2)

states2 = states2 / i
expect2 = expect2 / i
Smean = Smean / i
Pmean = Pmean / i
concmean = np.array(concmean) / i


print(concmean)

plt.plot(perturb_times, concmean)
plt.show()


# print(Qobj(states2))
# print((expect2[5]+expect2[8]).mean())
density_matrix = Qobj([[expect2[5][timesteps - 1], expect2[6][timesteps - 1]],
                       [expect2[7][timesteps - 1], expect2[8][timesteps - 1]]])
# print(density_matrix)
# result3 = mesolve(H0(omega, J, N), Qobj(states2), t2, [], e_ops=Exps, options=opts)