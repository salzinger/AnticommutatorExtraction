import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from qutip.solver import Options, Result, config, _solver_safety_check

opts = Options(store_states=True, store_final_state=True, ntraj=200)


def twobasis():
    return np.array([basis(2, 0), basis(2, 1)], dtype=object)


def productstateZ(up_atom, down_atom, N):
    up, down = twobasis()
    pbasis = np.full(N, down)
    pbasis[up_atom] = down
    pbasis[down_atom] = up
    return tensor(pbasis)


def productstateX(m, j, N):
    up, down = twobasis()
    pbasis = np.full(N, down)
    pbasis[m] = (up + down).unit()
    pbasis[j] = (up + down).unit()
    return tensor(pbasis)


def sigmap(m, N):
    up, down = twobasis()
    oplist = np.full(N, identity(2))
    oplist[m] = up * down.dag()
    return tensor(oplist)


def sigmam(m, N):
    up, down = twobasis()
    oplist = np.full(N, identity(2))
    oplist[m] = down * up.dag()
    return tensor(oplist)


def sigmaz(j, N):
    oplist = np.full(N, identity(2))
    oplist[j] = Qobj([[1, 0], [0, -1]])
    return tensor(oplist)


def sigmax(j, N):
    oplist = np.full(N, identity(2))
    oplist[j] = Qobj([[0, 1], [1, 0]])
    return tensor(oplist)


def sigmay(j, N):
    oplist = np.full(N, identity(2))
    oplist[j] = Qobj([[0, -1j], [1j, 0]])
    return tensor(oplist)


def MagnetizationZ(N):
    sum = 0
    for j in range(0, N):
        sum += sigmaz(j, N)
    return sum / N


def MagnetizationX(N):
    sum = 0
    for j in range(0, N):
        sum += sigmax(j, N)
    return sum / N


def MagnetizationY(N):
    sum = 0
    for j in range(0, N):
        sum += sigmax(j, N)
    return sum / N


def H0(omega, N):
    H = 0
    for j in range(0, N):
        H += 1 * omega / 2 * sigmaz(j, N)
    return H


def H1(Omega_R, N):
    H = 0
    for j in range(0, N):
        H += Omega_R * (sigmap(j, N))
    return H


def H2(Omega_R, N):
    H = 0
    for j in range(0, N):
        H += Omega_R * (sigmam(j, N))
    return H


N = 1

j = 0

omega = 2. * np.pi * 30

Omega_R = 2. * np.pi * 1

timesteps = 400
endtime = 1

t1 = np.linspace(0, endtime, timesteps)
t2 = np.linspace(0, endtime, timesteps)

t = np.linspace(0, endtime / 2, timesteps)
random_phase = 0.005 * np.random.randn(t.shape[0])

func1 = lambda t: np.exp(-1j * t * omega)
# noisy_data1 = func1(t)
noisy_func1 = lambda t: func1(t + random_phase)
noisy_data1 = noisy_func1(t)
S1 = Cubic_Spline(t[0], t[-1], noisy_data1)

func2 = lambda t: np.exp(1j * t * omega)
# noisy_data2 = func2(t)
noisy_func2 = lambda t: func2(t + random_phase)
noisy_data2 = noisy_func2(t)
S2 = Cubic_Spline(t[0], t[-1], noisy_data2)

Exps = [MagnetizationX(N), MagnetizationZ(N), MagnetizationY(N)]

Perturb = sigmax(0, 1)
Measure = sigmay(0, 1)

opts = Options(store_states=True, store_final_state=True, ntraj=200)

result_t1 = mesolve(H0(omega, N), productstateZ(0, 0, N), t1, [], Exps, options=opts,
                    progress_bar=True)

result_t1t2 = mesolve(H0(omega, N), result_t1.states[timesteps - 1], t2, [],
                      Exps, options=opts,
                      progress_bar=True)

result_AB = mesolve(H0(omega, N), Perturb * result_t1.states[timesteps - 1], t2, [],
                    Exps, options=opts,
                    progress_bar=True)

prod_AB = result_t1t2.states[timesteps - 1].dag() * Measure * result_AB.states[timesteps - 1]

prod_BA = result_AB.states[timesteps - 1].dag() * Measure * result_t1t2.states[timesteps - 1]

Commutator = prod_AB - prod_BA

AntiCommutator = prod_AB + prod_BA

result2 = mesolve([H0(omega, N), [H1(Omega_R, N), S1], [H2(Omega_R, N), S2]], result_t1.states[timesteps - 1],
                  t, [], Exps, options=opts, progress_bar=True)

result3 = mesolve(H0(omega, N), result2.states[timesteps - 1], t2, [],
                  Exps, options=opts,
                  progress_bar=True)

print('Initial state ....')
print(productstateZ(0, 0, N))

print('H0...')
print(H0(omega, N))
print('H1...')
print(H1(Omega_R, N))
print('H2...')
print(H2(Omega_R, N))

print('Commutator:', 1j * Commutator[0][0])
print('AntiCommutator: ', AntiCommutator[0][0])

fig, ax = plt.subplots(3, 3)
ax[0, 0].plot(t, np.imag(func1(t)))
ax[0, 0].plot(t, np.imag(noisy_data1), 'o')
ax[0, 0].plot(t, np.imag(S1(t)), lw=2)
ax[0, 0].set_xlabel('Time [2 Pi / Omega_Rabi]')
ax[0, 0].set_ylabel('Coupling Amplitude')
ax[0, 0].set_xlim([0, 0.1])

ax[0, 1].plot(t, S1(t), lw=2)
ax[0, 1].set_xlabel('Time [2 Pi / Omega_R]')

ax[0, 2].plot(t, np.imag(func2(t)))
ax[0, 2].plot(t, np.imag(noisy_data2), 'o')
ax[0, 2].plot(t, np.imag(S2(t)), lw=2)
ax[0, 2].set_xlabel('Time [2 Pi / Omega_R]')
ax[0, 2].set_xlim([0, 0.1])


ax[1, 0].plot(t1, result_t1.expect[0], label="MagnetizationX")
ax[1, 0].plot(t1, result_t1.expect[1], label="MagnetizationZ")
ax[1, 0].set_xlabel('Free Evolution Time [2 Pi / Omega_R]')
ax[1, 0].set_ylabel('Magnetization')
ax[1, 0].legend(loc="upper right")
ax[1, 0].set_ylim([-1.1, 1.1])

ax[1, 1].plot(t, result2.expect[0], label="MagnetizationX")
ax[1, 1].plot(t, result2.expect[1], label="MagnetizationZ")
ax[1, 1].set_xlabel('Perturbation Time [2 Pi / Omega_R]')
ax[1, 1].legend(loc="right")
ax[1, 1].set_ylim([-1.1, 1.1])

ax[1, 2].plot(t2, result3.expect[0], label="MagnetizationX")
ax[1, 2].plot(t2, result3.expect[1], label="MagnetizationZ")
ax[1, 2].set_xlabel('Free Evolution time [2 Pi / Omega_Rabi]')
ax[1, 2].legend(loc="right")
ax[1, 2].set_ylim([-1.1, 1.1])


ax[2, 0].plot(t1, result_t1.expect[0], label="MagnetizationX")
ax[2, 0].plot(t1, result_t1.expect[1], label="MagnetizationZ")
ax[2, 0].set_xlabel('Free Evolution time [2 Pi / Omega_Rabi]')
ax[2, 0].legend(loc="right")

ax[2, 1].plot(t2, result_AB.expect[0], label="MagnetizationX")
ax[2, 1].plot(t2, result_AB.expect[1], label="MagnetizationZ")
ax[2, 1].plot(t2, result_AB.expect[2], label="MagnetizationY")
ax[2, 1].set_xlabel('After Perturbation [2 Pi / Omega_Rabi]')
ax[2, 1].legend(loc="right")

ax[2, 2].plot(t2, result_t1t2.expect[0], label="MagnetizationX")
ax[2, 2].plot(t2, result_t1t2.expect[1], label="MagnetizationZ")
ax[2, 2].plot(t2, result_t1t2.expect[2], label="MagnetizationY")
ax[2, 2].set_xlabel('No Perturbation [2 Pi / Omega_Rabi]')
ax[2, 2].legend(loc="right")

plt.show()
