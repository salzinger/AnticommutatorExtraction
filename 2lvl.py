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
    return sum / 2


def MagnetizationX(N):
    sum = 0
    for j in range(0, N):
        sum += sigmax(j, N)
    return sum / 2


def Levels(Delta, j, N):
    up, down = twobasis()
    oplist = np.full(N, identity(2))
    oplist[j] = Delta * down * down.dag()
    return tensor(oplist)


def Probe(Omega_p, j, N):
    oplist = np.full(N, identity(4))
    oplist[j] = Omega_p / 2 * Qobj([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    return tensor(oplist)


def Coupling(Omega_c, j, N):
    oplist = np.full(N, identity(4))
    oplist[j] = Omega_c / 2 * Qobj([[0, 1, 0], [0, 1, 0], [0, 0, 0]])
    return tensor(oplist)


def H(J, N):
    H = 0
    for j in range(0, N):
        H += 0*sigmaz(j, N)
        for m in range(0, N):
            if m < j:
                H += J * (sigmap(m, N) * sigmam(j, N) + sigmam(m, N) * sigmap(j, N))
    return H


def H1(N):
    H = 0
    for j in range(0, N):
        H += 100*(sigmap(j, N) + sigmam(j, N))
    return H

print(H1(2))
def L(Gamma, N):
    up, down = twobasis()
    L = 0
    for j in range(0, N):
        oplist = np.full(N, identity(2))
        oplist[j] = down * up.dag()
        L += tensor(oplist)
    return L


def L_eff(Gamma, Omega, N):
    up, down = twobasis()
    L = 0
    for j in range(0, N):
        oplist = np.full(N, identity(2))
        oplist[j] = up * down.dag()
        L += tensor(oplist)
    return L


def L_eff_delta(Gamma, Omega, Delta, N):
    up, down = twobasis()
    L = 0
    for j in range(0, N):
        oplist = np.full(N, identity(2))
        oplist[j] = up * down.dag()
        L += tensor(oplist)
    return L


def H_eff_delta(Gamma, Omega, Delta, N):
    up, down = twobasis()
    H = 0
    for j in range(0, N):
        oplist = np.full(N, identity(2))
        oplist[j] = down * down.dag()
        H += tensor(oplist)
    return H



timesteps = 200
endtime = 1
perturbtime = 0.01

t = np.linspace(0, perturbtime, timesteps)
func = lambda t: np.sin(t * 5000)
noisy_func = lambda t: func(t+0.000*np.random.randn(t.shape[0]))
noisy_data = noisy_func(t)
S = Cubic_Spline(t[0], t[-1], noisy_data)
plt.figure()
plt.plot(t, func(t))
plt.plot(t, noisy_data, 'o')
plt.plot(t, S(t), lw=2)
plt.show()

times = np.linspace(0, endtime, timesteps)
perturb_times = np.linspace(0, perturbtime, timesteps)
J = 0
op = 0
Omega = 0
Gamma = 0
Delta = 0
N = 1
ops = [MagnetizationX(N), MagnetizationZ(N)]

opts = Options(store_states=True, store_final_state=True, ntraj=200)

result1 = mesolve(H(0, N), productstateZ(0, 0, N), times, [], ops, options=opts,
                  progress_bar=True)

result2 = mesolve([H(0, N), [H1(N), S]], result1.states[timesteps - 1],
                  perturb_times,
                  [], ops,
                  options=opts,
                  progress_bar=True)

result3 = mesolve(H(0, N), result2.states[timesteps - 1], times, [],
                  ops, options=opts,
                  progress_bar=True)




fig, ax = plt.subplots(1,3)
#ax[0].plot(times, result1.expect[0], label="MagnetizationX");
ax[0].plot(times, result1.expect[1], label="MagnetizationZ");
ax[0].set_xlabel('Time [1/Omega]');
ax[0].set_ylabel('');
ax[0].legend(loc="upper right")

#ax[1].plot(perturb_times, result2.expect[0], label="MagnetizationX");
ax[1].plot(perturb_times, result2.expect[1], label="MagnetizationZ");
ax[1].set_xlabel('Time [1/Omega]');
ax[1].set_ylabel('');
ax[1].legend(loc="right")


#ax[2].plot(times, result3.expect[0], label="MagnetizationX");
ax[2].plot(times, result3.expect[1], label="MagnetizationZ");
ax[2].set_xlabel('Time [1/Omega]');
ax[2].set_ylabel('');
ax[2].legend(loc="right")
plt.show()