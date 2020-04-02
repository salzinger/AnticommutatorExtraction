import numpy as np
import matplotlib.pyplot as plt

from qutip import *
from qutip.solver import Options, Result, config, _solver_safety_check


def fourbasis():
    return np.array([basis(4, 0), basis(4, 1), basis(4, 2), basis(4, 3)], dtype=object)


def productstateZ(up_atom, down_atom, N):
    up, down, excited, ground = fourbasis()
    pbasis = np.full(N, ground)
    pbasis[up_atom] = down
    pbasis[down_atom] = up
    return tensor(pbasis)


def productstateX(m, j, N):
    up, down, excited, ground = fourbasis()
    pbasis = np.full(N, ground)
    pbasis[m] = (up + down).unit()
    pbasis[j] = (up + down).unit()
    return tensor(pbasis)


def sigmap(m, N):
    up, down, excited, ground = fourbasis()
    oplist = np.full(N, identity(4))
    oplist[m] = up * down.dag()
    return tensor(oplist)


def sigmam(m, N):
    up, down, excited, ground = fourbasis()
    oplist = np.full(N, identity(4))
    oplist[m] = down * up.dag()
    return tensor(oplist)


def sigmaz(j, N):
    oplist = np.full(N, identity(4))
    oplist[j] = Qobj([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    return tensor(oplist)


def sigmax(j, N):
    oplist = np.full(N, identity(4))
    oplist[j] = Qobj([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
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


def Probe(Omega_p, j, N):
    oplist = np.full(N, identity(4))
    oplist[j] = Omega_p * Qobj([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    return tensor(oplist)


def Coupling(Omega_c, j, N):
    oplist = np.full(N, identity(4))
    oplist[j] = Omega_c * Qobj([[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
    return tensor(oplist)


def H(Omega_p, Omega_c, J, N):
    H = 0
    for j in range(0, N):
        H += Probe(Omega_p, j, N) + Coupling(Omega_c, j, N)
        for m in range(0, N):
            if m < j:
                H += J * (sigmap(m, N) * sigmam(j, N) + sigmam(m, N) * sigmap(j, N))
    return H


def L(Gamma, N):
    up, down, excited, ground = fourbasis()
    L = 0
    for j in range(0, N):
        oplist = np.full(N, identity(4))
        oplist[j] = Gamma * ground * excited.dag()
        L += tensor(oplist)
    return L


print(Coupling(1, 0, 2))
print(productstateZ(0, 1, 2))
print(Coupling(1, 0, 2) * productstateZ(0, 1, 2))
timesteps = 200
times = np.linspace(0, 10, timesteps)

op = 0
oc = 1
Gamma = 10
N = 2
opts = Options(store_states=True, store_final_state=True, ntraj=200)

result = mesolve(H(op, oc, 1, N), productstateX(0, 1, N), times, [L(Gamma, N)],
                 [MagnetizationX(N), MagnetizationZ(N), sigmaz(0, N), sigmax(0, N)], options=opts)

ups = np.zeros(timesteps)
downs = np.zeros(timesteps)
ground = np.zeros(timesteps)
excited = np.zeros(timesteps)

for t in range(0, timesteps):
    ups[t] = np.abs(result.states[t].ptrace(0)[0][0][0])
    downs[t] = np.abs(result.states[t].ptrace(0)[1][0][1])
    ground[t] = np.abs(result.states[t].ptrace(0)[3][0][3])
    excited[t] = np.abs(result.states[t].ptrace(0)[2][0][2])
    # print(result.states[t].ptrace(0)[0][0][0])

fig, ax = plt.subplots()
ax.plot(times, result.expect[0], label="MagnetizationX");
ax.plot(times, result.expect[3], label="Exp(SigmaX,0)", linestyle='--');
ax.plot(times, result.expect[1], label="MagnetizationZ");
ax.plot(times, result.expect[2], label="Exp(SigmaZ,0)", linestyle='--');
# ax.plot(times, np.abs(ups),label="Tr_1(rho,uu)",linestyle='--');
# ax.plot(times, np.abs(downs),label="Tr_1(rho,dd)",linestyle='-');
ax.set_xlabel('Time [1/J]');
ax.set_ylabel('');
leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)
plt.show(fig)
fig, ax = plt.subplots()
ax.plot(times, result.expect[0], label="MagnetizationX");
# ax.plot(times, result.expect[1],label="MagnetizationZ",linestyle='--',marker='o',markersize='2');
ax.plot(times, np.abs(ups), label="Tr_1(rho,uu)", linestyle='--');
ax.plot(times, np.abs(downs), label="Tr_1(rho,dd)", linestyle='--');
ax.plot(times, np.abs(excited), label="Tr_1(rho,ee)", linestyle='--');
ax.plot(times, np.abs(ground), label="Tr_1(rho,gg)", linestyle='--');

ax.set_xlabel('Time [1/J]');
ax.set_ylabel('');
leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)

# fig1, ax1 = plt.subplots()
# ax1.plot(times, result.expect[4],label="a2sig11");
# ax1.plot(times, result.expect[5],label="a2sig22");
# ax1.plot(times, result.expect[6],label="a2sig33");
# ax1.plot(times, result.expect[7],label="a2sig44");
# ax1.set_xlabel('Time');
# ax1.set_ylabel('Expectation value');
# leg1 = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
# leg1.get_frame().set_alpha(0.5)

plt.show(fig)
