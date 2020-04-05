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


def Levels(Delta, j, N):
    up, down, excited, ground = fourbasis()
    oplist = np.full(N, identity(4))
    oplist[j] = Delta * down * down.dag()
    return tensor(oplist)


def Probe(Omega_p, j, N):
    oplist = np.full(N, identity(4))
    oplist[j] = Omega_p / 2 * Qobj([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    return tensor(oplist)


def Coupling(Omega_c, j, N):
    oplist = np.full(N, identity(4))
    oplist[j] = Omega_c / 2 * Qobj([[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
    return tensor(oplist)


def H(Omega_p, Omega_c, Delta, J, N):
    H = 0
    for j in range(0, N):
        H += Probe(Omega_p, j, N) + Coupling(Omega_c, j, N) + Levels(Delta, j, N)
        for m in range(0, N):
            if m < j:
                H += J * (sigmap(m, N) * sigmam(j, N) + sigmam(m, N) * sigmap(j, N))
    return H


def L(Gamma, N):
    up, down, excited, ground = fourbasis()
    L = 0
    for j in range(0, N):
        oplist = np.full(N, identity(4))
        oplist[j] = np.sqrt(Gamma) * ground * excited.dag()
        L += tensor(oplist)
    return L


def L_eff(Gamma, Omega, N):
    up, down, excited, ground = fourbasis()
    L = 0
    for j in range(0, N):
        oplist = np.full(N, identity(4))
        oplist[j] = 1j * Omega / np.sqrt(Gamma) * ground * down.dag()
        L += tensor(oplist)
    return L


def L_eff_delta(Gamma, Omega, Delta, N):
    up, down, excited, ground = fourbasis()
    L = 0
    for j in range(0, N):
        oplist = np.full(N, identity(4))
        oplist[j] = np.sqrt(Gamma) * Omega / (2 * Delta - 1j * Gamma) * ground * down.dag()
        L += tensor(oplist)
    return L


def H_eff_delta(Gamma, Omega, Delta, N):
    up, down, excited, ground = fourbasis()
    H = 0
    for j in range(0, N):
        oplist = np.full(N, identity(4))
        oplist[j] = (Delta - Delta * Omega ** 2 / (4 * Delta ** 2 - 1j * Gamma ** 2)) * down * down.dag()
        H += tensor(oplist)
    return H


print(Coupling(1, 0, 2))
print(productstateZ(0, 1, 2))
print(Coupling(1, 0, 2) * productstateZ(0, 1, 2))
timesteps = 200
endtime = 0.1
times = np.linspace(0, endtime, timesteps)

J = 0
op = 0
Omega = 1
Gamma = 1
Delta = 10
N = 2
opts = Options(store_states=True, store_final_state=True, ntraj=200)

result = mesolve(H(op, Omega, Delta, J, N), productstateX(0, 1, N), times, [L(Gamma, N)],
                 [MagnetizationX(N), MagnetizationZ(N), sigmaz(0, N), sigmax(0, N)], options=opts)
result1 = mesolve(H_eff_delta(Gamma, Omega, Delta, N), productstateX(0, 1, N), times, [L_eff_delta(Gamma, Omega, Delta, N)],
                  [MagnetizationX(N), MagnetizationZ(N), sigmaz(0, N), sigmax(0, N)], options=opts)

ups = np.zeros(timesteps)
downs = np.zeros(timesteps)
ground = np.zeros(timesteps)
downs1 = np.zeros(timesteps)
ground1 = np.zeros(timesteps)
excited = np.zeros(timesteps)

for t in range(0, timesteps):
    ups[t] = np.abs(result.states[t].ptrace(0)[0][0][0])
    downs[t] = np.abs(result.states[t].ptrace(0)[1][0][1])
    ground[t] = np.abs(result.states[t].ptrace(0)[3][0][3])
    downs1[t] = np.abs(result1.states[t].ptrace(0)[1][0][1])
    ground1[t] = np.abs(result1.states[t].ptrace(0)[3][0][3])
    excited[t] = np.abs(result.states[t].ptrace(0)[2][0][2])
    # print(result.states[t].ptrace(0)[0][0][0])

fig, ax = plt.subplots(2,1)

#ax[0].plot(times, result.expect[0], label="MagnetizationX");
#ax[0].plot(times, result1.expect[0], label="MagnetizationX_eff", linestyle='', marker='o', markersize='1',color='green');
#ax[0].plot(times, (1-0.5*Gamma*Omega**2*times/(4*Delta**2+Gamma**2)),label='1-t*Gamma*Omega^2/(8*Delta^2+2*Gamma^2)')
#ax.plot(times, result.expect[3], label="Exp(SigmaX,0)", linestyle='--');
#ax[0].plot(times, result1.expect[1], label="MagnetizationZ_eff");
#ax.plot(times, result.expect[2], label="Exp(SigmaZ,0)", linestyle='--');
#ax[0].plot(times, np.abs(ups),label="Tr_1(rho,uu)",linestyle='--');
#ax[0].plot(times, np.abs(downs1),label="Tr_1(rho,dd)_eff",linestyle='--',color='green');
#ax[0].plot(times, np.abs(ground), label="Tr_1(rho,gg)", linestyle='--',color='grey');
#ax[0].set_xlabel('Time [1/Gamma]');
#ax[0].set_ylabel('');
#ax[0].legend(loc="upper right")

#ax[0].plot(times, result1.expect[0], label="MagnetizationX_eff");
#ax[0].plot(times, result.expect[0], label="MagnetizationX");
# ax.plot(times, 0.5*np.exp(-(oc/Gamma)*times),label='0.5*exp(-(oc/Gamma)*t)')
# ax.plot(times, result.expect[1],label="MagnetizationZ",linestyle='--',marker='o',markersize='2');
# ax.plot(times, np.abs(ups), label="Tr_1(rho,uu)", linestyle='--');
#plt.ylim(-.05,0.6)
ax[1].plot(times, np.abs(downs), label="Tr_1(rho,dd)", linestyle='-');
ax[1].plot(times, (np.abs(downs1)), label="Tr_1(rho,dd)_eff", linestyle='', marker='o', markersize='1',color='green');
ax[1].plot(times, np.heaviside(0.5-0.5*Gamma*Omega**2*times/(4*Delta**2+Gamma**2),0)*(0.5-0.5*Gamma*Omega**2*times/(4*Delta**2+Gamma**2)),label='0.5-t*Gamma*Omega^2/(8*Delta^2+2*Gamma^2)')
ax[1].set_xlabel('Time [1/Gamma]');
ax[1].set_ylabel('');
ax[1].legend(loc="right")


#ax[1].plot(times, np.abs(excited), label="Tr_1(rho,ee)", linestyle='--',color='orange');
ax[0].plot(times,np.abs(excited), label="Tr_1(rho,ee)", linestyle='--',color='orange');
#ax[1].plot(times, result.expect[0]-(1-Gamma*Omega**2*times/(8*Delta**2+2*Gamma**2)), label="MagnetizationX(t)-(1-t*Gamma*Omega^2/(8*Delta^2+2*Gamma^2))");
ax[0].plot(times, result.expect[0]-result1.expect[0], label="MagnetizationX(t)-MagnetizationX(t)_eff");
# ax.plot(times, np.abs(ground), label="Tr_1(rho,gg)", linestyle='--');
#ax[1].plot(times, (np.abs(ground1) - np.abs(ground)), label="eff - Tr_1(rho,gg)", linestyle='', marker='o',
        #markersize='1',color='gray');
ax[0].set_xlabel('Time [1/Gamma]');
ax[0].set_ylabel('');
ax[0].legend(loc="right")

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
