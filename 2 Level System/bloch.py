import numpy as np

import matplotlib.pyplot as plt

sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1.j], [1.j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])


def evo_state(t, g_0=0, e_0=1, d=0, Omega=np.pi):
    Omega_eff = np.sqrt(Omega ** 2 + d ** 2)

    out_state = np.array([[np.exp(1.j * d * t / 2) * (
                g_0 * np.cos(Omega_eff / 2 * t) - 1.j / Omega_eff * (d * g_0 + Omega * e_0) * np.sin(
            Omega_eff / 2 * t))],
                         [np.exp(1.j * d * t / 2) * (e_0 * np.cos(Omega_eff / 2 * t) + 1.j / Omega_eff * (
                                     d * e_0 - Omega * g_0) * np.sin(Omega_eff / 2 * t))]])

    return out_state


def exp(state, pauli):
    return np.vdot(state, np.dot(pauli, state))


t = np.linspace(0, 8, 160)

print("From down after pi/2")

print(evo_state(1/2))

print("x:", exp(evo_state(1/2), sigma_x))
print("y:", exp(evo_state(1/2), sigma_y))
print("z:", exp(evo_state(1/2), sigma_z))

print("From y onwards")

state = evo_state(0.005, evo_state(1/2)[0][0], evo_state(1/2)[1][0], d=2*np.pi*10, Omega=0.0)

print(state)

#print(np.vdot(evo_state(1), np.dot(sigma_z, evo_state(1))))

#print(np.vdot(evo_state(1), np.dot(sigma_z, evo_state(1))))

#print(evo_state(1 / 2, e_0=evo_state(1 / 2)[1], g_0=evo_state(1 / 2)[0]))
print("x:", exp(state, sigma_x))
print("y:", exp(state, sigma_y))
print("z:", exp(state, sigma_z))

state1 = evo_state(1/2, state[0][0], state[1][0], d=0, Omega=np.pi)

print(state1)

print("x:", exp(state1, sigma_x))
print("y:", exp(state1, sigma_y))
print("z:", exp(state1, sigma_z))


plt.plot(t, np.real(evo_state(t)[1][0]), linestyle="-", label="Re[c_e]")

plt.plot(t, np.imag(evo_state(t)[1][0]), linestyle="dotted", label="Im[c_e]")

plt.plot(t, np.real(evo_state(t)[0][0]), marker=".", linestyle="", label="Re[c_g]")

plt.plot(t, np.imag(evo_state(t)[0][0]), linestyle="--", label="Im[c_g]")

plt.legend()

plt.xlabel("$\Omega t [\pi]$")

plt.show()
