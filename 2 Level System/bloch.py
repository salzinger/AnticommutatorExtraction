import numpy as np
from qutip import *

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
    return np.real(np.round(np.vdot(state, np.dot(pauli, state)), decimals=4))


t = np.linspace(0, 8, 160)

print("From down after pi/2")

#print(evo_state(1/2))

x1 = exp(evo_state(1/2), sigma_x)
y1 = exp(evo_state(1/2), sigma_y)
z1 = exp(evo_state(1/2), sigma_z)

print("x:", x1)
print("y:", y1)
print("z:", z1)


print("From y onwards")

state = evo_state(0.04, evo_state(1/2)[0][0], evo_state(1/2)[1][0], d=np.pi*10, Omega=0.0)

#print(state)

#print(np.vdot(evo_state(1), np.dot(sigma_z, evo_state(1))))

#print(np.vdot(evo_state(1), np.dot(sigma_z, evo_state(1))))

#print(evo_state(1 / 2, e_0=evo_state(1 / 2)[1], g_0=evo_state(1 / 2)[0]))

x2 = exp(state, sigma_x)
y2 = exp(state, sigma_y)
z2 = exp(state, sigma_z)

print("x:", x2)
print("y:", y2)
print("z:", z2)

state1 = evo_state(1/2, state[0][0], state[1][0], d=0, Omega=np.pi)

#print(state1)


print("Dynamics afterwards")


x3 = exp(state1, sigma_x)
y3 = exp(state1, sigma_y)
z3 = exp(state1, sigma_z)

print("x:", x3)
print("y:", y3)
print("z:", z3)




c = Bloch()
c.make_sphere()
c.vector_color = ["grey", 'black', '#85bb65']
c.point_color = ['grey']
c.point_size = [1]

c.point_marker=['o']
vec1 = [-np.real(x2)+0.051, 0, 0]
vec2 = [-np.real(x2), np.real(y2), np.real(z2)]
vec3 = [-np.real(x3)/1.5, np.real(y3)/1.5, np.real(z3)]
th = np.linspace(0, 2*np.pi, 200)

xz = -np.ones(200)*np.real(x3)
yz = np.sin(th)*np.real(z3)
zz = np.cos(th)*np.real(z3)
c.add_points([xz, yz, zz], 'm')


xz = -np.ones(200)*np.real(x3)/1.5
yz = np.sin(th)*np.real(z3)
zz = np.cos(th)*np.real(z3)
c.add_points([xz, yz, zz], 'm')

c.add_vectors(vec1)
c.add_vectors(vec2)
c.add_vectors(vec3)
c.render()
c.clear()

#plt.plot(t, np.real(evo_state(t)[1][0]), linestyle="-", label="Re[c_e]")

#plt.plot(t, np.imag(evo_state(t)[1][0]), linestyle="dotted", label="Im[c_e]")

#plt.plot(t, np.real(evo_state(t)[0][0]), marker=".", linestyle="", label="Re[c_g]")

#plt.plot(t, np.imag(evo_state(t)[0][0]), linestyle="--", label="Im[c_g]")

#plt.legend()

#plt.xlabel("$\Omega t [\pi]$")

plt.show()



t = np.linspace(0, 8, 160)

print("From down after pi/2")

#print(evo_state(1/2))

x1 = exp(evo_state(1/2), sigma_x)
y1 = exp(evo_state(1/2), sigma_y)
z1 = exp(evo_state(1/2), sigma_z)

print("x:", x1)
print("y:", y1)
print("z:", z1)


print("From y onwards")

state = evo_state(0.04, evo_state(1/2)[0][0], evo_state(1/2)[1][0], d=np.pi*10, Omega=0.0)

#print(state)

#print(np.vdot(evo_state(1), np.dot(sigma_z, evo_state(1))))

#print(np.vdot(evo_state(1), np.dot(sigma_z, evo_state(1))))

#print(evo_state(1 / 2, e_0=evo_state(1 / 2)[1], g_0=evo_state(1 / 2)[0]))

x2 = exp(state, sigma_x)
y2 = exp(state, sigma_y)
z2 = exp(state, sigma_z)

print("x:", x2)
print("y:", y2)
print("z:", z2)

state1 = evo_state(1/2, state[0][0], state[1][0], d=0, Omega=np.pi)

#print(state1)


print("Dynamics afterwards")


x3 = exp(state1, sigma_x)
y3 = exp(state1, sigma_y)
z3 = exp(state1, sigma_z)

print("x:", x3)
print("y:", y3)
print("z:", z3)




c = Bloch()
c.make_sphere()
c.vector_color = ["grey", 'black', '#85bb65']
c.point_color = ['black']
c.point_size = [1]

c.point_marker=['o']
vec1 = [-np.real(x2)+0.092, 0, 0]
vec2 = [-np.real(x2)+0.03, np.real(y2)+0.01, np.real(z2)]
#vec3 = [-np.real(x3)/1.5, np.real(y3)/1.5, np.real(z3)]
th = np.linspace(0, 2*np.pi, 200)

xz = -np.ones(200)*np.real(x3)
yz = np.sin(th)*np.real(z3)
zz = np.cos(th)*np.real(z3)
c.add_points([xz, yz, zz], 'm')

c.add_vectors(vec1)
c.add_vectors(vec2)
#c.add_vectors(vec3)
c.render()
c.clear()

#plt.plot(t, np.real(evo_state(t)[1][0]), linestyle="-", label="Re[c_e]")

#plt.plot(t, np.imag(evo_state(t)[1][0]), linestyle="dotted", label="Im[c_e]")

#plt.plot(t, np.real(evo_state(t)[0][0]), marker=".", linestyle="", label="Re[c_g]")

#plt.plot(t, np.imag(evo_state(t)[0][0]), linestyle="--", label="Im[c_g]")

#plt.legend()

#plt.xlabel("$\Omega t [\pi]$")

plt.savefig("BlochHerm.pdf")  # and BW %.2f.pdf" % (noise_amplitude, bandwidth))

plt.show()



c = Bloch()
c.make_sphere()
c.vector_color = ["grey", '#85bb65']
c.point_color = ['#85bb65']
c.point_size = [1]

c.point_marker=['o']
vec1 = [-np.real(x2)+0.092, 0, 0]
vec2 = [-np.real(x2), np.real(y2), np.real(z2)]
vec3 = [-np.real(x3)/1.5, np.real(y3)/1.5, np.real(z3)]
th = np.linspace(0, 2*np.pi, 200)


xz = -np.ones(200)*np.real(x3)/1.5
yz = np.sin(th)*np.real(z3)
zz = np.cos(th)*np.real(z3)
c.add_points([xz, yz, zz], 'm')

c.add_vectors(vec1)
#c.add_vectors(vec2)
c.add_vectors(vec3)
c.render()
c.clear()

#plt.plot(t, np.real(evo_state(t)[1][0]), linestyle="-", label="Re[c_e]")

#plt.plot(t, np.imag(evo_state(t)[1][0]), linestyle="dotted", label="Im[c_e]")

#plt.plot(t, np.real(evo_state(t)[0][0]), marker=".", linestyle="", label="Re[c_g]")

#plt.plot(t, np.imag(evo_state(t)[0][0]), linestyle="--", label="Im[c_g]")

#plt.legend()

#plt.xlabel("$\Omega t [\pi]$")


plt.savefig("BlochNonHerm.pdf")  # and BW %.2f.pdf" % (noise_amplitude, bandwidth))

plt.show()
