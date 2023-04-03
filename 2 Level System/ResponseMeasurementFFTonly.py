import numpy as np
import matplotlib.font_manager
from Atoms import *
from Driving import *
import matplotlib.pyplot as plt
from scipy import integrate

plt.rcParams.update({
  "text.usetex": True,
    "font.family":"sans-serif",
})

def configure_plots(fontsize_figure=None, fontsize_inset=None, usetex=True):
    if usetex:
        plt.rcParams.update(
            {
                # Use LaTeX to write all text
                "text.usetex": True,
                # Custom preamble
                "text.latex.preamble": "\n".join(
                    [
                        "\\usepackage[utf8]{inputenc}",
                        "\\usepackage[T1]{fontenc}",
                        "\\usepackage{lmodern}",
                        "\\usepackage{amsmath}",
                        "\\usepackage{bm}",
                        "\\usepackage{siunitx}",
                        "\\usepackage{xfrac}",
                        "\\usepackage{braket}",
                    ]
                ),
            }
        )
    else:
        plt.rcParams.update(
            {
                "text.usetex": False,
            }
        )

    tex_fonts = {
        # Serif font (default should be Computer Modern Roman)
        # "font.family": "serif",
        # Sans-serif font (default should be Computer Modern Serif)
        "font.family": "sans-serif",
    }

    if fontsize_figure is not None:
        tex_fonts.update(
            {
                # Use 9pt font in plots to match 9pt font in captions
                "axes.labelsize": fontsize_figure,
                "font.size": fontsize_figure,
                "xtick.labelsize": fontsize_figure,
                "ytick.labelsize": fontsize_figure,
            }
        )

    if fontsize_inset is not None:
        tex_fonts.update(
            {
                # Use smaller font size for legends and insets
                "legend.fontsize": fontsize_inset,
            }
        )

    plt.rcParams.update(tex_fonts)

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#plt.rc('font',**{'family':'sans-serif','sans-serif':['Latin Modern Sans']})


#configure_plots(fontsize_figure=None, fontsize_inset=None, usetex=True)

#plt.rc('figure', figsize=(8.27, 11.69))

N = 1

omega = 0  # MHz
Omega_R = 2 * np.pi * 13.6 # MHz

J = 0 * 10 ** 0  # MHz

Exps = [MagnetizationX(N), MagnetizationZ(N), MagnetizationY(N), sigmaz(0, N), sigmaz(N - 1, N), upup(0, N),
        sigmap(0, N), sigmam(0, N), downdown(0, N)]

opts = Options(store_states=True, store_final_state=True)  # , nsteps=50000)

timesteps = 100
endtime = 0.1 * 13.6
pertubation_length = endtime / 1
perturb_times = np.linspace(0, endtime, timesteps)
init_state = productstateZ(0, 0, N)

################### MASTER EQUATION ############################################ 444444444444444444

#plt.rc('figure', figsize=(11.69, 8.27))

#fig, ax = plt.subplots(1, 1, figsize=(8.27, 8.27))


with open('incoherent_perturb.txt') as f:
    linesm0 = f.readlines()
with open('coherent_perturb.txt') as f:
    linesm3 = f.readlines()

with open('incoherent_error.txt') as f:
    linesm0e = f.readlines()
with open('coherent_error.txt') as f:
    linesm3e = f.readlines()

x0 = []
y0 = []
y0e = []
y3 = []
y3e = []
y5 = []
y5e = []
y15 = []
y15e = []
y30 = []
y30e = []

'''
3.953125
0.378901455848414
52.449438202247194
2.007774167644925
'''

for element in range(1, 10):
    x0.append(float(linesm0[element][0:5]) * 13.6)

    y0.append((float(linesm0[element][8:18])))
    y3.append((float(linesm3[element][8:18])))

    y3e.append(0.011718)

    y0e.append(0.012739)

ft = np.fft.fft(y0)

freq = np.fft.fftfreq(ft.size, x0[1])


def psd(data, samples, sample_time):
    long = data
    fs = samples / sample_time
    F, P = signal.welch(
        long, fs,
        nperseg=samples, scaling='spectrum', nfft=5000, return_onesided=0)
    return F, P


x0 = np.array(x0[0:9])

y0 = np.array(y0[0:9])

y3 = np.array(y3[0:9])

f = psd(y0, 7, 0.1)

f1 = psd(y3, 7, 0.1)

omegas = np.linspace(-1.5, 1.5, num=500)

integrals = []
integrals0 = []

x1 = np.linspace(x0[0], x0[-1]) #np.array(x0[0:9])

y0 = np.array(y0[0:9])

y31 = np.array(y3[0:9]) #-np.sin(x1*2*np.pi)*0.16

print("y3", y3)
print("x0", x0)
print("omegas", omegas)


y0e = y0e[0:len(y0)]

y3e = y3e[0:len(y0)]

ax2 = plt.subplot(222)

ax2.errorbar(x0, y0, y0e, marker="o", color='#85bb65',
                  label=r'Non-Hermitian $\langle \{ \sigma_z(0),\sigma_z(t) \} \rangle$', linestyle='', markersize="3")

ax2.plot(perturb_times, np.cos(2 * np.pi * perturb_times) * 0.16, color='#85bb65', linestyle='-')

ax2.errorbar(x0, y3, y3e, marker="o", color='black',
                  label=r'Hermitian $\langle[ \sigma_z(0),\sigma_z(t)] \rangle$', linestyle='', markersize="3")

ax2.plot(perturb_times, -np.sin(2 * np.pi * perturb_times) * 0.16, color='black', linestyle='-')



ydiv = []

for n in range(0, len(y0)):
    ydiv.append(y0[n] / y3[n])

#ax2.set_xlabel('Time [$2\pi/\Omega_R$]', fontsize=8)
#ax2.set_ylabel(r'$\langle S_z \rangle - \langle S_z \rangle_0$', fontsize=8)
#ax.legend(loc="lower center", fontsize=32)
ax2.set_ylim([-0.2, 0.2])
ax2.set_xlim([-0.005, 1.372])
ax2.tick_params(axis="both", labelsize=8)



#plt.savefig("ResponseMeasureTimeDomain.pdf")


#plt.show()

#plt.rc('figure', figsize=(11.69, 8.27))

#fig, ax = plt.subplots(1, 1, figsize=(8.27, 8.27/2))


for o in omegas:
    integrals.append(2*np.pi*integrate.simps(y3*np.exp(-1j*o*x0*2*np.pi), x0))

for o in omegas:
    integrals0.append(2*np.pi*integrate.simps(y0*np.exp(-1j*o*x0*2*np.pi), x0))

print("integrals", integrals)




def lorentzian(frequencies, amplitude, omega_0, gamma):
    func = lambda omega: amplitude / gamma / np.pi / (2 * ((omega - omega_0) / gamma) ** 2 + 1 / 2)
    return func(frequencies)


ferror = 0

f1error = 0

Nsamples = 10

y0samples = []

y3samples = []

for e in range(0, len(y0e)):
    ferror += y0e[e] ** 2
    f1error += y3e[e] ** 2


for e in range(0, len(y0e)):
    y0samples.append(np.random.normal(y0[e], y0e[e], Nsamples))
    y3samples.append(np.random.normal(y3[e], y3e[e], Nsamples))


samplederrory0 = []


for n in range(0, Nsamples):
    y0reshuffled = []
    for e in range(0, len(y0e)):
        y0reshuffled.append(y0samples[e][n])
    ftty0samples = np.real(np.fft.fft(y0reshuffled, n=int(1.5 * len(y0))))
    samplederrory0.append(ftty0samples)


print(np.sqrt(np.var(samplederrory0, axis=1)))

print(np.sqrt(np.var(samplederrory0, axis=0)))

print(np.sqrt(ferror))

#print(samplederrory0)



ftt = psd(-np.sin(2 * np.pi *13.6* np.array(x0)) * 0.16, 7, 0.1)

fttcos = psd(np.cos(2 * np.pi * np.array(x0)) * 0.16, 7, 0.1)

x0 = x0[0:]

y0 = y0[0:]

y3 = y3[0:]

ftty0 = np.fft.fft(y0, n=int(1.5 * len(y0)))

ftty3 = np.fft.fft(y3, n=int(1.5 * len(y0)))

ft = np.fft.fft(y0, n=int(1.5 * len(y0)))

freq = np.fft.fftfreq(ft.size, x0[1])

freqsamples=np.fft.fftfreq(ftty0samples.size, x0[1])

#print(len(ftty0))
#print(len(freq))


fserror = np.ones(len(omegas)) * np.sqrt(ferror)

f1serror = np.ones(len(omegas)) * np.sqrt(f1error)

ferror = np.ones(len(ftty0)) * np.sqrt(ferror)

f1error = np.ones(len(ftty3)) * np.sqrt(f1error)


# ax[0, 1].errorbar(freq, np.real(ftty0), ferror, marker="o", color='#85bb65', linestyle='', markersize="3", label=r"$ NonHermitian  Re FT \langle \{ \sigma_z(0),\sigma_z(t) \} \rangle $")

# ax[0, 1].errorbar(freq, np.imag(ftty3), f1error, marker="s", color='black', linestyle='', markersize="3", label=r"$ Hermitian  Im FT \langle \[ \sigma_z(0),\sigma_z(t) \] \rangle $")

#ax[0, 1].errorbar(freq, np.real(ftty0), ferror, marker="o", color='#85bb65', linestyle='', markersize="3",
#                  label=r'Non-Hermitian Re(FT($ \langle \{ \sigma_z(0),\sigma_z(t) \} \rangle$)')

#ax[0, 1].errorbar(freqsamples, np.real(ftty0samples), marker="o", color='#85bb65', linestyle='', markersize="3",
#                  label=r'SAMPLED Non-Hermitian Re(FT($ \langle \{ \sigma_z(0),\sigma_z(t) \} \rangle$)')

#ax[0, 1].errorbar(freq, np.imag(ftty3), ferror, marker="o", color='black', linestyle='', markersize="3",
#                  label=r'Hermitian Im(FT($ \langle [ \sigma_z(0),\sigma_z(t) ] \rangle$)')

ax1=plt.subplot(212)
ax1.errorbar(omegas, np.imag(integrals), fserror*0, marker="", color='black', linestyle='-', markersize="4",
                  )

ax1.errorbar(omegas, np.real(integrals0), f1serror*0, marker="", color='#85bb65', linestyle='-', markersize="4",
                  )

om=np.linspace(-1.5,1.5,500)

Temp = 10*10**(-6)
#ax[0, 1].errorbar(omegas, (1 - 2/(np.exp(2*omegas/Temp/10**4/6.558) + 1))*np.real(integrals0), marker="o", color='#85bb65', linestyle='--', markersize="0", linewidth='1.5',
#                  label=r"$T=10 \mu $K")

#used to be 2/6.558/10**4 but!: 2*np.pi*13.6*10^6 MHz * hbar/k_b = 2*np.pi* 1.04 * 10^-4

prefactor= 2*np.pi*1.04 * 10**(-4)


ax1.errorbar(omegas, (1 - 2/(np.exp(prefactor*om/Temp) + 1))*np.real(integrals0), f1serror*0, marker="", color='purple', linestyle='-', markersize="6",
                  )


omegas=np.linspace(-1.5, 1.5, 9)

integralsshort=[]
integralsshort0=[]
for o in omegas:
    integralsshort.append(2*np.pi*integrate.simps(y3*np.exp(-1j*o*x0*2*np.pi), x0))

for o in omegas:
    integralsshort0.append(2*np.pi*integrate.simps(y0*np.exp(-1j*o*x0*2*np.pi), x0))

ax1.errorbar(omegas, np.imag(integralsshort), fserror[0:9], marker="o", color='black', linestyle='', markersize="3",
                  label=r'$\chi^{\prime\prime}(\omega)=\mathcal{I}(\mathcal{F}\langle [ \hat{s}_z(0),\hat{s}_z(t) ] \rangle)$')


ax1.errorbar(omegas, np.real(integralsshort0), f1serror[0:9], marker="o", color='#85bb65', linestyle='', markersize="3",
                  label=r'$S(\omega)=\mathcal{R}(\mathcal{F}\langle \{ \hat{s}_z(0),\hat{s}_z(t) \} \rangle)$')


ax1.errorbar(omegas, (1 - 2/(np.exp(prefactor*omegas/Temp) + 1))*np.real(integralsshort0), f1serror[0:9], marker="o", color='purple', linestyle='', markersize="3",
                  label=r'$S(\omega) \tanh{\frac{\hbar\omega}{k_B T}}$ at $T=10$ $\mu$K')

#ax[0, 1].fill_between(omegas, np.imag(integrals)+ferror[0],np.imag(integrals)-ferror[0], color='grey',
#                  alpha=0.5)





#ax[0, 1].errorbar(freq, (np.heaviside(freq, 1) - np.heaviside(-freq, 1)) * np.imag(ftty3), ferror, marker="o",
#                  color='purple', linestyle='', markersize="0.5",
#                  label=r'Coth(T=0) * Im(FT($ \langle [ \sigma_z(0),\sigma_z(t) ] \rangle$)')

#ax[0, 1].axvline(x=0., color="grey", ymin=0.05, ymax=0.95)
print("ftt: ", np.max(ftt[0]))
Omega = 1
T = 1.36
#om = ftt[0] / 13.6
om = np.linspace(-1.525,1.525,10000)

#print(om)

'''

ax[0, 1].errorbar(om, 0.16 * (Omega * np.sin(2 * np.pi * om * T) * np.cos(2 * np.pi * Omega * T) - om * np.sin(
    2 * np.pi * Omega * T) * np.cos(2 * np.pi * om * T))
                  / ((om) ** 2 - Omega ** 2), marker="o", color='black', linestyle='', markersize="1", label="Analytical Hermitian response function")

ax[0, 1].errorbar(om, 0.16 * (om * np.sin(2 * np.pi * om * T) * np.cos(2 * np.pi * Omega * T) - Omega * np.sin(
    2 * np.pi * Omega * T) * np.cos(2 * np.pi * om * T))
                  / ((om) ** 2 - Omega ** 2), marker="o", color='#85bb65', linestyle='', markersize="1", label="Analytical Non-Hermitian response function")
'''
#ax[0, 1].errorbar(om, (np.heaviside(om, 1) - np.heaviside(-om, 1)), marker="o", color='grey', linestyle='',
#                  markersize="0.05", label="Coth(T=0)")


'''
ax[0, 1].errorbar(om, (np.heaviside(om, 1) - np.heaviside(-om, 1)), marker="o", color='purple', linestyle='',
                  markersize="0.1")#, label="Tanh(T=0)")


ax[0, 1].axvline(x=0., color="purple", ymin=0.049, ymax=0.951, linewidth='2')
'''


#Temp = 10**(-6)

#ax[0, 1].errorbar(omegas, 1 - 2/(np.exp(2*omegas/Temp/10**4) + 1), marker="o", color='grey', linestyle='-',
#                  markersize="1", label="Coth(T)")


Temp = 50*10**(-6)
#ax[0, 1].errorbar(omegas, (1 - 2/(np.exp(2*omegas/Temp/10**4/6.558) + 1))*np.real(integrals0), marker="o", color='#85bb65', linestyle='--', markersize="0", linewidth='1.5',
#                  label=r"$T=10 \mu $K")

#used to be 2/6.558/10**4 but!: 2*np.pi*13.6*10^6 MHz * hbar/k_b = 2*np.pi* 1.04 * 10^-4

prefactor= 2*np.pi*1.04 * 10**(-4)


#ax[0, 1].errorbar(om, (1 - 2/(np.exp(prefactor*om/Temp) + 1)), marker="o", color='purple', linestyle='', markersize="0.05", linewidth='0')

#Temp = 1.5*10**(-5)
#ax[0, 1].errorbar(omegas, (1 - 2/(np.exp(2*omegas/Temp/10**4/6.558) + 1))*np.real(integrals0), marker="o", color='#85bb65', linestyle=':', markersize="0", linewidth='1.5',
#                  label=r"$T=15 \mu $K"
#                  )

#ax[0, 1].errorbar(om, (1 - 2/(np.exp(prefactor*om/Temp) + 1)), marker="o", color='purple', linestyle='', markersize="0.05", linewidth='0')

'''
Temp = 50*10**(-6)

#ax[0, 1].errorbar(om, (1 - 2/(np.exp(prefactor*om/Temp) + 1)), marker="o", color='purple', linestyle='', markersize="0.1", linewidth='0')

#Temp = 2*10**(-6)

ax[0, 1].errorbar(om, (1 - 2/(np.exp(prefactor*om/Temp) + 1)), marker="o", color='purple', linestyle='', markersize="0.1", linewidth='0', label=r"tanh($\frac{h\omega}{k_B T}$) for $T=0,30,40 \mu $K")
'''

#ax[0, 1].errorbar(om, 1 - 2/(np.exp(2*om/Temp/10**4/6.527) + 1), marker="o", color='purple', linestyle='',
#                  markersize="0.1", label=r"Tanh($\frac{h\Omega_R}{k_B T}$) for $T=0,1,5,10 \mu $K")

#ax[0, 1].errorbar(om,  (1 - 2/(np.exp(2*om/Temp/10**4) + 1))*(0.15 * (om * np.sin(2 * np.pi * om * T) * np.cos(2 * np.pi * Omega * T) - Omega * np.sin(
#    2 * np.pi * Omega * T) * np.cos(2 * np.pi * om * T))
#                  / ((om) ** 2 - Omega ** 2)), marker="o", color='purple', linestyle='',
#                  markersize="0.1")#, label=r"Tanh($T=5*10^{-6}$K)*Re(FT($ \langle \{ \sigma_z(0),\sigma_z(t) \} \rangle$)")

#Temp = 5*10**(-6)

#ax[0, 1].errorbar(omegas, 1 - 2/(np.exp(2*omegas/Temp/10**4) + 1), marker="o", color='grey', linestyle='-',
#                  markersize="1", label="Coth(T)")

#ax[0, 1].errorbar(om, 1 - 2/(np.exp(2*om/Temp/10**4/6.558) + 1), marker="o", color='purple', linestyle='',
#                  markersize="0.03")#, label="Tanh($T=40\mu $K)")

#ax[0, 1].errorbar(om,  (1 - 2/(np.exp(2*om/Temp/10**4) + 1))*(0.15 * (om * np.sin(2 * np.pi * om * T) * np.cos(2 * np.pi * Omega * T) - Omega * np.sin(
#    2 * np.pi * Omega * T) * np.cos(2 * np.pi * om * T))
#                  / ((om) ** 2 - Omega ** 2)), marker="o", color='purple', linestyle='',
#                  markersize="0.02")#, label=r"Tanh($T=5*10^{-5}$K)*Re(FT($ \langle \{ \sigma_z(0),\sigma_z(t) \} \rangle$)")



Temp = 20*10**(-6)

'''
ax[0, 1].errorbar(om,  (1 - 2/(np.exp(prefactor*om/Temp) + 1))*(0.16 * (om * np.sin(2 * np.pi * om * T) * np.cos(2 * np.pi * Omega * T) - Omega * np.sin(
    2 * np.pi * Omega * T) * np.cos(2 * np.pi * om * T))
                  / ((om) ** 2 - Omega ** 2)), marker="o", color='purple', linestyle='', linewidth='0.5',
                  markersize="0.1", label=r" Non-Hermitian $\cdot$ tanh($\frac{\hbar\omega}{2 k_B T}$) for $T=20,200$ $\mu$K")

om=np.linspace(-1.5,1.5,500)

ax[0, 1].errorbar(omegas, (1 - 2/(np.exp(prefactor*om/Temp) + 1))*np.real(integrals0), f1serror*0, marker="", color='purple', linestyle='-', markersize="6",
                  label=r'Measured Non-Hermitian  *tanh(T=20 $\mu$K)')




Temp = 200*10**(-6)

#ax[0, 1].errorbar(omegas, 1 - 2/(np.exp(2*omegas/Temp/10**4) + 1), marker="o", color='purple', linestyle='',
#                  markersize="1")#, label="Coth(T)")

#ax[0, 1].errorbar(om, 1 - 2/(np.exp(2*om/Temp/10**4/6.558) + 1), marker="o", color='purple', linestyle='',
    #              markersize="0.02")#, label="Tanh($T=10^{-4}$K)")

ax[0, 1].errorbar(om,  (1 - 2/(np.exp(prefactor*om/Temp) + 1))*(0.16 * (om * np.sin(2 * np.pi * om * T) * np.cos(2 * np.pi * Omega * T) - Omega * np.sin(
    2 * np.pi * Omega * T) * np.cos(2 * np.pi * om * T))
                  / ((om) ** 2 - Omega ** 2)), marker="o", color='purple', linestyle='', linewidth='0.5',
                  markersize="0.015")


ax[0, 1].errorbar(omegas, (1 - 2/(np.exp(prefactor*om/Temp) + 1))*np.real(integrals0), f1serror, marker="o", color='blue', linestyle='', markersize="6",
                  label=r'Measured Non-Hermitian *tanh(T=200 $\mu$K)')
'''


'''
Temp = 2*10**(-6)


ax[0, 1].errorbar(om,  (1 - 2/(np.exp(prefactor*om/Temp) + 1))*(0.16 * (om * np.sin(2 * np.pi * om * T) * np.cos(2 * np.pi * Omega * T) - Omega * np.sin(
    2 * np.pi * Omega * T) * np.cos(2 * np.pi * om * T))
                  / ((om) ** 2 - Omega ** 2)), marker="o", color='purple', linestyle='', linewidth='0.5',
                  markersize="0.01")
'''




'''
om = om[int(len(om) / 2):len(om)]

ax[0, 1].errorbar(om, (np.heaviside(om, 1) - np.heaviside(-om, 1)) * (0.16 * (om * np.sin(2 * np.pi * om * T) * np.cos(2 * np.pi * Omega * T) - Omega * np.sin(
    2 * np.pi * Omega * T) * np.cos(2 * np.pi * om * T))
                  / ((om) ** 2 - Omega ** 2)), marker="o",
                  color='grey', linestyle='-', markersize="0", linewidth='2', label=r" Non-Hermitian $\cdot$ tanh($\frac{\hbar\omega}{2 k_B T}$) for $T=5,50,100\mu K$")

#ax[0, 1].errorbar(omegas, (np.heaviside(omegas, 1) - np.heaviside(-omegas, 1)) * np.imag(integrals), fserror, marker="o", color='purple', linestyle='', markersize="0",
#                  label=r'Coth(T=0)*Im(FT($ \langle [ \sigma_z(0),\sigma_z(t) ] \rangle$)')

#ax[0, 1].errorbar(omegas, (np.heaviside(omegas, 1) - np.heaviside(-omegas, 1)) * np.real(integrals0), fserror, marker="o", color='grey', linestyle='', markersize="0",
#                  label=r'Tanh(T=0)*Re(FT($ \langle \{ \sigma_z(0),\sigma_z(t) \} \rangle$))')

'''
#\vert \Psi_0 \rangle = \frac{\vert\uparrow\rangle + \vert\downarrow\rangle}{\sqrt{2}}
ax1.set_xlabel('Frequency $\omega$ [$\Omega_R$]', fontsize=8)
ax1.set_ylabel(r'Correlation Spectrum', fontsize=8)
ax1.legend(loc="lower right", fontsize=8, frameon=0) #loc="lower center",
#ax1.tick_params(axis="both", labelsize=8)
ax1.set_xlim([-1.525, 1.525])
ax1.tick_params(axis="both", labelsize=8)

#ax[0, 1].set_ylim([-1.5, 1.5])





#fig.tight_layout()
plt.savefig("ResponseMeasureFFT.pdf")
plt.show()








'''

#NUMERICAL INTEGRAL MANY

omegas = np.linspace(-1.5, 1.5, num=250)

integrals = []
integrals0 = []

x1 = np.linspace(x0[0], x0[-1])#np.array(x0[0:9])

y0 = np.array(y0[0:9])

y31 = np.array(y3[0:9]) #-np.sin(x1*2*np.pi)*0.16

#print("y3", y3)
#print("x0", x0)
#print("omegas", omegas)

for o in omegas:
    integrals.append(2*np.pi*integrate.simps(y3*np.exp(-1j*o*x0*2*np.pi), x0))

for o in omegas:
    integrals0.append(2*np.pi*integrate.simps(y0*np.exp(-1j*o*x0*2*np.pi), x0))

print("integrals", integrals)


fserror = np.ones(len(omegas)) * np.sqrt(ferror[0])

f1serror = np.ones(len(omegas)) * np.sqrt(f1error[0])

ax[1, 0].errorbar(omegas,  np.imag(integrals), color='black', linestyle='-', markersize="0", marker='o', linewidth='1.5',
                  label=r'Hermitian Im(FT($ \langle [ \sigma_z(0),\sigma_z(t) ] \rangle$)')

Temp = 10**(-9)
ax[1, 0].errorbar(omegas, (1 - 2/(np.exp(2*omegas/Temp/10**4/6.558) + 1))*np.real(integrals0), marker="o", color='#85bb65', linestyle='-', markersize="0", linewidth='1.5',
                  label=r"tanh($\frac{h\omega}{k_B T}$)*Re(FT($ \langle \{ \sigma_z(0),\sigma_z(t) \} \rangle$)")



ax[1, 0].fill_between(omegas, (1 - 2/(np.exp(2*omegas/Temp/10**4/6.558) + 1))*np.real(integrals0)+f1serror/2, (1 - 2/(np.exp(2*omegas/Temp/10**4/6.558) + 1))*np.real(integrals0)-f1serror/2,  color='#85bb65', alpha=0.1)

ax[1, 0].fill_between(omegas, np.imag(integrals)+fserror/2, np.imag(integrals)-fserror/2,  color='black', alpha=0.1)



Temp = 1*10**(-5)
ax[1, 0].errorbar(omegas, (1 - 2/(np.exp(2*omegas/Temp/10**4/6.558) + 1))*np.real(integrals0), marker="o", color='#85bb65', linestyle='--', markersize="0", linewidth='1.5',
                  label=r"$T=10 \mu $K")

ax[1, 0].errorbar(omegas, (1 - 2/(np.exp(2*omegas/Temp/10**4/6.558) + 1)), marker="o", color='purple', linestyle='--', markersize="0", linewidth='1.5',
)

Temp = 1.5*10**(-5)
ax[1, 0].errorbar(omegas, (1 - 2/(np.exp(2*omegas/Temp/10**4/6.558) + 1))*np.real(integrals0), marker="o", color='#85bb65', linestyle=':', markersize="0", linewidth='1.5',
                  label=r"$T=15 \mu $K"
                  )

ax[1, 0].errorbar(omegas, (1 - 2/(np.exp(2*omegas/Temp/10**4/6.558) + 1)), marker="o", color='purple', linestyle=':', markersize="0", linewidth='1.5')

Temp = 5*10**(-6)

ax[1, 0].errorbar(omegas, (1 - 2/(np.exp(2*omegas/Temp/10**4/6.558) + 1)), marker="o", color='purple', linestyle='-', markersize="0", linewidth='0.5')

Temp = 2*10**(-6)

ax[1, 0].errorbar(omegas, (1 - 2/(np.exp(2*omegas/Temp/10**4/6.558) + 1)), marker="o", color='purple', linestyle='-', markersize="0", linewidth='0.5')

#ax[1, 0].errorbar(omegas, (np.heaviside(omegas, 1) - np.heaviside(-omegas, 1)) * np.imag(integrals), marker="o",
#                  color='purple', linestyle='--', markersize="0",label=r'Coth(T=0)*Im(FT($ \langle [ \sigma_z(0),\sigma_z(t) ] \rangle$')

ax[1, 0].errorbar(omegas, (np.heaviside(omegas, 1) - np.heaviside(-omegas, 1)), marker="o",
                  color='purple', linestyle='-', markersize="0", linewidth="2")

ax[1, 0].axvline(x=0., color="purple", ymin=0.0499, ymax=0.951)

Omega = 1
T = 1.3
om = ftt[0] / 13.6

#print(om)

#ax[1, 0].errorbar(om, 0.15 * (om * np.sin(2 * np.pi * om * T) * np.cos(2 * np.pi * Omega * T) - Omega * np.sin(
#    2 * np.pi * Omega * T) * np.cos(2 * np.pi * om * T))
#                  / ((om) ** 2 - Omega ** 2), marker="o", color='#85bb65', linestyle='', markersize="1")

#ax[1, 0].errorbar(om, (0.15 * (Omega * np.sin(2 * np.pi * om * T) * np.cos(2 * np.pi * Omega * T) - om * np.sin(
#        2 * np.pi * Omega * T) * np.cos(2 * np.pi * om * T))
#                                                                      / ((om) ** 2 - Omega ** 2)), marker="o",
#                  color='black', linestyle='', markersize="1")

#ax[0, 1].errorbar(om, (np.heaviside(om, 1) - np.heaviside(-om, 1)), marker="o", color='grey', linestyle='',
#                  markersize="0.05", label="Coth(T=0)")

#ax[1, 0].errorbar(om, 0.15 * (Omega * np.sin(2 * np.pi * om * T) * np.cos(2 * np.pi * Omega * T) - om * np.sin(
#    2 * np.pi * Omega * T) * np.cos(2 * np.pi * om * T))
#                  / ((om) ** 2 - Omega ** 2), marker="o", color='black', linestyle='', markersize="1")

om = om[int(len(om) / 2):len(om)]
#ax[1, 0].errorbar(om, (np.heaviside(om, 1) - np.heaviside(-om, 1)) * (0.15 * (
#            Omega * np.sin(2 * np.pi * om * T) * np.cos(2 * np.pi * Omega * T) - om * np.sin(
#        2 * np.pi * Omega * T) * np.cos(2 * np.pi * om * T))
#                                                                      / ((om) ** 2 - Omega ** 2)), marker="o",
#                  color='purple', linestyle='--', markersize="0")

ax[1, 0].set_xlabel(r'Frequency $\omega$ [$\Omega$]', fontsize=20)
ax[1, 0].set_ylabel(r'Correlation Spectrum', fontsize=20)
ax[1, 0].legend(loc="lower right", fontsize=16)
ax[1, 0].set_xlim([-2.5, 2.5])
ax[1, 0].tick_params(axis="both", labelsize=16)

Omega = 1
T = 1.3
om = ftt[0] / 13.6



omegas1 = np.linspace(-1.5, 1.5, num=25)

#integrals = []
#integrals0 = []

x1 = np.linspace(x0[0], x0[-1])#np.array(x0[0:9])

y0 = np.array(y0[0:9])

y31 = np.array(y3[0:9]) #-np.sin(x1*2*np.pi)*0.16

print("y3", y3)
print("x0", x0)
print("omegas", omegas1)

#for o in omegas1:
#    integrals.append(2*np.pi*integrate.simps(y3*np.exp(-1j*o*x0*2*np.pi), x0))

#for o in omegas:
    #integrals0.append(2*np.pi*integrate.simps(y0*np.exp(-1j*o*x0*2*np.pi), x0))





#print(om)

#ax[1, 1].errorbar(freq, np.real(ftty0), ferror, marker="o", color='#85bb65', linestyle='', markersize="6",
  #                label=r'Non-Hermitian Re(FT($ \langle \{ \sigma_z(0),\sigma_z(t) \} \rangle$)')

#ax[1, 1].errorbar(freq, np.imag(ftty3), ferror, marker="o", color='black', linestyle='', markersize="6",
  #                label=r'Hermitian Im(FT($ \langle [ \sigma_z(0),\sigma_z(t) ] \rangle$)')

#ax0[1, 1].errorbar(freq, (np.heaviside(freq, 1) - np.heaviside(-freq, 1)) * np.imag(ftty3), ferror, marker="o",
    #              color='purple', linestyle='', markersize="0.5",
     #             label=r'Coth(T=0) * Im(FT($ \langle [ \sigma_z(0),\sigma_z(t) ] \rangle$)')


#ax[1, 1].errorbar(om, 0.15 * (om * np.sin(2 * np.pi * om * T) * np.cos(2 * np.pi * Omega * T) - Omega * np.sin(
#    2 * np.pi * Omega * T) * np.cos(2 * np.pi * om * T))
#                  / ((om) ** 2 - Omega ** 2), marker="o", color='#85bb65', linestyle='', markersize="1")

#ax[1, 1].errorbar(om, (np.heaviside(om, 1) - np.heaviside(-om, 1)), marker="o", color='purple', linestyle='',
#                  markersize="0.05", label="Coth(T=0)")

#ax[1, 1].errorbar(om, 0.15 * (Omega * np.sin(2 * np.pi * om * T) * np.cos(2 * np.pi * Omega * T) - om * np.sin(
#    2 * np.pi * Omega * T) * np.cos(2 * np.pi * om * T))
#                  / ((om) ** 2 - Omega ** 2), marker="o", color='black', linestyle='', markersize="1")

#om = om[int(len(om) / 2):len(om)]

#ax[1, 1].errorbar(om, (np.heaviside(om, 1) - np.heaviside(-om, 1)) * (0.15 * (
#            Omega * np.sin(2 * np.pi * om * T) * np.cos(2 * np.pi * Omega * T) - om * np.sin(
#        2 * np.pi * Omega * T) * np.cos(2 * np.pi * om * T))
#                                                                      / ((om) ** 2 - Omega ** 2)), marker="o",
#                  color='purple', linestyle='', markersize="0.05")

ax[1, 1].axvline(x=0., color="purple", ymin=0.05, ymax=0.95)

#Temp = 10**(-7)

#ax[1, 1].errorbar(omegas,  (1 - 2/(np.exp(2*omegas/Temp/10**4/6.558) + 1))*np.real(integrals0)- np.imag(integrals), marker="o", color='grey', linestyle='', linewidth='0.5',
                  #markersize="5", label=r"Tanh($\frac{h\Omega_R}{k_B T}$)*Re(FT($ \langle \{ \sigma_z(0),\sigma_z(t) \} \rangle$)-Im")

#ax[1, 1].errorbar(omegas,  np.imag(integrals), marker="o", color='grey', linestyle='', linewidth='0.5',
#                  markersize="5", label=r"Im(FT[])")




#ax[1, 1].errorbar(om,  (1 - 2/(np.exp(2*om/Temp/10**4/6.558) + 1))*(0.16 * (om * np.sin(2 * np.pi * om * T) * np.cos(2 * np.pi * Omega * T) - Omega * np.sin(
   # 2 * np.pi * Omega * T) * np.cos(2 * np.pi * om * T))
  #                / ((om) ** 2 - Omega ** 2)) -

    #              0.16 * (Omega * np.sin(2 * np.pi * om * T) * np.cos(2 * np.pi * Omega * T) - om * np.sin(
   # 2 * np.pi * Omega * T) * np.cos(2 * np.pi * om * T))
     #             / ((om) ** 2 - Omega ** 2)

       #           , marker="o", color='black', linestyle='-', linewidth='0.5',
      #            markersize="0.0", label=r"Tanh($\frac{h\Omega_R}{k_B T}$)*Re(FT($ \langle \{ \sigma_z(0),\sigma_z(t) \} \rangle$)-IMA")


#fserror = np.ones(len(omegas1)) * fserror[0]

print(fserror)

#f1serror = np.ones(len(omegas)) * np.sqrt(f1error[0])

ax[1, 1].errorbar(omegas,  np.imag(integrals), color='black', linestyle='-', markersize="0", marker='o', linewidth='1.5',
                  label=r'Hermitian Im(FT($ \langle [ \sigma_z(0),\sigma_z(t) ] \rangle$)')

#ax[1, 1].errorbar(om, 0.16 * (Omega * np.sin(2 * np.pi * om * T) * np.cos(2 * np.pi * Omega * T) - om * np.sin(
#    2 * np.pi * Omega * T) * np.cos(2 * np.pi * om * T))
#                  / ((om) ** 2 - Omega ** 2), marker="o", color='black', linestyle='', markersize="1")

Temp = 10**(-9)
ax[1, 1].errorbar(omegas, (1 - 2/(np.exp(2*omegas/Temp/10**4/6.558) + 1))*np.real(integrals0), marker="o", color='#85bb65', linestyle='-', markersize="0", linewidth='1.5',
                  label=r"Re(FT($ \langle \{ \sigma_z(0),\sigma_z(t) \} \rangle$ Tanh($\frac{h\Omega_R}{k_B T}$))")


ax[1, 1].fill_between(omegas, (1 - 2/(np.exp(2*omegas/Temp/10**4/6.558) + 1))*np.real(integrals0)+f1serror/2, (1 - 2/(np.exp(2*omegas/Temp/10**4/6.558) + 1))*np.real(integrals0)-f1serror/2,  color='#85bb65', alpha=0.2)

ax[1, 1].fill_between(omegas, np.imag(integrals)+fserror/2, np.imag(integrals)-fserror/2,  color='black', alpha=0.1)



Temp = 1*10**(-5)
ax[1, 1].errorbar(omegas, (1 - 2/(np.exp(2*omegas/Temp/10**4/6.558) + 1))*np.real(integrals0), marker="o", color='#85bb65', linestyle='--', markersize="0", linewidth='1.5',
                  label=r"$T=10 \mu $K")

ax[1, 1].errorbar(omegas, (1 - 2/(np.exp(2*omegas/Temp/10**4/6.558) + 1)), marker="o", color='purple', linestyle='--', markersize="0", linewidth='1.5',
)

Temp = 1.5*10**(-5)
ax[1, 1].errorbar(omegas, (1 - 2/(np.exp(2*omegas/Temp/10**4/6.558) + 1))*np.real(integrals0), marker="o", color='#85bb65', linestyle=':', markersize="0", linewidth='1.5',
                  label=r"$T=15 \mu $K"
                  )

ax[1, 1].errorbar(omegas, (1 - 2/(np.exp(2*omegas/Temp/10**4/6.558) + 1)), marker="o", color='purple', linestyle=':', markersize="0", linewidth='1.5')

Temp = 5*10**(-6)

ax[1, 1].errorbar(omegas, (1 - 2/(np.exp(2*omegas/Temp/10**4/6.558) + 1)), marker="o", color='purple', linestyle='-', markersize="0", linewidth='0.5')

Temp = 2*10**(-6)

ax[1, 1].errorbar(omegas, (1 - 2/(np.exp(2*omegas/Temp/10**4/6.558) + 1)), marker="o", color='purple', linestyle='-', markersize="0", linewidth='0.5')



#ax[1, 0].errorbar(omegas, (np.heaviside(omegas, 1) - np.heaviside(-omegas, 1)) * np.imag(integrals), marker="o",
#                  color='purple', linestyle='--', markersize="0",label=r'Coth(T=0)*Im(FT($ \langle [ \sigma_z(0),\sigma_z(t) ] \rangle$')

ax[1, 1].errorbar(omegas, (np.heaviside(omegas, 1) - np.heaviside(-omegas, 1)), marker="o",
                  color='purple', linestyle='-', markersize="0", linewidth="2")

ax[1, 1].axvline(x=0., color="purple", ymin=0.0499, ymax=0.951)

# ax[0, 1].errorbar(om, ftt[1], marker="o",  color='black', linestyle='', markersize="1")


def psd(data, samples, sample_time):
    long = data
    fs = samples / sample_time
    F, P = signal.welch(
        long, fs,
        nperseg=samples, scaling='spectrum', return_onesided=0)
    return F, P


f = psd(y0, 7, 0.1)

f1 = psd(y3, 7, 0.1)

# ax[0, 1].errorbar(f1[0]/13.6, f1[1], np.ones(len(f1[1]))*f1error[0]**2, marker="o", color='black', label=r'Hermitian', linestyle='', markersize="6")

# ax[0, 1].errorbar(f[0]/13.6, f[1], np.ones(len(f1[1]))*ferror[0]**2, marker="o", linestyle='', color='#85bb65', label=r'Non-Hermitian', markersize="6")


ax[1, 1].set_xlabel('Frequency [$\Omega_R$]', fontsize=20)

ax[1, 1].set_xlabel('Frequency [$\Omega_R$]', fontsize=20)
ax[1, 1].set_ylabel(r'Correlation Spectrum', fontsize=20)
ax[1, 1].legend(loc="lower right", fontsize=16)
ax[1, 1].set_xlim([-1.5, 1.5])
ax[1, 1].tick_params(axis="both", labelsize=16)



ft1 = np.fft.fft(y3)

freq1 = np.fft.fftfreq(ft1.size, x0[1])


# ax[1, 0].set_xlim([-3, 3])


ftty0 = ftty0[0:8]

ftty3 = ftty3[0:8]

freq = freq[0:8]

f1error = f1error[0:8]

#ax[1, 1].errorbar(freq, np.real(ftty0) / np.imag(ftty3), f1error, marker="s", color='black', linestyle='-',
#                  markersize="3", label="nonherm response/herm response")

om = freq

T = 1.3

#ax[1, 1].errorbar(om, 0.15 * (om * np.sin(2 * np.pi * om * T) * np.cos(2 * np.pi * Omega * T) - Omega * np.sin(
#    2 * np.pi * Omega * T) * np.cos(2 * np.pi * om * T))
#                  / ((om) ** 2 - Omega ** 2) / (0.15 * (
#            Omega * np.sin(2 * np.pi * om * T) * np.cos(2 * np.pi * Omega * T) - om * np.sin(
#        2 * np.pi * Omega * T) * np.cos(2 * np.pi * om * T))
#                                                / ((om) ** 2 - Omega ** 2)), marker="o", color='#85bb65', linestyle='-',
#                  markersize="1", label="integral/integral")
'''

# plt.savefig("gamma =  %.2f.png" % (gamma))  # and BW %.2f.pdf" % (noise_amplitude, bandwidth))
