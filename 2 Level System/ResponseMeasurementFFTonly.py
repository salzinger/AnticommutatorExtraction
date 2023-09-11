import numpy as np
import matplotlib.font_manager
from Atoms import *
from Driving import *
import matplotlib.pyplot as plt
from scipy import integrate

from lmfit import Model, Parameters


plt.rcParams.update({
    "text.usetex": 1,
    "font.family": "sans-serif",
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


# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# plt.rc('font',**{'family':'sans-serif','sans-serif':['Latin Modern Sans']})


# configure_plots(fontsize_figure=None, fontsize_inset=None, usetex=True)

# plt.rc('figure', figsize=(8.27, 11.69))

N = 1

omega = 0  # MHz
Omega_R = 2 * np.pi * 13.6  # MHz

hermfactor = 1/0.06/2/np.pi  # 1/0.06/2/np.pi  # Delta*t_perturb
nonhermfactor = 0.75/0.25  # 0.75 goes from trace normalised s_z to non-normalised s_z. non-normalised s_z gives 1/0.25 as factor for response function


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

# plt.rc('figure', figsize=(11.69, 8.27))

# fig, ax = plt.subplots(1, 1, figsize=(8.27, 8.27))


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
ynhf0 = []
y0e = []
ynhf0e = []
y3 = []
yhf3 = []
y3e = []
yhf3e = []
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

    ynhf0.append((float(linesm0[element][8:18]))*nonhermfactor)
    yhf3.append((float(linesm3[element][8:18]))*hermfactor)

    yhf3e.append(0.011718*hermfactor)

    ynhf0e.append(0.012739*nonhermfactor)

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


x0 = np.array(x0)

y0 = np.array(y0)

y3 = np.array(y3)

f = psd(y0, 7, 0.1)

f1 = psd(y3, 7, 0.1)

omegas = np.linspace(-2.1, 2.1, num=5001)

integrals = []
integrals0 = []

x1 = np.linspace(x0[0], x0[-1])  # np.array(x0[0:9])

y0 = np.array(y0)

y31 = np.array(y3)  # -np.sin(x1*2*np.pi)*0.16

print("y3", y3)
print("x0", x0)
print("omegas", omegas)

x = np.asarray(x0)
y = np.asarray(y3)
y11 = np.asarray(y0)


def damped_cosine(t, a, d, p, f):
    return (a * np.cos(2 * np.pi * (f * t + p))) * np.exp(-d * t)


def damped_sine(t, a, d, p, f):
    return (a * np.sin(2 * np.pi * (f * t + p))) * np.exp(-d * t)


params = Parameters()
params.add('a', value=-0.1)
params.add('d', value=0, min=0, vary=0)
params.add('p', value=0, vary=0)
params.add('f', value=1, vary=0)

dmodel = Model(damped_sine)
result = dmodel.fit(y, params, weights=np.ones_like(y3) / y3e[0], t=x)
print(result.fit_report())

params1 = Parameters()
params1.add('a', value=0.1)
params1.add('d', value=0, min=0, vary=0)
params1.add('p', value=0, vary=0)
params1.add('f', value=1, vary=0)

dmodel1 = Model(damped_cosine)
result1 = dmodel1.fit(y11, params1, weights=np.ones_like(y3) / y3e[0], t=x)
print(result1.fit_report())

# result.plot_fit(show_init=True)

# plt.show()


y0e = y0e[0:len(y0)]

y3e = y3e[0:len(y0)]

ax2 = plt.subplot(222)

ax2.errorbar(x0, y0, y0e, marker="o", color='#85bb65', linestyle='', markersize="3")#,
             #label=r'$\langle \{ \hat{s}_z(0),\hat{s}_z(t) \} \rangle$')
             #label=r'$\langle \{ \hat{s}_z(0),\hat{s}_z(t) \} \rangle$', linestyle='', markersize="3")

#ax2.plot(perturb_times, np.cos(2 * np.pi * perturb_times) * 0.16, color='#85bb65', linestyle='-')

#ax2.plot(perturb_times, damped_sine(perturb_times, a=result.params.valuesdict()["a"], d=result.params.valuesdict()["d"],
 #                                   p=result.params.valuesdict()["p"], f=result.params.valuesdict()["f"]),
  #       color="black", linestyle='-')

ax2.plot(perturb_times,
         damped_cosine(perturb_times, a=result1.params.valuesdict()["a"], d=result1.params.valuesdict()["d"],
                       p=result.params.valuesdict()["p"], f=result.params.valuesdict()["f"])
         , color='#85bb65', linestyle='--', label=r'$\langle \{ \hat{s}_z(0),\hat{s}_z(t) \} \rangle$')

#ax2.errorbar(x0, y3, y3e, marker="o", color='black', linestyle='', markersize="3",
 #            label=r'$\langle [\hat{s}_z(0),\hat{s}_z(t) ] \rangle$')
             #label=r'$\langle[ \hat{s}_z(0),\hat{s}_z(t)] \rangle$', linestyle='', markersize="3")

#ax2.plot(perturb_times, -np.sin(2 * np.pi * perturb_times) * 0.16, color='black', linestyle='-')

ydiv = []

for n in range(0, len(y0)):
    ydiv.append(y0[n] / y3[n])

ax2.set_xlabel('Time [$2\pi/\Omega_R$]', fontsize=14)
ax2.set_ylabel(r'$\langle \hat{s}_z \rangle - \langle \hat{s}_z \rangle_0$', fontsize=14)
ax2.legend(loc="lower center", fontsize=8, frameon=False)
ax2.set_ylim([-0.25, 0.25])
ax2.set_xlim([-0.005, 1.372])
ax2.tick_params(axis="both", labelsize=8)


'''
# Supplementary Figure Susceptibility
ax = plt.subplot(221)


ax.errorbar(x0, ynhf0, ynhf0e, marker="o", color='#85bb65', linestyle='', markersize="3")
             #label=r'$\langle \{ \hat{s}_z(0),\hat{s}_z(t) \} \rangle$', linestyle='', markersize="3")

#ax2.plot(perturb_times, np.cos(2 * np.pi * perturb_times) * 0.16, color='#85bb65', linestyle='-')

ax.plot(perturb_times, nonhermfactor*damped_sine(perturb_times, a=result.params.valuesdict()["a"], d=result.params.valuesdict()["d"],
                                    p=result.params.valuesdict()["p"], f=result.params.valuesdict()["f"]),
         color="black", linestyle='-')

ax.plot(perturb_times,
         hermfactor*damped_cosine(perturb_times, a=result1.params.valuesdict()["a"], d=result1.params.valuesdict()["d"],
                       p=result.params.valuesdict()["p"], f=result.params.valuesdict()["f"])
         , color='#85bb65', linestyle='-')

ax.errorbar(x0, yhf3, yhf3e, marker="o", color='black', linestyle='', markersize="3")
             #label=r'$\langle[ \hat{s}_z(0),\hat{s}_z(t)] \rangle$', linestyle='', markersize="3")

#ax2.plot(perturb_times, -np.sin(2 * np.pi * perturb_times) * 0.16, color='black', linestyle='-')

ydiv = []

for n in range(0, len(y0)):
    ydiv.append(y0[n] / y3[n])

ax.set_xlabel('Time [$2\pi/\Omega_R$]', fontsize=14)
ax.set_ylabel(r'$\chi_{\hat{s}_z, \hat{P}_{\downarrow}}$', fontsize=18)
ax.legend(loc="lower center", fontsize=8, frameon=False)
ax.set_ylim([-0.6, 0.6])
ax.set_xlim([-0.005, 1.372])
ax.tick_params(axis="both", labelsize=8)
'''

ax = plt.subplot(221)


#ax.errorbar(x0, ynhf0, ynhf0e, marker="o", color='#85bb65', linestyle='', markersize="3")
             #label=r'$\langle \{ \hat{s}_z(0),\hat{s}_z(t) \} \rangle$', linestyle='', markersize="3")

#ax2.plot(perturb_times, np.cos(2 * np.pi * perturb_times) * 0.16, color='#85bb65', linestyle='-')

ax.plot(perturb_times, damped_sine(perturb_times, a=result.params.valuesdict()["a"], d=result.params.valuesdict()["d"],
                                    p=result.params.valuesdict()["p"], f=result.params.valuesdict()["f"]),
         color="black", linestyle='--', label=r'$\langle [ \hat{s}_z(0),\hat{s}_z(t) ] \rangle$')


ax.errorbar(x0, y3, y3e, marker="o", color='black', linestyle='', markersize="3")#,
            #label=r'$\langle [\hat{s}_z(0),\hat{s}_z(t) ] \rangle$')
             #label=r'$\langle[ \hat{s}_z(0),\hat{s}_z(t)] \rangle$', linestyle='', markersize="3")

#ax2.plot(perturb_times, -np.sin(2 * np.pi * perturb_times) * 0.16, color='black', linestyle='-')

ydiv = []

for n in range(0, len(y0)):
    ydiv.append(y0[n] / y3[n])

ax.set_xlabel('Time [$2\pi/\Omega_R$]', fontsize=14)
ax.set_ylabel(r'$\langle \hat{s}_z \rangle - \langle \hat{s}_z \rangle_0$', fontsize=14)
ax.legend(loc="lower center", fontsize=8, frameon=False)
ax.set_ylim([-0.25, 0.25])
ax.set_xlim([-0.005, 1.372])
ax.tick_params(axis="both", labelsize=8)




# plt.savefig("ResponseMeasureTimeDomain.pdf")


# plt.show()

# plt.rc('figure', figsize=(11.69, 8.27))

# fig, ax = plt.subplots(1, 1, figsize=(8.27, 8.27/2))


startpoint = 0
endpoint = 9

for o in omegas:
    integrals.append(
        2 * np.pi * integrate.simps(yhf3[startpoint:endpoint] * np.exp(-1j * o * x0[startpoint:endpoint] * 2 * np.pi),
                                    x0[startpoint:endpoint]))

for o in omegas:
    integrals0.append(
        2 * np.pi * integrate.simps(ynhf0[startpoint:endpoint] * np.exp(-1j * o * x0[startpoint:endpoint] * 2 * np.pi),
                                    x0[startpoint:endpoint]))

#t("integrals", integrals)


def lorentzian(frequencies, amplitude, omega_0, gamma):
    func = lambda omega: amplitude / gamma / np.pi / (2 * ((omega - omega_0) / gamma) ** 2 + 1 / 2)
    return func(frequencies)


ferror = 0

f1error = 0

Nsamples = 10

y0samples = []

y3samples = []

for e in range(0, len(y0e)):
    ferror += ynhf0e[e] ** 2
    f1error += yhf3e[e] ** 2

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

# print(samplederrory0)


ftt = psd(-np.sin(2 * np.pi * 13.6 * np.array(x0)) * 0.16, 7, 0.1)

fttcos = psd(np.cos(2 * np.pi * np.array(x0)) * 0.16, 7, 0.1)

x0 = x0[0:]

y0 = y0[0:]

y3 = y3[0:]

ftty0 = np.fft.fft(y0, n=int(1.5 * len(y0)))

ftty3 = np.fft.fft(y3, n=int(1.5 * len(y0)))

ft = np.fft.fft(y0, n=int(1.5 * len(y0)))

freq = np.fft.fftfreq(ft.size, x0[1])

freqsamples = np.fft.fftfreq(ftty0samples.size, x0[1])

# print(len(ftty0))
# print(len(freq))


fserror = np.ones(len(omegas)) * np.sqrt(ferror)

f1serror = np.ones(len(omegas)) * np.sqrt(f1error)

ferror = np.ones(len(ftty0)) * np.sqrt(ferror)

f1error = np.ones(len(ftty3)) * np.sqrt(f1error)

# ax[0, 1].errorbar(freq, np.real(ftty0), ferror, marker="o", color='#85bb65', linestyle='', markersize="3", label=r"$ NonHermitian  Re FT \langle \{ \sigma_z(0),\sigma_z(t) \} \rangle $")

# ax[0, 1].errorbar(freq, np.imag(ftty3), f1error, marker="s", color='black', linestyle='', markersize="3", label=r"$ Hermitian  Im FT \langle \[ \sigma_z(0),\sigma_z(t) \] \rangle $")

# ax[0, 1].errorbar(freq, np.real(ftty0), ferror, marker="o", color='#85bb65', linestyle='', markersize="3",
#                  label=r'Non-Hermitian Re(FT($ \langle \{ \sigma_z(0),\sigma_z(t) \} \rangle$)')

# ax[0, 1].errorbar(freqsamples, np.real(ftty0samples), marker="o", color='#85bb65', linestyle='', markersize="3",
#                  label=r'SAMPLED Non-Hermitian Re(FT($ \langle \{ \sigma_z(0),\sigma_z(t) \} \rangle$)')

# ax[0, 1].errorbar(freq, np.imag(ftty3), ferror, marker="o", color='black', linestyle='', markersize="3",
#                  label=r'Hermitian Im(FT($ \langle [ \sigma_z(0),\sigma_z(t) ] \rangle$)')

ax1 = plt.subplot(212)



#ax1.errorbar(omegas, np.imag(integrals), fserror , marker="", color='black', linestyle='',
   #          markersize="4",
   #          )

# ax1.errorbar(omegas, nonhermfactor*np.real(integrals0), f1serror*0, marker="", color='#85bb65', linestyle='--', markersize="4",
#                  )

om = np.linspace(-1.5, 1.5, 5001)

Temp = 99 * 10 ** (-6)
# ax[0, 1].errorbar(omegas, (1 - 2/(np.exp(2*omegas/Temp/10**4/6.558) + 1))*np.real(integrals0), marker="o", color='#85bb65', linestyle='--', markersize="0", linewidth='1.5',
#                  label=r"$T=10 \mu $K")

# used to be 2/6.558/10**4 but!: 2*np.pi*13.6 MHz * hbar/k_b = 6.527 * 10^-4

prefactor = 6.527 * 10 ** (-4)

# ax1.errorbar(omegas, nonhermfactor*(1 - 2/(np.exp(prefactor*om/Temp) + 1))*np.real(integrals0), f1serror*0, marker="", color='purple', linestyle='--', markersize="6",
#                  )

'''
ax1.errorbar(omegas, hermfactor / (1 - 2 / (np.exp(prefactor * omegas / Temp) + 1)) * np.imag(integrals), marker="",
             color='purple', linestyle='dotted', markersize="6",
             label="coth * $\chi^{\prime \prime}$")
'''
# ax1.errorbar(om, 1/(1 - 2/(np.exp(prefactor*om/Temp) + 1)), marker="", color='purple', linestyle='--', markersize="6",
#                  label="coth")

# Temp=400*10**(-6)
# ax1.errorbar(omegas, nonhermfactor*(1 - 2/(np.exp(prefactor*omegas/Temp) + 1))*np.real(integrals0), marker="", color='blue', linestyle='-', markersize="1",
# )#label=r'$S(\omega) \tanh{\frac{\hbar\omega}{2 k_B T}}$ at $T=400$ $\mu$K')

# Temp=600*10**(-6)
# ax1.errorbar(omegas, nonhermfactor*(1 - 2/(np.exp(prefactor*omegas/Temp) + 1))*np.real(integrals0), marker="", color='red', linestyle='-', markersize="1",
#            )# label=r'$S(\omega) \tanh{\frac{\hbar\omega}{2 k_B T}}$ at $T=600$ $\mu$K')


omegas = np.linspace(-1.5, 1.5, 7)

integralsshort = []
integralsshort0 = []

for o in omegas:
    integralsshort.append(2 * np.pi * integrate.simps(yhf3 * np.exp(-1j * o * x0 * 2 * np.pi), x0))

for o in omegas:
    integralsshort0.append(2 * np.pi * integrate.simps(ynhf0 * np.exp(-1j * o * x0 * 2 * np.pi), x0))

# ax1.errorbar(omegas, hermfactor*np.imag(integralsshort), hermfactor*fserror[0:len(integralsshort)], marker="o", color='black', linestyle='', markersize="3",
#                  label=r'$\chi^{\prime\prime}(\omega)=\mathcal{I}(\mathcal{F}\langle [ \hat{s}_z(0),\hat{s}_z(t) ] \rangle)$')


# ax1.errorbar(omegas, hermfactor*np.imag(integralsshort)/nonhermfactor/np.real(integralsshort0)/100, marker="", color='purple', linestyle='-', markersize="6",
#                  )

# ax1.errorbar(omegas, (1 - 2/(np.exp(prefactor*omegas/Temp) + 1))/10, marker="", color='purple', linestyle='-', markersize="6",
#                  )


# ax1.errorbar(omegas, nonhermfactor*np.real(integralsshort0), nonhermfactor*f1serror[0:len(integralsshort)], marker="o", color='#85bb65', linestyle='', markersize="3",
#                  label=r'$S(\omega)=\mathcal{R}(\mathcal{F}\langle \{ \hat{s}_z(0),\hat{s}_z(t) \} \rangle)$')

#print("freqs", np.fft.fftfreq(len(y3), d=x0[1]))



padding = 0

yhf3 = yhf3[startpoint:endpoint]
ynhf0 = ynhf0[startpoint:endpoint]
yhf3 = np.pad(yhf3, pad_width=(padding, padding * 0), constant_values=(0, 0))
ynhf0 = np.pad(ynhf0, pad_width=(padding, padding * 0), constant_values=(0, 0))

#print(yhf3)

# y3.append(np.zeros(50))

# ynew = signal.resample(y3, 100)

#t("ffts", np.real(np.fft.ifft(y0)))

'''
ax1.errorbar(np.fft.fftfreq(len(ynhf0), d=x0[1]),
             np.real(np.fft.fft(ynhf0, norm="backward")),
             marker="o", color='#85bb65', linestyle='', markersize="6")

ax1.errorbar(np.fft.fftfreq(len(ynhf0), d=x0[1]),
             np.imag(np.fft.fft(yhf3, norm="backward")),
             marker="o", color='black', linestyle='', markersize="6")
'''





padding = 1000

yhf3 = yhf3[startpoint:endpoint]
ynhf0 = ynhf0[startpoint:endpoint]
yhf3 = np.pad(yhf3, pad_width=(padding*0, padding*1), constant_values=(0, 0))
ynhf0 = np.pad(ynhf0, pad_width=(padding*0, padding*1), constant_values=(0, 0))

#print(y3)

# y3.append(np.zeros(50))

# ynew = signal.resample(y3, 100)





def tanh(t, Temp):
    return 1 - 2 / (np.exp(prefactor * t / Temp) + 1)


params = Parameters()
params.add('Temp', value=10**(-3))

dmodel = Model(tanh)
result = dmodel.fit(np.imag(np.fft.fft(yhf3, norm="backward"))[149:int(len(ynhf0) / 2 - 323)]/np.real(np.fft.fft(ynhf0, norm="backward"))[149:int(len(ynhf0) / 2 - 323)]
                    , params, weights=np.ones_like(np.fft.fftfreq(len(ynhf0), d=x0[1])[149:int(len(ynhf0) / 2 - 323)])/ np.sqrt(2) / fserror[0],
                    t=np.fft.fftfreq(len(ynhf0), d=x0[1])[149:int(len(ynhf0) / 2 - 323)])
print(result.fit_report())



print("freq of begin", np.fft.fftfreq(len(ynhf0), d=x0[1])[149])

print("freq of end", np.fft.fftfreq(len(ynhf0), d=x0[1])[int(len(ynhf0) / 2 - 323)])

print("freq of begin", np.fft.fftfreq(len(ynhf0), d=x0[1])[int(len(ynhf0) / 2) + 324])

print("freq of end", np.fft.fftfreq(len(ynhf0), d=x0[1])[int(len(ynhf0)/2) + 356])


#print("freq of begin",np.fft.fftfreq(len(ynhf0), d=x0[1])[149])

print("maximum hf", np.imag(np.fft.fft(yhf3, norm="backward"))[166:int(len(ynhf0) / 2 - 333)])

print("max hf", np.max(np.imag(np.fft.fft(yhf3, norm="backward"))))

print("maximum nhf", np.real(np.fft.fft(ynhf0, norm="backward"))[176:int(len(ynhf0) / 2 - 323)])

print("freq of maximum hf", np.fft.fftfreq(len(ynhf0), d=x0[1])[166:int(len(ynhf0) / 2 - 333)])

print("freq of maximum nhf", np.fft.fftfreq(len(ynhf0), d=x0[1])[176:int(len(ynhf0) / 2 - 323)])

def tanh(t, Temp):
    return 1 - 2 / (np.exp(prefactor * t / Temp) + 1)


params = Parameters()
params.add('Temp', value=10**(-3))

dmodel = Model(tanh)
result = dmodel.fit(np.imag(np.fft.fft(yhf3, norm="backward"))[int(len(ynhf0) / 2) + 324:int(len(ynhf0)/2) + 356]/np.real(np.fft.fft(ynhf0, norm="backward"))[int(len(ynhf0) / 2) + 324:int(len(ynhf0)/2) + 356]
                    , params, weights=np.ones_like(np.fft.fftfreq(len(ynhf0), d=x0[1])[149:int(len(ynhf0) / 2 - 323)])/ np.sqrt(2) / fserror[0],
                    t=np.fft.fftfreq(len(ynhf0), d=x0[1])[int(len(ynhf0) / 2) + 324:int(len(ynhf0)/2) + 356])
print(result.fit_report())



params = Parameters()
params.add('Temp', value=10**(-3))

dmodel = Model(tanh)
result = dmodel.fit(np.concatenate((np.imag(np.fft.fft(yhf3, norm="backward"))[int(len(ynhf0) / 2) + 324:int(len(ynhf0)/2) + 356],
                    np.imag(np.fft.fft(yhf3, norm="backward"))[145:int(len(ynhf0) / 2 - 323)]) )/
                    np.concatenate((np.real(np.fft.fft(ynhf0, norm="backward"))[int(len(ynhf0) / 2) + 324:int(len(ynhf0)/2) + 356],
                    np.real(np.fft.fft(ynhf0, norm="backward"))[145:int(len(ynhf0) / 2 - 323)]) ),
                     params, weights=np.ones_like(np.concatenate((np.imag(np.fft.fft(yhf3, norm="backward"))[int(len(ynhf0) / 2) + 324:int(len(ynhf0)/2) + 356],
                    np.imag(np.fft.fft(yhf3, norm="backward"))[145:int(len(ynhf0) / 2 - 323)]) ))/ np.sqrt(2)/ fserror[0],
                    t=np.concatenate((np.fft.fftfreq(len(ynhf0), d=x0[1])[int(len(ynhf0) / 2) + 324:int(len(ynhf0)/2) + 356],
                    np.fft.fftfreq(len(ynhf0), d=x0[1])[145:int(len(ynhf0) / 2 - 323)])))
print(result.fit_report())


params = Parameters()
params.add('Temp', value=10**(-3))

dmodel = Model(tanh)
result = dmodel.fit(np.concatenate((np.imag(np.fft.fft(yhf3, norm="backward"))[int(len(ynhf0) / 2) + 327:int(len(ynhf0)/2) + 338],
                    np.imag(np.fft.fft(yhf3, norm="backward"))[168:int(len(ynhf0) / 2 - 325)]) )/
                    np.concatenate((np.real(np.fft.fft(ynhf0, norm="backward"))[int(len(ynhf0) / 2) + 327:int(len(ynhf0)/2) + 338],
                    np.real(np.fft.fft(ynhf0, norm="backward"))[168:int(len(ynhf0) / 2 - 325)]) ),
                     params, weights=np.ones_like(np.concatenate((np.imag(np.fft.fft(yhf3, norm="backward"))[int(len(ynhf0) / 2) + 327:int(len(ynhf0)/2) + 338],
                    np.imag(np.fft.fft(yhf3, norm="backward"))[168:int(len(ynhf0) / 2 - 325)]) ))/ np.sqrt(2)/fserror[0],
                    t=np.concatenate((np.fft.fftfreq(len(ynhf0), d=x0[1])[int(len(ynhf0) / 2) + 327:int(len(ynhf0)/2) + 338],
                    np.fft.fftfreq(len(ynhf0), d=x0[1])[168:int(len(ynhf0) / 2 - 325)])))
print(result.fit_report())

print("residual vector: ", result.residual)

print("maximum hf", np.imag(np.fft.fft(yhf3, norm="backward"))[166:int(len(ynhf0) / 2 - 333)])

print("max hf", np.max(np.imag(np.fft.fft(yhf3, norm="backward"))))

print("maximum nhf", np.real(np.fft.fft(ynhf0, norm="backward"))[176:int(len(ynhf0) / 2 - 323)])

print("freq of maximum hf",np.fft.fftfreq(len(ynhf0), d=x0[1])[166:int(len(ynhf0) / 2 - 333)])

print("freq of maximum nhf",np.fft.fftfreq(len(ynhf0), d=x0[1])[176:int(len(ynhf0) / 2 - 323)])

print("fserrlr: ",  fserror[0])

print("maximum hf", np.imag(np.fft.fft(yhf3, norm="backward"))[int(len(ynhf0) / 2) + 330:int(len(ynhf0)/2) + 345])

print("maximum nhf", np.real(np.fft.fft(ynhf0, norm="backward"))[int(len(ynhf0) / 2) + 320:int(len(ynhf0)/2) + 335])

print("freq of maximum hf",np.fft.fftfreq(len(ynhf0), d=x0[1])[int(len(ynhf0) / 2) + 330:int(len(ynhf0)/2) + 345])

print("freq of maximum nhf",np.fft.fftfreq(len(ynhf0), d=x0[1])[int(len(ynhf0) / 2) + 320:int(len(ynhf0)/2) + 335])

print("arctan(0.9975): ", np.arctanh(0.9975))

print("Temp no error: ", prefactor/2/np.arctanh(0.9975))

print("One std error: ", prefactor/2/np.arctanh(0.9975-0.08245))


print("Monte Carlo error: ", np.nanstd(prefactor/2/np.arctanh(0.9975 + 0.08245 * np.random.randn(1000))))

'''
ax1.errorbar(np.fft.fftfreq(len(ynhf0), d=x0[1])[168:int(len(ynhf0) / 2 - 325)],
             np.imag(np.fft.fft(yhf3, norm="backward"))[168:int(len(ynhf0) / 2 - 325)]/np.real(np.fft.fft(ynhf0, norm="backward"))[168:int(len(ynhf0) / 2 - 325)],
             marker="o", color='blue', linestyle='', markersize="2", label="$\chi^{\prime \prime}(\omega) / S(\omega)$")

ax1.errorbar(np.fft.fftfreq(len(ynhf0), d=x0[1])[int(len(ynhf0) / 2) + 327:int(len(ynhf0)/2) + 338],
             np.imag(np.fft.fft(yhf3, norm="backward"))[int(len(ynhf0) / 2) + 327:int(len(ynhf0)/2) + 338]/np.real(np.fft.fft(ynhf0, norm="backward"))[int(len(ynhf0) / 2) + 327:int(len(ynhf0)/2) + 338],
             marker="o", color='blue', linestyle='', markersize="2", label="$\chi^{\prime \prime}(\omega) / S(\omega)$")


ax1.errorbar(np.fft.fftfreq(len(ynhf0), d=x0[1])[0:int(len(ynhf0) / 2)],
             (
                         1 - 2 / (np.exp(prefactor * np.fft.fftfreq(len(ynhf0), d=x0[1])[0:int(len(ynhf0) / 2)] / Temp) + 1)),
             marker="",
             color='blue', linestyle='--', markersize="6")

ax1.errorbar(np.fft.fftfreq(len(ynhf0), d=x0[1])[int(len(ynhf0) / 2) + 1:int(len(ynhf0))],
             (1 - 2 / (np.exp(
                 prefactor * np.fft.fftfreq(len(ynhf0), d=x0[1])[int(len(ynhf0) / 2) + 1:int(len(ynhf0))] / Temp) + 1)), marker="",
             color='blue', linestyle='--', markersize="6",
             label=r"$tanh( \frac{\hbar \omega}{2 k_B T}) ,  T=%.0f \mu$K"% (Temp*10**6))  # and BW %.2f.pdf" % (noise_amplitude, bandwidth))
'''


ax1.errorbar(np.fft.fftfreq(len(ynhf0), d=x0[1])[0:int(len(ynhf0) / 2)],
             np.real(np.fft.fft(ynhf0, norm="backward"))[0:int(len(ynhf0) / 2)],
             marker="", color='#85bb65', linestyle='-', markersize="2",
             label=r'Fluctuation power spectrum $S(\omega)$')

ax1.errorbar(np.fft.fftfreq(len(ynhf0), d=x0[1])[int(len(ynhf0) / 2) + 1:int(len(ynhf0))],
             np.real(np.fft.fft(ynhf0, norm="backward"))[int(len(ynhf0) / 2) + 1:int(len(ynhf0))],
             marker="", color='#85bb65', linestyle='-', markersize="2")

ax1.errorbar(np.fft.fftfreq(len(ynhf0), d=x0[1])[0:int(len(ynhf0) / 2)],
             np.imag(np.fft.fft(yhf3, norm="backward")[0:int(len(ynhf0) / 2)]),
             marker="", color='black', linestyle='-', markersize="2",
             label=r'Dissipative susceptibility  $\hbar\chi^{\prime \prime}(\omega)$')

ax1.errorbar(np.fft.fftfreq(len(ynhf0), d=x0[1])[int(len(ynhf0) / 2) + 1:int(len(ynhf0))],
             np.imag(np.fft.fft(yhf3, norm="backward")[int(len(ynhf0) / 2) + 1:int(len(ynhf0))]),
             marker="", color='black', linestyle='-', markersize="2")


'''
ax1.errorbar(np.fft.fftfreq(len(ynhf0), d=x0[1])[0:int(len(ynhf0) / 2)],
             np.real(np.fft.fft(ynhf0, norm="backward"))[0:int(len(ynhf0) / 2)]*(
                         1 - 2 / (np.exp(prefactor * np.fft.fftfreq(len(ynhf0), d=x0[1])[0:int(len(ynhf0) / 2)] / Temp) + 1)),
             marker="",
             color='green', linestyle='--', markersize="6")
             
'''

ax1.errorbar(np.fft.fftfreq(len(ynhf0), d=x0[1])[int(len(ynhf0) / 2) + 1:int(len(ynhf0))],
             np.real(np.fft.fft(ynhf0, norm="backward"))[int(len(ynhf0) / 2) + 1:int(len(ynhf0))]*(1 - 2 / (np.exp(
                 prefactor * np.fft.fftfreq(len(ynhf0), d=x0[1])[int(len(ynhf0) / 2) + 1:int(len(ynhf0))] / Temp) + 1)), marker="",
             color='blue', linestyle='--', markersize="6", alpha=0.6,
             #label=r"$S(\omega) \tanh( \frac{\hbar \omega}{2 k_B T})$ at $T=%.0f \mu$K"% (Temp*10**6))
            label=r"$S(\omega) \tanh( \frac{\hbar \omega}{2 k_B T})$ for $k_B T= 0.15 \hbar \Omega_R $")

ax1.errorbar(np.fft.fftfreq(len(ynhf0), d=x0[1])[0:int(len(ynhf0) / 2)],
             np.real(np.fft.fft(ynhf0, norm="backward"))[0:int(len(ynhf0) / 2)]*(1 - 2 / (np.exp(
                 prefactor * np.fft.fftfreq(len(ynhf0), d=x0[1])[0:int(len(ynhf0) / 2)] / Temp) + 1)), marker="",
             color='blue', linestyle='--', markersize="6", alpha=0.6)

ax1.fill_between(np.fft.fftfreq(len(ynhf0), d=x0[1])[int(len(ynhf0) / 2) + 1:int(len(ynhf0))],
                 ((1 - 2 / (np.exp(
                     prefactor * np.fft.fftfreq(len(ynhf0), d=x0[1])[int(len(ynhf0) / 2) + 1:int(len(ynhf0))] / Temp) + 1))
                   * np.real(np.fft.fft(ynhf0, norm="backward"))[int(len(ynhf0) / 2) + 1:int(len(ynhf0))] - fserror[
                                                                                                   0:int(len(ynhf0) / 2)]),
                 ((1 - 2 / (np.exp(
                     prefactor * np.fft.fftfreq(len(ynhf0), d=x0[1])[int(len(ynhf0) / 2) + 1:int(len(ynhf0))] / Temp) + 1))
                   * np.real(np.fft.fft(ynhf0, norm="backward"))[int(len(ynhf0) / 2) + 1:int(len(ynhf0))] + fserror[
                                                                                                   0:int(len(ynhf0) / 2)]),
                 color="blue", alpha=0.2)


ax1.fill_between(np.fft.fftfreq(len(ynhf0), d=x0[1])[0:int(len(ynhf0) / 2)],
                 ((1 - 2 / (
                             np.exp(prefactor * np.fft.fftfreq(len(ynhf0), d=x0[1])[0:int(len(ynhf0) / 2)] / (Temp)) + 1))
                   * np.real(np.fft.fft(ynhf0, norm="backward"))[0:int(len(ynhf0) / 2)] - fserror[0:int(len(ynhf0) / 2)]),
                 ((1 - 2 / (
                         np.exp(prefactor * np.fft.fftfreq(len(ynhf0), d=x0[1])[0:int(len(ynhf0) / 2)] / (Temp)) + 1))
                   * np.real(np.fft.fft(ynhf0, norm="backward"))[0:int(len(ynhf0) / 2)] + fserror[0:int(len(ynhf0) / 2)]),
                 color="blue", alpha=0.2)


'''
ax1.errorbar(np.fft.fftfreq(len(ynhf0), d=x0[1])[0:int(len(ynhf0) / 2)],
             1 / (
                         1 - 2 / (np.exp(prefactor * np.fft.fftfreq(len(ynhf0), d=x0[1])[0:int(len(ynhf0) / 2)] / Temp) + 1))
             * np.imag(np.fft.fft(yhf3, norm="backward")[0:int(len(ynhf0) / 2)]), marker="",
             color='grey', linestyle='--', markersize="6")


ax1.errorbar(np.fft.fftfreq(len(ynhf0), d=x0[1])[int(len(ynhf0) / 2) + 1:int(len(ynhf0))],
             1 / (1 - 2 / (np.exp(
                 prefactor * np.fft.fftfreq(len(ynhf0), d=x0[1])[int(len(ynhf0) / 2) + 1:int(len(ynhf0))] / Temp) + 1))
             * np.imag(np.fft.fft(yhf3, norm="backward")[int(len(ynhf0) / 2) + 1:int(len(ynhf0))]), marker="",
             color='grey', linestyle='--', markersize="6",
             label=r"coth$( \frac{\hbar \omega}{2 k_B T})\chi^{\prime \prime} ,  T=%.0f \mu$K"% (Temp*10**6))

ax1.fill_between(np.fft.fftfreq(len(ynhf0), d=x0[1])[int(len(ynhf0) / 2) + 1:int(len(ynhf0))],
                 (1 / (1 - 2 / (np.exp(
                     prefactor * np.fft.fftfreq(len(ynhf0), d=x0[1])[int(len(ynhf0) / 2) + 1:int(len(ynhf0))] / Temp) + 1))
                   * np.imag(np.fft.fft(yhf3, norm="backward")[int(len(ynhf0) / 2) + 1:int(len(ynhf0))]) - fserror[
                                                                                                   0:int(len(ynhf0) / 2)]),
                 (1 / (1 - 2 / (np.exp(
                     prefactor * np.fft.fftfreq(len(ynhf0), d=x0[1])[int(len(ynhf0) / 2) + 1:int(len(ynhf0))] / Temp) + 1))
                   * np.imag(np.fft.fft(yhf3, norm="backward")[int(len(ynhf0) / 2) + 1:int(len(ynhf0))]) + fserror[
                                                                                                   0:int(len(ynhf0) / 2)]),
                 color="purple", alpha=0.2)


ax1.fill_between(np.fft.fftfreq(len(ynhf0), d=x0[1])[0:int(len(ynhf0) / 2)],
                 (1 / (1 - 2 / (
                             np.exp(prefactor * np.fft.fftfreq(len(ynhf0), d=x0[1])[0:int(len(ynhf0) / 2)] / Temp) + 1))
                   * np.imag(np.fft.fft(yhf3, norm="backward")[0:int(len(ynhf0) / 2)]) - fserror[0:int(len(ynhf0) / 2)]),
                 (1 / (1 - 2 / (
                         np.exp(prefactor * np.fft.fftfreq(len(ynhf0), d=x0[1])[0:int(len(ynhf0) / 2)] / Temp) + 1))
                   * np.imag(np.fft.fft(yhf3, norm="backward")[0:int(len(ynhf0) / 2)]) + fserror[0:int(len(ynhf0) / 2)]),
                 color="purple", alpha=0.2)

'''

ax1.fill_between(np.fft.fftfreq(len(ynhf0), d=x0[1])[0:int(len(ynhf0) / 2)],
                 (np.real(np.fft.fft(ynhf0, norm="backward")[0:int(len(ynhf0) / 2)]) - fserror[
                                                                                  0:int(len(ynhf0) / 2)]),
                 (np.real(np.fft.fft(ynhf0, norm="backward")[0:int(len(ynhf0) / 2)]) + fserror[
                                                                                  0:int(len(ynhf0) / 2)]),
                 color="#85bb65", alpha=0.25)

ax1.fill_between(np.fft.fftfreq(len(ynhf0), d=x0[1])[0:int(len(ynhf0) / 2)],
                 (np.imag(np.fft.fft(yhf3, norm="backward")[0:int(len(ynhf0) / 2)]) - fserror[
                                                                                  0:int(len(ynhf0) / 2)]),
                 (np.imag(np.fft.fft(yhf3, norm="backward")[0:int(len(ynhf0) / 2)]) + fserror[
                                                                                  0:int(len(ynhf0) / 2)]),
                 color="grey", alpha=0.3)
'''
ax1.fill_between(np.fft.fftfreq(len(ynhf0), d=x0[1])[0:int(len(ynhf0) / 2)],
                 (np.real(np.fft.fft(ynhf0, norm="backward")[0:int(len(ynhf0) / 2)]) - fserror[
                                                                                  0:int(len(ynhf0) / 2)]),
                 (np.real(np.fft.fft(ynhf0, norm="backward")[0:int(len(ynhf0) / 2)]) + fserror[
                                                                                  0:int(len(ynhf0) / 2)]),
                 color="#85bb65", alpha=0.25)
'''
ax1.fill_between(np.fft.fftfreq(len(yhf3), d=x0[1])[int(len(ynhf0) / 2) + 1:int(len(ynhf0))],
                 (np.imag(np.fft.fft(yhf3, norm="backward")[int(len(ynhf0) / 2) + 1:int(len(ynhf0))]) - fserror[
                                                                                                 int(len(
                                                                                                     ynhf0) / 2) + 1:int(
                                                                                                     len(ynhf0))]),
                 (np.imag(np.fft.fft(yhf3, norm="backward")[int(len(ynhf0) / 2) + 1:int(len(ynhf0))]) + fserror[
                                                                                                 int(len(
                                                                                                     ynhf0) / 2) + 1:int(
                                                                                                     len(ynhf0))]),
                 color="grey", alpha=0.3)

ax1.fill_between(np.fft.fftfreq(len(ynhf0), d=x0[1])[int(len(ynhf0) / 2) + 1:int(len(ynhf0))],
                 (np.real(np.fft.fft(ynhf0, norm="backward")[int(len(ynhf0) / 2) + 1:int(len(ynhf0))]) - fserror[
                                                                                                 int(len(
                                                                                                     ynhf0) / 2) + 1:int(
                                                                                                     len(ynhf0))]),
                 (np.real(np.fft.fft(ynhf0, norm="backward")[int(len(ynhf0) / 2) + 1:int(len(ynhf0))]) + fserror[
                                                                                                 int(len(
                                                                                                     ynhf0) / 2) + 1:int(
                                                                                                     len(ynhf0))]),
                 color="#85bb65", alpha=0.25)

# \vert \Psi_0 \rangle = \frac{\vert\uparrow\rangle + \vert\downarrow\rangle}{\sqrt{2}}
ax1.set_xlabel('Frequency $\omega$ [$\Omega_R$]', fontsize=14)
ax1.set_ylabel(r'Spectrum', fontsize=14)
ax1.legend(loc=(.53, 0), fontsize=8, frameon=0)  # loc="lower center",
# ax1.tick_params(axis="both", labelsize=8)
ax1.set_xlim([-1.7, 1.7])
ax1.tick_params(axis="both", labelsize=8)
ax1.set_ylim([-2.6, 2.6])


plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)

# fig.tight_layout()
plt.savefig("ResponseMeasureFFT.pdf")
plt.show()


times = np.linspace(0, 8, 100)

'''
ax1.errorbar(np.fft.fftfreq(len(times), d=times[1]),
             np.real(np.fft.fft(np.cos(2*np.pi*times)*np.exp(-0.26*times))) * nonhermfactor/2/np.pi,
             marker="", color='green', linestyle='-', markersize="2",
             label=r'$S(\omega)$ from fit')
'''
x1 = 2

x2 = -2

d = 0.26

'''
ax1.errorbar(om,
             -d ** 2 / (2 * np.exp(d * x1) * (d ** 3 + 4 * d * om ** 2)) - (2 * om ** 2) / (
                         np.exp(d * x1) * (d ** 3 + 4 * d * om ** 2))
             - (d ** 2 * np.cos(2 * om * x1)) / (2 * np.exp(d * x1) * (d ** 3 + 4 * d * om ** 2)) + (
                         d * om * np.sin(2 * om * x1)) / (np.exp(d * x1) * (d ** 3 + 4 * d * om ** 2))
             - (-d ** 2 / (2 * np.exp(d * x2) * (d ** 3 + 4 * d * om ** 2)) - (2 * om ** 2) / (
                         np.exp(d * x2) * (d ** 3 + 4 * d * om ** 2))
                - (d ** 2 * np.cos(2 * om * x2)) / (2 * np.exp(d * x2) * (d ** 3 + 4 * d * om ** 2)) + (
                            d * om * np.sin(2 * om * x2)) / (np.exp(d * x2) * (d ** 3 + 4 * d * om ** 2))), marker="",
             color='black', linestyle='-', markersize="2",
             label=r'$S(\omega)$ analytic')

'''

#print(len(y3[startpoint:endpoint]))
#print(x0[1])

#print(y3[startpoint:endpoint])

Temp = 5 * 10 ** (-6)
# ax1.errorbar(omegas, nonhermfactor*(1 - 2/(np.exp(prefactor*omegas/Temp) + 1))*np.real(integralsshort0), nonhermfactor*f1serror[0:len(integralsshort)], marker="o", color='purple', linestyle='', markersize="3",
#                  label=r'$S(\omega) \tanh{\frac{\hbar\omega}{2 k_B T}}$ at $T=5$ $\mu$K')


# ax[0, 1].fill_between(omegas, np.imag(integrals)+ferror[0],np.imag(integrals)-ferror[0], color='grey',
#                  alpha=0.5)


# ax[0, 1].errorbar(freq, (np.heaviside(freq, 1) - np.heaviside(-freq, 1)) * np.imag(ftty3), ferror, marker="o",
#                  color='purple', linestyle='', markersize="0.5",
#                  label=r'Coth(T=0) * Im(FT($ \langle [ \sigma_z(0),\sigma_z(t) ] \rangle$)')

# ax[0, 1].axvline(x=0., color="grey", ymin=0.05, ymax=0.95)
#print("ftt: ", np.max(ftt[0]))
Omega = 1
T = 1.36
# om = ftt[0] / 13.6
om = np.linspace(-1.525, 1.525, 10000)

# print(om)

'''

ax[0, 1].errorbar(om, 0.16 * (Omega * np.sin(2 * np.pi * om * T) * np.cos(2 * np.pi * Omega * T) - om * np.sin(
    2 * np.pi * Omega * T) * np.cos(2 * np.pi * om * T))
                  / ((om) ** 2 - Omega ** 2), marker="o", color='black', linestyle='', markersize="1", label="Analytical Hermitian response function")

ax[0, 1].errorbar(om, 0.16 * (om * np.sin(2 * np.pi * om * T) * np.cos(2 * np.pi * Omega * T) - Omega * np.sin(
    2 * np.pi * Omega * T) * np.cos(2 * np.pi * om * T))
                  / ((om) ** 2 - Omega ** 2), marker="o", color='#85bb65', linestyle='', markersize="1", label="Analytical Non-Hermitian response function")
'''
# ax[0, 1].errorbar(om, (np.heaviside(om, 1) - np.heaviside(-om, 1)), marker="o", color='grey', linestyle='',
#                  markersize="0.05", label="Coth(T=0)")


'''
ax[0, 1].errorbar(om, (np.heaviside(om, 1) - np.heaviside(-om, 1)), marker="o", color='purple', linestyle='',
                  markersize="0.1")#, label="Tanh(T=0)")


ax[0, 1].axvline(x=0., color="purple", ymin=0.049, ymax=0.951, linewidth='2')
'''

# Temp = 10**(-6)

# ax[0, 1].errorbar(omegas, 1 - 2/(np.exp(2*omegas/Temp/10**4) + 1), marker="o", color='grey', linestyle='-',
#                  markersize="1", label="Coth(T)")


Temp = 5 * 10 ** (-6)
# ax[0, 1].errorbar(omegas, (1 - 2/(np.exp(2*omegas/Temp/10**4/6.558) + 1))*np.real(integrals0), marker="o", color='#85bb65', linestyle='--', markersize="0", linewidth='1.5',
#                  label=r"$T=10 \mu $K")

# used to be 2/6.558/10**4 but!: 2*np.pi*13.6*10^6 MHz * hbar/k_b = 2*np.pi* 1.04 * 10^-4

prefactor = 2 * np.pi * 1.04 * 10 ** (-4)

# ax[0, 1].errorbar(om, (1 - 2/(np.exp(prefactor*om/Temp) + 1)), marker="o", color='purple', linestyle='', markersize="0.05", linewidth='0')

# Temp = 1.5*10**(-5)
# ax[0, 1].errorbar(omegas, (1 - 2/(np.exp(2*omegas/Temp/10**4/6.558) + 1))*np.real(integrals0), marker="o", color='#85bb65', linestyle=':', markersize="0", linewidth='1.5',
#                  label=r"$T=15 \mu $K"
#                  )

# ax[0, 1].errorbar(om, (1 - 2/(np.exp(prefactor*om/Temp) + 1)), marker="o", color='purple', linestyle='', markersize="0.05", linewidth='0')

'''
Temp = 50*10**(-6)

#ax[0, 1].errorbar(om, (1 - 2/(np.exp(prefactor*om/Temp) + 1)), marker="o", color='purple', linestyle='', markersize="0.1", linewidth='0')

#Temp = 2*10**(-6)

ax[0, 1].errorbar(om, (1 - 2/(np.exp(prefactor*om/Temp) + 1)), marker="o", color='purple', linestyle='', markersize="0.1", linewidth='0', label=r"tanh($\frac{h\omega}{k_B T}$) for $T=0,30,40 \mu $K")
'''

# ax[0, 1].errorbar(om, 1 - 2/(np.exp(2*om/Temp/10**4/6.527) + 1), marker="o", color='purple', linestyle='',
#                  markersize="0.1", label=r"Tanh($\frac{h\Omega_R}{k_B T}$) for $T=0,1,5,10 \mu $K")

# ax[0, 1].errorbar(om,  (1 - 2/(np.exp(2*om/Temp/10**4) + 1))*(0.15 * (om * np.sin(2 * np.pi * om * T) * np.cos(2 * np.pi * Omega * T) - Omega * np.sin(
#    2 * np.pi * Omega * T) * np.cos(2 * np.pi * om * T))
#                  / ((om) ** 2 - Omega ** 2)), marker="o", color='purple', linestyle='',
#                  markersize="0.1")#, label=r"Tanh($T=5*10^{-6}$K)*Re(FT($ \langle \{ \sigma_z(0),\sigma_z(t) \} \rangle$)")

# Temp = 5*10**(-6)

# ax[0, 1].errorbar(omegas, 1 - 2/(np.exp(2*omegas/Temp/10**4) + 1), marker="o", color='grey', linestyle='-',
#                  markersize="1", label="Coth(T)")

# ax[0, 1].errorbar(om, 1 - 2/(np.exp(2*om/Temp/10**4/6.558) + 1), marker="o", color='purple', linestyle='',
#                  markersize="0.03")#, label="Tanh($T=40\mu $K)")

# ax[0, 1].errorbar(om,  (1 - 2/(np.exp(2*om/Temp/10**4) + 1))*(0.15 * (om * np.sin(2 * np.pi * om * T) * np.cos(2 * np.pi * Omega * T) - Omega * np.sin(
#    2 * np.pi * Omega * T) * np.cos(2 * np.pi * om * T))
#                  / ((om) ** 2 - Omega ** 2)), marker="o", color='purple', linestyle='',
#                  markersize="0.02")#, label=r"Tanh($T=5*10^{-5}$K)*Re(FT($ \langle \{ \sigma_z(0),\sigma_z(t) \} \rangle$)")


Temp = 20 * 10 ** (-6)

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
