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

hermfactor = 1 / 0.06 / 2 / np.pi  # 1/0.06/2/np.pi  # Delta*t_perturb
nonhermfactor = 0.75 / 0.25  # 0.75 goes from trace normalised s_z to non-normalised s_z. non-normalised s_z gives 1/0.25 as factor for response function

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


with open('control_pulse.txt') as f:
    linesm0 = f.readlines()
with open('perturbation_pulse.txt') as f:
    linesm3 = f.readlines()

with open('control_error.txt') as f:
    linesm0e = f.readlines()
with open('perturbation_error.txt') as f:
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

for element in range(1, 22 - 5):
    x0.append(float(linesm0[element][0:5]))

    y0.append((float(linesm0[element][8:18])))
    y3.append((float(linesm3[element][8:18])))

    y3e.append((float(linesm0e[element][8:18])))

    y0e.append((float(linesm0e[element][8:18])))

print(y0)


def damped_cosine(t, a):
    return (a * np.cos(2 * np.pi * (t)))


def damped_sine(t, a, d, p, f):
    return (a * np.sin(2 * np.pi * (f * t + p)) * np.exp(-d * t))


def lin_fit(t, a):
    return (a * t - 0.5)


params = Parameters()
params.add('a', value=0.5, vary=0)
params.add('d', value=40, vary=0, min=0)
params.add('p', value=-0.25, vary=0)
params.add('f', value=3, vary=1)

dmodel = Model(damped_sine)
result = dmodel.fit(y3[0:7], params, weights=np.ones_like(y3[0:7]) / y3e[0:7], t=x0[0:7])
print(result.fit_report())

paramslin = Parameters()
paramslin.add('a', value=30, vary=1)

dmodellin = Model(lin_fit)
resultlin = dmodellin.fit(y3[0:7], paramslin, weights=np.ones_like(y3[0:7]) / y3e[0:7], t=x0[0:7])
print(resultlin.fit_report())

params1 = Parameters()
params1.add('a', value=0.5, vary=0)
params1.add('d', value=0.01, min=0)
params1.add('p', value=-0.25, vary=0)
params1.add('f', value=17.7, vary=1)

dmodel1 = Model(damped_sine)
result1 = dmodel1.fit(y0, params1, weights=np.ones_like(y0) / y0e, t=x0)
print(result1.fit_report())

perturb_times = np.linspace(x0[0], x0[-1], 100)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

ax.plot(perturb_times,
        damped_sine(perturb_times, a=result1.params.valuesdict()["a"], d=result1.params.valuesdict()["d"],
                    p=result1.params.valuesdict()["p"],
                    f=result1.params.valuesdict()["f"])
        , color='black', linestyle='-')

ax.plot(perturb_times,
        damped_sine(perturb_times, a=result.params.valuesdict()["a"], d=result.params.valuesdict()["d"],
                    p=result.params.valuesdict()["p"],
                    f=result.params.valuesdict()["f"])
        , color='#85bb65', linestyle='-')

ax.plot(perturb_times,
        lin_fit(perturb_times, a=resultlin.params.valuesdict()["a"])
        , color='#85bb65', linestyle=':')

ax.errorbar(x0, y0, y0e, marker="o", color='black', linestyle='', markersize="6")

ax.errorbar(x0, y3, y3e, marker="o", color='#85bb65', linestyle='', markersize="6")

ax.plot(x0, np.ones_like(x0) * 0.5, linestyle="--", color="grey")

ax.plot(x0, np.ones_like(x0) * 0., linestyle="--", color="grey")

ax.plot(x0, -np.ones_like(x0) * 0.5, linestyle="--", color="grey")

ax.set_xlim([0, 0.0755])
ax.set_ylim([-0.6, 0.6])
ax.tick_params(axis="both", labelsize=18)
ax.set_xticks(ticks=np.array([0., 0.03, 0.06]))
ax.set_yticks(ticks=np.array([-0.5, 0., 0.5]))
ax.set_ylabel(r"$0.5(N_a - N_\downarrow)/(N_a + N_\downarrow) $", fontsize="24")
ax.set_xlabel(r"Time [$\mu$s]", fontsize="24")

plt.subplots_adjust(left=0.15, bottom=0.15, right=None, top=None, wspace=None, hspace=None)



plt.savefig("SuppFig.pdf")  # and BW %.2f.pdf" % (noise_amplitude, bandwidth))

plt.show()
