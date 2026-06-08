#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.constants import c, hbar, epsilon_0
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import UnivariateSpline

# =========================================================
# USER SETTINGS
# =========================================================
OPTICAL_FILE = "optical_xx.dat"
EPS_RE_FILE = "epsilon_re.dat"
EPS_IM_FILE = "epsilon_im.dat"

ENERGY_MAX = 8.0
THICKNESS = 2.512e-10
OUTFILE = "optical_proerties.png"

SCIENTIFIC_THRESHOLD = 1e3
RY_TO_EV = 13.605693

# =========================================================
# JOURNAL STYLE
# =========================================================
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "axes.linewidth": 1.4,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

# =========================================================
# SPLINE FITTING (NO EXTRAPOLATION)
# =========================================================
def spline_smooth(x, y, density=10, smooth_factor=0):

    mask = np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    spline = UnivariateSpline(x, y, k=3, s=smooth_factor)

    x_new = np.linspace(x.min(), x.max(), density * len(x))
    y_new = spline(x_new)

    return x_new, y_new

# =========================================================
# AUTO SCI FORMAT
# =========================================================
def auto_scientific_yaxis(ax, data):
    if np.nanmax(np.abs(data)) >= SCIENTIFIC_THRESHOLD:
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(fmt)

# =========================================================
# SAFE LOG
# =========================================================
def apply_safe_log(ax, y):
    if np.nanmin(y) > 0:
        ax.set_yscale("log")

# =========================================================
# VISIBLE SPECTRUM
# =========================================================
def add_visible_spectrum_background(ax):
    ax.axvspan(0.0, 1.65, color="lightgray", alpha=0.20)
    ax.axvspan(1.65, 1.98, color="#ff0000", alpha=0.20)
    ax.axvspan(1.98, 2.10, color="#ff7f00", alpha=0.20)
    ax.axvspan(2.10, 2.19, color="#ffff00", alpha=0.20)
    ax.axvspan(2.19, 2.48, color="#00ff00", alpha=0.20)
    ax.axvspan(2.48, 2.56, color="#00ffff", alpha=0.20)
    ax.axvspan(2.56, 2.75, color="#0000ff", alpha=0.20)
    ax.axvspan(2.75, 3.26, color="#7f00ff", alpha=0.20)
    ax.axvspan(3.26, ENERGY_MAX, color="#dcd6f7", alpha=0.20)

# =========================================================
# READ optical_xx.dat (NO EXTRAPOLATION)
# =========================================================
def read_optical_dat(filename):

    raw = np.loadtxt(filename)

    E = raw[:, 0] * RY_TO_EV
    eps1 = raw[:, 2]
    eps2 = raw[:, 3]
    n = raw[:, 4]
    k = raw[:, 5]
    R = raw[:, 6]
    alpha = raw[:, 7]

    return E, eps1, eps2, n, k, R, alpha

# =========================================================
# READ Birefringence (NO EXTRAPOLATION)
# =========================================================
def read_eps_tensor(re_file, im_file):

    re = np.loadtxt(re_file)
    im = np.loadtxt(im_file)

    E = re[:, 0] * RY_TO_EV

    exx, ezz = re[:, 1], re[:, 3]
    imxx, imzz = im[:, 1], im[:, 3]

    def calc_n(e1, e2):
        mag = np.sqrt(e1**2 + e2**2)
        return np.sqrt((mag + e1) / 2)

    nx = calc_n(exx, imxx)
    nz = calc_n(ezz, imzz)

    biref = nz - nx

    return E, biref

# =========================================================
# COMPUTE
# =========================================================
def compute_all(E, eps1, eps2, n, k, R, alpha):

    omega = E * 1.602176634e-19 / hbar

    delta = np.where(alpha > 0, 1e7 / alpha, np.nan)
    T = (1 - R) * np.exp(-alpha * 1e2 * THICKNESS)

    sigma1 = epsilon_0 * omega * eps2
    sigma2 = epsilon_0 * omega * eps1

    ELF = eps2 / (eps1**2 + eps2**2)

    return dict(
        E=E, eps1=eps1, eps2=eps2,
        n=n, k=k, alpha=alpha, delta=delta,
        R=R, T=T,
        sigma1=sigma1, sigma2=sigma2,
        ELF=ELF
    )

# =========================================================
# LOAD DATA
# =========================================================
E, e1, e2, n, k, R, alpha = read_optical_dat(OPTICAL_FILE)
data = compute_all(E, e1, e2, n, k, R, alpha)

E_b, biref = read_eps_tensor(EPS_RE_FILE, EPS_IM_FILE)

# =========================================================
# COLOR SCHEME
# =========================================================
COLORS = [
    "#1f77b4", "#d62728", "#2ca02c",
    "#9467bd", "#ff7f0e", "#17becf"
]

# =========================================================
# PLOT
# =========================================================
fig, ax = plt.subplots(6, 2, figsize=(7.4, 15.5), constrained_layout=True)
ax = ax.flatten()

def panel(i, x, y, title, ylabel, log=False, color="black"):

    add_visible_spectrum_background(ax[i])

    xs, ys = spline_smooth(
        x,
        y,
        density=10,
        smooth_factor=0
    )

    ax[i].plot(xs, ys, color=color, linewidth=2.6)

    ax[i].set_xlim(x.min(), ENERGY_MAX)

    ax[i].set_title(title)
    ax[i].set_ylabel(ylabel)
    ax[i].set_xlabel("Photon energy (eV)")

    if log:
        positive = ys[ys > 0]
        if len(positive) > 0:
            ax[i].set_yscale("log")
    else:
        auto_scientific_yaxis(ax[i], ys)

# =========================================================
# PANELS
# =========================================================
panel(0, data["E"], data["eps1"], r"$\varepsilon_1$", r"$\varepsilon_1$", color=COLORS[0])
panel(1, data["E"], data["eps2"], r"$\varepsilon_2$", r"$\varepsilon_2$", color=COLORS[1])
panel(2, data["E"], data["n"], "Refractive index", r"$n$", color=COLORS[2])
panel(3, data["E"], data["k"], "Extinction coefficient", r"$k$", color=COLORS[3])

panel(4, data["E"], data["alpha"], "Absorption",
      r"$\alpha$ (cm$^{-1}$)", log=True, color=COLORS[4])

panel(5, data["E"], data["delta"], "Penetration depth",
      r"$\delta$ (nm)", log=True, color=COLORS[5])

panel(6, data["E"], data["R"], "Reflectivity", r"$R$", color=COLORS[0])
panel(7, data["E"], data["T"], "Transmittance", r"$T$", color=COLORS[1])

panel(8, data["E"], data["sigma1"], "Conductivity real",
      r"$\sigma_1$", log=True, color=COLORS[2])

panel(9, data["E"], data["sigma2"], "Conductivity imag",
      r"$\sigma_2$", color=COLORS[3])

panel(10, data["E"], data["ELF"], "Energy loss",
      r"Im$[-1/\varepsilon]$", color="black")

ax[10].set_ylim(0, 0.2)

panel(11, E_b, biref, "Birefringence", r"$\Delta n$", color="#8c564b")

plt.savefig(OUTFILE, dpi=800, bbox_inches='tight')

print("✅ High-quality figures saved (PNG)")

