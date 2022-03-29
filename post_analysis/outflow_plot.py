sys_dir = '/home/yuxuan'
dat_dir = '/home/yuxuan/wind_obs/mcmc_fit'
import sys
sys.path.append('/avatar/yuxuan/research_module/despotic/')
sys.path.append('/home/yuxuan/wind_obs/mcmc_fit/')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import brentq
import matplotlib.cm as cm
import matplotlib.colors as colors
from despotic import emitter
from despotic.winds import pwind, zetaM, sxMach, pM
from astropy.io import fits
# Constants; switch to cgs
from scipy.constants import G, c, m_p, m_e, h
from scipy.constants import k as kB
from astropy.units import Msun, yr, Angstrom, pc
import emcee
import wind_obs_diag_pkg as wodp

c = 1e2*c
hP = h*1e7
G = 1e3*G
kB = 1e7*kB
mH = (m_e + m_p)*1e3
muH = 1.4
Msun = Msun.to('g')
yr = yr.to('s')
Myr = 1e6*yr
Gyr = 1e9*yr
ang = Angstrom.to('cm')
pc = pc.to('cm')
kpc = 1e3*pc
# Fiducial parameters
mdotstar = 4.1*Msun/yr
epsff = 0.01
mach = 100.0
v0 = 120e5*np.sqrt(2)
r0 = 250*pc
m0 = v0**2*r0/(2.0*G)
rho0 = 3.0*m0/(4.0*np.pi*r0**3)
tc = r0/v0
temp = 50.
dist = 3.5e3*kpc
sx = sxMach(mach) # dispersion of density PDF

line = 'CO_2_1'
obj = 'M82'
side='north'
ex = 'area'
p = 'isothermal'
v_cut=0


driver = 'radiation'
pot = 'point'
expansion = 'area'

[phi, theta_in, theta_out, lg_mdot, tau0, uh] = [1.73, 57.29, 78.98, 0.74, 65.55, 0.]


driver = 'radiation'
pot = 'isothermal'
expansion = 'area'

[phi, theta_in, theta_out, lg_mdot, tau0, uh] = [-0.52, 29.19, 80.07, 0.53, 75.18, 0.]

fc=1
phi = phi/90.0*np.pi/2.0
theta_in = theta_in/90.0*np.pi/2.0
theta_out = theta_out/90.*np.pi/2.0
    
# create wind object
mdot = 10**lg_mdot/ (np.cos(theta_in) - np.cos(theta_out)) *Msun/yr # isotropic
eta = (np.cos(theta_in) - np.cos(theta_out)) * mdot / mdotstar
Gamma = brentq(
    lambda g: zetaM(np.log(g), sxMach(mach))/epsff - eta,
    1e-6, 100.)
pw = pwind(Gamma, mach, driver=driver, potential=pot, expansion=expansion, geometry='cone sheath', theta=theta_out, theta_in=theta_in, tau0=tau0, phi=phi, uh=uh, fcrit=fc)
                      
print('mass loading factor is %.2f'%eta)                  
print('x_crit: %.2f'%np.log(Gamma))

def outflow_rate_a(a, mdot, u_c=0.):
    # mass outflow rate at radius a.
    x_m = pw.X(u_c, a) 
    outflow_rate = np.zeros(len(a))
    for i, xmxm in enumerate(x_m):
        x = np.linspace(np.log(Gamma)-300., np.log(Gamma), 3000)
        x1 = np.linspace(np.log(Gamma)-300, xmxm, 3000)
        frac = np.trapz(x1, pM(x1, sx)) / np.trapz(x, pM(x, sx))
        outflow_rate[i] = (np.cos(theta_in) - np.cos(theta_out)) * mdot * frac / (Msun/yr)
    return outflow_rate
    

a = np.linspace(1.1,40.,200)
outf_r_a = outflow_rate_a(a, mdot)
outf_r_a2 = outflow_rate_a(a, mdot, u_c=1)    

plt.figure(1) 
plt.plot(a*0.25, outf_r_a, label='outflow with v>0')
plt.plot(a*0.25, outf_r_a2, label = r'outflow with v>v$_{\rm esc}$')
plt.xlabel("radius [kpc]",fontsize=16);plt.ylabel(r'Mass outflow rate [M$_\odot$/yr]',fontsize=16);plt.legend(loc='upper center', fontsize = '16')
plt.tick_params(axis="y", labelsize=15);plt.tick_params(axis="x", labelsize=15)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.tight_layout()

plt.savefig('figure_pub/outflow_rate_a.pdf')

a = np.linspace(1.1, 40.,200)
f_hvc = f_hvc_a(a)   
print(f_hvc)  
plt.figure(3)
plt.semilogy(a*0.25, f_hvc)

plt.xlabel("radius [kpc]",fontsize=16);plt.ylabel(r'Mass fraction',fontsize=16);
plt.tick_params(axis="y", labelsize=15);plt.tick_params(axis="x", labelsize=15)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.tight_layout()

plt.savefig('figure_pub/f_hvc_a.pdf')   
        
        