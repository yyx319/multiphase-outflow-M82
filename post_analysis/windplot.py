"""
Script to produce plots describing the wind in Yuan, Krumholz, & Martin (2021)
"""

from despotic.winds import pwind
from despotic import emitter
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from multiprocessing import Pool, cpu_count

# Constants for the galaxy
M0 = 8.2e8 * u.Msun
v0 = 170 * u.km/u.s
r0 = 0.25 * u.kpc
SFR = 4.1 * u.Msun/u.yr

# Driver, expansion, potential, geometry choices for all cases
driver = 'ideal'
potential = 'point'
expansion = 'area'
geometry = 'cone sheath'

#####################################
# HI 21 cm -- North
#####################################

# Parameters for highest-likelihood model returned by the MCMC
GammaHIN = 0.28686539242939585
MachHIN = 10.**0.63285017
phiHIN = 7.40147048 * np.pi/180.  # Degree to radian
theta0HIN = 59.5415857 * np.pi/180.
theta1HIN = 82.44179159 * np.pi/180.
saHIN = 2.*np.pi*(np.cos(theta0HIN) - np.cos(theta1HIN))
MdotHIN = 13.95951217 * u.Msun / u.yr * saHIN/(4*np.pi)

# Construct pwind object representing model
pwHIN = pwind(GammaHIN, MachHIN, driver = driver, potential = potential,
              expansion = expansion, geometry = geometry,
              theta = theta1HIN, theta_in = theta0HIN,
              phi = phiHIN)

#####################################
# HI 21 cm -- South
#####################################

# Parameters for highest-likelihood model returned by the MCMC
GammaHIS = 0.24397622820600418
MachHIS = 10.**1.00832836
phiHIS = 8.10014953 * np.pi/180.  # Degree to radian
theta0HIS = 68.18541005 * np.pi/180.
theta1HIS = 81.3888468 * np.pi/180.
saHIS = 2.*np.pi*(np.cos(theta0HIS) - np.cos(theta1HIS))
MdotHIS = 41.09921085 * u.Msun / u.yr * saHIS/(4*np.pi)

# Construct pwind object representing model
pwHIS = pwind(GammaHIS, MachHIS, driver = driver, potential = potential,
              expansion = expansion, geometry = geometry,
              theta = theta1HIS, theta_in = theta0HIS,
              phi = phiHIS)


#####################################
# CO 2-1 -- North
#####################################

# Parameters for highest-likelihood model returned by the MCMC
#GammaCON = 0.16356043722937696
#MachCON = 10.**0.99405173
#phiCON = -4.46018991 * np.pi/180.  # Degree to radian
#theta0CON = 10.03801425 * np.pi/180.
#theta1CON = 78.57394613 * np.pi/180.
#saCON = 2.*np.pi*(np.cos(theta0CON) - np.cos(theta1CON))
#MdotCON = 3.53894698 * u.Msun / u.yr * saCON/(4*np.pi)
GammaCON = 0.15986319559291706
MachCON = 10**1.01308681
phiCON = -3.06784127 * np.pi/180.
theta0CON = 13.26710684 * np.pi/180.
theta1CON = 75.48878296 * np.pi/180.
saCON = 2.*np.pi*(np.cos(theta0CON) - np.cos(theta1CON))
MdotCON = 3.83168797  * u.Msun / u.yr * saCON/(4*np.pi)

# Construct pwind object representing model
pwCON = pwind(GammaCON, MachCON, driver = driver, potential = potential,
              expansion = expansion, geometry = geometry,
              theta = theta1CON, theta_in = theta0CON,
              phi = phiCON)

#####################################
# CO 2-1 -- South
#####################################

# Parameters for highest-likelihood model returned by the MCMC
GammaCOS = 0.040180202352394445
MachCOS = 10.**1.97971914
phiCOS = 16.4825320 * np.pi/180.  # Degree to radian
theta0COS = 8.39444821 * np.pi/180.
theta1COS = 51.7877410 * np.pi/180.
saCOS = 2.*np.pi*(np.cos(theta0COS) - np.cos(theta1COS))
MdotCOS = 2.74728123 * u.Msun / u.yr * saCOS/(4*np.pi)

# Construct pwind object representing model
pwCOS = pwind(GammaCOS, MachCOS, driver = driver, potential = potential,
              expansion = expansion, geometry = geometry,
              theta = theta1COS, theta_in = theta0COS,
              phi = phiCOS)

#####################################
# Compute profiles
#####################################

# Radial range
r = np.logspace(np.log10(0.25)+1e-5, 0.75, 200)*u.kpc
a = (r/r0).to('')

# Units and scalings from dimensionless to dimensional units
rhounit = u.g / u.cm**3
pdotunit = u.Msun/u.yr * u.km/u.s
Edotunit = u.erg/u.s
rhoHINscale = (MdotHIN / (saHIN * r0**2 * v0)).to(rhounit)
rhoCONscale = (MdotCON / (saCON * r0**2 * v0)).to(rhounit)
pdotHINscale = (MdotHIN * v0).to(pdotunit)
pdotCONscale = (MdotCON * v0).to(pdotunit)
EdotHINscale = (MdotHIN * v0**2).to(Edotunit)
EdotCONscale = (MdotCON * v0**2).to(Edotunit)
rhoHISscale = (MdotHIS / (saHIS * r0**2 * v0)).to(rhounit)
rhoCOSscale = (MdotCOS / (saCOS * r0**2 * v0)).to(rhounit)
pdotHISscale = (MdotHIS * v0).to(pdotunit)
pdotCOSscale = (MdotCOS * v0).to(pdotunit)
EdotHISscale = (MdotHIS * v0**2).to(Edotunit)
EdotCOSscale = (MdotCOS * v0**2).to(Edotunit)

# Profiles
rhoHIN = pwHIN.rho(a) * rhoHINscale
rhoCON = pwHIN.rho(a) * rhoCONscale
pdotHIN = pwHIN.pdot(a) * pdotHINscale
pdotCON = pwCON.pdot(a) * pdotCONscale
EdotHIN = pwHIN.Edot(a) * EdotHINscale
EdotCON = pwCON.Edot(a) * EdotCONscale
rhoHIS = pwHIS.rho(a) * rhoHISscale
rhoCOS = pwHIS.rho(a) * rhoCOSscale
pdotHIS = pwHIS.pdot(a) * pdotHISscale
pdotCOS = pwCOS.pdot(a) * pdotCOSscale
EdotHIS = pwHIS.Edot(a) * EdotHISscale
EdotCOS = pwCOS.Edot(a) * EdotCOSscale


#####################################
# Make plot
#####################################

xlim = np.array([-0.65, 0.75])
rholim = np.array([-25, -20.5])
vlim = np.array([0, 170])
pdotlim = np.array([1,3.2])
Edotlim = np.array([39, 40.7])

plt.figure(1, figsize=(3.5,7))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.clf()

# First panel: density
ax = plt.subplot(4,1,1)
ax.plot(np.log10(r/u.kpc), np.log10(rhoHIN/rhounit), 'C0',
        label=r"H~\textsc{i} (N)")
ax.plot(np.log10(r/u.kpc), np.log10(rhoCON/rhounit), 'C1',
        label=r"CO (N)")
ax.plot(np.log10(r/u.kpc), np.log10(rhoHIS/rhounit), 'C0--',
        label=r"H~\textsc{i} (S)")
ax.plot(np.log10(r/u.kpc), np.log10(rhoCOS/rhounit), 'C1--',
        label=r"CO (S)")
ax.plot([0.3, 0.6], -23.6 + np.array([0, -0.6]), 'k:', alpha=0.5)
ax.text(0.4, -24, "$\propto r^{-2}$", rotation=-20)
ax.plot(np.log10([0.25,0.25]), rholim, 'k', lw=1, alpha=0.25)
ax.text(np.log10(0.25)+0.02, -24, r"$r_0$", rotation=-90)
ax.set_xlim(xlim)
ax.set_ylim(rholim)
ax.set_ylabel(r"$\log\rho$ [g cm$^{-3}$]")
ax.legend(ncol=2, fontsize=8)
ax.set_xticklabels([])
tx = plt.twinx(ax)
tx.set_ylim(rholim - np.log10(2.34e-24))
tx.set_ylabel(r"$\log n_\mathrm{H}$ [cm$^{-3}$]")
ty = plt.twiny(ax)
ty.set_xlim(xlim - np.log10(0.25))
ty.set_xlabel(r"$\log a = \log(r/r_0)$")

# Second panel: mean velocity
ax = plt.subplot(4,1,2)
ax.plot(np.log10(r/u.kpc), np.sqrt(2*EdotHIN/MdotHIN).to(u.km/u.s), 'C0')
ax.plot(np.log10(r/u.kpc), np.sqrt(2*EdotHIS/MdotHIS).to(u.km/u.s), 'C0--')
ax.plot(np.log10(r/u.kpc), (pdotHIN/MdotHIN).to(u.km/u.s), 'C0', alpha=0.5)
ax.plot(np.log10(r/u.kpc), (pdotHIS/MdotHIS).to(u.km/u.s), 'C0--', alpha=0.5)
ax.plot(np.log10(r/u.kpc), np.sqrt(2*EdotCON/MdotCON).to(u.km/u.s), 'C1')
ax.plot(np.log10(r/u.kpc), np.sqrt(2*EdotCOS/MdotCOS).to(u.km/u.s), 'C1--')
ax.plot(np.log10(r/u.kpc), (pdotCON/MdotCON).to(u.km/u.s), 'C1', alpha=0.5)
ax.plot(np.log10(r/u.kpc), (pdotCOS/MdotCOS).to(u.km/u.s), 'C1--', alpha=0.5)
ax.plot(np.log10([0.25,0.25]), vlim, 'k', lw=1, alpha=0.25)
ax.text(np.log10(0.25)+0.02, 110, r"$r_0$", rotation=-90)
ax.plot([-100,-100], [-100,-100], 'k', label=r"$\langle v\rangle_{\dot{M}}$")
ax.plot([-100,-100], [-100,-100], 'k', alpha=0.5, label=r"$\langle v\rangle_M$")
ax.legend(fontsize=8)
ax.set_xlim(xlim)
ax.set_ylim(vlim)
ax.set_ylabel(r"$v$ [km s$^{-1}$]")
ax.set_xticklabels([])
tx = plt.twinx(ax)
tx.set_ylim( vlim / (v0/(u.km/u.s)) )
tx.set_ylabel(r"$v/v_0$")
ty = plt.twiny(ax)
ty.set_xlim(xlim - np.log10(0.25))
ty.set_xticklabels([])

# Third panel: momentum flux
ax = plt.subplot(4,1,3)
ax.plot(np.log10(r/u.kpc), np.log10(pdotHIN/pdotunit), 'C0')
ax.plot(np.log10(r/u.kpc), np.log10(pdotCON/pdotunit), 'C1')
ax.plot(np.log10(r/u.kpc), np.log10(pdotHIS/pdotunit), 'C0--')
ax.plot(np.log10(r/u.kpc), np.log10(pdotCOS/pdotunit), 'C1--')
ax.plot(np.log10(r/u.kpc),
        np.log10((pdotHIN + pdotHIS + pdotCON + pdotCOS)/pdotunit),
        'k', label=r"Sum")
ax.plot(np.log10([0.25,0.25]), pdotlim, 'k', lw=1, alpha=0.25)
ax.text(np.log10(0.25)+0.02, 2.9, r"$r_0$", rotation=-90)
ax.set_xlim(xlim)
ax.set_ylim(pdotlim)
ax.legend(ncol=2, fontsize=8)
ax.set_ylabel(r"$\log\, \dot{p}$ [M$_\odot$ yr$^{-1}$ km s$^{-1}$]")
ax.set_xticklabels([])
tx = plt.twinx(ax)
tx.set_ylim( np.log10((10.**pdotlim*pdotunit/SFR) / (u.km/u.s)) )
tx.set_ylabel(r"$\log\, \dot{p}/\dot{M}_*$ [km s$^{-1}$]")
ty = plt.twiny(ax)
ty.set_xlim(xlim - np.log10(0.25))
ty.set_xticklabels([])

# Fourth panel: energy flux
ax = plt.subplot(4,1,4)
ax.plot(np.log10(r/u.kpc), np.log10(EdotHIN/Edotunit), 'C0')
ax.plot(np.log10(r/u.kpc), np.log10(EdotCON/Edotunit), 'C1')
ax.plot(np.log10(r/u.kpc), np.log10(EdotHIS/Edotunit), 'C0--')
ax.plot(np.log10(r/u.kpc), np.log10(EdotCOS/Edotunit), 'C1--')
ax.plot(np.log10(r/u.kpc),
        np.log10((EdotHIN + EdotCON + EdotHIS + EdotCOS)/Edotunit),
        'k')
ax.plot(np.log10([0.25,0.25]), Edotlim, 'k', lw=1, alpha=0.25)
ax.text(np.log10(0.25)+0.02, 40.2, r"$r_0$", rotation=-90)
ax.set_xlim(xlim)
ax.set_ylim(Edotlim)
ax.set_ylabel(r"$\log\,\dot{E}$ [erg s$^{-1}$]")
ax.set_xlabel(r"$\log r$ [kpc]")
tx = plt.twinx(ax)
tx.set_ylim( np.log10((10.**Edotlim*Edotunit/SFR) / (u.erg/u.Msun)) )
tx.set_ylabel(r"$\log\,\dot{E}/\dot{M}_*$ [erg M$_\odot^{-1}$]")
ty = plt.twiny(ax)
ty.set_xlim(xlim - np.log10(0.25))
ty.set_xticklabels([])

# Final spacing asjustment
plt.subplots_adjust(left=0.2, right=0.8, bottom=0.08, top=0.92, hspace=0.1)

# Save
plt.savefig("wind_structure.pdf")



#####################################
# CO 2-1 X factor calculation
#####################################

# Set properties of molecules
T = 50.                                 # Temp
abd = 1.1e-4                            # CO abundance
mCO = 2.34e-24*u.g / abd                # Mass per CO molecule
em = emitter('CO', abd)                 # Read CO molecular data
em.setLevPopLTE(T)                      # Set level populations in LTE
tX = em.tX(mCO/u.g)[1]*u.s              # t_X value for the 2-1 transition
Xthin = em.Xthin()[1]                   # X_CO in optically thin limit
MdotCON_iso = MdotCON * 4*np.pi/(saCON) # Isotropic mass loss rate
twN = (M0/MdotCON_iso).to(u.s)          # Wind mass removal timescale
MdotCOS_iso = MdotCOS * 4*np.pi/(saCOS) # Isotropic mass loss rate
twS = (M0/MdotCOS_iso).to(u.s)          # Wind mass removal timescale


# Compute grid in parallel
x = np.linspace(0, 10, 150)
z = np.linspace(0, 10, 150)
xx, zz = np.meshgrid(x, z)
args = [b for b in np.broadcast(xx.flat, zz.flat)]
def XfacNFunc(arg):
    print(arg)
    return pwCON.Xfac(T, em, twN, varpi=arg[1], varpi_t=arg[0], trans=1)
def XfacSFunc(arg):
    print(arg)
    return pwCOS.Xfac(T, em, twS, varpi=arg[1], varpi_t=arg[0], trans=1)
try:
    XfacN = np.load("XfacN.npy")
except:
    if __name__ == '__main__':
        with Pool(cpu_count()) as p:
            XfacN = p.map(XfacNFunc, args)
        #XfacN = [XfacNFunc(a) for a in args]  # Serial version
        XfacN = np.array(XfacN).reshape(xx.shape)
        np.save("XfacN.npy", XfacN)
try:
    XfacS = np.load("XfacS.npy")
except:
    if __name__ == '__main__':
        with Pool(cpu_count()) as p:
            XfacS = p.map(XfacSFunc, args)
        #XfacS = [XfacSFunc(a) for a in args]  # Serial version
        XfacS = np.array(XfacS).reshape(xx.shape)
        np.save("XfacS.npy", XfacS)

# Plot
plt.figure(2, figsize=(3.5,4.6))
NHlim = np.array([19.3497, 20.15])
alphalim = np.log10(10.**NHlim * 2.34e-24 / (1.99e33/3.09e18**2))
plt.clf()
plt.imshow(np.ma.masked_where(XfacN==0.0, np.log10(XfacN)),
           vmin=NHlim[0], vmax=NHlim[1], origin='lower',
           extent=[0,2.5,0,2.5], aspect='equal')
plt.imshow(np.ma.masked_where(XfacN[:,::-1]==0.0, np.log10(XfacN)[:,::-1]),
           vmin=NHlim[0], vmax=NHlim[1], origin='lower',
           extent=[-2.5,0,0,2.5], aspect='equal')
plt.imshow(np.ma.masked_where(XfacS==0.0, np.log10(XfacS))[::-1,:],
           vmin=NHlim[0], vmax=NHlim[1], origin='lower',
           extent=[0,2.5,-2.5,0], aspect='equal')
plt.imshow(np.ma.masked_where(XfacS==0.0, np.log10(XfacS))[::-1,::-1],
           vmin=NHlim[0], vmax=NHlim[1], origin='lower',
           extent=[-2.5,0,-2.5,0], aspect='equal')
xtmp = np.array([0,2.5])
m = np.sqrt( (np.cos(2*theta1CON) + np.cos(2*phiCON)) /
             (1 - np.cos(2*theta1CON)) )
plt.plot(xtmp, m*xtmp, 'k--', lw=2, alpha=0.5)
plt.plot(-xtmp, m*xtmp, 'k--', lw=2, alpha=0.5)
m = np.sqrt( (np.cos(2*theta0CON) + np.cos(2*phiCON)) /
             (1 - np.cos(2*theta0CON)) )
plt.plot(xtmp, m*xtmp, 'k--', lw=2, alpha=0.5)
plt.plot(-xtmp, m*xtmp, 'k--', lw=2, alpha=0.5)
m = np.sqrt( (np.cos(2*theta1COS) + np.cos(2*phiCOS)) /
             (1 - np.cos(2*theta1COS)) )
plt.plot(xtmp, -m*xtmp, 'k--', lw=2, alpha=0.5)
plt.plot(-xtmp, -m*xtmp, 'k--', lw=2, alpha=0.5)
m = np.sqrt( (np.cos(2*theta0COS) + np.cos(2*phiCOS)) /
             (1 - np.cos(2*theta0COS)) )
plt.plot(xtmp, -m*xtmp, 'k--', lw=2, alpha=0.5)
plt.plot(-xtmp, -m*xtmp, 'k--', lw=2, alpha=0.5)
xtmp = np.linspace(-0.25, 0.25, 200)
plt.fill_between(xtmp, np.sqrt(0.25**2-xtmp**2), -np.sqrt(0.25**2-xtmp**2),
                 color="lightgray", zorder=3)
plt.fill_between([-2.5,2.5], [0.8, 0.8], [-0.8,-0.8], facecolor='none',
                 edgecolor='k', hatch='////', alpha=0.5)
plt.xlim([-2.5,2.5])
plt.ylim([-2.5,2.5])
plt.xlabel(r'$x$ [kpc]')
plt.ylabel(r'$z$ [kpc]')

# Adjust position to make room for color bar
pos = plt.gca().get_position()
pos.y0 -= 0.2
pos.x0 += 0.03
pos.x1 += 0.05
plt.gca().set_position(pos)

# Add color bar
pos = plt.gca().get_position()
pos.y0 = pos.y1 + 0.14
pos.y1 = pos.y0 + 0.025
axcbar = plt.axes(pos)
cbar = plt.colorbar(orientation='horizontal', cax=axcbar)
cbar.ax.set_xlabel(
    r'$\log\,X_{\mathrm{CO}(2\to 1)}$ [cm$^{-2}$ / (K km s$^{-1}$)]')
cbarax2 = cbar.ax.twiny()
cbarax2.set_xlim(alphalim)
cbarax2.set_xlabel(
    r'$\log\,\alpha_{\mathrm{CO}(2\to 1)}$ [M$_\odot$ pc$^{-2}$ / (K km s$^{-1}$)]')
    
# Save
plt.savefig('XCO.pdf')
