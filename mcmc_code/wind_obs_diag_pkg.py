#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 07:52:04 2020

@author: yuxuan
"""
import os
import sys
dat_dir = '/home/yuxuan/wind_obs/wind_data' 
mcmc_dir = '/avatar/yuxuan/wind_obs/mcmc_dat/'
sys.path.append('/avatar/yuxuan/research_module/despotic')
# package for wind observational diagnostic with DESPOTIC
#
import time
import signal
from astropy.units import Msun, yr, Angstrom, pc
from scipy import stats
from scipy.constants import k as kB
from scipy.constants import G, c, m_p, m_e, h
from scipy import interpolate
from astropy.io import fits
from despotic.winds import pwind, zetaM, sxMach
from despotic import emitter
import matplotlib.colors as colors
import matplotlib.cm as cm
from scipy.optimize import brentq
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import sys
# sys.path.append('%s/research_module'%sys_dir)

# Constants; switch to cgs

#import yt
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
v0 = 120e5*np.sqrt(2)
r0 = 250*pc
m0 = v0**2*r0/(2.0*G)
rho0 = 3.0*m0/(4.0*np.pi*r0**3)
tc = r0/v0
dist = 3.5e3*kpc


def signal_handler(signum, frame):
    raise Exception("Timed out!")


def rebin(a, *args):
    '''rebin ndarray data into a smaller ndarray of the same rank whose dimensions
    are factors of the original dimensions. eg. An array with 6 columns and 4 rows
    can be reduced to have 6,3,2 or 1 columns and 4,2 or 1 rows.
    example usages:
    >>> a=rand(6,4); b=rebin(a,3,2)
    >>> a=rand(6); b=rebin(a,2)
    '''
    shape = a.shape
    lenShape = len(shape)
    factor = np.array(np.asarray(shape)/np.asarray(args), dtype=int)
    evList = ['a.reshape('] + \
             ['args[%d],factor[%d],' % (i, i) for i in range(lenShape)] + \
             [')'] + ['.mean(%d)' % (i+1) for i in range(lenShape)]
    print(''.join(evList))
    return eval(''.join(evList))


def rotate(ppv, center, angle, n_g):
    # rotate the spatial part of the ppv cube and zoom in
    # note that n_g should be even
    (x0, y0) = center
    n_v = np.shape(ppv)[0]
    ppv_rot = np.zeros((n_v, n_g, n_g))
    if angle == 0:
        ppv_rot = ppv[:, int(x0-n_g/2):int(x0+n_g/2),
                      int(y0-n_g/2):int(y0+n_g/2)]
    elif angle != 0:
        c = np.cos(angle*np.pi/180.)
        s = np.sin(angle*np.pi/180.)
        for x in range(n_g):
            for y in range(n_g):
                src_x = c*(x-n_g/2) + s*(y-n_g/2) + x0
                src_y = -s*(x-n_g/2) + c*(y-n_g/2) + y0
                ppv_rot[:, x, y] = ppv[:, int(src_x), int(src_y)]
    return ppv_rot, n_g, n_g


def zeroth_moment_map(v, cube_noise, noise):
    dv = (v[-1]-v[0])/(len(v))
    L_int = np.sum(cube_noise*dv, axis=0)
    noise_int = noise*dv*np.sqrt(len(v))
    good_pix_x_a, good_pix_y_a = np.where(L_int > 3*noise_int)
    cube_noise[cube_noise < 3*noise] = 0

    # calculate second moment in 'good' pixels
    n_v, n_pix_x, n_pix_y = np.shape(cube_noise)
    m0 = np.zeros((n_pix_x, n_pix_y))
    for x, y in zip(good_pix_x_a, good_pix_y_a):
        lum_los = cube_noise[:, x, y]  # lum array in beam
        m0[x, y] = np.sum(lum_los)*dv
    return m0


def first_moment_map(v, cube_noise, noise):
    # output: zeroth moment map m0; first moment map m1
    m0 = zeroth_moment_map(v, cube_noise, noise)

    dv = (v[-1]-v[0])/(len(v))
    L_int = np.sum(cube_noise*dv, axis=0)
    noise_int = noise*dv*np.sqrt(len(v))
    good_pix_x_a, good_pix_y_a = np.where(L_int > 3*noise_int)
    cube_noise[cube_noise < 3*noise] = 0

    # calculate second moment in 'good' pixels
    n_v, n_pix_x, n_pix_y = np.shape(cube_noise)

    m1 = np.zeros((n_pix_x, n_pix_y))
    for x, y in zip(good_pix_x_a, good_pix_y_a):
        lum_los = cube_noise[:, x, y]  # lum array in beam
        m1[x, y] = np.sum(v*lum_los*dv)/m0[x, y]
    return m0, m1


def second_moment_map(v, cube_noise, noise):
    # output: zeroth moment map m0; first moment map m1; second moment map m2
    m0, m1 = first_moment_map(v, cube_noise, noise)

    dv = (v[-1]-v[0])/(len(v))
    L_int = np.sum(cube_noise*dv, axis=0)
    noise_int = noise*dv*np.sqrt(len(v))
    good_pix_x_a, good_pix_y_a = np.where(L_int > 3*noise_int)
    cube_noise[cube_noise < 3*noise] = 0

    # calculate second moment in 'good' pixels
    n_v, n_pix_x, n_pix_y = np.shape(cube_noise)

    m2 = np.zeros((n_pix_x, n_pix_y))
    for x, y in zip(good_pix_x_a, good_pix_y_a):
        lum_los = cube_noise[:, x, y]               # lum array in beam
        m2[x, y] = np.sum((v-m1[x, y])**2*lum_los*dv)/m0[x, y]
    m2 = np.sqrt(m2)
    return m0, m1, m2

###################
# import obs data #
###################


def read_wind_data(obj, line, side, v_cut, v_sm_edge):
    if obj == 'M82':
        if line == 'CO_2_1':
            print('CO')
            # CO cold molecular gas; Leroy15 CO 2->1
            CO_2_1 = fits.open(
                '%s/%s/Leroy15/ngc3034_hans_I_CO_J1-2_lwb2009.fits' % (dat_dir, obj))

            CO_2_1 = CO_2_1[0].data
            CO_2_1 = CO_2_1[::-1, :, :]
            # CO_2_1
            CO_2_1[np.isnan(CO_2_1) == True] = 0

            # rotate the 2D data to make the major axis x axis, and minor axis y axis
            # major axis is oriented at PA=67
            ppv_rot, n_co, n_co = rotate(
                CO_2_1, center=(105, 143), angle=-23, n_g=75)
            cdel_co = 1.11e-3*60  # in kpc
            # scale of x and y
            ex_co = cdel_co*n_co
            ex_co = cdel_co*n_co

            # generating moment map
            # CO_2_1 moment map
            # v in obs frame
            v_r = 828009.375667/1.0e3
            v_l = 828009.375667/1.0e3 - 5201.61474610/1.0e3*231  # in km/s
            # v in M82 frame
            v_r = v_r-211.
            v_l = v_l-211.
            v = np.linspace(v_l, v_r, 232)
            dv = (v_r-v_l)/231
            sigma = 0.007  # the cube has rms noise per 5.2 km s-1 channel of 7 mK in units of main beam temperature
            x = np.linspace(-ex_co/2., ex_co/2., n_co)
            y = np.linspace(-ex_co/2., ex_co/2., n_co)

        elif line == 'HI':
            print('HI')
            # cold neutron gas; Martini18 HI 21cm
            HI = fits.open(
                '%s/%s/Martini18/m82_hi_24as_rotated.fits' % (dat_dir, obj))
            HI = HI[0].data
            HI = HI[::-1, ::-1, ::-1]
            HI[np.isnan(HI) == True] = 0
            HI[HI < 0] = 0
            # rotate the 2D data to make the major axis x axis, and minor axis y axis
            # major axis is oriented at PA=67
            ppv_rot, n_hi, n_hi = rotate(
                HI, center=(1000, 1000), angle=0, n_g=300)
            cdel_hi = 2.78e-4*60  # in kpc
            # scale of x and y
            ex_hi = cdel_hi*n_hi

            # Area(square)/Area(circle) = (24)**2 / pi / 12**2 SNR = sqrt(1.273) * SNR_beam
            # HI moment map
            v_r = 5.696760655716e5/1.0e3
            v_l = 5.696760655716e5/1.0e3 - 5.000005412925e3/1.0e3*154  # in km/s
            # v in M82 frame
            v_r = v_r-211.
            v_l = v_l-211.
            v = np.linspace(v_l, v_r, 155)
            dv = (v_r-v_l)/154
            sigma = 0.4
            x = np.linspace(-ex_hi/2., ex_hi/2., n_hi)
            y = np.linspace(-ex_hi/2., ex_hi/2., n_hi)

        if line == 'CO_2_1' or line == 'HI':
            if line == 'CO_2_1':
                pos_a_u = [1.2]*4+[0.8]*5
                pos_t_u = [-.8, -.4, 0., .8] + [-.8, -.4, 0., .4, .8]
                pos_a_l = [-1.2]*4+[-.8]*5
                pos_t_l = [-.8, -.4, 0., .4] + [-.8, -.4, 0., .4, .8]
            elif line == 'HI':
                pos_a_u = [2.0]*5+[1.5]*5+[1.]*5
                pos_t_u = [-1., -.5, 0., .5, 1.]*3
                pos_a_l = [-2.]*5+[-1.5]*5+[-1.]*5
                pos_t_l = [-1., -.5, 0., .5, 1.]*3

            if side == 'north':
                pos_t = pos_t_u
                pos_a = pos_a_u
            elif side == 'south':
                pos_t = pos_t_l
                pos_a = pos_a_l
            elif side == 'use_for_Halpha':
                pos_t = [1.05, -1.43]
                pos_a = [0, 0]
            pos_t = np.array(pos_t)
            pos_a = np.array(pos_a)
            pos_iy_a = np.zeros(len(pos_a))
            pos_ix_a = np.zeros(len(pos_a))
            for i, p_t, p_a in zip(range(len(pos_a)), pos_t, pos_a):
                pos_iy = abs(y-p_a).argmin()
                pos_ix = abs(x-p_t).argmin()
                pos_iy_a[i] = pos_iy
                pos_ix_a[i] = pos_ix

            pos_ix_a = np.array(pos_ix_a, dtype=int)
            pos_iy_a = np.array(pos_iy_a, dtype=int)

            # save spectrum data to T_B_dat
            spct_dat = []
            for i, pos_iy, pos_ix in zip(range(len(pos_iy_a)), pos_iy_a, pos_ix_a):
                if line == 'CO_2_1':
                    spct = np.average(
                        ppv_rot[:, pos_iy-2:pos_iy+2, pos_ix-2:pos_ix+2], axis=(1, 2))
                if line == 'HI':
                    spct = np.average(
                        ppv_rot[:, pos_iy-11:pos_iy+11, pos_ix-11:pos_ix+11], axis=(1, 2))
                spct, _, _ = stats.binned_statistic(
                    v, spct, 'mean', bins=v_sm_edge)
                spct_dat.append(spct)
            v_sm = (v_sm_edge[:-1]+v_sm_edge[1:])/2
            spct_dat = np.array(spct_dat)

            # velocity cutoff
            if v_cut != 0:
                vcl_i = abs(v+v_cut).argmin()
                vcr_i = abs(v-v_cut).argmin()
                v = np.concatenate((v[0:vcl_i], v[vcr_i:]))
                spct_dat = np.concatenate(
                    (spct_dat[:, 0:vcl_i], spct_dat[:, vcr_i:]), axis=1)

            if line == 'CO_2_1':
                sigma_spct = sigma/np.sqrt(5**2/2.5**2/np.pi)
            if line == 'HI':
                sigma_spct = sigma/np.sqrt(23**2/12**2/np.pi)

            return ppv_rot, v, sigma, spct_dat, sigma_spct, v_sm, pos_t, pos_a, pos_ix_a, pos_iy_a

        elif line == 'Halpha':
            print('Halpha')
            f, spct = np.loadtxt('%s/%s/Martin98/M82_%s.txt' %
                                 (dat_dir, obj, side), usecols=(0, 1), unpack=True, skiprows=1)
            Ha_e = np.loadtxt('%s/%s/Martin98/M82_%s_rms.txt' %
                              (dat_dir, obj, side), usecols=(1), unpack=True, skiprows=1)
            sigma_spct = np.sqrt(np.average(Ha_e**2))
            v = (f/6562.801 - 1)*c/1.e5  # in
            v = v-211.
            v_sm = (v_sm_edge[:-1]+v_sm_edge[1:])/2
            spct, _, _ = stats.binned_statistic(
                v, spct, 'mean', bins=v_sm_edge)
            pos_t = 0
            if side == 'north':
                pos_a = 1.05
            elif side == 'south':
                pos_a = -1.43

            # normalize the spectrum to Kennicutt08
            spct *= 0.05
            sigma_spct *= 0.05

            return spct, sigma_spct, v_sm, pos_t, pos_a


def log_pcl(theta, theta_2):
    # log of the prior of clumping factor
    alpha_ha = 4e-13  # in cgs
    e_ha = 3.03e-12
    # surface brightness of the data
    SB = 4*np.pi*4.008e-5/1.3  # in cgs

    driver, p, ex = theta_2
    if ex == 'solid':
        ex = 'solid angle'
    phi, theta_in, theta_out, lg_mdot, A = theta

    mach = 10**lg_mach
    phi = phi/90.0*np.pi/2.0
    theta_in = theta_in/90.0*np.pi/2.0
    theta_out = theta_out/90.*np.pi/2.0
    md = 10**lg_mdot
    md = md*Msun/yr
    eta = (np.cos(theta_in) - np.cos(theta_out)) * md / mdotstar
    '''
    Gamma = brentq(
            lambda g: zetaM(np.log(g), sxMach(mach))/epsff - eta,
            1e-6, 100.)
    '''
    pw = pwind(Gamma=0.1, mach=mach, driver=driver, potential=p, expansion=ex, geometry='cone sheath',
               theta=theta_out, theta_in=theta_in, tau0=0,
               phi=phi, uh=0, fcrit=1.)

    # path length through the wind
    los = pw.s_crit(1*kpc/r0, 0)
    L = (los[1] - los[0]) + (los[3] - los[2])
    # calculate the density derived from clumping factor
    n_c = np.sqrt(SB/alpha_ha/e_ha/(L*r0)*10**lg_c_rho)
    #print(L)
    # volume density of M82 at aout 1 kpc is
    log_pcl = -(np.log10(n_c) - np.log10(13.))**2 / (2*0.005**2)
    return log_pcl, n_c


# Velocity grid
def em_line_spec(line, mach, phi=5, theta_in=30, theta_out=50, pos_t=0, pos_a=1, driver='hot', mdot=100, uh=10, tau0=5,
                 expansion='solid', pot='isothermal', fc=1., u=np.linspace(-3, 3, 100)):
    # functions to calculate wind spectra
    #
    # mdot: isotropic mass outflow rate
    #
    if expansion == 'solid':
        expansion = 'solid angle'
    # note that pos_a and pos_t could also be array

    varpi_t = pos_t*kpc/r0
    varpi = pos_a*kpc/r0
    phi = phi/90.0*np.pi/2.0
    theta_in = theta_in/90.0*np.pi/2.0
    theta_out = theta_out/90.*np.pi/2.0
    mdot = mdot*Msun/yr
    # The sound speed is c_s = sqrt(k_B T / m),where m = mean particle mass. For atomic hydrogen gas m ~= 2.34 x 10^-24 g (assuming 90% H and 10% He by number), so plugging in T = 5000 K gives c_s ~= 5.4 km/s, and thus for our adopted 40 km/s velocity dispersion in the galaxy, we have Mach = sigma / c_s ~= 7.4
    if line == 'Halpha':
        temp = 1.e5
    elif line == 'CO_2_1':
        temp = 50.
    elif line == 'HI':
        temp = 5000.
    print("mach number=", mach)

    # create wind object
    try:
        md = mdot
        eta = (np.cos(theta_in) - np.cos(theta_out)) * md / mdotstar
        Gamma = brentq(
            lambda g: zetaM(np.log(g), sxMach(mach))/epsff - eta,
            1e-6, 100.)
        ex = expansion
        p = pot
        # print('Gamma=%f'%Gamma)
        if line == 'CO_2_1' or line == 'HI':
            pw3 = pwind(Gamma, mach, driver=driver, potential=p,
                        expansion=ex, geometry='cone sheath',
                        theta=theta_out, theta_in=theta_in, tau0=tau0,
                        phi=phi, uh=uh, fcrit=fc)  # , interpabs=5e-3,interprel=5e-3)
            #print("generate wind successfully")
        elif line == 'Halpha':
            pw3 = pwind(Gamma, mach, driver=driver, potential=p,
                        expansion=ex, geometry='cone sheath',
                        theta=theta_out, theta_in=theta_in, tau0=tau0,
                        phi=phi, uh=uh, fcrit=fc, epsabs=1e-7, epsrel=1e-5)  # , interpabs=5e-3,interprel=5e-3)
    except:
        print('fail to generate wind')
        error = 1
        if hasattr(pos_a, "__len__") == True:
            return np.zeros((len(pos_a), len(u))), error
        elif hasattr(pos_a, "__len__") == False:
            return np.zeros(len(u)), error

    tw = m0 / mdot
    ###################
    # Halpha spectrum #
    ###################
    if line == 'Halpha':
        Ha = np.zeros(len(u))
        # Cooling constant
        lam_e = 3.9e-25
        # Wavelength grid
        lam0 = 6562.801*ang

        Ha = 1e17*2.0/(36.*np.pi) * lam_e * r0 * (rho0/mH)**2 * lam0 / v0 * (tc/tw)**2 \
            * pw3.Xi(u, varpi=varpi, varpi_t=varpi_t)
        spct = Ha

    ###############
    # CO emission #
    ###############
    elif (line == 'CO_1_0' or line == 'CO_2_1' or line == 'CO_3_2'):
        co = emitter('CO', 1.1e-4)
        # CO emission profiles
        if hasattr(pos_a, "__len__") == True:
            CO_TB = np.zeros((len(pos_a), len(u)))
            for i, vpi, vpt in zip(range(len(pos_a)), varpi, varpi_t):
                #print([pos_t[i], pos_a[i]])
                if line == 'CO_1_0':
                    CO_TB[i, :] = pw3.temp_LTE(
                        u, temp, emit=co, tw=tw, varpi=vpi, varpi_t=vpt)
                if line == 'CO_2_1':
                    CO_TB[i, :] = pw3.temp_LTE(
                        u, temp, emit=co, tw=tw, trans=1, varpi=vpi, varpi_t=vpt)
                    # print(CO_TB[i,:])
        elif hasattr(pos_a, "__len__") == False:
            CO_TB = np.zeros(len(u))
            if line == 'CO_1_0':
                CO_TB = pw3.temp_LTE(
                    u, temp, emit=co, tw=tw, varpi=varpi, varpi_t=varpi_t)
            if line == 'CO_2_1':
                CO_TB = pw3.temp_LTE(
                    u, temp, emit=co, tw=tw, trans=1, varpi=varpi, varpi_t=varpi_t)
        spct = CO_TB

    elif line == 'HI':
        # HI emission profiles
        HI_TB = np.zeros(len(u))
        abd = 1.0  # this is the abundance of the emitting species relative to hydrogen nuclei, which is just 1 since the species here is atomic hydrogen
        Omega = 5.75e-12  # this is the dimensionless oscillator strength of the transition
        wl = 21.1  # this is the wavelength of the line in cm
        fj = 0.25  # this is the fraction of atoms in the lower energy state
        boltzfac = np.exp(-hP*c/kB/wl/temp)
        if hasattr(pos_a, "__len__") == True:
            HI_TB = np.zeros((len(pos_a), len(u)))
            for i, vpi, vpt in zip(range(len(pos_a)), varpi, varpi_t):
                HI_TB[i, :] = pw3.temp_LTE(u, temp, tw=tw, abd=abd, Omega=Omega, wl=wl,
                                           fj=fj, boltzfac=boltzfac, varpi=vpi, varpi_t=vpt, correlated=False)
        elif hasattr(pos_a, "__len__") == False:
            HI_TB = np.zeros(len(u))
            HI_TB = pw3.temp_LTE(u, temp, tw=tw, abd=abd, Omega=Omega, wl=wl, fj=fj,
                                 boltzfac=boltzfac, varpi=varpi, varpi_t=varpi_t, correlated=False)
        spct = HI_TB

    if pw3.error_stat == None:
        error = 0
        pass
    elif pw3.error_stat != None:
        print(pw3.error_stat)
        error = 1
    pw3.clear_err()
    return spct, error


def em_line_ppv(line, mach, phi, theta_in, theta_out, driver, uh, tau0, mdot, expansion,
                pot, sp_r, sp_res, v, side, fc=1.):
    phi = phi/90.0*np.pi/2.0
    theta_in = theta_in/90.0*np.pi/2.0
    theta_out = theta_out/90.*np.pi/2.0
    mdot = mdot*Msun/yr
    if line == 'CO_2_1':
        temp = 50.
    elif line == 'HI':
        temp = 5000.
    # create wind object
    md = mdot
    eta = (np.cos(theta_in) - np.cos(theta_out)) * md / mdotstar
    Gamma = brentq(
        lambda g: zetaM(np.log(g), sxMach(mach))/epsff - eta,
        1e-6, 100.)
    tw = m0 / mdot
    ex = expansion
    p = pot

    try:
        data = np.load('m82_%s_%s_%s_%s_%s_ppv.npz' %
                       (line, side, driver, p, ex))
        T_B_ppv = data['T_B_ppv']
        return T_B_ppv
    except:
        pass

    pw = pwind(Gamma, mach, driver=driver, potential=p,
               expansion=ex, geometry='cone sheath',
               theta=theta_out, theta_in=theta_in, phi=phi, tau0=tau0,
               uh=uh, fcrit=fc)

    sp_res_half = int((sp_res+1)/2)
    # Define the spatial and velocity grids we will use
    u = v*1e5/v0
    if side == 'both':
        varpi_a = np.linspace(sp_r*kpc, -sp_r*kpc, sp_res)/r0
        varpi_t = np.linspace(-sp_r*kpc, sp_r*kpc, sp_res)/r0
    elif side == 'north':
        varpi_a = np.linspace(0, sp_r*kpc, sp_res_half)/r0
        varpi_t = np.linspace(-sp_r*kpc, sp_r*kpc, sp_res)/r0
    elif side == 'south':
        varpi_a = np.linspace(-sp_r*kpc, 0, sp_res_half)/r0
        varpi_t = np.linspace(-sp_r*kpc, sp_r*kpc, sp_res)/r0

    vpt2, vpa2 = np.meshgrid(varpi_t, varpi_a, indexing='xy')

    if line == 'Halpha':
        # Cooling constant
        lam_e = 3.9e-25

    if (line == 'CO_1_0' or line == 'CO_2_1' or line == 'CO_3_2'):
        # Import the required molecular data
        co = emitter('CO', 1.1e-4)
        # Compute ppv
        T_B_ppv = np.zeros((u.shape[0], varpi_a.shape[0], varpi_t.shape[0]))
        # try:
        #    data = np.load('m82_%s_mdot%.0f_ex_%s_ppv.npz'%(line, mdot, ex))
        #    intTA = data['T_B']
        # except IOError:
        #    pass
        for i in range(len(varpi_a)):
            for j in range(len(varpi_t)):
                if line == 'CO_1_0':
                    T_B_ppv[:, i, j] = pw.temp_LTE(u, temp, emit=co, tw=tw, trans=0,
                                                   varpi=vpa2[i, j],
                                                   varpi_t=vpt2[i, j])
                elif line == 'CO_2_1':
                    T_B_ppv[:, i, j] = pw.temp_LTE(u, temp, emit=co, tw=tw, trans=1,
                                                   varpi=vpa2[i, j],
                                                   varpi_t=vpt2[i, j])
                elif line == 'CO_3_2':
                    T_B_ppv[:, i, j] = pw.temp_LTE(u, temp, emit=co, tw=tw, trans=2,
                                                   varpi=vpa2[i, j],
                                                   varpi_t=vpt2[i, j])
    elif line == 'HI':
        # Import the required molecular data
        # Compute ppv
        T_B_ppv = np.zeros((u.shape[0], varpi_a.shape[0], varpi_t.shape[0]))
        # try:
        #    data = np.load('m82_%s_mdot%.0f_ex_%s_ppv.npz'%(line, mdot, ex))
        #    intTA = data['T_B']
        # except IOError:
        #    pass
        abd = 1.0  # this is the abundance of the emitting species relative to hydrogen nuclei, which is just 1 since the species here is atomic hydrogen
        Omega = 5.75e-12  # this is the dimensionless oscillator strength of the transition
        wl = 21.1  # this is the wavelength of the line in cm
        fj = 0.25  # this is the fraction of atoms in the lower energy state
        boltzfac = np.exp(-hP*c/kB/wl/temp)
        for i in range(len(varpi_a)):
            print(i)
            for j in range(len(varpi_t)):
                T_B_ppv[:, i, j] = pw.temp_LTE(u, temp, tw=tw, abd=abd, Omega=Omega, wl=wl, fj=fj,
                                               boltzfac=boltzfac, varpi=vpa2[i, j], varpi_t=vpt2[i, j], correlated=False)
    # manipulate ppv
    np.savez('m82_%s_%s_%s_%s_%s_ppv.npz' %
             (line, side, driver, p, ex), T_B_ppv=T_B_ppv)
    return T_B_ppv


def cal_Gamma(lg_mdot, lg_mach):
    mdot_t = 10**lg_mdot*Msun/yr  # true mass outflow rate (not isotropic)
    mach = 10**lg_mach
    # create wind object
    eta = mdot_t / mdotstar
    try:
        Gamma = brentq(
            lambda g: zetaM(np.log(g), sxMach(mach))/epsff - eta,
            1e-6, 100.)
    except:
        Gamma = 100.  # as long as the value is larger than 1.
    return Gamma


# MCMC fitting
# input:
# theta: parameters of the wind model, including phi, theta_in, theta_out,
# lg_mdot log of effective outflow rate
# tau0, uh
# theta_2: driver, potential and expansion law of the wind model
# sigma: noise level
# pos_t, pos_a: array of position of the observed spectrums
# spct_dat: array of observed spectrum


def log_prior(theta, theta_2, line, incl_mach):
    # print(theta)

    if line == 'CO_2_1' or line == 'HI':
        driver, p, ex = theta_2
    elif line=='Halpha':
        driver, p, ex, c_rho = theta_2

    if incl_mach==0:
        if line == 'CO_2_1' or line=='Halpha':
            lg_mach = 2.
        if line == 'HI':
            lg_mach = np.log10(7.4)

    if incl_mach==1:
        lg_mach = theta[-1]
        if 0<lg_mach<3:
            pass
        else:
            return -np.inf
        
    if driver == 'ideal':
        phi, theta_in, theta_out, lg_mdot = theta[:4]
        if 0.0 < theta_in < 70. and np.max([theta_in+10, 50.]) < theta_out < 90. and -2. < lg_mdot < 3. and -60. < phi < 60.:
            prior = np.abs(np.cos(phi*np.pi/180))  # flat over \sin phi
            return np.log(prior)
        else:
            return -np.inf

    elif driver == 'radiation':
        phi, theta_in, theta_out, lg_mdot, tau0 = theta[:5]
        Gamma = cal_Gamma(lg_mdot=lg_mdot, lg_mach=lg_mach )
        if 0.0 < theta_in < 70. and np.max([theta_in+10, 50.]) < theta_out < 90. and -2. < lg_mdot < 3. and -60. < phi < 60. and 0. < tau0 < 300. and Gamma*tau0 >= 3.:
            prior = np.abs(np.cos(phi*np.pi/180))  # flat over \sin phi
            return np.log(prior)
        else:
            return -np.inf
    
    elif driver == 'hot':
        phi, theta_in, theta_out, lg_mdot, uh = theta[:5]
        if 0.0 < theta_in < 70. and np.max([theta_in+10, 50.]) < theta_out < 90. and -2. < lg_mdot < 3. and -60. < phi < 60. and 0. < uh < 30.:
            prior = np.abs(np.cos(phi*np.pi/180))  # flat over \sin phi          
            return np.log(prior)
        else:
            return -np.inf






def log_likelihood(theta, theta_2, line, pos_t, pos_a, v, spct_dat, spct_dat_shift, sigma, side, shift, spct_hi_dat, incl_mach):
    # generate despotic spectrum
    # get parameter
    if line == 'CO_2_1' or line == 'HI':
        driver, p, ex = theta_2
        if driver == 'ideal':
            phi, theta_in, theta_out, lg_mdot = theta[:4]
            uh = 1
            tau0 = 1  # arbitrary chosen
        elif driver == 'radiation':
            phi, theta_in, theta_out, lg_mdot, tau0 = theta[:5]
            uh = 0
        elif driver == 'hot':
            phi, theta_in, theta_out, lg_mdot, uh = theta[:5]
            tau0 = 0

    elif line == 'Halpha':
        driver, p, ex, c_rho = theta_2
        if driver == 'ideal':
            phi, theta_in, theta_out, lg_mdot, A = theta
            uh = 1
            tau0 = 1  # arbitrary chosen
        elif driver == 'radiation':
            phi, theta_in, theta_out, lg_mdot, tau0, A = theta
            uh = 0
        elif driver == 'hot':
            phi, theta_in, theta_out, lg_mdot, uh, A = theta
            tau0 = 0

    if incl_mach==0:
        if line=='CO_2_1':
            mach = 100
        elif line=='HI':
            mach = 7.4
        elif line=='Halpha':
            mach = 100         
    elif incl_mach==1:
        lg_mach = theta[-1]
        mach = 10**lg_mach

    f_A = np.cos(theta_in/90.0*np.pi/2.0) - np.cos(theta_out/90.0*np.pi/2.0)
    mdot = 10**lg_mdot/f_A  # isotropic mass outflow rate
    # eliminate unphysical parameter combinations
    if log_prior(theta, theta_2, line, incl_mach) == -np.inf:
        return -np.inf

    # make spectrum
    u = v*1.0e5/v0  # dimensionless velocity
    if line == 'CO_2_1' or line == 'HI':
        spct, error = em_line_spec(line=line, mach=mach, phi=phi, theta_in=theta_in, theta_out=theta_out, pos_t=pos_t, pos_a=pos_a, driver=driver, mdot=mdot, uh=uh, tau0=tau0, expansion=ex, pot=p, fc=1., u=u)

    elif line == 'Halpha':
        spct_ha, error = em_line_spec(line=line, mach=mach, phi=phi, theta_in=theta_in, theta_out=theta_out, pos_t=pos_t, pos_a=pos_a, driver=driver, mdot=mdot, uh=uh, tau0=tau0, expansion=ex, pot=p, fc=1., u=u)
        if side == 'north':
            spct_hi = spct_hi_dat[0]
        else:
            spct_hi = spct_hi_dat[1]
        spct = c_rho*spct_ha+A*spct_hi

    # save non-convergence parameters
    spct_has_nan = np.isnan(np.sum(spct))
    if spct_has_nan == False and error == 0:
        pass
    elif spct_has_nan == True or error == 1:
        print('not converge')
        spct[spct != spct] = 0  # make NaN value to be 0
        filename = '/avatar/yuxuan/wind_obs/mcmc_dat/non_converge_points/%s/%s/%s_%s_%s_%s_non_converge_points.txt' % (
              side, line, line, driver, p, ex) 
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'ab') as f:
            np.savetxt(f, [theta], fmt='%5f', delimiter=',')

    # calculating kai sq
    if line == 'CO_2_1' or line == 'HI':
        #shift = np.shape()
        if shift == True:
            n_sh = np.shape(spct_dat_shift)[0]
            delta_spct = ([spct]*n_sh - spct_dat_shift)**2  # 2D
            # average over spectrum velocity; delta_sum shift*position
            delta_aa = np.average(delta_spct, axis=2)
            # for each position minimize the kaisq along axis of velocity shift, delta 1D array position
            delta_a = np.min(delta_aa, axis=0)
            delta_sum = np.sum(delta_a)/sigma**2  # sum delta sq over position
        if shift == False:
            delta_spct = (spct-spct_dat)**2
            # average over spectrum velocity; delta_sum shift*position
            delta_a = np.average(delta_spct, axis=1)
            delta_sum = np.sum(delta_a)/sigma**2  # sum delta sq over position
        kaisq = -0.5 * delta_sum
    elif line == 'Halpha':
        if shift == True:
            n_sh = np.shape(spct_dat_shift)[0]
            delta_spct = ([spct]*n_sh - spct_dat_shift)**2  # 2D
            # average over spectrum velocity; delta_sum shift*position
            delta_a = np.average(delta_spct, axis=1)
            # for each position minimize the kaisq along axis of velocity shift, delta 1D array position
            delta = np.min(delta_a)
            delta_sum = delta**2/sigma**2
        elif shift ==  False:
            delta_sum = np.mean((spct-spct_dat)**2 / sigma**2)
        kaisq = -0.5 * delta_sum
    print('kaisq=%f' % kaisq)
    return kaisq


def log_prob(theta, theta_2, line, pos_t, pos_a, v, spct_dat, spct_dat_shift, sigma, side, shift, spct_hi_dat, incl_mach):
    lp = log_prior(theta, theta_2, line, incl_mach)
    ll = log_likelihood(theta, theta_2, line, pos_t, pos_a,
                        v, spct_dat, spct_dat_shift, sigma, side, shift, spct_hi_dat, incl_mach)
    return lp + ll, ll
