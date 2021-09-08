#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 11:19:24 2020

@author: yuxuan
"""

# python /home/yuxuan/wind_obs/mcmc_fit/wind_mcmc_fit.py 56 M82 north CO_2_1 ideal isothermal area 0 
#############################
# modules import and set up #
#############################
from contextlib import closing
from multiprocessing import cpu_count
from multiprocessing import Pool
import os
import wind_obs_diag_pkg as wodp
import emcee
from astropy.units import Msun, yr, Angstrom, pc
from scipy.constants import k as kB
from scipy.constants import G, c, m_p, m_e, h
from astropy.io import fits
from despotic.winds import pwind, zetaM, sxMach
from despotic import emitter
import matplotlib.colors as colors
import matplotlib.cm as cm
from scipy.optimize import brentq
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift
import numpy as np
import sys
sys.path.append('/avatar/yuxuan/research_module/despotic')
sys.path.append('/home/yuxuan/wind_obs/mcmc_code')
mcmc_dat_dir = '/avatar/yuxuan/wind_obs/mcmc_dat'

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
mach = 100.0
uh = 10.0
v0 = 120e5*np.sqrt(2)
r0 = 250*pc
m0 = v0**2*r0/(2.0*G)
rho0 = 3.0*m0/(4.0*np.pi*r0**3)
tc = r0/v0
temp = 50.
dist = 3.5e3*kpc


ncpus = int(sys.argv[1])
obj = sys.argv[2]
side = sys.argv[3]
line = sys.argv[4]
driver = sys.argv[5]
p = sys.argv[6]
ex = sys.argv[7]
if line == 'Halpha':
    c_rho = sys.argv[8]
    c_rho = float(c_rho)
incl_rot = True

incl_mach = sys.argv[8] # change to 0 if you don't want to include mach number


###################
# import obs data #
###################
# directory of data
print('loading data')

v_cut = 0

# v in M82 frame

################
# MCMC fitting #
################
if line == 'CO_2_1' or line == 'HI':
    v_r = 250
    v_l = -250
    v_sm_edge = np.linspace(v_l, v_r, 51)
    ppv_rot, _, sigma, spct_dat, sigma_spct, v, pos_t, pos_a, pos_ix_a, pos_iy_a = wodp.read_wind_data(
        obj=obj, line=line, side=side, v_cut=v_cut, v_sm_edge=v_sm_edge)
elif line == 'Halpha':
    v_r = 250
    v_l = -250
    v_sm_edge = np.linspace(v_l, v_r, 51)
    spct_dat, sigma_spct, v, pos_t, pos_a = wodp.read_wind_data(
        obj=obj, line=line, side=side, v_cut=v_cut, v_sm_edge=v_sm_edge)
    ppv_rot, _, sigma, spct_hi_dat, sigma_spct, v, pos_t, pos_a, pos_ix_a, pos_iy_a = wodp.read_wind_data(obj='M82', line='HI', side='use_for_Halpha', v_cut=0, v_sm_edge=v_sm_edge)

spct_dat_shift = []
for i in range(-5, 6):
    if line == 'CO_2_1' or line == 'HI':
        # np.roll(spct_dat, i, axis=1) )
        spct_dat_shift.append(shift(spct_dat, [0, i], cval=0.))
    elif line == 'Halpha':
        spct_dat_shift.append(shift(spct_dat, i, cval=0.))
spct_dat_shift = np.array(spct_dat_shift)


print('mcmc fitting')
nwalkers = 40   
nsteps = 500

# initial guess
if line == 'CO_2_1' or line == 'HI':
    if incl_mach==0:
        if driver == 'ideal':
            ndim = 4
            pos = np.array([0, 30, 60, 0.8]) + np.array([10, 5, 8, 0.2]) * np.random.randn(nwalkers, ndim)
        elif driver == 'radiation':
            ndim = 5
            pos = np.array([0, 30, 60, 0.8, 60]) + np.array([10, 5, 8, 0.2, 5]) * np.random.randn(nwalkers, ndim)
        elif driver == 'hot':
            ndim = 5
            pos = np.array([0, 30, 60, 0.8, 10]) + np.array([10, 5, 8, 0.2, 1]) * np.random.randn(nwalkers, ndim)
    elif incl_mach==1:
        if driver == 'ideal':
            ndim = 5
            pos = np.array([0, 30, 60, 0.8, 1.5]) + np.array([10, 5, 8, 0.2, .5]) * np.random.randn(nwalkers, ndim)
        elif driver == 'radiation':
            ndim = 6
            pos = np.array([0, 30, 60, 0.8, 60, 1.5]) + np.array([10, 5, 8, 0.2, 5, .5]) * np.random.randn(nwalkers, ndim)
        elif driver == 'hot':
            ndim = 6
            pos = np.array([0, 30, 60, 0.8, 10, 1.5]) + np.array([10, 5, 8, 0.2, 1, .5]) * np.random.randn(nwalkers, ndim)

elif line == 'Halpha':
    if driver == 'ideal':
        ndim = 5
        pos = np.array([0, 30, 60, 0.8, .5]) + np.array([10, 5, 8, 0.2, 0.1]) * np.random.randn(nwalkers, ndim)
    elif driver == 'radiation':
        ndim = 6
        pos = np.array([0, 30, 60, 0.8, 60, .5]) + np.array([10, 5, 8, 0.2, 5, 0.1]) * np.random.randn(nwalkers, ndim)
    elif driver == 'hot':
        ndim = 6
        pos = np.array([0, 30, 60, 0.8, 10, .5]) + np.array([10, 5, 8, 0.2, 1, 0.1]) * np.random.randn(nwalkers, ndim)

# save the chain in h5 file
if line == 'CO_2_1' or line == 'HI':
    filename = "%s/mk_chain/%s_%s/%s_%s_%s_%s_%s_%s.h5" % (
        mcmc_dat_dir, side, line, obj, side, line, driver, p, ex)
elif line == 'Halpha':
    filename = "%s/mk_chain/%s_%s/%s_%s_%s_%s_%s_%s_c%.0f.h5" % (
        mcmc_dat_dir, side,line, obj, side, line, driver, p, ex, c_rho)
os.makedirs(os.path.dirname(filename), exist_ok=True)
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)


# uncomment below line if you would like to find the parameters that raise the non-convergence problem

filename = '%s/non_converge_points/%s_%s/%s_%s_%s_%s_non_converge_points.txt' % (
            mcmc_dat_dir, side, line, line, driver, p, ex)
os.makedirs(os.path.dirname(filename), exist_ok=True)
with open(filename, 'wb') as f:
    if line == 'CO_2_1' or line == 'HI':
        if driver == 'ideal':
            np.savetxt(f, [['phi', 'theta_in', 'theta_out',
                       'lg_mdot']], fmt='%5s', delimiter=',')
        if driver == 'radiation':
            np.savetxt(f, [['phi', 'theta_in', 'theta_out',
                       'lg_mdot', 'tau_0']], fmt='%5s', delimiter=',')
        if driver == 'hot':
            np.savetxt(f, [['phi', 'theta_in', 'theta_out',
                       'lg_mdot', 'u_h']], fmt='%5s', delimiter=',')
    elif line == 'Halpha':
        if driver == 'ideal':
            np.savetxt(f, [['phi', 'theta_in', 'theta_out',
                       'lg_mdot', 'A']], fmt='%5s', delimiter=',')
        if driver == 'radiation':
            np.savetxt(f, [['phi', 'theta_in', 'theta_out',
                       'lg_mdot', 'tau_0', 'A']], fmt='%5s', delimiter=',')
        if driver == 'hot':
            np.savetxt(f, [['phi', 'theta_in', 'theta_out',
                       'lg_mdot', 'u_h', 'A']], fmt='%5s', delimiter=',')



print('start mcmc')
os.environ["OMP_NUM_THREADS"] = "56"

# multiprocessing
from multiprocessing import Pool
from multiprocessing import cpu_count
from contextlib import closing

if line == 'CO_2_1' or line == 'HI':
    theta_2 = (driver, p, ex)
elif line == 'Halpha':
    theta_2 = (driver, p, ex, c_rho)
print(ncpus)
Pool(processes=ncpus)
dtype = [('log_likelihood', float)]

if line=='CO_2_1' or 'HI':
    sampler = emcee.EnsembleSampler(nwalkers, ndim, wodp.log_prob, pool=None, args=(
        theta_2, line, pos_t, pos_a, v, spct_dat, spct_dat_shift, sigma_spct, side, incl_rot, None, incl_mach), backend=backend, blobs_dtype=dtype)
    sampler.run_mcmc(pos, nsteps, progress=True)

if line=='Halpha':
    sampler = emcee.EnsembleSampler(nwalkers, ndim, wodp.log_prob, pool=None, args=(
        theta_2, line, pos_t, pos_a, v, spct_dat, spct_dat_shift, sigma_spct, side, incl_rot, spct_hi_dat, incl_mach), backend=backend, blobs_dtype=dtype)
    sampler.run_mcmc(pos, nsteps, progress=True)

