#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 23:13:32 2020
@author: yuxuan
python plot_pub.py HI radiation point area med
"""
import os
os.environ["DESPOTIC_HOME"] = "/Users/yuxuan/Documents/Research_module/despotic"
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage.interpolation import shift
import emcee
import sys
import plot_util as pu
mcmc_dir = '/Users/yuxuan/Desktop/'
sys.path.append('/Users/yuxuan/Desktop/mcmc_code/')
import wind_obs_diag_pkg as wodp
v0 = 120e5*np.sqrt(2)

obj='M82'
line= 'Halpha'#['CO_2_1','HI']
side=sys.argv[1]
dm_a = ['ideal', 'radiation', 'hot'] # ['ideal', 'radiation', 'hot']
pot_a  = ['point', 'isothermal'] # ['point', 'isothermal']
ex_a = ['area', 'intermediate', 'solid'] # ['area', 'intermediate', 'solid']
c_rho_a = [10, 100, 1000]

dm_best = 'ideal'
p_best = 'point'
ex_best = 'solid'
sel_pol = sys.argv[2] # selection policy: best, med
incl_mach=1




post_analysis_mk_chain=1
comp_spct=1

cut_off_phi_l = 0
cut_off_phi_u = 10
alpha=0.6

# read wind data
if comp_spct==1 or comp_moment==1:
  v_cut=0
  if line=='Halpha':
    v_r = 300
    v_l = -300
    v_sm_edge = np.linspace(v_l,v_r,41)
    #ppv_rot, v_ppv, sigma, spct_hi, sigma_spct, v, pos_t, pos_a, pos_ix_a, pos_iy_a = wodp.read_wind_data(obj='M82', line='HI', side='use_for_Halpha', v_cut=0, v_sm_edge=v_sm_edge)
    ppv_rot, _, sigma, spct_hi_dat, sigma_spct, v, pos_t, pos_a, pos_ix_a, pos_iy_a = wodp.read_wind_data(obj='M82', line='HI', side='use_for_Halpha', v_cut=0, v_sm_edge=v_sm_edge)
    spct_dat, sigma_spct, v, pos_t, pos_a = wodp.read_wind_data(obj=obj, line=line, side=side, v_cut=v_cut, v_sm_edge=v_sm_edge)

  T_B_dat = spct_dat
  u = v*1e5/v0 # dimensionless velocity

  if side == 'north':
    spct_hi = spct_hi_dat[0]
  else:
    spct_hi = spct_hi_dat[1]

print(pos_t, pos_a)

from matplotlib import rcParams

#################################
# post analysis on markov chain #
#################################
nwalkers=10
discard = 50
dm = dm_best
p = p_best
ex = ex_best
ndim, labels, samples, flat_samples, log_likelihood_samps, fit_par_best, fit_par_med, fit_par_w = pu.read_chain(line, obj, side, dm, p, ex, c_rho=100, discard=discard, thin=1, nwalkers=nwalkers)

rcParams["font.size"] = 20
plt.rc('text',usetex=True)
plt.rc('font', family='serif',size=20)
if post_analysis_mk_chain==1:
  print('analyse chain')
  # corner plot


  import corner
  if line=='Halpha':
    idx = np.where( flat_samples[:,0]<20 )[0]
  print(len(idx) )
  flat_samples = flat_samples[idx,:]
  #we are just fitting one atmosphere
  flat_samples[:,3] = flat_samples[:,3]-np.log10(2.)
  print(np.shape(flat_samples) )
  if dm=='ideal':
    fig = corner.corner(flat_samples, labels=labels, range=[(-90.,90.), (0.,90.), (0., 90.), (-1, 2), (0,1), (0,3)], fontsize=18);
  elif dm=='radiation':
    fig = corner.corner(flat_samples, labels=labels, range=[(-90.,90.), (0.,90.), (0., 90.), (-1, 2), (0,1), (20, 100), (0,3)], fontsize=18)
  elif dm=='hot':
    fig = corner.corner(flat_samples, labels=labels, range=[(-90.,90.), (0.,90.), (0., 90.), (-1, 2), (0,1), (5, 20), (0,3)], fontsize=18)
  #fig.subplots_adjust(wspace=0, hspace=0)
  #fig.subplots_adjust(left=0.05, right=0.95,bottom=0.05, top=0.95)
  #plt.rc('text', usetex=True)
  #plt.rc('font', family='serif',size=20)
  fig.savefig('figure_pub/corner_%s_%s_%s_%s.pdf'%(line, dm, p, ex))


if line=='Halpha':
    nrow=3
    ncol=1
    figsize=(5, 12.)

##################################
# compare observation and theory #
##################################
# choose 20 walkers
# compare spectra
if comp_spct==1:
  # example spectrum
  u = v*1e5/v0 # dimensionless velocity
  dm = dm_best
  p = p_best
  ex = ex_best
  print(p,ex)




  # walkers
  fig, axs = plt.subplots(3,1, figsize=figsize, sharex=True, sharey=True, tight_layout=True, gridspec_kw = {'wspace':0., 'hspace':0.})
  for i, c_rho in zip(range(len(c_rho_a)), c_rho_a ):
    ndim, labels, samples, flat_samples, log_likelihood_samps, fit_par_best, fit_par_med, fit_par_w = pu.read_chain(line, obj, side, dm, p, ex, c_rho=c_rho, discard=discard, thin=1, nwalkers=nwalkers)
    phi_b, theta_in_b, theta_out_b, lg_mdot_b, A_b, tau0_b, uh_b, lg_mach_b, mdot_b  = eval('fit_par_'+sel_pol)
    phi_a, theta_in_a, theta_out_a, lg_mdot_a, A_a, tau0_a, uh_a, lg_mach_a, mdot_a = fit_par_w
    print('fitted parameter', phi_b, theta_in_b, theta_out_b, lg_mdot_b, A_b, tau0_b, uh_b, lg_mach_b, mdot_b)

    mach_b=10**lg_mach_b
    mach_a=10**lg_mach_a

    axs[i].fill_between(v, spct_dat-sigma_spct, spct_dat+sigma_spct, color = 'grey', alpha=0.5)
    spct_ha, error = wodp.em_line_spec(line='Halpha', mach=mach_b, phi=phi_b, theta_in=theta_in_b, theta_out=theta_out_b, pos_t=pos_t, pos_a=pos_a, driver=dm, mdot=mdot_b, uh=uh_b, tau0=tau0_b, expansion=ex, pot=p, fc = 1., u=u)

    spct = c_rho*spct_ha+A_b*spct_hi
    print(spct_ha)
    spct[spct!=spct]=0
    color, ls, lw = pu.plot_style(dm, p, ex, dm_best, p_best, ex_best)

    axs[i].plot(v, spct, alpha=0.6, ls=ls, lw=lw, label = r'$c_\rho = %d$'%(int(c_rho)) )

    for phi, theta_in, theta_out, lg_mdot, A, tau0, uh, mdot, mach in zip(phi_a, theta_in_a, theta_out_a, lg_mdot_a, A_a, tau0_a, uh_a, mdot_a, mach_a):
        spct_ha, error = wodp.em_line_spec(line=line, mach=mach, phi=phi, theta_in=theta_in, theta_out=theta_out, pos_t=pos_t, pos_a=pos_a, driver=dm, mdot=mdot, uh=uh, tau0=tau0, expansion=ex, pot=p, fc = 1., u=u)
        print(spct_ha)
        spct = c_rho*spct_ha+A*spct_hi
        axs[i].plot(v, spct, 'orange', linewidth='1.5', alpha=alpha)

    axs[i].legend()
    axs[i].set_ylabel(r'$j_\nu$ [erg/s/Hz/Sr]', fontsize=20)
    if i==nrow-1:
        axs[i].set_xlabel('$v$ [km s$^{-1}$]', fontsize=20);
    axs[i].tick_params(axis="x", labelsize=20)
    axs[i].tick_params(axis="y", labelsize=20)
    axs[i].axis([-300,300,0,10.])
  plt.subplots_adjust(left=0.05, right=0.99,bottom=0.08, top=0.99)
  plt.savefig('figure_pub/%s_spec_comp.pdf'%(line))
