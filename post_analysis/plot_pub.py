#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 23:13:32 2020
@author: yuxuan
python plot_pub.py HI north ideal point area best central
python plot_pub.py CO_2_1 north ideal point area best
"""
import os
os.environ["DESPOTIC_HOME"] = '/data/ERCblackholes3/yuxuan/wind_obs/despotic'
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage.interpolation import shift
import emcee
import sys
import plot_util as pu
mcmc_dir = '/data/ERCblackholes3/yuxuan/wind_obs'
sys.path.append('/data/ERCblackholes3/yuxuan/wind_obs/multiphase-outflow-M82/mcmc_code')
import wind_obs_diag_pkg as wodp
v0 = 120e5*np.sqrt(2)

obj='M82'
line=sys.argv[1] #['CO_2_1','HI']
side=sys.argv[2]
dm_a = ['ideal', 'radiation', 'hot']     # ['ideal', 'radiation', 'hot']
pot_a = ['point', 'isothermal']         # ['point', 'isothermal']
ex_a = ['area', 'intermediate', 'solid'] # ['area', 'intermediate', 'solid']

dm_best =sys.argv[3]
p_best =sys.argv[4]
ex_best =sys.argv[5]
sel_pol = sys.argv[6] # selection policy: best, med
incl_mach=1
if line=='HI':
  fov=sys.argv[7]
else:
  fov='central'


post_analysis_mk_chain=0
comp_spct=0
comp_moment=1 


cut_off_phi_l = 0
cut_off_phi_u = 10
alpha=0.6

fontsize=25

# read wind data
if comp_spct==1 or comp_moment==1:
  v_cut=0
  if line=='CO_2_1' or line=='HI':
    v_r = 250
    v_l = -250
    v_sm_edge = np.linspace(v_l,v_r,51)
    ppv_rot, v_ppv, sigma, spct_dat, sigma_spct, v, pos_t, pos_a, pos_ix_a, pos_iy_a = wodp.read_wind_data(obj=obj, line=line, side=side, v_cut=v_cut, v_sm_edge=v_sm_edge, fov=fov)
  elif line=='Halpha':
    ppv_rot, sigma, spct_hi, sigma_spct, v, pos_t, pos_a, pos_ix_a, pos_iy_a = wodp.read_wind_data(obj='M82', line='HI', side='use_for_Halpha', v_cut=0, v_sm_edge=v_sm_edge)
    spct_dat, sigma_spct, v, pos_t, pos_a = wodp.read_wind_data(obj=obj, line=line, side=side, v_cut=v_cut, v_sm_edge=v_sm_edge)
    v_r = 300
    v_l = -300
    v_sm_edge = np.linspace(v_l,v_r,41)
  T_B_dat = spct_dat
  u = v*1e5/v0 # dimensionless velocity


from matplotlib import rcParams

#################################
# post analysis on markov chain #
#################################

# read parameter
nwalkers=10
discard = 0
dm = dm_best
p = p_best
ex = ex_best
ndim, labels, samples, flat_samples, log_likelihood_samps, fit_par_best, fit_par_med, fit_par_w = pu.read_chain(line, obj, side, dm, p, ex, discard=discard, thin=1, nwalkers=nwalkers, fov=fov)
phi_b, theta_in_b, theta_out_b, lg_mdot_b, tau0_b, uh_b, mdot_b, lg_mach_b = eval('fit_par_'+sel_pol)
Gamma = wodp.cal_Gamma(lg_mdot_b, lg_mach_b)


# make plots
rcParams["font.size"] = 25
plt.rc('text',usetex=True)
plt.rc('font', family='serif',size=25)
if post_analysis_mk_chain==1:
  print('analyse chain')
  # corner plot
  import corner
  #if line=='CO_2_1':
  #  idx = np.where( (flat_samples[:,0]<95) & (flat_samples[:,0]>-95) &  (flat_samples[:,1]>25) )[0]
  #elif line=='HI':
  #  # idx = np.where( (flat_samples[:,0]<10) & (flat_samples[:,0]>-10) &  (flat_samples[:,1]>45) & (flat_samples[:,2]>60) & (flat_samples[:,2]-flat_samples[:,1]>10) )[0]
  #  idx = np.where( flat_samples[:,0]<90 )[0]

  fig, axes = plt.subplots(ndim, figsize=(10, 15), sharex=True)
  for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
  axes[-1].set_xlabel("step number");
  fig.savefig('figure_pub/chain_%s_%s_%s_%s_%s.pdf'%(line, dm, p, ex, fov))

  #flat_samples = flat_samples[idx,:]
  #we are just fitting one atmosphere
  flat_samples[:,3] = flat_samples[:,3]-np.log10(2.)
  if dm=='ideal':
    fig = corner.corner(flat_samples, labels=labels, range=[(-90.,90.), (0.,90.), (0., 90.), (-1, 2), (0,2)], max_n_ticks=3)
  elif dm=='radiation':
    fig = corner.corner(flat_samples, labels=labels, range=[(-90.,90.), (0.,90.), (0., 90.), (-1, 2), (20, 100), (0, 2)], max_n_ticks=3)
  elif dm=='hot':
    fig = corner.corner(flat_samples, labels=labels, range=[(-90.,90.), (0.,90.), (0., 90.), (-1, 2), (5, 20), (0,2)], max_n_ticks=3)
    #fig = corner.corner(flat_samples, labels=labels, range=[(-30.,30.), (0.,60.), (40., 90.), (-1, 2), (5, 20), (0,2)], fontsize=18)

  #fig.subplots_adjust(wspace=0, hspace=0)
  fig.subplots_adjust(left=0.1, right=0.98,bottom=0.1, top=0.98)
  #plt.rc('text', usetex=True)
  #plt.rc('font', family='serif',size=20)
  fig.savefig('figure_pub/corner_%s_%s_%s_%s_%s.pdf'%(line, dm, p, ex, fov))

if line=='CO_2_1':
    nrow=2
    ncol=5
    figsize = (20., 8.)
elif line=='HI' and fov=='central':
    nrow=3
    ncol=5
    figsize=(20., 12.)
elif line=='HI' and fov=='full':
    nrow=3
    if side=='north':
        ncol=4
        figsize=(20., 10.)
    elif side=='south':
        ncol=6
        figsize=(20., 8.)


##################################
# compare observation and theory #
##################################


if comp_spct==1:
  # example spectrum
  u = v*1e5/v0 # dimensionless velocity
  dm = dm_best
  p = p_best
  ex = ex_best
  phi_b, theta_in_b, theta_out_b, lg_mdot_b, tau0_b, uh_b, mdot_b = eval('fit_par_'+sel_pol)[:7]
  phi_a, theta_in_a, theta_out_a, lg_mdot_a, tau0_a, uh_a, mdot_a = fit_par_w[:7]
  if incl_mach==1:
    lg_mach_b = eval('fit_par_'+sel_pol)[-1]
    lg_mach_a = fit_par_w[-1]
    mach_b=10**lg_mach_b
    mach_a=10**lg_mach_a

  if incl_mach==0:
    if line=='CO_2_1':
      mach_b = 100
      mach_a = [100]*9
    elif line=='HI':
      mach_b = 7.4
      mach_a = [7.4]*15
        
  
  ###########
  # walkers #
  ###########
  v_sh_a = []
  fig, axs = plt.subplots(nrow, ncol, figsize=figsize, sharex=True, sharey=True, tight_layout=True, gridspec_kw = {'wspace':0., 'hspace':0.})
  for i, p_t, p_a in zip(range(len(pos_a)), pos_t, pos_a):
    if line=='CO_2_1':
        if i<3: k = i; j = 0
        else: k = (i+1)%5; j = int( (i+1-k)/5 )
    elif line=='HI' and fov=='central':
        k = i%5; j = int( (i-k)/5 )
    elif line=='HI' and fov=='full':
      if i==0:
        continue
      else:
        k = (i-1)%ncol; j = int( (i-1-k)/ncol )

    # observation
    axs[j][k].fill_between(v, T_B_dat[i]-sigma_spct, T_B_dat[i]+sigma_spct, color = 'grey', alpha=0.5, label = r'(%.1f, %.1f)'%(p_t, p_a))
    spct, error = wodp.em_line_spec(line=line, mach=mach_b, phi=phi_b, theta_in=theta_in_b, theta_out=theta_out_b, pos_t=p_t, pos_a=p_a, driver=dm, mdot=mdot_b, uh=uh_b, tau0=tau0_b, expansion=ex, pot=p, fc = 1., u=u)
    spct[spct!=spct]=0
    v_sh = pu.cal_v_sh(T_B_dat[i], spct)
    v_sh_a.append(v_sh)
    color, ls, lw = pu.plot_style(dm, p, ex, dm_best, p_best, ex_best)
    axs[j][k].plot(v, shift(spct, [v_sh], cval=0.), alpha=0.6, ls=ls, lw=lw )

    for phi, theta_in, theta_out, lg_mdot, tau0, uh, mdot, mach in zip(phi_a, theta_in_a, theta_out_a, lg_mdot_a, tau0_a, uh_a, mdot_a, mach_a):
        spct, error = wodp.em_line_spec(line=line, mach=mach, phi=phi, theta_in=theta_in, theta_out=theta_out, pos_t=p_t,
                                        pos_a=p_a, driver=dm, mdot=mdot, uh=uh, tau0=tau0, expansion=ex, pot=p, fc = 1., u=u)
        v_sh = pu.cal_v_sh(T_B_dat[i], spct)
        axs[j][k].plot(v, shift(spct, [v_sh], cval=0.), 'orange', linewidth='1.5', alpha=alpha)

    if k==0:
        axs[j][k].set_ylabel('$T_B$ [K]', fontsize=fontsize);
    if j==nrow-1:
        axs[j][k].set_xlabel('$v$ [km s$^{-1}$]', fontsize=fontsize);
    axs[j][k].legend(loc='upper right' ,fontsize=22)
    axs[j][k].set_xticks([-100, 0, 100])
    axs[j][k].tick_params(axis="x", labelsize=fontsize)
    axs[j][k].tick_params(axis="y", labelsize=fontsize)

    if line=='CO_2_1':
        axs[j][k].axis([-220,220,0, 0.22])
    elif line=='HI':
        axs[j][k].axis([-180,180,0,10.])
  plt.subplots_adjust(left=0.03, right=0.99,bottom=0.04, top=0.99)
  plt.savefig('figure_pub/%s_spec_comp_%s.pdf'%(line, fov))

    
  plt.figure()
  # map of velocity shift
  if line=='HI' and fov=='full':
    plt.scatter(pos_t[1:], pos_a[1:], c=v_sh_a, cmap="bwr")
  else:
    plt.scatter(pos_t, pos_a, c=v_sh_a, cmap="bwr")
  plt.xlabel('x [kpc]'); plt.ylabel('y [kpc]')
  cb = plt.colorbar()
  cb.set_label(r'vel shift [km/s]')
  plt.savefig('figure_pub/v_sh_map_%s.pdf'%fov)

  
  ###################
  # parameter study #
  ###################
  fig, axs = plt.subplots(3,5, figsize=(20., 12.), sharex=True, sharey=True, tight_layout=True, gridspec_kw = {'wspace':0., 'hspace':0.})
  for i, p_t, p_a in zip(range(len(pos_a)), pos_t, pos_a):
    if line=='CO_2_1':
        if i<3: k = i; j = 0
        else: k = (i+1)%5; j = int( (i+1-k)/5 )
    elif line=='HI':
        k = i%5; j = int( (i-k)/5 )
    # select the maximum brightness
    phi, theta_in, theta_out, lg_mdot, tau0, uh, mdot = eval('fit_par_'+sel_pol)[:7]
    if incl_mach==1:
      lg_mach=eval('fit_par_'+sel_pol)[-1]
      mach=10**lg_mach

    if p==p_best and ex==ex_best:
      axs[j][k].fill_between(v, T_B_dat[i]-sigma_spct, T_B_dat[i]+sigma_spct, color = 'grey', alpha=0.5, label = r'(%.1f, %.1f)'%(p_t, p_a))
    spct, error = wodp.em_line_spec(line=line, mach=mach, phi=phi, theta_in=theta_in, theta_out=theta_out, pos_t=p_t,
                                    pos_a=p_a, driver=dm, mdot=mdot, uh=uh, tau0=tau0, expansion=ex, pot=p, fc = 1., u=u)
    spct[spct!=spct]=0
    spct1, error = wodp.em_line_spec(line=line, mach=mach, phi=phi+10, theta_in=theta_in, theta_out=theta_out, pos_t=p_t,
                                    pos_a=p_a, driver=dm, mdot=mdot, uh=uh, tau0=tau0, expansion=ex, pot=p, fc = 1., u=u)
    spct1[spct1!=spct1]=0
    mdot2 = mdot*(np.cos(theta_in*np.pi/180.) - np.cos(theta_out*np.pi/180.) ) / (np.cos( (theta_in-10.)*np.pi/180.) - np.cos( (theta_out-10.)*np.pi/180.) )
    spct2, error = wodp.em_line_spec(line=line, mach=mach, phi=phi, theta_in=theta_in-10., theta_out=theta_out-10., pos_t=p_t,
                                     pos_a=p_a, driver=dm, mdot=mdot2, uh=uh, tau0=tau0, expansion=ex, pot=p, fc = 1., u=u)
    spct2[spct2!=spct2]=0
    spct3, error = wodp.em_line_spec(line=line, mach=mach, phi=phi, theta_in=theta_in, theta_out=theta_out, pos_t=p_t,
                                    pos_a=p_a, driver=dm, mdot=mdot*2, uh=uh, tau0=tau0, expansion=ex, pot=p, fc = 1., u=u)
    spct3[spct3!=spct3]=0

    v_sh = pu.cal_v_sh(T_B_dat[i], spct)
    v_sh1 = pu.cal_v_sh(T_B_dat[i], spct1)
    v_sh2 = pu.cal_v_sh(T_B_dat[i], spct2)
    v_sh3 = pu.cal_v_sh(T_B_dat[i], spct3)
    color, ls, lw = pu.plot_style(dm, p, ex, dm_best, p_best, ex_best)
    axs[j][k].plot(v, shift(spct, [v_sh], cval=0.), ls=ls, lw=lw  , alpha=alpha)
    axs[j][k].plot(v, shift(spct1, [v_sh1], cval=0.), ls=ls, lw=3 , alpha=alpha)
    axs[j][k].plot(v, shift(spct2, [v_sh2], cval=0.), ls=ls, lw=3 , alpha=alpha)
    axs[j][k].plot(v, shift(spct3, [v_sh3], cval=0.), ls=ls, lw=3 , alpha=alpha)
    if k==0:
        axs[j][k].set_ylabel('$T_B$ [K]', fontsize=fontsize);
    if j==2:
        axs[j][k].set_xlabel('$v$ [km/s]', fontsize=fontsize);
    axs[j][k].legend(fontsize=22)
    axs[j][k].tick_params(axis="x", labelsize=fontsize)
    axs[j][k].tick_params(axis="y", labelsize=fontsize)
    if line=='CO_2_1':
        axs[j][k].axis([-220,220,0,0.2])
    elif line=='HI':
        axs[j][k].axis([-180,180,0,10])
  axs[0][4].legend(fontsize=fontsize)
  plt.subplots_adjust(left=0.05, right=0.99,bottom=0.08, top=0.99)
  #plt.rc('text', usetex=True)
  #plt.rc('font', family='serif',size=20)
  plt.savefig('figure_pub/%s_spec_comp_par.pdf'%(line))

  
  ###############################
  # potential and expansion law #
  ###############################
  print('\n\n\n')
  print('Compare different potential and expansion law \n')
  fig, axs = plt.subplots(3,5, figsize=(20., 12.), sharex=True, sharey=True, tight_layout=True, gridspec_kw = {'wspace':0., 'hspace':0.})
  # example spectrum
  u = v*1e5/v0 #dimensionless velocity
  dm = dm_best
  for p in pot_a:
    for ex in ex_a:
      ndim, labels, samples, flat_samples, log_likelihood_samps, fit_par_best, fit_par_med, fit_par_w = pu.read_chain(line, obj, side, dm, p, ex, discard=discard, thin=1, nwalkers=nwalkers)
      phi, theta_in, theta_out, lg_mdot, tau0, uh, mdot = eval('fit_par_'+sel_pol)[:7]
      if incl_mach==1:
        lg_mach=eval('fit_par_'+sel_pol)[-1]
        mach=10**lg_mach
      for i, p_t, p_a in zip(range(len(pos_a)), pos_t, pos_a):
        if line=='CO_2_1':
          if i<3: k = i; j = 0
          else: k = (i+1)%5; j = int( (i+1-k)/5 )
        elif line=='HI':
          k = i%5; j = int( (i-k)/5 )

        # select the maximum brightness
        if p==p_best and ex==ex_best:
          axs[j][k].fill_between(v, T_B_dat[i]-sigma_spct, T_B_dat[i]+sigma_spct, color = 'grey' , label = r'(%.1f, %.1f)'%(p_t, p_a))
        spct, error = wodp.em_line_spec(line=line, mach=mach, phi=phi, theta_in=theta_in, theta_out=theta_out, pos_t=p_t,
                                        pos_a=p_a, driver=dm, mdot=mdot, uh=uh, tau0=tau0, expansion=ex, pot=p, fc = 1., u=u)
        spct[spct!=spct]=0
        v_sh = pu.cal_v_sh(T_B_dat[i], spct)
        color, ls, lw = pu.plot_style(dm, p, ex, dm_best, p_best, ex_best)

        if j==0 and k==0 and p==p_best:
          axs[j][k].plot(v, shift(spct, [v_sh], cval=0.), label=r'ex=%s'%ex, color = color, ls=ls, lw=lw, alpha=alpha)
        elif j==0 and k==1 and ex==ex_best:
          axs[j][k].plot(v, shift(spct, [v_sh], cval=0.), label=r'p=%s'%p, color = color, ls=ls, lw=lw, alpha=alpha)
        else:
          axs[j][k].plot(v, shift(spct, [v_sh], cval=0.), color = color, ls=ls, lw=lw, alpha=alpha)

        if k==0:
            axs[j][k].set_ylabel('$T_B$ [K]');
        if j==nrow-1:
            axs[j][k].set_xlabel('$v$ [km/s]');
        axs[j][k].legend(fontsize=22)
        axs[j][k].tick_params(axis="x", labelsize=fontsize)
        axs[j][k].tick_params(axis="y", labelsize=fontsize)

        if line=='CO_2_1':
            axs[j][k].axis([-220,220,0,0.2])
        elif line=='HI':
            axs[j][k].axis([-180,180,0,10.])
  #plt.subplots_adjust(wspace=0, hspace=0)
  plt.subplots_adjust(left=0.05, right=0.99,bottom=0.08, top=0.99)
  plt.savefig('figure_pub/%s_spec_comp_p_ex.pdf'%(line) )

  
  '''
  #####################
  # driving mechanism #
  #####################
  fig, axs = plt.subplots(3,5, figsize=(20., 12.), sharex=True, sharey=True, tight_layout=True, gridspec_kw = {'wspace':0., 'hspace':0.})
  p = p_best
  ex = ex_best
  for dm in dm_a:
    ndim, labels, samples, flat_samples, log_likelihood_samps, fit_par_best, fit_par_med, fit_par_w = pu.read_chain(line, obj, side, dm, p, ex, discard=discard, thin=1, nwalkers=nwalkers)
    phi, theta_in, theta_out, lg_mdot, tau0, uh, mdot = eval('fit_par_'+sel_pol)[:7]
    if incl_mach==1:
      lg_mach=eval('fit_par_'+sel_pol)[-1]
      mach=10**lg_mach

    for i, p_t, p_a in zip(range(len(pos_a)), pos_t, pos_a):
      if line=='CO_2_1':
        if i<3: k = i; j = 0
        else: k = (i+1)%5; j = int( (i+1-k)/5 )
      elif line=='HI':
        k = i%5; j = int( (i-k)/5 )
      # select the maximum brightness
      if dm==dm_best:
        axs[j][k].fill_between(v, T_B_dat[i]-sigma_spct, T_B_dat[i]+sigma_spct, color = 'grey' , label = r'(%.1f, %.1f)'%(p_t, p_a))
      spct, error = wodp.em_line_spec(line=line, mach=mach, phi=phi, theta_in=theta_in, theta_out=theta_out, pos_t=p_t,
                                      pos_a=p_a, driver=dm, mdot=mdot, uh=uh, tau0=tau0, expansion=ex, pot=p, fc = 1., u=u)
      spct[spct!=spct]=0

      v_sh = pu.cal_v_sh(T_B_dat[i], spct)

      color, ls, lw = pu.plot_style(dm, p, ex, dm_best, p_best, ex_best)


      if j==0 and k==0:
        axs[j][k].plot(v, shift(spct, [v_sh], cval=0.), label=r'dm=%s'%(dm), ls=ls, lw=lw, alpha=alpha)
      else:
        axs[j][k].plot(v, shift(spct, [v_sh], cval=0.), ls=ls, lw=lw, alpha=alpha)

      if k==0:
          axs[j][k].set_ylabel('$T_B$ [K]', fontsize=16);
      if j==nrow-1:
          axs[j][k].set_xlabel('$v$ [km/s]', fontsize=16);
      axs[j][k].legend(fontsize=16)
      axs[j][k].tick_params(axis="x", labelsize=15)
      axs[j][k].tick_params(axis="y", labelsize=15)

      if line=='CO_2_1':
          axs[j][k].axis([-220,220, 0.,0.2])
      elif line=='HI':
          axs[j][k].axis([-180,180, 0.,10.])
  plt.subplots_adjust(left=0.05, right=0.99,bottom=0.08, top=0.99)
  plt.savefig('figure_pub/%s_spec_comp_dm.pdf'%(line))
  
  '''


fontsize=16
######################
# compare moment map #
######################
if comp_moment==1:
  print('compare moment map')
  rcParams["font.size"] = 16
  plt.rc('font', family='serif',size=16)
  nwalkers=0
  dm = dm_best
  p = p_best
  ex = ex_best
  ndim, labels, samples, flat_samples, log_likelihood_samps, fit_par_best, fit_par_med, fit_par_w = pu.read_chain(line, obj, side, dm, p, ex, discard=discard, thin=1, nwalkers=nwalkers)
  phi, theta_in, theta_out, lg_mdot, tau0, uh, mdot = eval('fit_par_'+sel_pol)[:7]
  if incl_mach==1:
    lg_mach=eval('fit_par_'+sel_pol)[-1]
    mach=10**lg_mach
  sp_r=2.5
  x = np.linspace(0, sp_r, 100)
  z1 = x*np.sqrt( (np.cos(2*theta_in*np.pi/180) + np.cos(2*phi*np.pi/180)) / (1-np.cos(2*theta_in*np.pi/180)) )
  z2 = x*np.sqrt( (np.cos(2*theta_out*np.pi/180) + np.cos(2*phi*np.pi/180)) / (1-np.cos(2*theta_out*np.pi/180)) )


  print('compare moment map')
  # first and second moment map
  # velocity gradient along minor axis
  if line=='CO_2_1':
    sp_res = 75
  elif line=='HI':
    sp_res = 75
  print(np.shape(ppv_rot))
  T_B_ppv = np.zeros( (v.shape[0], sp_res, sp_res) )
  T_B_ppv_north = wodp.em_line_ppv(line=line, mach=mach, phi=phi, theta_in=theta_in, theta_out=theta_out, driver=dm, uh=uh, tau0= tau0, mdot=mdot,
                             expansion=ex, pot=p, sp_r=2.5, sp_res=sp_res, v=v, side=side)
  #T_B_ppv[:,int(sp_res/2):,:] = T_B_ppv_north
  T_B_ppv = T_B_ppv_north

  if line=='HI':
    ppv_rot = wodp.rebin(ppv_rot, 155, sp_res, sp_res)
  #if line=='CO_2_1':
  #    ppv_rot = wodp.rebin(ppv_rot, 232, sp_res, sp_res)
  m0, m1, m2 = wodp.second_moment_map(v_ppv, ppv_rot[:, int(sp_res/2):, :], noise=sigma)
  m0[m0==0]=np.nan
  m1[m1==0]=np.nan
  m2[m2==0]=np.nan
  m0_despotic, m1_despotic, m2_despotic = wodp.second_moment_map(v, T_B_ppv, noise=0)
  m1_despotic[m1_despotic==0]=np.nan; m2_despotic[m2_despotic==0]=np.nan
  
  np.savetxt('figure_pub/%s_%s_m0.txt'%(line, side), m0)
  np.savetxt('figure_pub/%s_%s_m1.txt'%(line, side), m1)
  np.savetxt('figure_pub/%s_%s_m2.txt'%(line, side), m2)
  np.savetxt('figure_pub/%s_%s_m0_mod.txt'%(line, side), m0_despotic)
  np.savetxt('figure_pub/%s_%s_m1_mod.txt'%(line, side), m1_despotic)
  np.savetxt('figure_pub/%s_%s_m2_mod.txt'%(line, side), m2_despotic)
  
  if os.path.exists('figure_pub/%s_%s_m0.txt'%(line, side) ):
    m0=np.loadtxt('figure_pub/%s_%s_m0.txt'%(line, side))
    m1=np.loadtxt('figure_pub/%s_%s_m1.txt'%(line, side))
    m2=np.loadtxt('figure_pub/%s_%s_m2.txt'%(line, side))
    m0_despotic=np.loadtxt('figure_pub/%s_%s_m0_mod.txt'%(line, side))
    m1_despotic=np.loadtxt('figure_pub/%s_%s_m1_mod.txt'%(line, side))
    m2_despotic=np.loadtxt('figure_pub/%s_%s_m2_mod.txt'%(line, side))
  
  
  extent = [-sp_r,sp_r, 0, sp_r ]
  row = 2; col=3;
  fig, axs = plt.subplots(row, col, figsize=(16,5) )#, sharex=True, sharey=True, tight_layout=True, gridspec_kw = {'wspace':0.05, 'hspace':0.00})
  if line=='CO_2_1':
      vmin=1e0; vmax=1e3
  elif line=='HI':
      vmin=1e2; vmax=3e3
  p0=axs[0][0].imshow(m0, origin='lower',  extent=extent,
             vmin=vmin, vmax=vmax)
  cb0 = fig.colorbar(p0, ax = axs[0][0], shrink=0.6)
  cb0.set_label('Intensity [K km/s]',size=fontsize)
  axs[0][0].axis(extent)

  p0=axs[0][1].imshow(m0_despotic, origin='lower', extent=extent, vmin=vmin, vmax=vmax)
  cb0 = fig.colorbar(p0, ax = axs[0][1], shrink=0.6)
  cb0.set_label('Intensity [K km/s]',size=fontsize)
  idx = np.where(z2<sp_r)[0]
  axs[0][1].plot(x[idx], z1[idx], 'k', x[idx], z2[idx], 'k',-x[idx], z1[idx], 'k', -x[idx], z2[idx], 'k')
  axs[0][1].axis(extent)

  p0=axs[0][2].imshow((m0-m0_despotic)/m0, origin='lower', extent=extent, cmap='bwr', vmin=-1., vmax=1.)
  cb0 = fig.colorbar(p0, ax = axs[0][2], shrink=0.6)
  cb0.set_label('residue/observed',size=fontsize)
  axs[0][2].plot(x[idx], z1[idx], 'k', x[idx], z2[idx], 'k',-x[idx], z1[idx], 'k', -x[idx], z2[idx], 'k')
  axs[0][2].axis(extent)

  #CO_2_1_m2[CO_2_1_m2>100]=np.nan
  p2=axs[1][0].imshow(m2, origin='lower', extent=extent,  vmin=10, vmax=120 )
  cb2 = fig.colorbar(p2, ax = axs[1][0], shrink=0.6)
  cb2.set_label('linewidth [km/s]',size=fontsize)
  axs[1][0].axis(extent)

  p2=axs[1][1].imshow(m2_despotic, origin='lower', extent=extent,  vmin=10, vmax=120)
  cb2 = fig.colorbar(p2, ax =axs[1][1], shrink=0.6)
  cb2.set_label('linewidth [km/s]',size=fontsize)
  axs[1][1].plot(x[idx], z1[idx], 'k', x[idx], z2[idx], 'k',-x[idx], z1[idx], 'k', -x[idx], z2[idx], 'k')
  axs[1][1].axis(extent)

  p2=axs[1][2].imshow( (m2-m2_despotic)/np.sqrt(m2**2 + sigma_spct), origin='lower', extent=extent, cmap='bwr', vmin=-1., vmax=1.)
  cb2 = fig.colorbar(p2, ax =axs[1][2], shrink=0.6)
  cb2.set_label(r'residue/observed',size=fontsize)
  axs[1][2].plot(x[idx], z1[idx], 'k', x[idx], z2[idx], 'k',-x[idx], z1[idx], 'k', -x[idx], z2[idx], 'k')

  axs[1][2].axis(extent)
  x = np.linspace(-2.5, 2.5, 100)
  for i in range(row):
    for j in range(col):
      axs[i][j].fill_between(x, -0.6, 0.6, color=(0.75, 0.75, 0.75))
      if i==row-1:
        axs[i][j].set_xlabel("$x$ [kpc]",fontsize=18)
      if j==0:
        axs[i][j].set_ylabel("$z$ [kpc]",fontsize=18)
      axs[i][j].tick_params(axis="x", labelsize=16)
      axs[i][j].tick_params(axis="y", labelsize=16)

  plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95)
  plt.savefig('figure_pub/%s_moment_obs_t.pdf'%line)

