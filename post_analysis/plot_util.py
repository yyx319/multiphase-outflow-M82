import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage.interpolation import shift
import emcee
import os
os.environ["DESPOTIC_HOME"] = '/data/ERCblackholes3/yuxuan/wind_obs/despotic'
import sys
mcmc_dir = '/data/ERCblackholes3/yuxuan/wind_obs'
sys.path.append('/data/ERCblackholes3/yuxuan/wind_obs/multiphase-outflow-M82/mcmc_code')
import wind_obs_diag_pkg as wodp
v0 = 120e5*np.sqrt(2)
incl_mach=1 #

def cal_v_sh(T_B_dat, spct):
    # add velocity shift to the spectra to minimize kaisq
    T_B_dat_shift = []
    for j in range(-5,6):
        T_B_dat_shift.append( shift(T_B_dat, j, cval=0.) )
    # ax0: vel_sh ax1: spec vel
    T_B_dat_shift=np.array(T_B_dat_shift)
    delta_spct = ( [spct]*11 - T_B_dat_shift )**2 # 2D
    delta_aa = np.sum( delta_spct, axis=1 ) # sum over spectrum velocity; delta_sum shift*position
    delta_a = np.min(delta_aa, axis=0 ) # for each position minimize the kaisq along axis of velocity shift, delta 1D array position
    s_idx = np.argmin(delta_aa)
    v_sh = 5 - s_idx
    return v_sh



def read_chain(line, obj, side, dm, p, ex, c_rho=100, discard=100, thin=1, nwalkers=5, fov='central', verbose=False):
    #################################
    # post analysis on markov chain #
    #################################
    if line=='CO_2_1' or line=='HI':
        if fov=='central':
            filename='%s/mk_chain/%s_%s/%s_%s_%s_%s_%s_%s.h5'%(mcmc_dir, side, line, obj, side, line, dm, p, ex)
        elif fov=='full':
            filename='%s/mk_chain/%s_%s_full/%s_%s_%s_%s_%s_%s_full.h5'%(mcmc_dir, side, line, obj, side, line, dm, p, ex)
        if dm=='ideal':
            ndim=4
            labels = [r"$\phi$", r"$\theta_{\mathrm{in}}$", r"$\theta_{\mathrm{out}}$",r"$\log\dot{M}$"]
        elif dm=='radiation':
            ndim=5
            labels = [r"$\phi$", r"$\theta_{\mathrm{in}}$", r"$\theta_{\mathrm{out}}$",r"$\log\dot{M}$", r'$\tau_0$']
        elif dm=='hot':
            ndim=5
            labels = [r"$\phi$", r"$\theta_{\mathrm{in}}$", r"$\theta_{\mathrm{out}}$",r"$\log\dot{M}$", r'u$_h$']
    elif line=='Halpha':
        filename='%s/mk_chain/%s_%s/%s_%s_%s_%s_%s_%s_c%d.h5'%(mcmc_dir, side, line, obj, side, line, dm, p, ex, c_rho)
        if dm=='ideal':
            ndim=6
            labels = [r"$\phi$", r"$\theta_{\mathrm{in}}$", r"$\theta_{\mathrm{out}}$",r"$\log\dot{M}$", 'A']
        elif dm=='radiation':
            ndim=7
            labels = [r"$\phi$", r"$\theta_{\mathrm{in}}$", r"$\theta_{\mathrm{out}}$",r"$\log\dot{M}$", 'A', r'$\tau_0$']
        elif dm=='hot':
            ndim=7
            labels = [r"$\phi$", r"$\theta_{\mathrm{in}}$", r"$\theta_{\mathrm{out}}$",r"$\log\dot{M}$", 'A', r'u$_h$']
    if incl_mach==1:
        ndim += 1
        labels += [r"$\log \mathcal{M}$"]

    sampler = emcee.backends.HDFBackend(filename)
    # comparing model and data
    try:
        tau = sampler.get_autocorr_time()
        print('autocorrelation time of the chain is:', tau)
    except:
        pass
    samples = sampler.get_chain()
    log_likelihood_samps = sampler.get_blobs()['log_likelihood']
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    flat_log_likelihood_samps = sampler.get_blobs(discard=discard, thin=thin, flat=True)['log_likelihood']
    print('read chain successfully')

    
    # fit parameters that give the maximum likelihood
    idx = flat_log_likelihood_samps.argmax()
    print(idx)
    if line=='CO_2_1' or line=='HI':
        fit_par = np.zeros(7+incl_mach)
    elif line=='Halpha':
        fit_par = np.zeros(8+incl_mach)
    fit_par[:4] = flat_samples[idx, :4] # phi, theta_in, theta_out, lgmdot
    if dm=='radiation':
        fit_par[4] = flat_samples[idx, 4] # tau_0
    elif dm=='hot':
        fit_par[5] = flat_samples[idx, 4] # u_h
    f_A = np.cos(fit_par[1]/90.0*np.pi/2.0)-np.cos(fit_par[2]/90.0*np.pi/2.0)
    mdot = 10**fit_par[3]/f_A # isotropic mass outflow rate
    fit_par[6] = mdot # mdot

    if incl_mach==1:
        if line=='Halpha':
            fit_par[7] = flat_samples[idx, -2] # conversion parameter A
            fit_par[8] = flat_samples[idx, -1] # log_mach
        else:
            fit_par[7] = flat_samples[idx, -1] # log_mach
    elif incl_mach==0 and line=='Halpha':
        fit_par[7] = flat_samples[idx, -1] # conversion parameter A
    fit_par_best = fit_par

    if verbose==True:
        if line=='CO_2_1' or line=='HI':
            print('fitted parameter best (phi, theta_in, theta_out, lgmdot, tau_0, u_h, mdot, lgmach)', fit_par_best)
        elif line=='Halpha':
            print('fitted parameter best (phi, theta_in, theta_out, lgmdot, tau_0, u_h, mdot, A, lgmach)', fit_par_best)

    # medium of fitted parameters
    if line=='CO_2_1' or line=='HI':
        fit_par = np.zeros(8)
    elif line=='Halpha':
        fit_par = np.zeros(9)

    # phi, theta_in, theta_out, lgmdot
    for i in range(4):
        fit_par[i] = np.percentile(flat_samples[:, i], 50)
    # tau_0
    if dm=='radiation':
        fit_par[4] = np.percentile(flat_samples[:, 4], 50)
    # u_h
    elif dm=='hot':
        fit_par[5] = np.percentile(flat_samples[:, 4], 50)
    f_A = np.cos(fit_par[1]/90.0*np.pi/2.0)-np.cos(fit_par[2]/90.0*np.pi/2.0)
    # isotropic mass outflow rate
    mdot = 10**fit_par[3]/f_A # isotropic mass outflow rate
    fit_par[6] = mdot
    if line=='Halpha':
        fit_par[-2] = np.percentile(flat_samples[:, -2], 50)

    if incl_mach==1:
        if line=='Halpha':
            fit_par[7] = np.percentile(flat_samples[:, -2], 50)  # conversion parameter A
            fit_par[8] = np.percentile(flat_samples[:, -1], 50)  # log_mach
        else:
            fit_par[7] = np.percentile(flat_samples[:, -1], 50)  # log_mach
    elif incl_mach==0 and line=='Halpha':
        fit_par[7] = np.percentile(flat_samples[:, -1], 50) # conversion parameter A
    fit_par_med = fit_par
    
    if verbose==True:
        if line=='CO_2_1' or line=='HI':
            print('fitted parameter medium (phi, theta_in, theta_out, lgmdot, tau_0, u_h, mdot, lgmach)', fit_par_med)
        elif line=='Halpha':
            print('fitted parameter medium (phi, theta_in, theta_out, lgmdot, tau_0, u_h, mdot, A, lgmach)', fit_par_med)

    #with open('/home/yuxuan/wind_obs/post_analysis/%s_%s_best_fit_par.txt'%(side, line), 'ab') as f:
        #np.savetxt(f, [[dm, p, ex]], fmt='%5s', delimiter=',')
        #np.savetxt(f, [[phi, theta_in, theta_out, lg_mdot, tau0, uh]], fmt='%5f', delimiter=',')


    # fitted parameters of walkers
    if line=='CO_2_1' or line=='HI':
        fit_par_w = np.zeros( (nwalkers, 8) )
    elif line=='Halpha':
        fit_par_w = np.zeros( (nwalkers, 9) )
    fit_par_w[:,:4] = samples[-1,:nwalkers,:4] # phi, theta_in, theta_out, lgmdot
    if dm=='radiation':
        fit_par_w[:,4] = samples[-1, :nwalkers, 4] # tau_0
    elif dm=='hot':
        fit_par_w[:,5] = samples[-1, :nwalkers, 4] # u_h
    f_A_a = np.cos(fit_par_w[:,1] /90.0*np.pi/2.0) - np.cos(fit_par_w[:,2]/90.0*np.pi/2.0)
    mdot_a=10**fit_par_w[:,3]/f_A_a # isotropic mass outflow rate
    fit_par_w[:,6] = mdot_a

    if incl_mach==1:
        if line=='Halpha':
            fit_par_w[:,7] = samples[-1, :nwalkers, -2] # conversion parameter A
            fit_par_w[:,8] = samples[-1, :nwalkers, -1] # log_mach
        else:
            fit_par_w[:,7] = samples[-1, :nwalkers, -1] # log_mach
    elif incl_mach==0 and line=='Halpha':
        fit_par[:,7] = samples[-1, :nwalkers, -1] # conversion parameter A
    fit_par_w = np.transpose(fit_par_w)
    if verbose==True:
        if line=='CO_2_1' or line=='HI':
            print('fitted parameter walkers (phi, theta_in, theta_out, lgmdot, tau_0, u_h, mdot, lgmach)', fit_par_w)
        elif line=='Halpha':
            print('fitted parameter walkers (phi, theta_in, theta_out, lgmdot, tau_0, u_h, mdot, A, lgmach)', fit_par_w)
    return ndim, labels, samples, flat_samples, log_likelihood_samps, fit_par_best, fit_par_med, fit_par_w


def plot_style(dm, p, ex, dm_best, p_best, ex_best):
    if p=='point': ls = '-'
    elif p=='isothermal': ls = '--'

    if ex=='area': colors = 'blue'
    elif ex=='intermediate': colors = 'orange'
    elif ex=='solid': colors = 'red'

    if dm==dm_best and p==p_best and ex==ex_best: lw = 8.
    else: lw = 4.

    return colors, ls, lw



#def plot_spct():
