#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 23:14:02 2020

@author: yuxuan
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import emcee
mcmc_dir = '/Users/yuxuan/Desktop'
import sys
# comparing different models
obj='M82'
line='Halpha'
side=sys.argv[1]
dm_a = ['ideal','radiation','hot']#, 'radiation','hot'] #['ideal','radiation','hot']
p_a = ['point', 'isothermal']
ex_a= ['area','intermediate','solid']#, 'intermediate', 'solid']#['area', 'intermediate', 'solid']
c_rho = int(sys.argv[2])

AIC_a = np.zeros( (len(dm_a), len(p_a), len(ex_a) ) )
par_a = np.zeros( (len(dm_a), len(p_a), len(ex_a), 7 ) )
spar1_a = np.zeros( (len(dm_a), len(p_a), len(ex_a), 7 ) )
spar2_a = np.zeros( (len(dm_a), len(p_a), len(ex_a), 7 ) )

for i, dm in enumerate(dm_a):
  if line=='Halpha':
    if dm=='ideal':
      ndim=6
    else:
      ndim=7
  for j, p in enumerate(p_a):
    for k, ex in enumerate(ex_a):
      try:
        print('enter line 38')
        if line=='Halpha':
          filename='%s/mk_chain/%s_%s/%s_%s_%s_%s_%s_%s_c%d.h5'%(mcmc_dir, side, line, obj, side, line, dm, p, ex, c_rho)
        print(filename)
        sampler = emcee.backends.HDFBackend(filename)
        flat_samples = sampler.get_chain(discard=0, thin=1, flat=True)
        print("flat sample size",flat_samples.size)
        log_likelihood_samps = sampler.get_blobs(discard=0, thin=1)['log_likelihood']
        flat_log_likelihood_samps = sampler.get_blobs(discard=0, thin=1,flat=True)['log_likelihood']
        #idx = ( np.abs(flat_samples[:, 2] - flat_samples[:, 1]) > 10 )
        #flat_log_likelihood_samps = flat_log_likelihood_samps[idx]
        # maximum likelihood function
        logL = np.max( flat_log_likelihood_samps )
        AIC = 2*ndim-2*logL # Akaike information criteriea
        AIC_a[i,j,k] = AIC
        print('AIC of the %s, %s, %s is :%f'%(dm, p, ex, AIC) )

        # parameter
        # mass outflow rate of one hemisphere
        flat_samples[:, 3] = flat_samples[:, 3] - np.log10(2)
        # mcmc index to table index

        for x in range(ndim):
          mcmc = np.percentile(flat_samples[:, x], [16, 50, 84])
          print(mcmc)
          q = np.diff(mcmc)
          par_a[i,j,k,x] = mcmc[1]
          spar1_a[i,j,k,x] = q[0]
          spar2_a[i,j,k,x] = q[1]
          print('code go to line 66')
        print('code run to line 75.')
      except:
        AIC_a[i,j,k] = np.inf


print(AIC_a)
print(np.min(AIC_a) )
delta = AIC_a - np.min(AIC_a)
w = np.exp(-delta/2)
w_nor = w/np.sum(w)
# Akaike weight
print('Akaike weight')
for i, dm in enumerate(dm_a):
    for j, p in enumerate(p_a):
        for k, ex in enumerate(ex_a):
            print('Akaike weight of the %s, %s, %s is :%f'%(dm, p, ex, w_nor[i,j,k]) )
            #print('log of Akaike weight for the %s, %s, %s is :%f'%(dm, p, ex, np.log10(w_nor) ) )

# print the parameter of the model with the highest A weight

#np.savetxt(f, [['phi', 'theta_in', 'theta_out', 'lg_mdot', 'tau_0', 'u_h', 'lg_mach', 'A']], fmt='%5s', delimiter=',')
with open('%s_%s_best_fit_par.txt'%(side, line), 'wb') as f:
  if line=='CO_2_1' or line=='HI':
    np.savetxt(f, [[' Dm ', ' Pot ', ' Exp law ', ' $w$ ',' $\\phi$ ', ' $\\theta_{\\rm in}$ ', ' $\\theta_{\\rm out}$ ', '$ \\log \\Dot{M} $', 'A', '\\log \\mathcal{M}' , ' $\\tau_0$ ', ' $u_h$ \\\\',]], fmt='%5s', delimiter='&')

#list = [['ideal', 'ideal', 'radiation'], ['point', 'point', 'isothermal'], ['','',''] ]

for i, dm in enumerate(dm_a):
  for j, p in enumerate(p_a):
    for k, ex in enumerate(ex_a):
      with open('%s_%s_best_fit_par.txt'%(side, line), 'ab') as f:
        if dm=='ideal':
          np.savetxt(f, [[' %s '%dm if j==0 and k==0 else ' ', ' %s '%p if k==0 else ' ',
                          ' %s '%ex, ' %.2f '%w_nor[i,j,k] if w_nor[i,j,k]>=5e-3 else ' 0 ',
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,0], spar1_a[i,j,k,0], spar2_a[i,j,k,0]),
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,1], spar1_a[i,j,k,1], spar2_a[i,j,k,1]),
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,2], spar1_a[i,j,k,2], spar2_a[i,j,k,2]),
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,3], spar1_a[i,j,k,3], spar2_a[i,j,k,3]),
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,4], spar1_a[i,j,k,4], spar2_a[i,j,k,4]),
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,5], spar1_a[i,j,k,5], spar2_a[i,j,k,5]),
                          ' - ', ' - \\\\']], fmt='%5s', delimiter='&')
        elif dm=='radiation':
          np.savetxt(f, [[' %s '%dm if j==0 and k==0 else ' ', ' %s '%p if k==0 else ' ',
                          ' %s '%ex, ' %.2f '%w_nor[i,j,k] if w_nor[i,j,k]>=5e-3 else ' 0 ',
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,0], spar1_a[i,j,k,0], spar2_a[i,j,k,0]),
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,1], spar1_a[i,j,k,1], spar2_a[i,j,k,1]),
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,2], spar1_a[i,j,k,2], spar2_a[i,j,k,2]),
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,3], spar1_a[i,j,k,3], spar2_a[i,j,k,3]),
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,5], spar1_a[i,j,k,5], spar2_a[i,j,k,5]),
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,6], spar1_a[i,j,k,6], spar2_a[i,j,k,6]),
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,4], spar1_a[i,j,k,4], spar2_a[i,j,k,4]),
                          ' - \\\\']], fmt='%5s', delimiter='&')
        elif dm=='hot':
          dm2='hot gas'
          np.savetxt(f, [[' %s '%dm2 if j==0 and k==0 else ' ', ' %s '%p if k==0 else ' ',
                          ' %s '%ex, ' %.2f '%w_nor[i,j,k] if w_nor[i,j,k]>=5e-3 else ' 0 ',
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,0], spar1_a[i,j,k,0], spar2_a[i,j,k,0]),
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,1], spar1_a[i,j,k,1], spar2_a[i,j,k,1]),
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,2], spar1_a[i,j,k,2], spar2_a[i,j,k,2]),
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,3], spar1_a[i,j,k,3], spar2_a[i,j,k,3]),
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,5], spar1_a[i,j,k,5], spar2_a[i,j,k,5]),
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,6], spar1_a[i,j,k,6], spar2_a[i,j,k,6]),
                          ' - ',
                          ' $%.2f_{-%.2f}^{+%.2f}$ \\\\'%(par_a[i,j,k,4], spar1_a[i,j,k,4], spar2_a[i,j,k,4])
                          ]], fmt='%5s', delimiter='&')





