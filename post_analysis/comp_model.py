#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 23:14:02 2020

@author: yuxuan

python comp_model.py CO_2_1 north central
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import emcee
mcmc_dir = '/Users/yuxuan/Desktop'
import sys
# comparing different models
obj='M82'
line=sys.argv[1]
side=sys.argv[2]
fov=sys.argv[3]
if line=='Halpha':
  c_rho = int(sys.argv[4])

print('%s map'%fov )
dm_a = ['ideal','radiation','hot']#, 'radiation','hot'] #['ideal','radiation','hot']
p_a = ['point', 'isothermal']
ex_a= ['area','intermediate','solid']#, 'intermediate', 'solid']#['area', 'intermediate', 'solid']
dm2_a =  ['Ideal','Radiation','Hot gas']
p2_a = ['Point', 'Isothermal']
ex2_a = ['Area','Intermediate','Solid']

AIC_a = np.zeros( (len(dm_a), len(p_a), len(ex_a) ) )
par_a = np.zeros( (len(dm_a), len(p_a), len(ex_a), 7 ) )
spar1_a = np.zeros( (len(dm_a), len(p_a), len(ex_a), 7 ) )
spar2_a = np.zeros( (len(dm_a), len(p_a), len(ex_a), 7 ) )



for i, dm in enumerate(dm_a):
  if line=='CO_2_1' or line=='HI':
    if dm=='ideal':
      ndim=5
    else:
      ndim=6
  elif line=='Halpha':
    if dm=='ideal':
      ndim=6
    else:
      ndim=7
  for j, p in enumerate(p_a):
    for k, ex in enumerate(ex_a):
      try:
        if line=='CO_2_1' or line=='HI':
          if fov=='full':
            filename='%s/mk_chain/%s_%s_full/%s_%s_%s_%s_%s_%s_full.h5'%(mcmc_dir, side, line, obj, side, line, dm, p, ex)
          else:
            filename='%s/mk_chain/%s_%s/%s_%s_%s_%s_%s_%s.h5'%(mcmc_dir, side, line, obj, side, line, dm, p, ex)
        elif line=='Halpha':
          filename='%s/mk_chain/%s_%s/%s_%s_%s_%s_%s_%s_c%d.h5'%(mcmc_dir, side, line, obj, side, line, dm, p, ex, c_rho)
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
    np.savetxt(f, [[' Dm ', ' Pot ', ' Exp law ', ' $w$ ',' $\\phi$ ', ' $\\theta_{\\rm in}$ ', ' $\\theta_{\\rm out}$ ', '$ \\log \\Dot{M} $', '\\log \\mathcal{M}' , ' $\\tau_0$ ', ' $u_h$ \\\\',]], fmt='%5s', delimiter='&')

for i, dm in enumerate(dm2_a):
  for j, p in enumerate(p2_a):
    for k, ex in enumerate(ex2_a):
      print('%s_%s_%s'%(dm, p, ex))
      with open('%s_%s_best_fit_par.txt'%(side, line), 'ab') as f:
        if dm=='Ideal':
          np.savetxt(f, [[' %s '%dm if j==0 and k==0 else ' ', ' %s '%p if k==0 else ' ',
                          ' %s '%ex, ' %.2f '%w_nor[i,j,k] if w_nor[i,j,k]>=5e-3 else ' 0 ',
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,0], spar1_a[i,j,k,0], spar2_a[i,j,k,0]),
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,1], spar1_a[i,j,k,1], spar2_a[i,j,k,1]),
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,2], spar1_a[i,j,k,2], spar2_a[i,j,k,2]),
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,3], spar1_a[i,j,k,3], spar2_a[i,j,k,3]),
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,4], spar1_a[i,j,k,4], spar2_a[i,j,k,4]),
                          ' - ', ' - \\\\']], fmt='%5s', delimiter='&')
        elif dm=='Radiation':
          np.savetxt(f, [[' %s '%dm if j==0 and k==0 else ' ', ' %s '%p if k==0 else ' ',
                          ' %s '%ex, ' %.2f '%w_nor[i,j,k] if w_nor[i,j,k]>=5e-3 else ' 0 ',
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,0], spar1_a[i,j,k,0], spar2_a[i,j,k,0]),
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,1], spar1_a[i,j,k,1], spar2_a[i,j,k,1]),
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,2], spar1_a[i,j,k,2], spar2_a[i,j,k,2]),
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,3], spar1_a[i,j,k,3], spar2_a[i,j,k,3]),
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,5], spar1_a[i,j,k,5], spar2_a[i,j,k,5]),
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,4], spar1_a[i,j,k,4], spar2_a[i,j,k,4]),
                          ' - \\\\']], fmt='%5s', delimiter='&')
        elif dm=='Hot gas':
          np.savetxt(f, [[' %s '%dm if j==0 and k==0 else ' ', ' %s '%p if k==0 else ' ',
                          ' %s '%ex, ' %.2f '%w_nor[i,j,k] if w_nor[i,j,k]>=5e-3 else ' 0 ',
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,0], spar1_a[i,j,k,0], spar2_a[i,j,k,0]),
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,1], spar1_a[i,j,k,1], spar2_a[i,j,k,1]),
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,2], spar1_a[i,j,k,2], spar2_a[i,j,k,2]),
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,3], spar1_a[i,j,k,3], spar2_a[i,j,k,3]),
                          ' $%.2f_{-%.2f}^{+%.2f}$ '%(par_a[i,j,k,5], spar1_a[i,j,k,5], spar2_a[i,j,k,5]),
                          ' - ',
                          ' $%.2f_{-%.2f}^{+%.2f}$ \\\\'%(par_a[i,j,k,4], spar1_a[i,j,k,4], spar2_a[i,j,k,4])
                          ]], fmt='%5s', delimiter='&')





