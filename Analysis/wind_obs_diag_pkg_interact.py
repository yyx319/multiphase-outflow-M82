#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 07:52:04 2020

@author: yuxuan
"""

# analysis package for wind observational diagnostic with DESPOTIC
# 
import sys
sys.path.append('/Users/yuxuan/Documents/Research_module/despotic')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import brentq
import matplotlib.cm as cm
import matplotlib.colors as colors
from despotic import emitter
from despotic.winds import pwind, zetaM, sxMach

# Constants; switch to cgs
from scipy.constants import G, c, m_p, m_e, h
from scipy.constants import k as kB
from astropy.units import Msun, yr, Angstrom, pc

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
#temp = 50.
dist = 3.5e3*kpc

def rotate(ppv, center, angle, n_g, zoom, L):
    # rotate the spatial part of the ppv cube
    
    (x0, y0) = center
    c = np.cos(angle*np.pi/180.)
    s = np.sin(angle*np.pi/180.)
    n_v=np.shape(ppv)[0]
    ppv_rot = np.zeros((n_v,n_g,n_g))
    for x in range(n_g):
        for y in range(n_g):
            src_x = c*(x-n_g/2) + s*(y-n_g/2) + x0
            src_y = -s*(x-n_g/2) + c*(y-n_g/2) + y0
            ppv_rot[:,x,y] = ppv[:,int(src_x),int(src_y)]
    if zoom==False:
        pass
    elif zoom==True: 
        mid = (n_g-1)/2
        x_min = mid-L/2
        x_max = mid+L/2
        ppv_rot = ppv_rot[:,x_min:x_max:1,x_min:x_max:1]
    n_v, n_y, n_x = np.shape(ppv_rot)
    return ppv_rot, n_y, n_x


def zeroth_moment_map(v, cube_noise, noise):
    cube_noise[cube_noise<3*noise]=0
    dv = (v[-1]-v[0])/(len(v))
    L_int = np.sum(cube_noise*dv, axis=0)
    noise_int = noise*dv*np.sqrt(len(v))
    
    good_pix_x_a, good_pix_y_a = np.where(L_int>3*noise_int)
    
    # calculate second moment in 'good' pixels
    n_v, n_pix_x,n_pix_y = np.shape(cube_noise)
    m0 = np.zeros((n_pix_x,n_pix_y))
    for x, y in zip(good_pix_x_a, good_pix_y_a):
        lum_los = cube_noise[:,x,y] # lum array in beam 
        m0[x,y] = np.sum(lum_los)
    return m0



def first_moment_map(v, cube_noise, noise):
    # output: zeroth moment map m0; first moment map m1
    cube_noise[cube_noise<3*noise]=0
    dv = (v[-1]-v[0])/(len(v))
    L_int = np.sum(cube_noise*dv, axis=0)
    noise_int = noise*dv*np.sqrt(len(v))
    
    good_pix_x_a, good_pix_y_a = np.where(L_int>3*noise_int)
    
    # calculate second moment in 'good' pixels
    n_v, n_pix_x,n_pix_y = np.shape(cube_noise)
    m0 = zeroth_moment_map(v, cube_noise, noise)
    m1 = np.zeros((n_pix_x,n_pix_y))
    for x, y in zip(good_pix_x_a, good_pix_y_a):
        lum_los = cube_noise[:,x,y] # lum array in beam 
        m1[x,y] = np.sum(v*lum_los)/m0[x,y]
    return m0, m1


def second_moment_map(v, cube_noise, noise):
    # output: zeroth moment map m0; first moment map m1; second moment map m2
    cube_noise[cube_noise<3*noise]=0
    dv = (v[-1]-v[0])/len(v)
    L_int = np.sum(cube_noise*dv, axis=0)
    noise_int = noise*dv*np.sqrt(len(v))
    good_pix_x_a, good_pix_y_a = np.where(L_int>3*noise_int)
    print(len(good_pix_x_a))
    # calculate second moment in 'good' pixels
    n_v, n_pix_x, n_pix_y = np.shape(cube_noise)
    m0, m1 = first_moment_map(v, cube_noise, noise)
    m2 = np.zeros((n_pix_x,n_pix_y))
    for x, y in zip(good_pix_x_a, good_pix_y_a):
        lum_los = cube_noise[:,x,y]               # lum array in beam 
        m2[x,y] = np.sum((v-m1[x,y])**2*lum_los)/m0[x,y]    
    m2 = np.sqrt(m2)
    return m0, m1, m2






##########
# functions to interact with ipynb
##########    
# Velocity grid


def em_line_spec_show(line, temp=50, mach=100, phi=5, theta_in=30, theta_out=50, pos_t=0, pos_a=1, driver = 'ideal', lg_mdot=1., uh=10, tau0=50, expansion='solid angle', pot='isothermal', fc = 1.):
    varpi_t = pos_t*kpc/r0
    varpi = pos_a*kpc/r0
    u = np.linspace(-3, 3, 100)
    phi = phi/90.0*np.pi/2.0
    theta_in = theta_in/90.0*np.pi/2.0
    theta_out = theta_out/90.*np.pi/2.0
    mdot = 10**lg_mdot/( np.cos(theta_in) - np.cos(theta_out)  )
    mdot = mdot*Msun/yr
    
    # create wind object
    md = mdot
    eta = (np.cos(theta_in) - np.cos(theta_out)) * md / mdotstar
    Gamma = brentq(
        lambda g: zetaM(np.log(g), sxMach(mach))/epsff - eta,
        1e-6, 100.)
    print('Gamma = %.3f'%Gamma)
    ex = expansion
    p = pot
    pw3 = pwind(Gamma, mach, driver=driver, potential=p,
                  expansion=ex, geometry='cone sheath',
                  theta=theta_out, theta_in=theta_in, tau0=tau0,
                  phi=phi, uh=uh, fcrit=fc)#, interpabs=5e-3,interprel=5e-3)
    
    tw = m0 / mdot
    ################### 
    # Halpha spectrum #
    ###################
    if line=='Halpha':
        Ha = np.zeros(len(u))
        # Cooling constant
        lam_e = 3.9e-25
        # Wavelength grid
        lam0 = 6562.801*ang
        lam = lam0*(1.0+u*v0/c)
        
        Ha = 1e17*2.0/(36.*np.pi) * lam_e * r0 * (rho0/mH)**2 * lam0 / v0 * (tc/tw)**2 * pw3.Xi(u, varpi=varpi, varpi_t=varpi_t)
        '''
        abd = 1. #this is the abundance of the emitting species relative to hydrogen nuclei, which is just 1 since the species here is atomic hydrogen
        Omega =  8.98 #this is the dimensionless oscillator strength of the transition, A_ul=6.25e8
        wl = lam0
        boltzfac = np.exp(-hP*c/kB/wl/temp)
        fj = 8./(8.+18.*boltzfac) #this is the fraction of atoms in the lower energy state
        Ha_TB = pw3.temp_LTE(u, temp, tw=tw, abd=abd, Omega = Omega, wl = wl, fj = fj, boltzfac = boltzfac, varpi=varpi, varpi_t=varpi_t, correlated=False)
        # convert between K to MJy Sr-1
        Ha = 2*hP*c/lam0**3 / ( np.exp( hP * c/kB / lam0 / Ha_TB ) - 1. )
        '''
        print(Ha)
        plt.plot(u*v0/1e7, np.log10(Ha[:]+1e-30), lw=2)
        plt.plot(np.ones(2)*v0/1e7, [-2, 3], 'k:')
        plt.plot(-np.ones(2)*v0/1e7, [-2, 3], 'k:')
        plt.xlim([-5,5])
        plt.ylim([-2,4])
        plt.show()
    
    ###############
    # CO emission #
    ###############
    elif (line=='CO_1-0' or line=='CO_2-1' or line=='CO_3-2'):    
        co = emitter('CO', 1.1e-4)
        # CO emission profiles
        CO_TB = np.zeros(len(u))
        if line == 'CO_1-0':
            CO_TB = pw3.temp_LTE(u, temp, emit=co, tw=tw, varpi=varpi, varpi_t=varpi_t)
        if line == 'CO_2-1':
            CO_TB = pw3.temp_LTE(u, temp, emit=co, tw=tw, trans=1, varpi=varpi, varpi_t=varpi_t)
        plt.plot(u*v0/1e7, np.log10(CO_TB[:]+1e-30), lw=2)
        plt.plot(np.ones(2)*v0/1e7, [-4, 0.5], 'k:')
        plt.plot(-np.ones(2)*v0/1e7, [-4, 0.5], 'k:')
        plt.xlim([-5, 5])
        plt.ylim([-3, 0.5])
        plt.show()

    elif (line=='HI'):
        # HI emission profiles
        HI_TB = np.zeros(len(u))
        #abd = 1.0  — this is the abundance of the emitting species relative to hydrogen nuclei, which is just 1 since the species here is atomic hydrogen
        #Omega = 5.75e-12 — this is the dimensionless oscillator strength of the transition
        #wl = 21.1 — this is the wavelength of the line in cm
        #fj = 0.25 — this is the fraction of atoms in the lower energy state
        #boltzfac = 1.0 — this is the Boltzmann factor exp(-E / k_B T), which is almost exactly unity for any astrophysically-reasonable temperature
        wl=21.1
        boltzfac = np.exp(-hP*c/kB/wl/temp)
        HI_TB = pw3.temp_LTE(u, temp, tw=tw, abd=1.0, Omega = 5.75e-12, wl = wl, fj = 0.25, boltzfac = boltzfac, varpi=varpi, varpi_t=varpi_t, correlated=False)
        plt.plot(u*v0/1e7, np.log10(HI_TB[:]+1e-30), lw=2)
        plt.plot(np.ones(2)*v0/1e7, [-4, 0.5], 'k:')
        plt.plot(-np.ones(2)*v0/1e7, [-4, 0.5], 'k:')
        plt.xlim([-5, 5])
        plt.ylim([-2, 1.5])
        plt.show()



def em_line_img(line='Halpha', phi=5, theta_in=30, theta_out=50, pos=1, driver = 'hot', uh=10, tau0= 5, mdot=100, expansion='solid angle', 
                pot='isothermal', fc = 1., spatial_resolution = 50):
    varpi_a = pos*kpc/r0
    phi = phi/90.0*np.pi/2.0
    theta_in = theta_in/90.0*np.pi/2.0
    theta_out = theta_out/90.*np.pi/2.0
    mdot = mdot*Msun/yr
    
    # create wind object
    md = mdot
    eta = (np.cos(theta_in) - np.cos(theta_out)) * md / mdotstar
    Gamma = brentq(
        lambda g: zetaM(np.log(g), sxMach(mach))/epsff - eta,
        1e-6, 100.)
    tw = m0 / mdot
    ex = expansion
    p = pot
    pw = pwind(Gamma, mach, driver=driver, potential=p, \
               expansion=ex, geometry='cone sheath', \
               theta=theta_out, theta_in=theta_in, phi=phi, tau0=tau0,
               uh=uh, fcrit=fc) #interpabs=5e-3,interprel=5e-3)
    
    # Define the spatial and velocity grids we will use
    varpi_a = np.linspace(0, 2*kpc, spatial_resolution)/r0
    varpi_t = np.linspace(0, 2*kpc, spatial_resolution)/r0
    vpa2, vpt2 = np.meshgrid(varpi_a, varpi_t, indexing='xy')
    
    if line=='Halpha':
    # Cooling constant
        lam_e = 3.9e-25
        # Get integrated emission everywhere; mask undefined values
        xi = 1.0/(36.*np.pi) * lam_e * r0 * (rho0/mH)**2 * pw.xi(vpa2, vpt2)
        xi[xi > 1.0e300] = np.nan
        xi[xi == 0.0] = np.nan

    
    
    
    if (line=='CO_1-0' or line=='CO_2-1' or line=='CO_3-2'):
        # Import the required molecular data
        co = emitter('CO', 1.1e-4)
        
        # Compute integrated antenna temperature
        intTA = np.zeros(vpa2.shape)
        try:
             data = np.load('m82_CO_int.npz')
             intTA = data['intTA']
        except IOError:
            pass
        for i in range(len(vpa2)):
            print(i)
            if np.amax(intTA[i,:]) == 0.0:
                if line=='CO_1-0':
                    intTA[i,:] = pw.intTA_LTE(v0/1e5, temp, emit=co, tw=tw,trans=0,
                                              varpi=vpa2[i,:],
                                              varpi_t=vpt2[i,:])
                elif line=='CO_2-1':
                    intTA[i,:] = pw.intTA_LTE(v0/1e5, temp, emit=co, tw=tw,trans=1,
                                              varpi=vpa2[i,:],
                                              varpi_t=vpt2[i,:])
                elif line=='CO_3-2':
                    intTA[i,:] = pw.intTA_LTE(v0/1e5, temp, emit=co, tw=tw,trans=2,
                                              varpi=vpa2[i,:],
                                              varpi_t=vpt2[i,:])
            np.savez('m82_CO_int.npz', intTA=intTA)
        
        # Fill in by symmetry
        intTA_big = np.empty(2*np.array(intTA.shape)-1)
        intTA_big[intTA.shape[0]-1:, intTA.shape[1]-1:] = intTA
        intTA_big[:intTA.shape[0], intTA.shape[1]-1:] = intTA[::-1,:]
        intTA_big[intTA.shape[0]-1:, :intTA.shape[1]] = intTA[:,::-1]
        intTA_big[:intTA.shape[0], :intTA.shape[1]] = intTA[::-1,::-1]
    return intTA_big

        
    
    
    
    
    
def em_line_ppv(line='Halpha', phi=5, theta_in=30, theta_out=50, driver = 'hot', uh=10, tau0= 5,mdot=100., expansion='solid angle', 
                pot='isothermal', fc = 1., sp_r=2., spatial_resolution = 50, v_l=-3.*v0/1e5,v_r=3.*v0/1e5, v_res=100):
    phi = phi/90.0*np.pi/2.0
    theta_in = theta_in/90.0*np.pi/2.0
    theta_out = theta_out/90.*np.pi/2.0
    mdot = mdot*Msun/yr
    
    # create wind object
    md = mdot
    eta = (np.cos(theta_in) - np.cos(theta_out)) * md / mdotstar
    Gamma = brentq(
        lambda g: zetaM(np.log(g), sxMach(mach))/epsff - eta,
        1e-6, 100.)
    tw = m0 / mdot
    ex = expansion
    p = pot
    pw = pwind(Gamma, mach, driver=driver, potential=p, \
               expansion=ex, geometry='cone sheath', \
               theta=theta_out, theta_in=theta_in, phi=phi, tau0=tau0,
               uh=uh, fcrit=fc) #interpabs=5e-3,interprel=5e-3)
    
    # Define the spatial and velocity grids we will use
    v_l = v_l*1e5/v0
    v_r = v_r*1e5/v0
    u = np.linspace(v_l, v_r, v_res)
    varpi_a = np.linspace(-2*kpc, 2*kpc, spatial_resolution)/r0
    varpi_t = np.linspace(-2*kpc, 2*kpc, spatial_resolution)/r0
    vpt2, vpa2 = np.meshgrid(varpi_t, varpi_a, indexing='xy')
    
    if line=='Halpha':
    # Cooling constant
        lam_e = 3.9e-25
    
    if (line=='CO_1-0' or line=='CO_2-1' or line=='CO_3-2'):
        # Import the required molecular data
        co = emitter('CO', 1.1e-4)

        # Compute ppv
        T_B_ppv = np.zeros((u.shape[0], varpi_a.shape[0], varpi_t.shape[0]))
        #try:
        #    data = np.load('m82_%s_mdot%.0f_ex_%s_ppv.npz'%(line, mdot, ex))
        #    intTA = data['T_B']
        #except IOError:
        #    pass
        for i in range(len(vpa2)):
            print(i)
            for j in range(len(vpt2)):
                if line=='CO_1-0':
                    T_B_ppv[:,i,j] = pw.temp_LTE(u, temp, emit=co, tw=tw, trans=0, 
                                                 varpi=vpa2[i,j], 
                                                 varpi_t=vpt2[i,j])
                elif line=='CO_2-1':
                    T_B_ppv[:,i,j] = pw.temp_LTE(u, temp, emit=co, tw=tw, trans=1, 
                                                 varpi=vpa2[i,j], 
                                                 varpi_t=vpt2[i,j])
                elif line=='CO_3-2':
                    T_B_ppv[:,i,j] = pw.temp_LTE(u, temp, emit=co, tw=tw, trans=2, 
                                                 varpi=vpa2[i,j], 
                                                 varpi_t=vpt2[i,j])
                    
        
    # manipulate ppv
    #T_B_ppv = T_B_ppv+1e-30
    T_B_ppv = T_B_ppv[:,::-1,:]
    np.savez('m82_%s_mdot%.0f_ex_%s_ppv.npz'%(line, mdot, ex), T_B=T_B_ppv)
            
    return T_B_ppv

        







def abs_line_spec_show(line, phi=5, theta_in=30, theta_out=50, pos=1, driver = 'hot', uh=10, tau=5,mdot=100, 
                  expansion='area', pot='points',fc=1.):
    varpi_a = pos*kpc/r0
    phi = phi/90.0*np.pi/2.0
    theta_in = theta_in/90.0*np.pi/2.0
    theta_out = theta_out/90.*np.pi/2.0
    mdot = mdot*Msun/yr
  
    # create wind object
    md = mdot
    eta = (np.cos(theta_in) - np.cos(theta_out)) * md / mdotstar
    Gamma = brentq(
        lambda g: zetaM(np.log(g), sxMach(mach))/epsff - eta,
        1e-6, 100.)
    ex = expansion
    p = pot
    pw3 = pwind(Gamma, mach, driver=driver, potential=p,
                expansion=ex, geometry='cone sheath',
                theta=theta_out, theta_in=theta_in,
                phi=phi, uh=uh, fcrit=fc, interpabs=5e-3,interprel=5e-3)
    
    
    
    
    
    lineid = [r'Mg II $\lambda\lambda 2796, 2804$',
              r'Na I $\lambda\lambda 5892, 5898$']
    lam0 = [[279.6, 280.4], [589.2, 589.8]]
    u_trans = [[0, (lam0[0][1]-lam0[0][0])/lam0[0][0]*c/v0],
               [0, (lam0[1][1]-lam0[1][0])/lam0[1][0]*c/v0]]
    tX = np.array([[190., 95.], [0.32, 0.16]])*Gyr
    tw = m0 / mdot
    # Wavelength grids
    nlam = 400
    lam = [np.linspace(278., 282., nlam), np.linspace(587., 592., nlam)]
    tau = np.zeros((2, nlam))
    for l, l0 in enumerate(lam0):
        u = (lam[l]/l0[0]-1.0)*c/v0
        tau[l,:] = pw3.tau_c(u, tXtw=tX[l]/tw, u_trans=u_trans[l], varpi=varpi_a)
    
    # MgII
    plt.plot(lam[0], np.exp(-tau[0,:]), lw=2)
    plt.plot(lam0[0][0]*np.ones(2), [0, 1], 'k:')
    plt.plot(lam0[0][1]*np.ones(2), [0, 1], 'k:')
    plt.ylim([0.0, 1])
    plt.show()
    
    # NaI
    plt.plot(lam[1], np.exp(-tau[1,:]), lw=2)
    plt.plot(lam0[1][0]*np.ones(2), [0, 1], 'k:')
    plt.plot(lam0[1][1]*np.ones(2), [0, 1], 'k:')
    plt.ylim([0.0, 1])
    plt.show()
    

# MCMC fitting


    
