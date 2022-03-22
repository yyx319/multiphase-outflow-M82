This repository contains the code used for the analysis described in Yuan, Krumholz & Martin 2022. The code use the DESPOTIC code to constrain the physical properties of galactic wind. Full description of DESPOTIC and model is in https://bitbucket.org/krumholz/despotic/src/master/ and https://ui.adsabs.harvard.edu/abs/2017MNRAS.471.4061K/abstract. More detailed guidance can be found in the website https://yyx319.github.io/multiphase-outflow-M82/

The code is quite extendable and can be used to study the properties of multi-phase outflow in other galaxies. So we attach a simple user guide for this suite of code.

User guide
There are three main steps to extend the code to analyse other observations: I. import data II. MCMC fitting III. Analyse the results

I. Import observation data
The function read_data in wind_obs_diag_pkg.py defines how we read in the data and select a sample of spectra for comparison.

II. MCMC fitting
wind_mcmc_fit.py: MCMC fitting for include CO (2-1), HI 21cm, and Halpha line by default. 

If you would like to add emission lines of your choice, add some code for generating the spectra for that line in function em_line_spec in wind_obs_diag_pkg.py. Please refer to the DESPOTIC document https://despotic.readthedocs.io/en/latest/ for the details of how to implement this.


III. Analyse the results

Three python routines for analyse the results are listed below:

comp_models.py distinguish different set of models characterised by different driving mechanisms, potential and expansion law.

comp_obs_th.py compare the theroretical spectra to observed spectra, for a single set of model of (dm, p, ex)

plot_pub.py Make publishable plots





