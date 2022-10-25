This repository contains the code used for the analysis described in Yuan, Krumholz & Martin 2022. The code use the DESPOTIC code to constrain the physical properties of galactic wind. Full description of DESPOTIC and model is in https://bitbucket.org/krumholz/despotic/src/master/, https://despotic.readthedocs.io/en/latest/ and https://ui.adsabs.harvard.edu/abs/2017MNRAS.471.4061K/abstract. 

The code is quite extendable and can be used to study the properties of multi-phase outflow in other galaxies. We attach a simple user guide for this suite of code. 

User guide
There are three main steps to extend the code to analyse other observations: I. import data II. MCMC fitting III. Analyse the results

I. Import observation data

The function read_data in wind_obs_diag_pkg.py defines how we read in the data and select a sample of spectra for comparison.

II. MCMC fitting

wind_mcmc_fit.py: this MCMC fitting module includes CO (2-1), HI 21cm, and Halpha line by default. If you would like to add emission lines of your choice, add some code for generating the spectra for that line in function em_line_spec in wind_obs_diag_pkg.py. Please refer to the DESPOTIC document https://despotic.readthedocs.io/en/latest/ for the details of how to implement this. Note that hot gas driven wind requires tabulate table. Please contact mark.krumholz@anu.edu.au for those tables.

Spcify the directory of DESPOTIC code, observational data, as well as MCMC output

Example of using the code:

If we want to run MCMC for LTE line such as CO_2_1, we execute

    python wind_mcmc_fit.py 56 M82 north CO_2_1 ideal isothermal area 1
    
where 56 is the number of CPU used, M82 is the object, north is the side, CO_2_1 is the line, ideal is the driving mechanism, isothermal is the potential, area is the expansion law, 1 is whether the MCMC include mach number of fitting or not.

If we want to do MCMC for subfcritical line such as Halpha, we also need to specify the clumping factor in the end. Below is an example of this:

python wind_mcmc_fit.py 56 M82 north Halpha ideal isothermal area 1 10
where 10 is the clumping factor.

III. Analyse the results

Three python routines for analyse the results are listed below:

comp_models.py: Distinguish different set of models characterised by different driving mechanisms, potential and expansion law.

    python comp_model.py CO_2_1 north central
    
where central is field of view

plot_pub.py: Make publishable plots, including comarison of spectra, moment map. Ex:

    python plot_pub.py HI north ideal point area best
    
where best is selection policy





