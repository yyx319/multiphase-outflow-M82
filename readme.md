This code

User guide

I. Import observation data

change the function read_data in 
wind_obs_diag_pkg.py


II. MCMC fitting

wind_mcmc_fit.py: MCMC fitting for include CO (2-1), HI 21cm, and Halpha line by default. 


add emission lines of your choice. 


III. Analyse the results

comp_models.py distinguish different set of models characterised by different driving mechanisms, potential and expansion law.

comp_obs_th.py compare the theroretical spectra to observed spectra, for a single set of model of (dm, p, ex)

Make publishable plots


