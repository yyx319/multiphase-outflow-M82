import os

ncpus = '56'
obj = 'M82'
side_a = ['north', 'south']
line_a = ['CO_2_1','HI']
dm_a = ['ideal', 'radiation', 'hot']
p_a = ['point', 'isothermal']
ex_a = ['area', 'intermediate', 'solid']
crho = '10'
incl_mach='1'

for side in side_a:
  for line in line_a:
    os.chdir( '/home/yuxuan/wind_obs/script_job/%s_%s'%(side, line) )
    for dm in dm_a:
      for p in p_a:
        for ex in ex_a:
          file_name='%s_%s_%s_%s.sh'%(line, dm, p, ex)
          with open(file_name, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('#PBS -N %s_%s_%s\n'%(dm, p, ex) )
            f.write('#PBS -l select=1:ncpus=%s\n'%ncpus )
            f.write('#### Request exclusive use of nodes #####\n')
            f.write('#PBS -l place=scatter:excl\n')
            f.write('#PBS -q smallmem\n')
            f.write('#PBS -m abe\n')
            f.write('cd /home/yuxuan/wind_obs/mcmc_code/\n')
            if line=='CO_2_1' or line=='HI':
              f.write('python wind_mcmc_fit.py %s %s %s %s %s %s %s %s'%(ncpus,obj,side,line,dm,p,ex, incl_mach) )
            elif line=='Halpha':
              # three different clumping factor
              f.write('python wind_mcmc_fit.py %s %s %s %s %s %s %s %s'%(ncpus,obj,side,line,dm,p,ex,'10') )
              f.write('python wind_mcmc_fit.py %s %s %s %s %s %s %s %s'%(ncpus,obj,side,line,dm,p,ex,'100') )
              f.write('python wind_mcmc_fit.py %s %s %s %s %s %s %s %s'%(ncpus,obj,side,line,dm,p,ex,'1000') ) 
            f.close() 


