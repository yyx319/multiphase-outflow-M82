#include <cstdio>
#include <cmath>
#include <vector>
#include "pwind_expansion.H"
#include "pwind_geom.H"
#include "pwind_potential.H"
#include "pwind_rad.H"
#include "pwind_hot.H"
#include "pwind_ideal.H"
#include <ctime>

int main(int argc, char **argv) {
  //ideal isothermal solid
  const double pi = 3.1415926;
  const double Gamma = 0.15417110388240216;
  const double mach = 65.6344680876;
  const double tau0 = 59.49062385;
  const double uh = 7.83277017;
  const double theta_in = 36.3664932*pi/180;
  const double theta_out = 66.06507513*pi/180;
  const double phi = 6.03335183*pi/180;  
  const double epsabs = 0.0001;
  const double epsrel = 0.01;
  const double fcrit = 1.0;
  pwind_geom_cone_sheath geom(theta_out, theta_in, phi);
  //pwind_rad_is pw(Gamma, mach, tau0, &geom, epsabs, epsrel);
  //const double interprel = 1e-2;
  //const double interpabs = 1e-2;
  //pwind_ideal_is pw(Gamma, mach, &geom, epsabs, epsrel);
  pwind_ideal_is pw(Gamma, mach, &geom);

  int i;
  for(i=1; i<=200; i=i+1){
  std::cout << pw.eta( -3.+6./199.*(i-1.), 1307.4913763643424, 0.25, 0.9999863623929567, false, -1.0, 6.0, 0.0, false) 
<< 
std::endl;
  }
}

