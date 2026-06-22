#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <cmath>
#include <string>

using namespace std;

const double pi = M_PI;

// Vortex creation
const string creation_condition = 
"GENERATE"; 											// Condition to GENERATE vortices, or GENERATE_WITHIN_TRAPS, or LOAD vortices from a file placed in ./resources

// System basic
const double R = 10.0; 									// Radius of the container or the star
const int nv = 5000; 									// Number of superfluid vortices within
const int npin_desired = 100000;						// Number of pinning sites within
const double k = 1; 									// Circulation strength
const double phi = 0.1; 								// Dissipation strength
const double I_c = 1.0;									// Moment of inertia of the crust
const double I_s = 1.0;									// Moment of inertia of the superfluid

// System derived
const double omega_0 = (nv * k)/(R * R); 				// Feynman rotation rate of superfluid // Used as the initial rotation rate of the container
//const double omega_0 = 40; 								// Feynman rotation rate of superfluid // Used as the initial rotation rate of the container
const double T_0 = (2*pi) / omega_0; 					// Corresponding time period // Used as the initial time period of the star
const double a = R * sqrt(pi / npin_desired); 			// Separation between pinning sites
const double Xi = 0.1 * a;								// Characteristic pinning radius
const double I_rel = I_s/I_c;							// Ratio of the moment of inertias

// System variables
inline double omega_c = omega_0; 						// Rotation rate of the container // Initialized to Feynman rotation rate
inline double omega_s = omega_0; 						// Rotation rate of the superfluid // Initialized to Feynman rotation rate

// Pinning
const string pin_config = "STANDARD"; 					// Condition to generate pinning sites with STANDARD, ANNULAR, or TRAPS strength variation
const int nmax = floor(2*R/a)+1; 						// nmax is the number of pinning sites along a diameter //+1 is a must for correct counting of pinning sites
// Standard
const double V_0 = 10000.0; 							// Pinning strength throughout the star
// Annular
const double r_annulus = R/sqrt(2); 					// Distance from the centre of the container to the edge of the inner disc
const double V_annular_in = 0; 							// Pinning strength within the inner disc
const double V_annular_out = 0; 						// Pinning strength within the outer annulus
// Traps
const double trap_size = 0.5; 							// Length of one size of the square trap // Set by user // 0.5 DEBUG
const double trap_separation = 1.0;								// Separation between two square traps
const int trap_number =
(pi*R*R) / (trap_separation*trap_separation); 			// Number of square traps within the star // Derived quantity
const double trap_padding = 0.05 * trap_size;      		// The buffer zone at trap boundaries where vortices are not initialized
const double V_trap_in = 10000.0; 						// Pinning strength within the trap
const double V_trap_out = 0.0; 							// Pinning strength outside the trap


// Triggers
const string trigger_type = "OFF"; 						// Condition to generate triggers of type SECTORIAL or STRESSWAVE or TRAPWISE. Triggers can be switched OFF too
const int number_triggers = 2; 						// Number of triggers
const double duration_triggers = 1 * T_0; 				// Duration of each trigger
// Sectorial
const int sector_count = 8; 							// Number of sectors, any one of which could be randomly triggered at the appropriate times
const string sectorial_region = "OUTER"; 				// Sectorial triggers could happen in OUTER, INNER, or FULLSECTOR regions
// Stresswave
const double stresswave_threshold = 0.7; 				// Reduction factor of the pinning strength across the star to emulate stresswaves (between 0 and 1)
// Trapwise
const int number_trapwise = 1;							// Number of traps to be triggered
const double r_trapwise_in = 0.0 * R;					// The inner limit of the radial region to be considered
const double r_trapwise_out = 1.0 * R;					// The outer limit of the radial region to be considered


// Dynamics
const double A_ext = (-0.25e-3) * (omega_0/T_0); 		// Spindown acceleration acting on the star
const double N_ext = A_ext * I_c; 						// Spindown torque acting on the star


// Glitch finding
const double epsilon_glitch = 1e-12; 					// Smallest glitch allowed

// Integration
const string integration_condition = "BARNESHUT"; 		// Condition to integrate using EXACT or BARNESHUT procedure
const double error_tolerance = 1e-5; 					// Error tolerance
// Barnes-Hut
const double size_bh_real = 2.0*R; 						// End-to-end size of the Barnes-Hut box for real vortices
const double size_bh_image = 100.0*R; 					// End-to-end size of the Barnes-Hut box for image vortices
const double theta_bh_real = 0.5; 						// Acceptable angle (~ size_box/distance_from_vortex) below which Barnes-Hut force calculation is allowed for real vortices
const double theta_bh_image = 0.5; 						// Acceptable angle (~ size_box/distance_from_vortex) below which Barnes-Hut force calculation is allowed for image vortices

// Run
const double del_t = T_0/200.0; 						// Maximum integration time step
const double stabilization_runtime = 2.0 * T_0; 		// Duration of stabilization
const double dynamics_runtime = 2.0 * T_0;			// Duration of dynamics
const int number_runs = 1; 								// Number of runs of the same simulation

// Filesystem
const string path = "./output/"; 						// Path in which the data output folders are created
inline string pathplace; 								// Mutable directory for output

// Writing
const int write_step = 10;								// The position data is recorded only for every write_step-th iteration // The time difference between two written data points is del_t * write_step 

#endif
