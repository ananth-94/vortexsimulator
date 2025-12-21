#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <cmath>
#include <string>

using namespace std;

const double pi = M_PI;

// Vortex creation
const string creation_condition = 
"GENERATE_WITHIN_TRAPS"; 								// Condition to GENERATE vortices, or GENERATE_WITHIN_TRAPS, or LOAD vortices from a file placed in ./resources

// System basic
const double R = 10.0; 									// Radius of the container or the star
const int nv = 10000; 									// Number of superfluid vortices within
const int npin_desired = 100000;						// Number of pinning sites within
const double k = 0.4; 									// Circulation strength
const double phi = 0.1; 								// Dissipation strength

// System derived
const double omega_0 = (nv * k)/(R * R); 				// Feynman rotation rate of superfluid // Used as the initial rotation rate of the container
const double T_0 = (2*pi) / omega_0; 					// Corresponding time period // Used as the initial time period of the star
const double a = R * sqrt(pi / npin_desired); 			// Separation between pinning sites
const double Xi = 0.5 * a; 								// Characteristic pinning radius

// System variables
inline double omega_c = omega_0; 						// Rotation rate of the container // Initialized to Feynman rotation rate
inline double omega_s = omega_0; 						// Rotation rate of the superfluid // Initialized to Feynman rotation rate

// Pinning
const string pin_config = "TRAPS"; 					// Condition to generate pinning sites with STANDARD, ANNULAR, or TRAPS strength variation
const int nmax = floor(2*R/a)+1; 						// nmax is the number of pinning sites along a diameter //+1 is a must for correct counting of pinning sites
// Standard
const double V_0 = 10000; 								// Pinning strength throughout the star
// Annular
const double r_annulus = R/sqrt(2); 					// Distance from the centre of the container to the edge of the inner disc
const double V_annular_in = 0; 							// Pinning strength within the inner disc
const double V_annular_out = 0; 						// Pinning strength within the outer annulus
/*// Traps1
const int trap_number = 500; 							// Number of square traps within the star // Set by user
const int trap_ratio = 4; 								// Ratio of the separation between the traps to the size of the traps // Set by user
const double trap_separation = 
(R * sqrt(pi/trap_number)); 							// Separation between two square traps // Derived quantity for now
const double trap_size = trap_separation / trap_ratio; 	// Length of one size of the square trap // Derived quantity for now
const double V_trap_in = 10000; 						// Pinning strength within the trap
const double V_trap_out = 0; 							// Pinning strength outside the trap*/
// Traps2
const double trap_size = 0.5; 							// Length of one size of the square trap // Set by user
const int trap_ratio = 2; 								// Ratio of the separation between the traps to the size of the traps // Set by user
const double trap_separation = 
trap_ratio * trap_size; 								// Separation between two square traps // Derived quantity
const int trap_number =
(pi*R*R) / (trap_separation*trap_separation); 			// Number of square traps within the star // Derived quantity
const double V_trap_in = 10000; 						// Pinning strength within the trap
const double V_trap_out = 0; 							// Pinning strength outside the trap


// Triggers
const string trigger_type = "TRAPWISE"; 						// Condition to generate triggers of type SECTORIAL or STRESSWAVE or TRAPWISE. Triggers can be switched OFF too
const int number_triggers = 20; 							// Number of triggers
const double duration_triggers = 1 * T_0; 				// Duration of each trigger
// Sectorial
const int sector_count = 8; 							// Number of sectors, any one of which could be randomly triggered at the appropriate times
const string sectorial_region = "OUTER"; 				// Sectorial triggers could happen in OUTER, INNER, or FULLSECTOR regions
// Stresswave
const double stresswave_threshold = 0.7; 				// Reduction factor of the pinning strength across the star to emulate stresswaves (between 0 and 1)
// Trapwise
const int number_trapwise = 1;							// Number of traps to be triggered
const double r_trapwise_in = 0 * R;						// The inner limit of the radial region to be considered
const double r_trapwise_out = 1 * R;					// The outer limit of the radial region to be considered


// Dynamics
const double N_ext = (-0.25e-3) * (omega_0/T_0); 		// Deceleration of the star

// Glitch finding
const double epsilon_glitch = 1e-12; 					// Smallest glitch allowed

// Integration
const string integration_condition = "BARNESHUT"; 		// Condition to integrate using EXACT or BARNESHUT procedure
const double error_tolerance = 1e-5; 					// Error tolerance
// Barnes-Hut
const double size_bh_real = 2*R; 						// End-to-end size of the Barnes-Hut box for real vortices
const double size_bh_image = 100*R; 					// End-to-end size of the Barnes-Hut box for image vortices
const double theta_bh_real = 0.5; 						// Acceptable angle (~ size_box/distance_from_vortex) below which Barnes-Hut force calculation is allowed for real vortices
const double theta_bh_image = 0.5; 						// Acceptable angle (~ size_box/distance_from_vortex) below which Barnes-Hut force calculation is allowed for image vortices

// Run
const double del_t = T_0/50; 							// Maximum integration time step
const double stabilization_runtime = 10 * T_0; 			// Duration of stabilization
const double dynamics_runtime = 2000 * T_0;				// Duration of dynamics
const int number_runs = 5; 								// Number of runs of the same simulation

// Filesystem
const string path = "./output/"; 						// Path in which the data output folders are created
inline string pathplace; 								// Mutable directory for output

// Writing
const bool write_fullpos = false; 						// Condition to write full position data (states vs time) after dynamics
const int minimal_step = 5;								// The full position data is reduced to (1/minimal_step) times its size

#endif
