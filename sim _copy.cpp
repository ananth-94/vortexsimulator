// Stabilization and dynamics of superfluid vortices in a neutron star - Anantharaman S V, Ashoka University

// Stabilization (Vortices - yes, pinning - yes, image vortices - yes , spindown - no, feedback - no)
// Dynamics or Simulation (Vortices - yes, pinning - yes, image vortices - yes , spindown - yes, feedback - yes)

// Version 1 of code completed - Aug 20, 2022, 16:00:00
// Version 2 of code completed - Apr 01, 2023, 05:00:00 
// Version 3 of code completed - Jan 07, 2024, 06:00:00
// Version 4 of code completed - May 29, 2024, 13:00:00
// Version 5 of code completed - Sep 29, 2024, 19:00:00

/* Latest changes:
	Gaussian trigger now called sectorial trigger.
	Stresswave was made to function akin to sectorial trigger.
*/

/* Future changes:
	-
*/

/* Required external dependencies:
	Boost v1.79.0
*/	

/* Purpose:
	The following code is used to simulate vortices in a neutron star, given a list of parameters.
	Collective vortex movement is claimed to be responsible for glitches in pulsars.
	This code is devised to help understand that thread of thought.
	We follow closely the methods proposed in Howitt-Melatos-Haskell 2020.
*/
	
/* Input and output files:
	
	Input:
	
	None
	
	Output:
	
	info_simulation.dat
	info_progress.dat
	info_gltich.dat
	init_vortex_pos.dat
	init_img_pos.dat
	stabilized_vortex_pos.dat
	stabilized_Hvalues.dat
	sim_vortex_pos.dat
	sim_vortex_pos_minimal.dat
	sim_omega_c.dat
	sim_omega_s.dat
	sim_triggers.dat
	sim_unpinned.dat
	data_glitch.dat
*/

/* Notes:

	Use optimization level -O3 at the time of compilation.
	
	Compile using: g++ ~/sim.cpp -O3 -I ~/boost_1_79_0 -o ~/sim.out -fopenmp -static-libstdc++
	Run using: ~/sim.out

*/


// ----- Headers and namespaces -----
	
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm> // Required for sort function
#include <boost/numeric/odeint.hpp> // Boost ODE integration package
#include <omp.h> // Required for OpenMP
#include <bits/stdc++.h> // Required for accumulate function -- Summing all contents of a vector

using namespace std;
using namespace boost::numeric::odeint;


// ----- Shorthands -----

#define xi f[i-1]
#define yi f[i-1+nv]
#define ri sqrt(pow(f[i-1],2)+pow(f[i-1+nv],2))
#define xij (f[i-1] - f[j-1])
#define yij (f[i-1+nv] - f[j-1+nv])
#define rij2  (pow((f[i-1] - f[j-1]),2) + pow((f[i-1+nv] - f[j-1+nv]),2))
#define xij_img (f[i-1] - imgvortex_vectx[j-1])
#define yij_img (f[i-1+nv] - imgvortex_vecty[j-1])
#define rij2_img  (pow((f[i-1] - imgvortex_vectx[j-1]),2) + pow((f[i-1+nv] - imgvortex_vecty[j-1]),2))
#define dxidt dfdt[i-1]
#define dyidt dfdt[i-1+nv]

// ----- Jugaad to set directory for outputs -----

string path = "./InOut/"; // Enter the absolute path of the folder
string var, pathplace; // Needed for storing multiple runs in different folders
const int numb_runs = 5; // Desired number of runs of the simulation
int run_id = 1; // Initialization of run_id

/* Follow the below example snippet to open a file of a desired name and perform read/write operations
	string filename = "Test.txt";
	string fullpath = pathplace+filename;
	ofstream file;	
	file.open(fullpath); // erasing content of vortexinit file
	file << "Result: "<< fullpath << endl; //file actions
	file.close();
*/


// ----- Definitions and declarations (Parameter conrols) -----

const double 	pi = M_PI;

// Basic
// For nv = 2000 simulations

const int 	nv = 2000, npin_desired = 20000, numb_trig = 0;
const double 	R = 10, k = 1.0, phi = 0.1, V_1 = 2000, V_2 = 2000, trig_duration = 0, decel = -0.25e-3,
							stab_time = 100, sim_time = 2000;
const string trig_type = "stresswave";

// Advanced							
// Common to all simulations

const string pin_config = "annular", trig_region = "fullsector", write_fullpos = "False";
const double k_img = k, omega_0 = (nv*k) / (R*R), T_0 = (2*pi) / omega_0, a = R * sqrt(pi / npin_desired),
						Xi = 0.1*a , N_ext_0 = (decel) * (omega_0/T_0), n_ratio = 1, r_annulus = (R/sqrt(2)),
						duration_trig = trig_duration*T_0, runtime_stabilization = stab_time*T_0, runtime_simulation = sim_time*T_0,
						del_t = 5e-3;
double N_ext = N_ext_0, omega_c = omega_0, omega_s = omega_0;
const int sector_count = 8;
const double stresswave_threshold = 0.7;
double epsilon_glitch = 1e-12;


/* List of parameters detailed

	numb_runs is the number of runs (repetitions) of the simulation required
	nv is number of vortices
	npin_desired is the desired number of pinning sites
	R is the Radius of container
	k is roughly the circulation of a vortex
	k_img is roughly the circulation of the image vortices
	omega_0 is the initial rotation rate of the star
	T_0 is the initial time period of the star
	phi is the dissipation due to spindown of crust leading to Magnus force
	del_t is the initial integration timestep. Sort of irrelevant for adaptive algorithms that need to respect minimum error
	a is the lattice spacing of pinning sites. Roughly, a = R * sqrt(pi / npin_desired)
	Xi is the characteristic width of pinning sites
	N_ext_0 The external deceleration acting on the neutron star
	N_ext Used so that a variable deceleration model can be easily included if necessary
	V_1 and V-2 are the pinning strengths for a two pinning model -- Set V_1 = V_2 for single pinning equivalence
	string pin_config is the chosen pinning configuration - Available: half, alternate, annular
	n_ratio ensures there are pinning sites of strength V_1 and V_2 populated in the ratio n_ratio:1 -- n_ratio is relevant for alternate pinning model
	r_annulus sets a radius within (beyond) which pins of strength V_1 (V_2) are populated -- r_ annulus is relevant for the annular pinning model
	trig_region is the location of the trigger - Available: inner, outer, fullsector
	numb_trig is the number of desired trigger events
	duration_trig is the duration of each trigger event -- Set value to zero to switch off trigger
	sector_count defines the number of sectors the star is divided into, much like pizza slices
	epsilon_glitch is the sie of the smallest glitch that the algorithm detects
	trig_type decides the type of trigger to be included - Available: sectorial, stresswave
	stresswave_threshold sets the reduction factor for the pinning strength when a stresswave passes
	write_fullpos decides if the full data of the positions should be written to disk. The minimal data is always written

*/

// Variable initializations

int npin = 0; // Needed to keep count of actual number of  pinning sites
int vortex_out_count = 0; // To count number of vortices outside the container
int nmax = floor(2*R/a)+1; 	//nmax is the number of pinning sites along a diameter
														//+1 is a must for correct counting of pinning sites
int type_counter = 0; // Relevant for setup of alternate pinning configuration
int trial_count = 0; // Keeping track of the integration iteration number since beginning
int count_trig = 0; // Tracking the number of trigger events
int state_trig = 0; // Trcking the state of the trigger -- If the trigger is in progress or not
int unpinned_count = 0; // Number of unpinned vortices
int vortex_in_count = 0; // Number of vortices within boundary
double H = 0; // H is rougly reflective of the energy of the vortex array
double kiss = 0; // K/I_s -- Proportionality constant involved in calculating angular speed of superfluid
double prog = 0; // Initializing progress meter
double time_start_trig = 0; // Noting the start time of a trigger
double t = 0; // Global time variable

// Vector declarations

typedef vector<double> state_type;

state_type f; // Define state variable f -- f is a one dimensional vector of size nv with nv/2 x-positions followed by nv/2 y-positions

vector<double> vortex_vectx(nv, 0.0); // Define global variable - x location of vortices
vector<double> vortex_vecty(nv, 0.0); // Define global variable - y location of vortices
vector<double> imgvortex_vectx(nv, 0.0); // Define global variable - x location of image vortices
vector<double> imgvortex_vecty(nv, 0.0); // Define global variable - y location of image vortices
vector<double> K(nv,k); // Initializing all entries as k. Declaration prevents memory access issues
vector<double> K_img(nv,k_img); // Initializing all entries as k_img. Declaration prevents memory access issues
vector<double> omega_s_vector={omega_0}; // Setting first entry of omega_s_vector -- Required for calculations
vector<double> omega_c_vector={omega_0}; // Setting first entry of omega_c_vector -- Required for calculations
vector<vector<double>> V_0(nmax,vector<double>(nmax,0.0)); // Vector of pinning strengths -- Can be altered after initialization
vector<vector<double>> V_0_original(nmax,vector<double>(nmax,0.0)); // Original vector of pinning strengths -- Should not be altered after initialization
vector<double> times_trig(numb_trig,0.0); // The times at which the triggers are implemented
vector<double> omega_s_forsum(nv,0.0); //To efficiently parallelize the below loop -- MANDATORY for parallelization

// Required for re-initializing at the beginning of a new runs

const vector<double> i_vortex_vectx(nv, 0.0);
const vector<double> i_vortex_vecty(nv, 0.0);
const vector<double> i_imgvortex_vectx(nv, 0.0);
const vector<double> i_imgvortex_vecty(nv, 0.0);
const vector<double> i_K(nv,k);
const vector<double> i_K_img(nv,k_img);
const vector<double> i_omega_s_vector={omega_0};
const vector<double> i_omega_c_vector={omega_0};
const vector<vector<double>> i_V_0(nmax,vector<double>(nmax,0.0));
const vector<vector<double>> i_V_0_original(nmax,vector<double>(nmax,0.0));
const vector<double> i_times_trig(numb_trig,0.0);
const vector<double> i_omega_s_forsum(nv,0.0);

//Below declarations required for saving data -- These must be cleared before beginning a new run

vector<double> Hvalues; // The values of H at different times during the stabilization phase
vector<double> tvalues; // The values of time
vector<vector<double>> vortex_states; // To record 'f', the state of the system, over time
vector<double> omega_c_vec; // omega_c over time
vector<double> omega_s_vec; // omega_s over time
vector<int> numboff_vec; // The number of pinning sites switched off corresponding to the trigger times
vector<int> sector_vec; // The sector ids of the triggers
vector<int> unpinned_vec; // Number of unpinned vortices
vector<int> vortex_in_vec; // Number of vortices within the boundary

// Stepper definitions

runge_kutta_cash_karp54<state_type> rkck54;
runge_kutta4<state_type> rk4;

// Function declarations

void reinitialize();
void create_output_files(int run_id);
void vortexinit1();
void initial_state();
void circulation_init();
void vortex_counter();
void pin_strength_initialize_alternate();
void pin_strength_initialize_half();
void pin_strength_initialize_annular();
void pin_config_set();
void pins_count();
void write_info_start();
void stabilization();
void write_data_post_stabilization();
void trig_init();
void kiss_fix();
void write_info_mid();
void simulation();
void write_info_end();
void write_data_post_simulation();
double randnumb01(); 
char* calc_time();
void write_progress(const double runtime, const string prefix);
void H_calculate_save();
void integ_adaptive();
void eom(const state_type &f, state_type &dfdt , double t);
void updates(const int  omega_s_condition);
void trigger_check(double duration_trig, double t);
void pin_off();
void pin_reduce();
int sector_check(double x, double y, int numb_sector);
void find_glitch(vector<double> omega_c_vec, vector<double> tvalues, double epsilon);


// ----- Algorithms -----

// Main function

int main()
{	
	for (run_id=1; run_id <= numb_runs; run_id++)
	{
	
		cout << "Run " << run_id << " of " << numb_runs <<endl;

		// Initial resets

		reinitialize();

		// Create output files

		create_output_files(run_id);

		// Populate vortices

		vortexinit1();
		initial_state();
		circulation_init(); // Initialize circulation values
		vortex_counter(); // To count vortices outside before beginning stabilization

		// Populate pinning sites

		pin_config_set();
		pins_count(); // To simply count the total number of pinning sites

		// Start stabilization
		
		write_info_start();
		cout << "Stabilization on" << endl;
		stabilization();
		write_data_post_stabilization(); // To write stabilized positions, velocities and Hvalues
		
		// Intermediate resets

		t = 0;
		tvalues.clear();
		prog = 0;

		// Start simulation

		trig_init(); // Initialize times for triggers to occur
		vortex_counter(); // To count vortices outside, inside, and unpinned before beginning simulation
		
		kiss_fix(); // To fix the constant of proportionality for angular momentum calculations
		write_info_mid();
		
		cout << "Simulation on" << endl;
		simulation();
		
		cout << "Writing to file" << endl;
		
		vortex_counter();
		write_info_end();
		write_data_post_simulation(); // To write all vortex motions during simulation, and omega vs t, saved thusfar in RAM.
		
		// Glitch finding
		
		cout << "Finding glitches and saving data" << endl;
		find_glitch(omega_c_vec, tvalues, epsilon_glitch);
		
		cout << "Done" << endl << endl;
	}
return 0;
}


// The below functions are arranged in the order of their exeecution in main
// The last section consists of support functions which are not explicitly called in main but are essential for the program


// Functions called in main
//------------------------------// 

// Reinitialize all variables -- Required when running several iterations of the simulation
void reinitialize()
{

	N_ext = N_ext_0;
	omega_c = omega_0;
	omega_s = omega_0;

	// Variable initializations

	npin = 0; 
	vortex_out_count = 0;
	nmax = floor(2*R/a)+1; 
	type_counter = 0; 
	trial_count = 0; 
	count_trig = 0; 
	state_trig = 0; 
	unpinned_count = 0;
	vortex_in_count = 0;
	H = 0; 
	kiss = 0; 
	prog = 0; 
	time_start_trig = 0;
	t = 0;
	

	// Vectors declaration

	vortex_vectx = i_vortex_vectx;
	vortex_vecty = i_vortex_vecty;
	imgvortex_vectx = i_imgvortex_vectx;
	imgvortex_vecty = i_imgvortex_vecty;
	K = i_K;
	K_img = i_K_img;
	omega_s_vector.clear(); omega_s_vector = i_omega_s_vector;
	omega_c_vector.clear(); omega_c_vector = i_omega_c_vector;
	V_0 = i_V_0;
	V_0_original = i_V_0_original;
	times_trig = i_times_trig;
	omega_s_forsum = i_omega_s_forsum;

	// Below declarations required for saving data

	f.clear();
	Hvalues.clear();
	tvalues.clear();
	vortex_states.clear();
	omega_c_vec.clear();
	omega_s_vec.clear();
	numboff_vec.clear(); 
	sector_vec.clear(); 
	unpinned_vec.clear();
	vortex_in_vec.clear();

}


//------------------------------//

// To create output files in the appropriate directory
void create_output_files(int run_id)
{
	// Setting the directory for the current run
	var = to_string(run_id);
	pathplace = path+"run"+var+"/";
	std::filesystem::create_directory(pathplace);
	
	// Erasing contents of output files and setting header labels
	string filename = "sim_vortex_pos.dat"; 
	string fullpath = pathplace+filename;	
	ofstream file;	
	file.open(fullpath); 
	file << "t/T_0";
	for (int i=1; i<=nv; i++)
	{
		file << '\t'<< "x" <<i;
	}
	for (int i=1; i<=nv; i++)
	{
		file << '\t'<< "y" <<i;
	}
	file << endl;
	file.close();
	
	filename = "sim_vortex_pos_minimal.dat"; 
	fullpath = pathplace+filename;	
	file.open(fullpath); 
	file << "t/T_0";
	for (int i=1; i<=nv; i++)
	{
		file << '\t'<< "x" <<i;
	}
	for (int i=1; i<=nv; i++)
	{
		file << '\t'<< "y" <<i;
	}
	file << endl;
	file.close();
	
	filename = "init_vortex_pos.dat"; 
	fullpath = pathplace+filename;
	file.open(fullpath); // Erasing content of vortexinit file
	file << "x" << "\t" << "y" << endl;
	file.close();
	
	filename = "init_img_pos.dat"; 
	fullpath = pathplace+filename;	
	file.open(fullpath); // Erasing content of imgvortexinit file
	file << "x" << "\t" << "y" << endl;
	file.close();
	
	filename = "stabilized_Hvalues.dat"; 
	fullpath = pathplace+filename;
	file.open(fullpath); // Erasing content of H_values file
	file << "t/T_0" << "\t" << "H" << endl;
	file.close();
	
	filename = "stabilized_vortex_pos.dat"; 
	fullpath = pathplace+filename;
	file.open(fullpath); // Erasing content of stabilized_pos file
	file.close();
	
	filename = "sim_omega_c.dat"; 
	fullpath = pathplace+filename;
	file.open(fullpath); // Erasing content of omega_c file
	file << "t/T_0" << "\t" << "omega_c/omega_0" << endl;
	file.close();
	
	filename = "sim_omega_s.dat"; 
	fullpath = pathplace+filename;
	file.open(fullpath); // Erasing content of omega_s file
	file << "t/T_0" << "\t" << "omega_s/omega_0" << endl;
	file.close();
	
	filename = "info_simulation.dat"; 
	fullpath = pathplace+filename;
	file.open(fullpath); // Erasing content of info_sim file
	file.close();
	
	filename = "info_progress.dat"; 
	fullpath = pathplace+filename;
	file.open(fullpath); // Erasing content of info_progress file
	file.close();

	filename = "sim_triggers.dat"; 
	fullpath = pathplace+filename;
	file.open(fullpath); // Erasing content of sim_triggers file
	file << "t/T_0" << "\t"<< "sector_id" << "\t" << "numboff" << endl;
	file.close();
	
	filename = "sim_unpinned.dat"; 
	fullpath = pathplace+filename;
	file.open(fullpath); // Erasing content of sim_unpinned file
	file << "t/T_0" << "\t" << "unpinned_count" << "\t" << "total_count" << "\t" << "unpinned_frac" << endl;
	file.close();
	
	filename = "info_glitch.dat"; 
	fullpath = pathplace+filename;
	file.open(fullpath); // Erasing content of info_glitch file
	file.close();
	
	filename = "data_glitch.dat"; 
	fullpath = pathplace+filename;
	file.open(fullpath); // Erasing content of data_glitch file
	file.close();
	
}


//------------------------------//

// Function to initialize vortices (actual and image) randomly
void vortexinit1()
{
	// Generate a circle of radius R units filled uniformly with points and write data to a file
	
	# pragma omp parallel for
	for (int i = 0; i < nv; i++)
	{ 	
	    	
		double r = R*sqrt(randnumb01());
		double theta = 2*pi*randnumb01();
	            
	    double x = r * cos(theta);
	    double y= r * sin(theta);
	     
		vortex_vectx[i] = x;
		vortex_vecty[i] = y;
		
		double r_img = pow(R,2) / r;
		double theta_img = theta;
		
		double x_img = r_img * cos(theta_img);
    	double y_img= r_img * sin(theta_img);
	    
   		imgvortex_vectx[i] = x_img;
		imgvortex_vecty[i] = y_img;	
	}
	    
	for (int i = 0; i<nv; i++) //Writing cannot be parallel
	{       
		string filename = "init_vortex_pos.dat";
		string fullpath = pathplace+filename;		        
		ofstream outfile;        
	    outfile.open(fullpath,ios::out | ios::app);
	    outfile.precision(15);
	    outfile << vortex_vectx[i] << "\t" << vortex_vecty[i] << endl;
	    outfile.close();
	    
	    filename = "init_img_pos.dat"; 
		fullpath = pathplace+filename;		        
	    outfile.open(fullpath,ios::out | ios::app);
	    outfile.precision(15);
	    outfile << imgvortex_vectx[i] << "\t" << imgvortex_vecty[i] << endl;
	    outfile.close(); 
	}

}

// Function to read the initial vortex positions and write them to the vector f -- Remnant from code where positions could be loaded from a file
void initial_state()
{
	f.clear();
    f.insert(f.begin(), vortex_vectx.begin(), vortex_vectx.end()); // To concatenate vectors
    f.insert(f.end(), vortex_vecty.begin(), vortex_vecty.end());
}

// To initialize circulation values
void circulation_init()
{
	#pragma omp parallel for
	for (int i = 1; i <= nv; i++)
	{
		double r = sqrt(pow(f[i-1],2) + pow(f[i-1+nv],2));
		
		if (K[i-1] == 0)
		{
			K[i-1] = 0;
			K_img[i-1] = 0;
		}
		
		else
		{
			if (r>R)
			{
				K[i-1] = 0;
				K_img[i-1] = 0;
			}
			
			else
			{
				K[i-1] = k;
				K_img[i-1] = k;
			}
		}
	}
}

// Function to count vortices outside boundary and the number unpinned
void vortex_counter()
{
	vortex_out_count = 0;
	vortex_in_count = 0;
	unpinned_count = 0;
		
	for (int i=1; i<=nv; i++)
	{	
		double r = sqrt(pow(f[i-1],2) + pow(f[i-1+nv], 2));
		if (r >= R)
		{
			vortex_out_count += 1;
		}
		else if (r < R)
		{
			vortex_in_count += 1;
			
			double x_pin_nearest = -(R) + (round((xi + R)/a) * a);
			double y_pin_nearest = -(R) + (round((yi + R)/a) * a);
			double r2_pin_nearest = pow((x_pin_nearest - xi),2) + pow((y_pin_nearest - yi),2);
			double r_pin_nearest = sqrt(r2_pin_nearest);
			
			if(r_pin_nearest > Xi)
			{
				unpinned_count += 1;	
			}
		}
	}
}


//------------------------------//

// The following functions are to initialize pinning strengths - Two pinning model
// Alternate pin model
void pin_strength_initialize_alternate()
{
for (double x = -R; x <= R; x = x+a)
  {  	
    for (double y = -R; y <= R; y = y+a)
    {  
      double dist_pin = sqrt(pow(x,2)+pow(y,2)); 
      if (dist_pin<R)
      {
        int nx = round((R+x)/a);
        int ny = round((R+y)/a);
        
        if (type_counter != n_ratio)
        {
          V_0[nx][ny] = V_1;
          type_counter = type_counter + 1;
        }
        else if (type_counter == n_ratio)
        {
          V_0[nx][ny] = V_2;
          type_counter = 0;
        }
      } 
    }
  }
}

// Volume-separated (Half) pin model
void pin_strength_initialize_half()
{
for (double x = -R; x <= R; x = x+a)
  {  	
    for (double y = -R; y <= R; y = y+a)
    {  
      double dist_pin = sqrt(pow(x,2)+pow(y,2)); 
      if (dist_pin<R)
      {
        int nx = round((R+x)/a);
        int ny = round((R+y)/a);
        
        if (x < 0)
        {
          V_0[nx][ny] = V_1;
        }
        else if (x >= 0)
        {
          V_0[nx][ny] = V_2;
        }
      } 
    }
  }
}

// Annnular pin model
void pin_strength_initialize_annular()
{
for (double x = -R; x <= R; x = x+a)
  {  	
    for (double y = -R; y <= R; y = y+a)
    {  
      double dist_pin = sqrt(pow(x,2)+pow(y,2)); 
      if (dist_pin<R)
      {
        int nx = round((R+x)/a);
        int ny = round((R+y)/a);
        
        if (dist_pin <= r_annulus)
        {
          V_0[nx][ny] = V_1;
        }
        else if (dist_pin > r_annulus)
        {
          V_0[nx][ny] = V_2;
        }
      } 
    }
  }
}

// Pin_config implement
void pin_config_set()
{
	if(pin_config == "half")
	{
		pin_strength_initialize_half(); // volume-serapated strength variation
	}
	
	else if(pin_config == "alternate")
	{
		pin_strength_initialize_alternate(); // alternate strength variation
	}

	else if(pin_config == "annular")
	{
		pin_strength_initialize_annular(); // annular strength variation
	}
	
	V_0_original = V_0;
}

// To simply count the number of pinning sites in within the container
void pins_count()
{	
	for (double x = -R; x <= R; x = x+a)
	{ 	    	
    	for (double y = -R; y <= R; y = y+a)
    	{
	                
	        double dist_pin = sqrt(pow(x,2)+pow(y,2)); 
	        if (dist_pin<R)
			{	        
		        npin = npin +1;
			} 
	    }
	}
}


//------------------------------//

// To write info file at start
void write_info_start()
{
	string filename = "info_simulation.dat"; 
	string fullpath = pathplace+filename;
	fstream outfile;
	outfile.open(fullpath,ios::out | ios::app);	// Open the file in append mode (ios::app) or printing mode (ios::out) both
	outfile.precision(15);

	outfile << "nv = " << nv  << endl;
	outfile << "npin = " << npin << endl;
	outfile << "pin_config = " << pin_config << endl;
	outfile << "r_annulus = " << r_annulus/R << " R" << endl;
	outfile << "R = " << R << endl;
	outfile << "omega_0 = " << omega_0 << endl;
	outfile << "T_0 = " << T_0 << endl;
	outfile << "phi = " << phi << endl;
	outfile << "V_1 = " << V_1 << endl;
	outfile << "V_2 = " << V_2 << endl;
	outfile << "n_ratio = " << n_ratio << endl;
	outfile << "a = " << a/R << " R" << endl;
	outfile << "Xi = " << Xi/R << " R" << endl;
	outfile << "N_ext = " << N_ext*(T_0/omega_0) << " omega_0/T_0"  << endl;
	outfile << "initial del_t = " << del_t/T_0 << " T_0" << endl << endl;
	outfile << "trig_region = " << trig_region << endl;
	outfile << "numb_trig = " << numb_trig << endl;
	outfile << "duration_trig = " << duration_trig/T_0 << " T_0" << endl;
	outfile << "sector_count = " << sector_count << endl;	
	outfile << "trig_type = " << trig_type << endl;

	outfile << "Stabilization runtime = " << runtime_stabilization/T_0 << " T_0" << endl << endl;
	outfile << "Simulation runtime = " << runtime_simulation/T_0 << " T_0" << endl << endl;
	
	outfile << "Start time = " << calc_time() << endl;
	outfile << "nv_out = " << vortex_out_count << endl;
	
	outfile << endl;
	outfile.close();
}

// Stabilization
void stabilization()
{		
  	while(t <= runtime_stabilization)
  	{
  		// Saving various quantities for writing later
		write_progress(runtime_stabilization, "Stabilization");
		tvalues.insert(tvalues.end(), t/T_0); // Time gets saved in units of T_0
		H_calculate_save();
			  
	  	// Integrate
		integ_adaptive();
		t += del_t;
		
		// Calculate new location of image vortices, and circulations
    	updates(0);  
	}
}

//  To write the stablized positions, velocities and Hvalues
void write_data_post_stabilization()
{
	fstream outfile;
	
	string filename = "stabilized_vortex_pos.dat"; 
   	string fullpath = pathplace+filename;
    outfile.open(fullpath, ios::out|ios::app);
    outfile.precision(15);
    outfile << "x" << "\t" << "y" << endl;
	for (int i=1; i <= nv; i++)
    {
    	outfile << f[i-1] << "\t" << f[i-1+nv]<< endl;
	}
    outfile << endl;
    outfile.close();
    
    // Writing H values and corresponding times	
	filename = "stabilized_Hvalues.dat"; 
   	fullpath = pathplace+filename;
	outfile.open(fullpath, ios::out|ios::app);
	outfile.precision(15);
	
	for(int i=0; i<Hvalues.size(); i++)
	{
		outfile << tvalues[i]<< "\t" << Hvalues[i] << endl;
	}
		
	outfile.close();
    
}


//------------------------------//

// Initialization of trigger times
void trig_init()
{
  for (int i=0; i<numb_trig; i++)
  {
   times_trig[i] = randnumb01() * runtime_simulation;
  }
  sort(times_trig.begin(), times_trig.end()); // Needs 'algorithm' header
  times_trig.push_back(runtime_simulation+5); // To ensure that the trigger is not activated for all times between the last random trigger time and the simulation runtime  
}

// Function to fix constant k/I_s (defined as kiss)
void kiss_fix()
{
	double sum=0;
	for (int i=1; i<=nv ;i++)
	{	
		double r = sqrt(pow(f[i-1],2) + pow(f[i-1+nv],2));
		if (r<R)
		{
			sum = sum + (pow(R,2)-pow(r,2));
		}
	}
	
	kiss = omega_0 / sum;	
}

// Function to write info file midway between stabilization and simulation
void write_info_mid()
{	
	string filename = "info_simulation.dat"; 
	string fullpath = pathplace+filename;
	fstream outfile;
	outfile.open(fullpath,ios::out | ios::app);	// Open the file in append mode (ios::app) or printing mode (ios::out) both
	outfile.precision(15);
	
	outfile << "Stabilization end time = " << calc_time() << endl;
	outfile << endl;
	outfile << "Simulation begins." << endl;
	outfile << "nv_out = " << vortex_out_count << endl;
	outfile << "unpinned_count = " << unpinned_count << endl;
	outfile << "omega_c = " << omega_c/omega_0 << " omega_0" << endl<< endl;
	outfile << "k/I_s = " << kiss << endl << endl;
	
	outfile << endl;
	outfile.close();
}

// Simulation

void simulation()
{	
	// f and all other global variables that are not a part of the intermediate resets retain their value from the end of stabilization
  	while(t <= runtime_simulation)
  	{
  		// Saving various quantities for writing later
  		write_progress(runtime_simulation, "Simulation");
		tvalues.insert(tvalues.end(), t/T_0); // Time gets saved in units of T_0
		vortex_states.insert(vortex_states.end(),f); // Vortex states get saved over time
		omega_c_vec.insert(omega_c_vec.end(),omega_c/omega_0); // Saving omega_c
		omega_s_vec.insert(omega_s_vec.end(),omega_s/omega_0); // Saving omega_s
		unpinned_vec.insert(unpinned_vec.end(),unpinned_count); // Saving unpinned_count
		vortex_in_vec.insert(vortex_in_vec.end(),vortex_in_count); // Saving vortex_in_count
		
		// Activate trigger if applicable
		if(trig_duration != 0)
		{
			trigger_check(duration_trig, t);	
		}

		// Spindown of the container
		omega_c += N_ext * del_t;

		// Integrate
		integ_adaptive();
		t += del_t;
		
		// Calculate new locations of image vortices, circulations, omega_s, and update vortex counts
   		updates(1);
    	
   		// Feedback onto container
		int n = omega_s_vector.size();
		double del_omega_s = omega_s_vector[n-1] - omega_s_vector[n-2];
		omega_c += -del_omega_s;
	}
}

// To write info file at end
void write_info_end()
{	
	string filename = "info_simulation.dat"; 
	string fullpath = pathplace+filename;
	fstream outfile;
	outfile.open(fullpath,ios::out | ios::app);	// Open the file in append mode (ios::app) or printing mode (ios::out) both
	outfile.precision(15);
	
	outfile << "End time = " << calc_time() << endl;
	outfile << "nv_out = " << vortex_out_count << endl;
	outfile << "omega_c = " << omega_c/omega_0 << " omega_0" << endl<< endl;
	
	outfile << endl;
	outfile.close();
}

// To write data post simulation
// All vortex motions, and omega vs t, recorded at an interval dictated by write_step)
void write_data_post_simulation()
{
	fstream outfile;

	// Writing positions
	
	string filename = "sim_vortex_pos.dat"; 
	string fullpath = pathplace+filename;
	
	if (write_fullpos == "True")
	{
		outfile.open(fullpath,ios::out | ios::app);
		outfile.precision(15);
		
		for (int i=0; i<tvalues.size(); i++)
		{
			outfile << tvalues[i];
			for (int j=1; j<=( 2*nv ); j++)
		  	{
		    	outfile << '\t'<< vortex_states[i][j-1];
			}
			outfile << endl;
		}
		
		outfile.close();
	}

	// Writing minimal positions - Reducing file size by a factor of ten
	
	filename = "sim_vortex_pos_minimal.dat"; 
	fullpath = pathplace+filename;
	outfile.open(fullpath,ios::out | ios::app);
	outfile.precision(15);
	
	for (int i=0; i<tvalues.size(); i=i+10)
	{
		outfile << tvalues[i];
		for (int j=1; j<=( 2*nv ); j++)
	  	{
	    	outfile << '\t'<< vortex_states[i][j-1];
		}
		outfile << endl;
	}
	
	outfile.close();
    
  	// Writing omega_c/omega_0 and corresponding times
	
	filename = "sim_omega_c.dat"; 
 	fullpath = pathplace+filename;
	outfile.open(fullpath, ios::out|ios::app);
	outfile.precision(15);
	
	for(int i=0; i<tvalues.size(); i++)
	{
		outfile << tvalues[i]<< "\t" << omega_c_vec[i] << endl;
	}
		
	outfile.close();
	
	// Writing omega_s/omega_0 and corresponding times
	
	filename = "sim_omega_s.dat"; 
 	fullpath = pathplace+filename;
	outfile.open(fullpath, ios::out|ios::app);
	outfile.precision(15);
	
	for(int i=0; i<tvalues.size(); i++)
	{
		outfile << tvalues[i]<< "\t" << omega_s_vec[i] << endl;
	}
		
	outfile.close();
	
	// Writing tigger details and corresponding times
  
	filename = "sim_triggers.dat"; 
	fullpath = pathplace+filename;
	outfile.open(fullpath, ios::out|ios::app);
	outfile.precision(15);
	

	if (trig_type == "sectorial")
	{
		for(int i=0; i<times_trig.size()-1; i++)
		{
			outfile << times_trig[i]/T_0 << "\t"<< sector_vec[i] << "\t" << numboff_vec[i] << endl;
		}
	}

	else if (trig_type == "stresswave")
	{
		for(int i=0; i<times_trig.size()-1; i++)
		{
			outfile << times_trig[i]/T_0 << endl;
		}
	}

	
	// times_trig.size()-1 has been used since the last entry of the vector was added by hand
	
	outfile.close();
	
	// Writing total number of vortices within boundary and unpinned_count
	
	filename = "sim_unpinned.dat"; 
	fullpath = pathplace+filename;
	outfile.open(fullpath, ios::out|ios::app);
	outfile.precision(15);
	
	for(int i=0; i<tvalues.size(); i++)
	{
		outfile << tvalues[i] << "\t" << unpinned_vec[i] << "\t" << vortex_in_vec[i] << "\t" << double(unpinned_vec[i])/double(vortex_in_vec[i]) << endl;
	}
	
	outfile.close();	
}


// Support functions
//------------------------------// 
// Function to generate a random number between 0 and 1
double randnumb01() 
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dist(0, 1);	
	
	return dist(gen);
 } 

// Current computer date and time calculation
char* calc_time()
{
	// Current date and time retrieved from the computer
	time_t now = time(0);
	
	// Convert now to string form
	char* dt = ctime(&now);
	
	return dt;
}

// Function to track progress
void write_progress(const double runtime, const string prefix)
{
	if ( t >= ((prog+1)/100)*runtime)
	{
		string filename = "info_progress.dat"; 
		string fullpath = pathplace+filename;
		fstream outfile;
		outfile.open(fullpath,ios::out | ios::app);
		outfile.precision(4);
		outfile << prefix << " progress " << prog+1 << " percent" << endl;
		outfile.close();

		prog = prog +1;
	}
}

// Function to calculate H and save the values in a vector
void H_calculate_save()
{

	H=0;
	
	#define rij  sqrt((pow((f[i-1] - f[j-1]),2) + pow((f[i-1+nv] - f[j-1+nv]),2)))
	
	vector<double> H_forsum(nv,0.0);
	
	#pragma omp parallel for
	for (int i=1; i<=nv ;i++)
	{
		double h_i =0;
		
		for (int j=1; j<=nv ;j++)
		{
			if(j != i)
			{
				h_i += log (rij);
			}
		}
		
		H_forsum[i-1] = h_i;
	}
	
	H = accumulate(H_forsum.begin(),H_forsum.end(),0.0);
	
	Hvalues.insert(Hvalues.end(),H);
}

// Function to perform adaptive integration where the maximum step size is capped by del_t
void integ_adaptive()
{
	double err_tol = 1e-5; // Error tolerance
	vector<double> ferr(nv,0.0);	
	state_type f_temp = f;
	
	rkck54.do_step( eom , f , t , del_t, ferr );
	
	// To make all entries of ferr positive
	for(int i=0; i<ferr.size(); i++)
	{
		if(ferr[i] < 0)
		{
			ferr[i] = (-1) * ferr[i];
		}
	}
	
	// Maximum error due to previous step of integration
	double maxerr = *max_element(ferr.begin(),ferr.end()); 
	
	// Correction scheme if error is beyond tolerance
	if(maxerr >= err_tol)
	{		
		f = f_temp; // Rewrite old state
		
		double corr = pow((err_tol/maxerr), 0.2); // Correction factor for time step
				
		double low10 = pow(10,floor(log10(corr))); // To bring it to the nearest lower 10^x
		
		// Conditions for obtaining optimal correction factor
		if(corr/low10 < 2)
		{
			corr = 1 * low10;
		}
		
		else if( corr/low10 >= 2 && corr/low10 < 5)
		{
			corr = 2 * low10;
		}
		
		else if( corr/low10 >=5)
		{
			corr = 5 * low10;
		}
		
		// Setting the required size of a timestep and the number of such steps required
		double del_t_temp = corr * del_t;
		int n_steps = 1/corr;
		
		for(int i = 1; i<=n_steps; i++)
		{
		rkck54.do_step( eom , f , t , del_t_temp, ferr );
		}
	}
	
	// At the end of the routine, no matter what, the total time elapsed due to integration is del_t
	// The increment in time is performed in the calling block
}

// Equations of motion
void eom(const state_type &f, state_type &dfdt , double t)
{	
	#pragma omp parallel for
	for (int i=1; i<=nv; i++)
	{
		if (K[i-1] == 0) // Don't do the math if the vortex doesn't exist
		{
			dxidt = 0;
			dyidt = 0;
		}
		
		else
		{

		dxidt =   (omega_c) * yi; // Differential equations for x - Part 1 (Rotation)
		dyidt = - (omega_c) * xi; // Differential equations for y - Part 1 (Rotation)	
		
		double Ki = K[i-1]; // To avoid repeated memory calls in the following loop. Especially expensive if cache is full and cpu reads directly from memory
		
		for (int j=1; j<=nv; j++)
		{
			if (K[j-1] == 0) // Don't do the math if the other vortex doesn't exist
			{
				// Add nothing;
			}
			
			else
			{
				if (i!=j)
				{
				dxidt += -K[j-1] * (Ki * yij / rij2); // Differential equations for x - Part 2 (Other vortices)
				dyidt +=  K[j-1] * (Ki * xij / rij2); // Differential equations for y - Part 2 (Other vortices)
				}
			
				dxidt += K_img[j-1] * (Ki * yij_img / rij2_img); // Differential equations for x - Part 3 (Image vortices)
				dyidt += -K_img[j-1] * (Ki  * xij_img / rij2_img); // Differential equations for y - Part 3 (Image vortices)	
			}
		}

		// Below we account only for the effect of the pin closest to the vortex, thus simplifying calculations	
		
		int nx = round((xi + R)/a) ;
	  	int ny = round((yi + R)/a) ;
	  	if (nx<=0 || ny<=0 || nx>=nmax || ny>=nmax) 
	  	{
	  		nx = 0;
	  		ny = 0;
		}
		
		// If xi or yi is slightly lesser than -R or greater than +R, nx and ny could be negative or out of bounds
		// If nx or ny is out of bounds, the above would lead to a pinning strength of zero (since V_0[0][0] = 0)
		// The above mechanism saves the code from segmentation fault
		
		double	x_pin_nearest = -(R) + (round((xi + R)/a) * a);
		double	y_pin_nearest = -(R) + (round((yi + R)/a) * a);
		double	r2_pin_nearest = pow((x_pin_nearest - xi),2) + pow((y_pin_nearest - yi),2);
		double	r_pin_nearest = sqrt(r2_pin_nearest);

		// Regular pinning term
		dxidt += ( (V_0[nx][ny])  * exp(- r2_pin_nearest/(2*pow(Xi,2))) * (yi - y_pin_nearest)); // Differential equations for x - Part 4 (Pinning)
		dyidt += - ( (V_0[nx][ny])  * exp(- r2_pin_nearest/(2*pow(Xi,2))) * (xi - x_pin_nearest)); // Differential equations for y - Part 4 (Pinning)

		// Ensuring out of bound vortices do not get updated
		dxidt *= K[i-1];
		dyidt *= K[i-1];
		
		// Including dissipation
		double	vel_x = dxidt;
		double	vel_y = dyidt;
		
		dxidt = cos(phi) * vel_x + sin(phi) * vel_y; 
		dyidt = - sin(phi) * vel_x + cos(phi) * vel_y;
		
		}
	}
}

// Function to update image vortices, circulation, omega_s, vortex counts in one shot
void updates(const int  omega_s_condition)
{	

	if(omega_s_condition == 0) // Without omega_s update - For stabilization
	{	
		#pragma omp parallel for
		for (int i=1; i<=nv ; i++)
		{
			if (K[i-1] == 0)
			{
				// Don't do the math if the vortex doesn't exist
			}
			
			else
			{
				// Updating image vortices
				double r = sqrt(pow(f[i-1],2) + pow(f[i-1+nv],2));
				double theta = atan2(f[i-1+nv] , f[i-1]);
				
				double r_img = pow(R,2) / r;
				double theta_img = theta;
		
				double x_img = r_img * cos(theta_img);
				double y_img= r_img * sin(theta_img);
		
				imgvortex_vectx[i-1]=x_img;
				imgvortex_vecty[i-1]=y_img;
			
				// Updating circulations
				if (K[i-1] == 0)
				{
					K[i-1] = 0;
					K_img[i-1] = 0;
				}
				
				else
				{
					if (r>R)
					{
						K[i-1]=0;
						K_img[i-1]=0;
					}
					else
					{
						K[i-1]=k;
						K_img[i-1]=k;
					}
				}
			}
		}			
	}

	if(omega_s_condition == 1) // With omega_s update - For simulation
	{	
		#pragma omp parallel for
		for (int i=1; i<=nv ; i++)
		{
			if (K[i-1] == 0)
			{
				// Don't do the math if the vortex doesn't exist
			}
			
			else
			{
				// Updating image vortices
				double r = sqrt(pow(f[i-1],2) + pow(f[i-1+nv],2));
				double theta = atan2(f[i-1+nv] , f[i-1]);
				
				double r_img = pow(R,2) / r;
				double theta_img = theta;
		
				double x_img = r_img * cos(theta_img);
				double y_img= r_img * sin(theta_img);
		
				imgvortex_vectx[i-1]=x_img;
				imgvortex_vecty[i-1]=y_img;
			
				// Updating circulations
				if (K[i-1] == 0)
				{
					K[i-1] = 0;
					K_img[i-1] = 0;
				}
				
				else
				{
					if (r>R)
					{
						K[i-1]=0;
						K_img[i-1]=0;
					}
					else
					{
						K[i-1]=k;
						K_img[i-1]=k;
					}
				}
			
				// Superfluid rotation due to a given vortex
				omega_s_forsum[i-1] =  K[i-1] * kiss * (pow(R,2)-pow(r,2)); // Usage of vector instead of in-place increment due to parallelization
			}
		}
		
		// Updating omega_s
		omega_s = accumulate(omega_s_forsum.begin(), omega_s_forsum.end(), 0.0);		
		omega_s_vector.push_back(omega_s);
		
		// Updating vortex_in_count and unpinned_count
		vortex_counter();
	}

}


// To check for trigger at every timestep
void trigger_check(double duration_trig, double t)
{
	if (trig_type == "stresswave")
	{
		if (t > times_trig[count_trig])
		{
		time_start_trig = t;
		state_trig == 1;
		count_trig= count_trig + 1;
		pin_reduce();
		}

		if (state_trig == 1)
		{
		if (t > time_start_trig + duration_trig)
		{
		  state_trig = 0;
		  V_0 = V_0_original;
		}
		}
	}

	else if (trig_type == "sectorial")
	{	
		if (t > times_trig[count_trig])
		{
		time_start_trig = t;
		state_trig == 1;
		count_trig= count_trig + 1;
		pin_off();
		}

		if (state_trig == 1)
		{
		if (t > time_start_trig + duration_trig)
		{
		  state_trig = 0;
		  V_0 = V_0_original;
		}
		}
	}
}

// To turn off pinning for randomly chosen pinning sites in a randomly chosen sector
void pin_off()
{

int numboff = 0; // The number of pins switched off for a given trigger
int sector_id;
sector_id = randnumb01() * sector_count; // To randomly choose a sector number between 0 and 7 - Each sector has an opening angle of 45 degrees
sector_vec.push_back(sector_id);

numboff = 0; // Reset numboff

for (double x = -R; x <= R; x = x+a)
  {  	
    for (double y = -R; y <= R; y = y+a)
    {  
    	double dist_pin = sqrt(pow(x,2)+pow(y,2)); 
      
        if (trig_region=="outer")
        {
			if (dist_pin<R)
			{
				int nx = round((R+x)/a);
				int ny = round((R+y)/a);
				
				if (dist_pin <= r_annulus)
				{
					V_0[nx][ny] = V_1;
				}
				
				else if (dist_pin > r_annulus)
				{
					V_0[nx][ny] = V_2;
					if (sector_check(x, y, sector_id) == 1)
					{
						if (randnumb01() <= 0.5) // To unpin with a probability of half
						{
							V_0[nx][ny] = 0;
							numboff = numboff + 1;
						}
					}
			
				}
			
			}
		}
			
		else if (trig_region=="inner")
		{
			if (dist_pin<R)
			{
				int nx = round((R+x)/a);
				int ny = round((R+y)/a);
				
				if (dist_pin <= r_annulus)
				{
					V_0[nx][ny] = V_1;
					if (sector_check(x, y, sector_id) == 1)
					{
						if (randnumb01() <= 0.5) // To unpin with a probability of half
						{
							V_0[nx][ny] = 0;
							numboff = numboff + 1;
						}
					}
				}
				
				else if (dist_pin > r_annulus)
				{
					V_0[nx][ny] = V_2;			
				}
			
			}
		}

		else if (trig_region=="fullsector")
		{
			if (dist_pin<R)
			{
				int nx = round((R+x)/a);
				int ny = round((R+y)/a);
				
				if (dist_pin <= r_annulus)
				{
					V_0[nx][ny] = V_1;
					if (sector_check(x, y, sector_id) == 1)
					{
						if (randnumb01() <= 0.5) // To unpin with a probability of half
						{
							V_0[nx][ny] = 0;
							numboff = numboff + 1;
						}
					}
				}
				
				else if (dist_pin > r_annulus)
				{
					V_0[nx][ny] = V_2;	
					if (sector_check(x, y, sector_id) == 1)
					{
						if (randnumb01() <= 0.5) // To unpin with a probability of half
						{
							V_0[nx][ny] = 0;
							numboff = numboff + 1;
						}
					}		
				}
			
			}
		}
    }
  }

numboff_vec.push_back(numboff);
}

// To reduce pinning strength across the star
void pin_reduce()
{
for (double x = -R; x <= R; x = x+a)
  {  	
    for (double y = -R; y <= R; y = y+a)
    {  
    	double dist_pin = sqrt(pow(x,2)+pow(y,2)); 

		if (dist_pin<R)
		{
			int nx = round((R+x)/a);
			int ny = round((R+y)/a);

			V_0[nx][ny] = stresswave_threshold * V_1;
		
		}
    }
  }
}

// To check if pinning site falls within triggered sector
int sector_check(double x, double y, int numb_sector)
{
  double pin_angle;

  if (x == 0)
  {
    x = 0.01; // To avoid any funny errors in atan2 function
  }
  if (y == 0)
  {
    y = 0.01; // To avoid any funny errors in atan2 function
  }
  if (y>0)
  {
    pin_angle = atan2(y,x) * 180 / pi;
  }
  if (y<0)
  {
    pin_angle = 360 + (atan2(y, x) * 180 / pi);
  }

  double angle_start =  numb_sector * (360/sector_count);
  double angle_end = angle_start + 360/sector_count;
  int check_value;

  if (pin_angle>angle_start && pin_angle<angle_end)
  {
    check_value = 1;
  }
  else
  {
    check_value = 0;
  }

  return check_value;
}

// Recognizing a glitch and storing relevant data
void find_glitch(vector<double> omega, vector<double> t, double epsilon)
{

bool glitch_state = false, glitch_init = false;
double t_i, omega_i, t_f, omega_f;

vector<double> t_glitch, t_rise, del_omega, omegadot;
 
 // Glitch finding and storing
for (int i = 1; i < t.size(); i++)
{
	double omegadot_val = (omega[i]-omega[i-1])/(t[i]-t[i-1]);
	omegadot.push_back(omegadot_val);
	
	if (glitch_state == false)
	{
		if (omegadot_val < 0)
		{
			glitch_state = false;
		}
		else if (omegadot_val > 0)
		{
			glitch_state = true;
			glitch_init = true;
		}
	}
	
	if (glitch_state == true)
	{
		if (glitch_init == true)
		{
		  t_i = t[i-1];
		  omega_i = omega[i-1];
		  glitch_init = false;
		} 
		else if (glitch_init == false)
		{
			//Do nothing
		}
		
		if (omegadot_val < 0)
		{
		    t_f = t[i-1];
			omega_f = omega[i-1];
		    double del_omega_temp = omega_f - omega_i;
		    if(del_omega_temp > epsilon)
			{
				t_glitch.push_back(t_i);
				t_rise.push_back(t_f - t_i);
				del_omega.push_back(omega_f - omega_i);	
			}	
			glitch_state = false;
		}
	}
}

// Calculate waiting times
vector<double> t_wait;
for(int i = 1; i < t_glitch.size(); i++)
{
	t_wait.push_back(t_glitch[i]-t_glitch[i-1]);
}

// Delete first glitch entries from t_glitch,t_rise,del_omega -- to make these vectors the same size as t_wait
t_glitch.erase(t_glitch.begin());
t_rise.erase(t_rise.begin());
del_omega.erase(del_omega.begin());

// Finding the mean glitch size
double sum = 0;
double mean = 0;

for (int i=0; i<del_omega.size(); i++)
{
	sum = sum + del_omega[i];
}

mean = sum / del_omega.size();

// Finding max and min glitch size
double max = *max_element(del_omega.begin(), del_omega.end());
double min = *min_element(del_omega.begin(), del_omega.end());

// Finding median glitch size
double median = 0.0;
vector<double> del_omega_copy = del_omega;
sort(del_omega_copy.begin(), del_omega_copy.end());

int len = del_omega_copy.size();

if(len % 2 == 0)
{
	median = (del_omega_copy[(len/2)- 1] + del_omega_copy[(len/2)]) / 2;
}
else
{
	median = del_omega_copy[((len-1)/2)];
}

// ----- Writing to file -----

fstream outfile;

// Glitch info
string filename = "info_glitch.dat"; 
string fullpath = pathplace+filename;
outfile.open(fullpath,ios::out | ios::app);
outfile.precision(15);

outfile << "Smallest allowed glitch size = " << epsilon << " omega_0" << endl; 
outfile << t_glitch.size() << " glitches found" << endl;
outfile << "The first glitch has been neglected since a waiting time cannot be associated with it." << endl;
outfile << "Biggest glitch size = " << max << " omega_0" << endl;
outfile << "Smallest glitch size = " << min << " omega_0" << endl;
outfile << "Mean glitch size = " << mean << " omega_0" << endl;  
outfile << "Median glitch size = " << median << " omega_0" << endl; 
 
outfile.close();

// Glitch data
filename = "data_glitch.dat"; 
fullpath = pathplace+filename;
outfile.open(fullpath,ios::out | ios::app);
outfile.precision(15);

outfile << "t_glitch" << "\t" << "t_rise" << "\t" << "glitch_size" << "\t" << "t_wait" << endl;

for(int i = 0; i < t_glitch.size(); i++)
{
	outfile << t_glitch[i] << "\t" << t_rise[i] << "\t" << del_omega[i] << "\t" << t_wait[i] << endl;
}

outfile.close();

}

//------------------------------//

// Code for debugging

/*	// To get a sense of the numbers - Place in EOM outside the for loop
	
	int i = 0;
	double temp_otherx=0, temp_imgx=0, temp_othery=0, temp_imgy=0, temp_pinx=0, temp_piny=0;
	
	cout << "x = " << xi << endl;
	cout << "y = " << yi << endl;
	cout << "omegax = " << (omega_c) * yi << endl; // Differential equations for x - Part 1 (Rotation)
	cout << "omegay = " << - (omega_c) * xi << endl; // Differential equations for y - Part 1 (Rotation)
	
	double Ki = K[i-1]; // To avoid repeated memory calls in the following loop. Especially expensive if cache is full and cpu reads directly from memory
	
	for (int j=1; j<=nv; j++)
	{
		if (K[j-1] == 0) // Don't do the math if the other vortex doesn't exist
		{
			// Add nothing;
		}
		
		else
		{
			if (i!=j)
			{
			temp_otherx += -K[j-1] * (Ki * yij / rij2); // Differential equations for x - Part 2 (Other vortices)
			temp_othery +=  K[j-1] * (Ki * xij / rij2); // Differential equations for y - Part 2 (Other vortices)
			}
		
			temp_imgx += K_img[j-1] * (Ki * yij_img / rij2_img); // Differential equations for x - Part 3 (Image vortices)
			temp_imgy += -K_img[j-1] * (Ki  * xij_img / rij2_img); // Differential equations for y - Part 3 (Image vortices)	
		}
	}
	
	cout << "otherx = " << temp_otherx << endl;
	cout << "othery = " << temp_othery << endl;
	cout << "imgx = " << temp_imgx << endl;
	cout << "imgy = " << temp_imgy << endl;
	
	// Below we account only for the effect of the pin closest to the vortex, thus simplifying calculations	
	
	int nx = round((xi + R)/a) ;
  	int ny = round((yi + R)/a) ;
  	if (nx<=0 || ny<=0 || nx>=nmax || ny>=nmax) 
  	{
  		nx = 0;
  		ny = 0;
	}
	
	// If xi or yi is slightly lesser than -R or greater than +R, nx and ny could be negative or out of bounds
	// If nx or ny is out of bounds, the above would lead to a pinning strength of zero (since V_0[0][0] = 0)
	// The above mechanism saves the code from segmentation fault
	
	double x_pin_nearest = -(R) + (round((xi + R)/a) * a);
	double y_pin_nearest = -(R) + (round((yi + R)/a) * a);
	double r2_pin_nearest = pow((x_pin_nearest - xi),2) + pow((y_pin_nearest - yi),2);
	double r_pin_nearest = sqrt(r2_pin_nearest);

	temp_pinx = ( (V_0[nx][ny])  * exp(- r2_pin_nearest/(2*pow(Xi,2))) * (yi - y_pin_nearest)); // Differential equations for x - Part 4 (Pinning)
	temp_piny = - ( (V_0[nx][ny])  * exp(- r2_pin_nearest/(2*pow(Xi,2))) * (xi - x_pin_nearest)); // Differential equations for y - Part 4 (Pinning)
	
	cout << "pinx = " << temp_pinx << endl;
	cout << "piny = " << temp_piny << endl;
	cout << "pin_nx = " << nx << endl;
	cout << "pin_ny = " << ny << endl; 
	getch();
*/