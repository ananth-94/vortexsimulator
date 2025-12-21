#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "parameters.h"
#include "utility.h"
#include "support.h"

using namespace std;

// Track progress
void write_progress(double t, int &progress, const double runtime, const string prefix)
{
	if ( (t/runtime)*100 >= progress+1)
	{
		string filename = "info_progress.dat"; 
		string fullpath = pathplace+filename;
		fstream outfile;
		outfile.open(fullpath,ios::out | ios::app);
		outfile.precision(4);
		outfile << prefix << " progress " << progress+1 << " percent" << endl;
		outfile.close();

		progress += 1;
	}
}

// Write info file at start
void write_info_start(vector<double> &f)
{
	string filename = "info_simulation.dat"; 
	string fullpath = pathplace+filename;
	fstream outfile;
	outfile.open(fullpath,ios::out | ios::app);	// Open the file in append mode (ios::app) or printing mode (ios::out) both
	outfile.precision(15);

	// Vortex creation
	outfile << "// Vortex creation" << endl;
	outfile << "creation_condition = " << creation_condition << endl << endl;

	// System basic
	outfile << "// System basic" << endl;
	outfile << "R = " << R << endl;
	outfile << "nv = " << nv << endl;
	outfile << "npin_desired = " << npin_desired << endl;
	outfile << "k = " << k << endl;
	outfile << "phi = " << phi << endl;
	// System derived
	outfile << "// System derived" << endl;
	outfile << "omega_0 = " << omega_0 << endl;
	outfile << "T_0 = " << T_0 << endl;
	outfile << "a = " << a/R << " R" << endl;
	outfile << "Xi = " << Xi/a << " a" << endl;
	// System variables
	outfile << "// System variables" << endl;
	outfile << "omega_c = " << omega_c/omega_0 << " omega_0" << endl;
	outfile << "omega_s = " << omega_s/omega_0 << " omega_0" << endl << endl;

	// Pinning
	outfile << "// Pinning" << endl;
	outfile << "pin_config = " << pin_config << endl;
	outfile << "nmax = " << nmax << endl;
	// Standard
	outfile << "// Standard" << endl;
	outfile << "V_0 = " << V_0 << endl;
	// Annular
	outfile << "// Annular" << endl;
	outfile << "r_annulus = " << r_annulus/R << " R" << endl;
	outfile << "V_annular_in = " << V_annular_in << endl;
	outfile << "V_annular_out = " << V_annular_out << endl;
	// Traps
	outfile << "// Traps" << endl;
	outfile << "trap_number = " << trap_number << endl;
	outfile << "trap_ratio = " << trap_ratio << endl;
	outfile << "trap_separation = " << trap_separation << endl;
	outfile << "trap_size = " << trap_size << endl;
	outfile << "V_trap_in = " << V_trap_in << endl;
	outfile << "V_trap_out = " << V_trap_out << endl << endl;

	// Triggers
	outfile << "// Triggers" << endl;
	outfile << "trigger_type = " << trigger_type << endl;
	outfile << "number_triggers = " << number_triggers << endl;
	outfile << "duration_triggers = " << duration_triggers << endl;
	//Sectorial
	outfile << "// Sectorial" << endl;
	outfile << "sector_count = " << sector_count << endl;
	outfile << "sectorial_region = " << sectorial_region << endl;
	//Stresswave
	outfile << "// Stresswave" << endl;
	outfile << "stresswave_threshold = " << stresswave_threshold << endl;
	// Trapwise
	outfile << "number_trapwise = " << number_trapwise << endl;
	outfile << "r_trapwise_in = " << r_trapwise_in << endl;
	outfile << "r_trapwise_out = " << r_trapwise_out << endl << endl;

	// Dynamics
	outfile << "// Dynamics" << endl;
	outfile << "N_ext = " << N_ext/(omega_0/T_0) << " omega_0/T_0" << endl << endl;

	// Glitch finding
	outfile << "// Glitch finding" << endl;
	outfile << "epsilon_glitch = " << epsilon_glitch << endl << endl;

	// Integration
	outfile << "// Integration" << endl;
	outfile << "integration_condition = " << integration_condition << endl;
	// Barnes-Hut
	outfile << "// Barnes-Hut" << endl;
	outfile << "size_bh_real = " << size_bh_real/R << " R" << endl;
	outfile << "size_bh_image = " << size_bh_image/R << " R" << endl;
	outfile << "theta_bh_real = " << theta_bh_real << endl;
	outfile << "theta_bh_image = " << theta_bh_image << endl << endl;

	// Run
	outfile << "// Run" << endl;
	outfile << "del_t = " << del_t/T_0 << " T_0" << endl;
	outfile << "stabilization_runtime = " << stabilization_runtime/T_0 << " T_0" << endl;
	outfile << "dynamics_runtime = " << dynamics_runtime/T_0 << " T_0" << endl << endl;

	// Filesystem
	outfile << "// Filesystem" << endl;
	outfile << "path = " << path << endl << endl;

	// Writing
	outfile << "// Filesystem" << endl;
	outfile << "write_fullpos = " << to_string(write_fullpos) << endl;
	outfile << "minimal_step = " << minimal_step << endl << endl;
	
	// Start
	outfile << "// Start" << endl;
	outfile << "start time = " << calc_time();
	outfile << "out_count = " << out_counter(f) << endl;
	outfile << "unpinned_count = " << unpinned_counter(f) << endl << endl;

	outfile.close();
}

// Write info file midway between stabilization and dynamics
void write_info_mid(vector<double> &f, double kiss)
{	
	string filename = "info_simulation.dat"; 
	string fullpath = pathplace+filename;
	fstream outfile;
	outfile.open(fullpath,ios::out | ios::app);	// Open the file in append mode (ios::app) or printing mode (ios::out) both
	outfile.precision(15);
	
	// Mid
	outfile << "// Mid" << endl;
	outfile << "stabilization end time = " << calc_time();
	outfile << "k/I_s = " << kiss << endl;
	outfile << "out_count = " << out_counter(f) << endl;
	outfile << "unpinned_count = " << unpinned_counter(f) << endl;
	outfile << "omega_c = " << omega_c/omega_0 << " omega_0" << endl<< endl;
	outfile.close();
}

// Write info file at end
void write_info_end(vector<double> &f)
{	
	string filename = "info_simulation.dat"; 
	string fullpath = pathplace+filename;
	fstream outfile;
	outfile.open(fullpath,ios::out | ios::app);	// Open the file in append mode (ios::app) or printing mode (ios::out) both
	outfile.precision(15);
	
	// End
	outfile << "// End" << endl;
	outfile << "end time = " << calc_time();
	outfile << "out_count = " << out_counter(f) << endl;
	outfile << "unpinned_count = " << unpinned_counter(f) << endl;
	outfile << "omega_c = " << omega_c/omega_0 << " omega_0" << endl<< endl;
	outfile.close();
}

//  Write the stablized positions and H values
void write_data_post_stabilization(vector<double> &t_vector, vector<double> &f, vector<double> &H_vector)
{
	fstream outfile;
	
	// Positions at the end of stabilization
	string filename = "stabilized_vortex_pos.dat"; 
   	string fullpath = pathplace+filename;
    outfile.open(fullpath, ios::out|ios::app);
    outfile.precision(15);
    outfile << "x" << "\t" << "y" << endl;
	for (int i=0; i < nv; i++)
    {
    	outfile << f[i] << "\t" << f[i+nv]<< endl;
	}
    outfile << endl;
    outfile.close();
    
    // H values and corresponding times	
	filename = "stabilized_Hvalues.dat"; 
   	fullpath = pathplace+filename;
	outfile.open(fullpath, ios::out|ios::app);
	outfile.precision(15);
	
	for(int i=0; i<H_vector.size(); i++)
	{
		outfile << t_vector[i]<< "\t" << H_vector[i] << endl;
	}
		
	outfile.close();  
}

// Write data post dynamics
// All vortex motions, and omega vs t, recorded at an interval del_t
void write_data_post_dynamics(vector<double> &t_vector, vector<vector<double>> &f_vector, vector<double> &omega_c_vector, vector<double> &omega_s_vector,
	vector<int> &number_off_vector, vector<int> &sector_id_vector, vector<double> &trigger_times, vector<int> &out_vector, vector<int> &unpinned_vector)
{
	fstream outfile;

	// Full positions
	string filename = "sim_vortex_pos.dat"; 
	string fullpath = pathplace+filename;
	
	if (write_fullpos == true)
	{
		outfile.open(fullpath, ios::out | ios::app);
		outfile.precision(15);
		
		for (int i=0; i<t_vector.size(); i++)
		{
			outfile << t_vector[i];
			for (int j=0; j< 2*nv ; j++)
		  	{
		    	outfile << '\t'<< f_vector[i][j];
			}
			outfile << endl;
		}
		
		outfile.close();
	}

	// Minimal positions - Reducing file size by a factor of ten
	filename = "sim_vortex_pos_minimal.dat"; 
	fullpath = pathplace+filename;
	outfile.open(fullpath,ios::out | ios::app);
	outfile.precision(15);
	
	for (int i=0; i<t_vector.size(); i=i+minimal_step)
	{
		outfile << t_vector[i];
		for (int j=0; j<( 2*nv ); j++)
	  	{
	    	outfile << '\t'<< f_vector[i][j];
		}
		outfile << endl;
	}
	
	outfile.close();
    
  	// omega_c/omega_0 and corresponding times
	filename = "sim_omega_c.dat"; 
 	fullpath = pathplace+filename;
	outfile.open(fullpath, ios::out|ios::app);
	outfile.precision(15);
	
	for(int i=0; i<t_vector.size(); i++)
	{
		outfile << t_vector[i]<< "\t" << omega_c_vector[i] << endl;
	}
		
	outfile.close();
	
	// omega_s/omega_0 and corresponding times
	filename = "sim_omega_s.dat"; 
 	fullpath = pathplace+filename;
	outfile.open(fullpath, ios::out|ios::app);
	outfile.precision(15);
	
	for(int i=0; i<t_vector.size(); i++)
	{
		outfile << t_vector[i]<< "\t" << omega_s_vector[i] << endl;
	}
		
	outfile.close();
	
	// Trigger details and corresponding times
	filename = "sim_triggers.dat"; 
	fullpath = pathplace+filename;
	outfile.open(fullpath, ios::out|ios::app);
	outfile.precision(15);
	

	if (trigger_type == "SECTORIAL")
	{
		// trigger_times.size()-1 has been used since the last entry of the vector was added by hand
		for(int i=0; i<trigger_times.size()-1; i++)
		{
			outfile << trigger_times[i]/T_0 << "\t"<< sector_id_vector[i] << "\t" << number_off_vector[i] << endl;
		}
	}

	else if (trigger_type == "STRESSWAVE")
	{
		// trigger_times.size()-1 has been used since the last entry of the vector was added by hand
		for(int i=0; i<trigger_times.size()-1; i++)
		{
			outfile << trigger_times[i]/T_0 << endl;
		}
	}

	else if (trigger_type == "TRAPWISE")
	{
		// trigger_times.size()-1 has been used since the last entry of the vector was added by hand
		for(int i=0; i<trigger_times.size()-1; i++)
		{
			outfile << trigger_times[i]/T_0 << endl;
		}
	}
	
	outfile.close();
	
	// Total number of vortices within boundary and unpinned_count
	filename = "sim_unpinned.dat"; 
	fullpath = pathplace+filename;
	outfile.open(fullpath, ios::out|ios::app);
	outfile.precision(15);
	
	for(int i=0; i<t_vector.size(); i++)
	{
		outfile << t_vector[i] << "\t" << unpinned_vector[i] << "\t" << (nv-out_vector[i]) << "\t" << double(unpinned_vector[i])/double(nv-out_vector[i]) << endl;
	}
	
	outfile.close();	
}
