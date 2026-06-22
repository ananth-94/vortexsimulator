// SUPERSPIN: A program to simulate superfluid vortices in a neutron star
// Anantharaman S V, Ashoka University
// June 2 2025, 17:25:00

#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <iomanip> // For setprecision()
#include <utility> // For pair function
#include "parameters.h"
#include "initialization.h"
#include "support.h"
#include "writes.h"
#include "integration.h"
#include "pinning.h"
#include "triggers.h"
#include "glitchfinder.h"

using namespace std;

int main()
{
	// Start message
	cout << endl;
	cout << "---------------------------------------------------------" << endl;
	cout << "Program to simulate superfluid vortices in a neutron star" << endl;
	cout << "Developer: Anantharaman SV, Ashoka University, June 2025" << endl;
	cout << "---------------------------------------------------------" << endl << endl;
	
	// Multiple runs with the same parameters
	for (int run_id = 1; run_id <= number_runs; run_id++)
	{	
		cout << "Run " << run_id << " of " << number_runs << endl;

		// Set the directory for the current run
		pathplace = path+"run"+to_string(run_id)+"/";

		// Start clock
		auto start = std::chrono::high_resolution_clock::now();
		double t = 0;
		vector<double> t_vector;
		int progress = 0;

		// Create output files
		create_output_files(run_id);

		// Create vortices
		vector<double> f = initialize_state_real(creation_condition); // State vector of real vortices // {x1,x2,x3,...y1,y2,y3...} // Size 2*nv
		vector<double> f_image = initialize_state_image(f); // State vector of corresponding image vortices // Size 2*nv
		vector<double> K(nv, k); // Circulations of nv real vortices having strength k // Size nv

		// Create pinning sites
		vector<vector<double>> pinning_vector(nmax, vector<double>(nmax, 0.0)); // Pinning strengths of all the sites on an (nmax x nmax) grid
		pinning_vector = pin_config_set();

		// Initialize stabilization variables
		vector<double> H_vector;

		// Reinitialize appropriate global variables -- Relevant for multiple runs
		omega_c = omega_0;
		omega_s = omega_0;

		// Write info start
		write_info_start(f);

		// Stabilization loop
		cout << "Stabilization on" << endl;
		while(t <= stabilization_runtime)
		{
			// Write progress
			write_progress(t, progress, stabilization_runtime, "Stabilization");

			// Save time and append to vector
			t_vector.insert(t_vector.end(), t/T_0); // Time gets saved in units of T_0

			// Calculate H and append to vector
			double H = H_calculate(f);
			H_vector.insert(H_vector.end(), H);

			// Adaptive integration
			if(integration_condition == "BARNESHUT")
			{
				adaptive_integrate_bh(f, f_image, K, pinning_vector,t);
			}
			else if(integration_condition == "EXACT")
			{
				adaptive_integrate_exact(f, f_image, K, pinning_vector,t);
			}

			// Increment time
			t += del_t;
		}

		// Write data
		write_data_post_stabilization(t_vector, f, H_vector);

		// Intermediate resets
		t = 0;
		t_vector.clear();
		progress = 0;

		// Initialize dynamics variables
		int out_count = out_counter(f); // Number of vortices outside the boundary 
		int unpinned_count = unpinned_counter(f); // Number of unpinned vortices
		double kiss = kiss_fix(f); // Fix the constant of proportionality for angular momentum calculations

		vector<double> omega_c_vector; // Rotation rate of container stored over time
		vector<double> omega_s_vector; // Rotation rate of superfluid stored over time
		vector<double> t_vector_write; // Times solely for writing positions
		vector<vector<double>> f_vector; // State of the system stored over time
		vector<int> out_vector; // Number of vortices outside the boundary stored over time
		vector<int> unpinned_vector; // Number of unpinned vortices stored over time

		// Initialize triggers
		vector<double> trigger_times = triggers_init();
		// Sectorial variables
		int count_trig = 0;
		int state_trig = 0;
		vector<int> number_off_vector;
		vector<int> sector_id_vector;

		// Write info mid
		write_info_mid(f, kiss);

		// Initialize other variables
		int f_write_count = 0;

		// Dynamics loop
		cout << "Dynamics on" << endl;
		while(t <= dynamics_runtime)
		{
			// Writes
			write_progress(t, progress, dynamics_runtime, "Dynamics");
			t_vector.insert(t_vector.end(), t/T_0); // Time gets saved in units of T_0
			omega_c_vector.insert(omega_c_vector.end(), omega_c/omega_0); // Saving omega_c in units of omega_0
			omega_s_vector.insert(omega_s_vector.end(), omega_s/omega_0); // Saving omega_s in units of omega_0
			
			if(f_write_count % write_step == 0)
			{
				t_vector_write.insert(t_vector_write.end(), t/T_0); // Time gets saved in units of T_0
				f_vector.insert(f_vector.end(), f); // Vortex states get saved over time
			}
			
			out_vector.insert(out_vector.end(), out_count); // Saving out_count
			unpinned_vector.insert(unpinned_vector.end(), unpinned_count); // Saving unpinned_count
			
			// Increment write count
			f_write_count += 1;

			// Activate trigger if applicable
			pair<int, int> triggers_info = triggers_check(t, pinning_vector, count_trig, state_trig, trigger_times);
			// Save info regarding trigger if of sectorial type
			int number_off = triggers_info.first;
			int sector_id = triggers_info.second;
			if (number_off != 0 || sector_id != 0) // Should be true only for sectorial triggers
			{
				number_off_vector.insert(number_off_vector.end(), number_off);
				sector_id_vector.insert(sector_id_vector.end(), sector_id);
			}

			// Spindown of the container
			omega_c += (N_ext/I_c) * del_t;

			// Adaptive integration
			if(integration_condition == "BARNESHUT")
			{
				adaptive_integrate_bh(f, f_image, K, pinning_vector,t);
			}
			else if(integration_condition == "EXACT")
			{
				adaptive_integrate_exact(f, f_image, K, pinning_vector,t);
			}

			// New superfluid rotation
			double omega_s_new = superfluid_rotation(f, K, kiss);
			// Feedback onto container
			double del_omega_s = omega_s_new - omega_s;
			omega_c -= I_rel * del_omega_s;
			// Update superfluid rotation
			omega_s = omega_s_new;

			// Update number of vortices outside boundary
			out_count = out_counter(f);
			// Update number of vortices unpinned
			unpinned_count = unpinned_counter(f);

			// Increment time
			t += del_t;

			// If the crust has completely spundown, the program halts immediately
			if(omega_c <= 0)
			{
				break;
			}
		}

		// Change to next line after displaying dynamics progress
		cout << endl;

		// Write info end
		write_info_end(f);

		// Write data
		cout << "Writing data" << endl;
		write_data_post_dynamics(t_vector, t_vector_write, f_vector, omega_c_vector, omega_s_vector, number_off_vector, sector_id_vector, trigger_times, out_vector, unpinned_vector, run_id);

		// Find glitches and write glitch data
		cout << "Finding glitches" << endl;
		find_glitches(omega_c_vector, t_vector, epsilon_glitch);

		// End clock
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
		cout << "Execution time: " << duration.count() << " seconds" << endl << endl ;
	}

	// Collating glitch data from all runs and writing
	cout << "Collating glitch data from all runs" << endl;
	bool collate_condition = collate_glitches();
	if (collate_condition == false)
	{
		cout << "No glitches to collate" << endl;
	}
	cout << endl;

	// End message
	cout << "---------------------------------------------------------" << endl;
	cout << "Fin" << endl;
	cout << "---------------------------------------------------------" << endl << endl;

	return 0;
}
