#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <omp.h>
#include <bits/stdc++.h> // For filesystem function
#include "parameters.h"
#include "utility.h"

using namespace std;

// Before stabilization
// Initializing the state corresponding to real vortices
vector<double> initialize_state_real(string condition)
{
	vector<double> f(2*nv, 0.0);
	
	// Generate vortices from existing positions file
	if(condition == "LOAD")
	{

		// Reading in data
		cout << "Loading positions of vortices" << endl;
		
		string filename = "loadfile_"+ to_string(nv) + ".dat"; 
		string fullpath = "./resources/" + filename; // The loadfile is supplied in the same directory as the code files
		ifstream infile;
		infile.open(fullpath);

		// If the appropriate loadfile does not exist
		if (!infile.is_open())
		{
			// Indicate an error and exit
    		std::cerr << "Error: Could not open " << fullpath << endl;
    		exit(0);
		}
		
		string header1, header2;
		infile >> header1 >> header2;
		
		for (int i = 0; i < nv; i++)
		{ 	
			double x,y;
			infile >> x >> y;
			//if(ifile.eof() == 1) break;
			
			f[i] = x;
			f[i+nv] = y;
		}
		
		infile.close();
	}

	// Generating vortices anew randomly and uniformly across the star
	else if(condition == "GENERATE")
	{
		// Generating data
		cout << "Generating positions of vortices" << endl;

		// Generate a circle of radius R units filled uniformly with points
		# pragma omp parallel for
		for (int i = 0; i < nv; i++)
		{ 	
		    	
			double r = R*sqrt(randnumb01());
			double theta = 2*pi*randnumb01();
		            
		    double x = r * cos(theta);
		    double y= r * sin(theta);
		     
			f[i] = x;
			f[i+nv] = y;
		}
	}

	// Generating vortices, all located within the traps
    else if(condition == "GENERATE_WITHIN_TRAPS")
	{
		// Generating data
		cout << "Generating positions of vortices within traps" << endl;

		// Generate a star of radius R with vortices present only in the trap locations
		# pragma omp parallel for
		for (int i = 0; i < nv; i++)
		{		    
		    bool vortex_check = false;
		    while (vortex_check == false)
		    {
		    	// Create a vortex randomly positioned
		    	double r = R*sqrt(randnumb01());
				double theta = 2*pi*randnumb01();
			            
			    double x = r * cos(theta);
			    double y= r * sin(theta);
		    	
   			 // If vortex is within a trap and not too close to the boundary
        double trap_centre_x = int(round(x/trap_separation)) * trap_separation;
        double trap_centre_y = int(round(y/trap_separation)) * trap_separation;       
			  if (((x-trap_padding) > (trap_centre_x-(trap_size/2.0))) && ((x+trap_padding) < (trap_centre_x+(trap_size/2.0))) && ((y-trap_padding) > (trap_centre_y-(trap_size/2.0))) && ((y+trap_padding) < (trap_centre_y+(trap_size/2.0))))
				{
          	// Consider vortex
  					f[i] = x;
  					f[i+nv] = y;
  					vortex_check = true;
				}
		    }
		}
	}
	
	// Write initial positions to file	      
	string filename = "init_vortex_pos.dat";
	string fullpath = pathplace+filename;		        
	ofstream outfile;
    outfile.open(fullpath,ios::out | ios::app);
    outfile.precision(15);
    for (int i = 0; i<nv; i++) //Writing cannot be parallel
	{ 
    	outfile << f[i] << "\t" << f[i+nv] << endl;
	}
    outfile.close();

    return f;
}

// Initializing the state corresponding to image vortices		
vector<double> initialize_state_image(vector<double> &f)
{   
	vector<double> f_image(2*nv, 0.0);

	for (int i = 0; i < nv; i++)
	{  
		double x = f[i];
		double y = f[i+nv];

		double r = sqrt(pow(x,2)+pow(y,2));
		double theta = atan2(y, x);

		double r_image = pow(R,2) / r;
		double theta_image = theta;

		double x_image = r_image * cos(theta_image);
		double y_image= r_image * sin(theta_image);

		f_image[i] = x_image;
		f_image[i+nv] = y_image;
	}

	// Write initial image positions to file	
	string filename = "init_img_pos.dat";
	string fullpath = pathplace+filename;
	ofstream outfile;	        
    outfile.open(fullpath,ios::out | ios::app);
    outfile.precision(15);
    for (int i = 0; i<nv; i++)
	{
    	outfile << f_image[i] << "\t" << f_image[i+nv] << endl;
	}
    outfile.close();

    return f_image;
}

// To initialize output files in the appropriate directory
void create_output_files(int run_id)
{	
	// Setting the directory for the current run
	std::filesystem::create_directory(pathplace);

	// Erasing contents of output files and setting header labels
	string filename = "sim_vortex_pos.dat"; 
	string fullpath = pathplace+filename;	
	ofstream file;	
	file.open(fullpath); 
	file << "t/T_0";
	for (int i=0; i<nv; i++)
	{
		file << '\t'<< "x" <<i+1;
	}
	for (int i=0; i<nv; i++)
	{
		file << '\t'<< "y" <<i+1;
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
	
	filename = "init_pins_pos.dat"; 
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
	file << "t_glitch" << "\t" << "t_rise" << "\t" << "glitch_size" << "\t" << "t_wait" << endl;
	file.close();

	filename = "data_glitch_multi.dat"; 
	fullpath = path+filename; // 'path' here, not 'pathplace' - Produces collated data file in the output folder and not inside the runs folder
	file.open(fullpath); // Erasing content of data_glitch_multi file
	file<<"t_glitch"<<"\t"<<"t_rise"<<"\t"<<"glitch_size"<<"\t"<<"t_wait"<<endl;
	file.close();
}

// Before dynamics
// Fix constant k/I_s (defined as kiss)
double kiss_fix(vector<double> &f)
{
	double sum = 0;
	for (int i=0; i<nv ;i++)
	{	
		double r = sqrt(pow(f[i],2) + pow(f[i+nv],2));
		if (r<R)
		{
			sum = sum + (pow(R,2)-pow(r,2));
		}
	}
	
	double kiss = omega_0 / sum;

	return kiss;

}
