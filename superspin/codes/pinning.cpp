#include <iostream>
#include <iomanip> // For setw and setfill
#include <fstream>
#include <string>
#include <vector>
#include "parameters.h"

using namespace std;

// Standard pinning model
vector<vector<double>> pinning_standard()
{
	vector<vector<double>> pinning_vector_temp(nmax, vector<double>(nmax, 0.0));
	for (double x = -R; x <= R; x = x+a)
	{  	
		for (double y = -R; y <= R; y = y+a)
		{  
			double dist_pin = sqrt(pow(x,2)+pow(y,2)); 
			if (dist_pin<R)
			{
				int nx = round((R+x)/a);
				int ny = round((R+y)/a);
				pinning_vector_temp[nx][ny] = V_0;
			} 
		}
	}
	
	return pinning_vector_temp;
}

// Annnular pinning model
vector<vector<double>>  pinning_annular()
{
	vector<vector<double>> pinning_vector_temp(nmax, vector<double>(nmax, 0.0));
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
					pinning_vector_temp[nx][ny] = V_annular_in;
				}
				else if (dist_pin > r_annulus)
				{
					pinning_vector_temp[nx][ny] = V_annular_out;
				}
			} 
		}
	}

	return pinning_vector_temp;
}

// Vortex traps model
vector<vector<double>>  pinning_traps()
{
	vector<vector<double>> pinning_vector_temp(nmax, vector<double>(nmax, 0.0));

	for (double x = -R; x <= R; x = x+a)
	{  	
		for (double y = -R; y <= R; y = y+a)
		{ 
			double dist_pin = sqrt(pow(x,2)+pow(y,2)); 
			// If the pinning site is within the star
			if (dist_pin < R)
			{
				int nx = round((R+x)/a);
				int ny = round((R+y)/a);
				
				// And is within the trap
        double trap_centre_x = int(round(x/trap_separation)) * trap_separation;
        double trap_centre_y = int(round(y/trap_separation)) * trap_separation;
				if ((x > (trap_centre_x-(trap_size/2.0))) && (x < (trap_centre_x+(trap_size/2.0))) && (y > (trap_centre_y-(trap_size/2.0))) && (y < (trap_centre_y+(trap_size/2.0))))
				{
					pinning_vector_temp[nx][ny] = V_trap_in;
				}
				else
				{
					pinning_vector_temp[nx][ny] = V_trap_out;
				}
			}
		}
	}

	return pinning_vector_temp;
}

// Write location of pinning sites
void write_pinning_sites(vector<vector<double>> pinning_vector_temp)
{
  // Holders for x and y location of sites
  vector<double> location_x;
  vector<double> location_y;
  
  // Find pinning sites with non-zero pinning strength
  for (double x = -R; x <= R; x = x+a)
	{  	
		for (double y = -R; y <= R; y = y+a)
		{  
			double dist_pin = sqrt(pow(x,2)+pow(y,2)); 
			if (dist_pin<R)
			{
				int nx = round((R+x)/a);
				int ny = round((R+y)/a);
				
				if(pinning_vector_temp[nx][ny] != 0)
				{
				  location_x.push_back(x);
				  location_y.push_back(y);
				}
			} 
		}
	}
	
  // Write initial pinning locations to file      
  string filename = "init_pins_pos.dat";
  string fullpath = pathplace+filename;		        
  ofstream outfile;
  outfile.open(fullpath,ios::out | ios::app);
  outfile.precision(15);
  for (int i = 0; i<location_x.size(); i++) //Writing cannot be parallel
  { 
	  outfile << location_x[i] << "\t" << location_y[i] << endl;
  }
  outfile.close();
  
}

// Implement pinning configuration
vector<vector<double>>  pin_config_set()
{	
	vector<vector<double>> pinning_vector_temp(nmax, vector<double>(nmax, 0.0));

	if(pin_config == "STANDARD")
	{
		pinning_vector_temp = pinning_standard(); // No variation in strength
	}

	else if(pin_config == "ANNULAR")
	{
		pinning_vector_temp = pinning_annular(); // Annular strength variation
	}
	else if(pin_config == "TRAPS")
	{
		pinning_vector_temp = pinning_traps(); // Traps
	}

        // Write pinning locations to file
        write_pinning_sites(pinning_vector_temp);

	return pinning_vector_temp;
}

// Count the number of pinning sites within the container
int pins_counter()
{	
	int npin = 0;
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

	return npin;
}
