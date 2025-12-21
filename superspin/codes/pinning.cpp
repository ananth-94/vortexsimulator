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
				if (int(round(x/trap_size)) % trap_ratio == 0 && int(round(y/trap_size)) % trap_ratio == 0)
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
