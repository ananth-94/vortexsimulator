// To find glitch size, rise time of glitch, waiting times between glitches  - Anantharaman S V, Ashoka University

// To be used alongside the code for 'Stabilization and dynamics of superfluid vortices in a neutron star' 

// Version 1 of code completed - May 29, 2025, 13:00:00
	
// ----- Headers and namespaces -----

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace std;

// ----- Definitions, declarations and common functions -----


double pi = M_PI;

// Parameters
double epsilon = 1e-12;

// Function declarations
void find_glitch(vector<double> omega, vector<double> t, double epsilon);

// ----- Main function -----

int main()
{

cout << "Finding glitches and saving data" << endl;

// Reading in data
cout << "Reading" << endl;
vector<double> t, omega;

ifstream ifile;
ifile.open("sim_omega_c.dat");

string header1, header2;
ifile >> header1 >> header2;

while (true)
{
double x,y;
ifile >> x >> y;
if(ifile.eof() == 1) break;
t.push_back(x);
omega.push_back(y);
}

ifile.close();


// Glitch finding
cout << "Finding" << endl;
find_glitch(omega, t, epsilon);		

return 0;

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
cout << "Writing" << endl;

fstream outfile;

// Glitch info
string filename = "info_glitch.dat"; 
outfile.open(filename,ios::out | ios::app);
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
outfile.open(filename,ios::out | ios::app);
outfile.precision(15);

outfile << "t_glitch" << "\t" << "t_rise" << "\t" << "glitch_size" << "\t" << "t_wait" << endl;

for(int i = 0; i < t_glitch.size(); i++)
{
	outfile << t_glitch[i] << "\t" << t_rise[i] << "\t" << del_omega[i] << "\t" << t_wait[i] << endl;
}

outfile.close();

}