#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include "parameters.h"

using namespace std;

// Global variables for collating glitch data
vector<double> collate_t_glitch, collate_t_rise, collate_del_omega, collate_t_wait;

// Recognizing a glitch and storing relevant data
void find_glitches(vector<double> omega, vector<double> t, double epsilon)
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

// Warn and return if no glitches
if(del_omega.size() < 2)
{
	cout << "No glitches found" << endl;
	return;
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

// Collate data
collate_t_glitch.insert(collate_t_glitch.end(),t_glitch.begin(),t_glitch.end());
collate_t_rise.insert(collate_t_rise.end(),t_rise.begin(),t_rise.end());
collate_del_omega.insert(collate_del_omega.end(),del_omega.begin(),del_omega.end());
collate_t_wait.insert(collate_t_wait.end(),t_wait.begin(),t_wait.end());

// Writing to file
fstream outfile;

// Glitch info
string filename = "info_glitch.dat"; 
string fullpath = pathplace+filename;
outfile.open(fullpath, ios::out | ios::app);
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
outfile.open(fullpath, ios::out | ios::app);
outfile.precision(15);

for(int i = 0; i < t_glitch.size(); i++)
{
	outfile << t_glitch[i] << "\t" << t_rise[i] << "\t" << del_omega[i] << "\t" << t_wait[i] << endl;
}

outfile.close();

}

// Write collated glitch data
bool collate_glitches()
{
	// Return false if no glitches to collate
	if(collate_del_omega.size() == 0)
	{
		return false;
	}

	string filename = "data_glitch_multi.dat"; 
	string fullpath = path+filename;	
	ofstream ofile; 
	ofile.open(fullpath, ios::out | ios::app);
	ofile.precision(15);
	for(int i = 0; i < collate_t_glitch.size() ;i++)
	{
		ofile<<collate_t_glitch[i]<<"\t"<<collate_t_rise[i]<<"\t"<<collate_del_omega[i]<<"\t"<<collate_t_wait[i]<<endl;
	}
	ofile.close();

	return true;
}