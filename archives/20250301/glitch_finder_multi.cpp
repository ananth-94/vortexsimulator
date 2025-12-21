// To extract glitch data from across multiple runs - Anantharaman S V, Ashoka University

// To find glitch size, rise time of glitch, and waiting times between glitches at all times
// To produce the same for a reduced dataset of omega vs t (So as to effectively smoothen the data)
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
	
	
// ----- Jugaad to set directory for outputs (Since relative paths don't seem work in C++ on linux) -----

	string path = "./"; // Enter the absolute path of the folder
	string var, pathplace; // Needed for storing multiple runs in different folders
	int numb_runs; // Desired number of runs of the simulation
	int run_id = 1;

	/* Follow the below example snippet to open a file of a desired name and perform read/write operations
		string filename = "Test.txt";
		string fullpath = pathplace+filename;
		ofstream file;	
		file.open(fullpath); // erasing content of vortexinit file
		file << "Result: "<< fullpath << endl; //file actions
		file.close();
	*/	


// ----- Definitions, declarations and common functions -----

double pi = M_PI;

// Parameters
double epsilon = 1e-12;

// Function declarations
void find_glitch_multi(vector<double> omega, vector<double> t, double epsilon);
void find_glitch_multi_reduced(vector<double> omega, vector<double> t, double epsilon);

// Global variables used for collating data
vector<double> collate_t_glitch, collate_t_rise, collate_del_omega, collate_t_wait;
vector<double> collate_t_glitch_reduced, collate_t_rise_reduced, collate_del_omega_reduced, collate_t_wait_reduced;

// ----- Main function -----

int main()
{

	cout << "Finding glitches across multiple sets of data" << endl << endl;
	cout << "Enter number of runs " ;
	cin >> numb_runs;

	for (run_id=1; run_id <= numb_runs; run_id++)
	{
		// Setting the directory for the current run
		var = to_string(run_id);
		pathplace = path+"run"+var+"/";
		
		cout << "Run " << run_id << " of " << numb_runs <<endl;
		
		// Reading in data
		cout << "Reading" << endl;
		vector<double> t, omega;
		
		string filename = "sim_omega_c.dat"; 
		string fullpath = pathplace+filename;	
		ifstream ifile;
		ifile.open(fullpath);
		
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
		find_glitch_multi(omega, t, epsilon);
		find_glitch_multi_reduced(omega, t, epsilon);
	}


	// ----- Writing to file -----
	cout << "Writing" << endl << endl;
	ofstream ofile;

	// Writing collated data
	ofile.open("data_glitch_multi.dat");
	ofile.precision(15);
	ofile<<"t_glitch"<<"\t"<<"t_rise"<<"\t"<<"glitch_size"<<"\t"<<"t_wait"<<endl;
	for(int i = 0; i < collate_t_glitch.size() ;i++)
	{
		ofile<<collate_t_glitch[i]<<"\t"<<collate_t_rise[i]<<"\t"<<collate_del_omega[i]<<"\t"<<collate_t_wait[i]<<endl;
	}
	ofile.close();

	// Writing collated data -- Reduced
	ofile.open("data_glitch_multi_reduced.dat");
	ofile.precision(15);
	ofile<<"t_glitch"<<"\t"<<"t_rise"<<"\t"<<"glitch_size"<<"\t"<<"t_wait"<<endl;
	for(int i = 0; i < collate_t_glitch_reduced.size() ;i++)
	{
		ofile<<collate_t_glitch_reduced[i]<<"\t"<<collate_t_rise_reduced[i]<<"\t"<<collate_del_omega_reduced[i]<<"\t"<<collate_t_wait_reduced[i]<<endl;
	}
	ofile.close();
	
	return 0;
}

// Recognizing a glitch and storing relevant data
void find_glitch_multi(vector<double> omega, vector<double> t, double epsilon)
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

	// Collating data

	collate_t_glitch.insert(collate_t_glitch.end(),t_glitch.begin(),t_glitch.end());
	collate_t_rise.insert(collate_t_rise.end(),t_rise.begin(),t_rise.end());
	collate_del_omega.insert(collate_del_omega.end(),del_omega.begin(),del_omega.end());
	collate_t_wait.insert(collate_t_wait.end(),t_wait.begin(),t_wait.end());
}

// Recognizing a glitch and storing relevant data -- For reduced omega vs t dataset
void find_glitch_multi_reduced(vector<double> omega, vector<double> t, double epsilon)
{

	bool glitch_state = false, glitch_init = false;
	double t_i, omega_i, t_f, omega_f;

	vector<double> t_glitch, t_rise, del_omega, omegadot;
	
	// Reducing dataset - Extracting every 10th entry from the original dataset
	vector<double> t_reduced, omega_reduced;

	for (int i = 0; i < t.size(); i=i+10)
	{
		t_reduced.push_back(t[i]);
		omega_reduced.push_back(omega[i]);
	}

	t = t_reduced;
	omega = omega_reduced;

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

	// Collating data

	collate_t_glitch_reduced.insert(collate_t_glitch_reduced.end(),t_glitch.begin(),t_glitch.end());
	collate_t_rise_reduced.insert(collate_t_rise_reduced.end(),t_rise.begin(),t_rise.end());
	collate_del_omega_reduced.insert(collate_del_omega_reduced.end(),del_omega.begin(),del_omega.end());
	collate_t_wait_reduced.insert(collate_t_wait_reduced.end(),t_wait.begin(),t_wait.end());
}