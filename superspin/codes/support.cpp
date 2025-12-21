#include <vector>
#include <string>
#include <omp.h>
#include <bits/stdc++.h> // Required for accumulate
#include "parameters.h"
#include "initialization.h"

using namespace std;

// Convert to vortices (vector<vector<double>>) from state f (vector<double>) and circulations K (vector<double>)
void state_to_vortices(vector<double> &f, vector<double> &K, vector<vector<double>> &vortices)
{
	//# pragma omp parallel for
	for(int i = 0; i < nv; i++)
	{
		vortices[i][0] = f[i];
		vortices[i][1] = f[i+nv];
		vortices[i][2] = K[i];
	}
}

// Calculate H
double H_calculate(vector<double> &f)
{

	double H = 0;	
	vector<double> H_forsum(nv,0.0);
	
	#pragma omp parallel for
	for (int i=0; i<nv ;i++)
	{
		double h_i =0;
		
		for (int j=0; j<nv ;j++)
		{
			if(j != i)
			{
				double rij = sqrt((pow((f[i] - f[j]),2) + pow((f[i+nv] - f[j+nv]),2)));
				h_i += log (rij);
			}
		}
		
		H_forsum[i] = h_i;
	}
	
	H = accumulate(H_forsum.begin(),H_forsum.end(),0.0);
	return H;	
}

// Update image vortices and circulations
void updates(vector<double> &f, vector<double> &f_image, vector<double> &K)
{	
	//#pragma omp parallel for
	for (int i=0; i<nv ; i++)
	{
		// Update image vortices and circulation
		// If the vortex exists
		if (K[i] != 0)
		{				
			// Updating image vortices
			double r = sqrt(pow(f[i],2) + pow(f[i+nv],2));
			double theta = atan2(f[i+nv] , f[i]);
			
			double r_img = pow(R,2) / r;
			double theta_img = theta;
	
			double x_img = r_img * cos(theta_img);
			double y_img= r_img * sin(theta_img);
	
			f_image[i] = x_img;
			f_image[i+nv] = y_img;
			
			// Circulation is zero if the vortex is outside the boundary
			if (r>=R)
			{
				K[i] = 0;
			}
		}	
	}
}

// Calculate current superfluid rotation
double superfluid_rotation(vector<double> &f, vector<double> &K, double kiss)
{
	// Superfluid rotation rate
	// Usage of vector instead of in-place increment due to parallelization
	vector<double> omega_s_forsum(nv, 0.0);
	#pragma omp parallel for
	for (int i=0; i<nv ; i++)
	{	
		// If the vortex exists	
		if (K[i] != 0)
		{
			double r = sqrt(pow(f[i],2) + pow(f[i+nv],2));
			omega_s_forsum[i] = kiss * (pow(R,2)-pow(r,2));
		}
	}

	// Updating omega_s
	double omega_s = accumulate(omega_s_forsum.begin(), omega_s_forsum.end(), 0.0);

	return omega_s;
}

// Count vortices outside boundary
int out_counter(vector<double> &f)
{
	int out_count = 0;
		
	for (int i=0; i<nv; i++)
	{	
		double r = sqrt(pow(f[i],2) + pow(f[i+nv], 2));
		if (r >=R)
		{
			out_count += 1;
		}
	}

	return out_count;
}

// Count vortices unpinned
int unpinned_counter(vector<double> &f)
{
	int unpinned_count = 0;

	for (int i=0; i<nv; i++)
	{	
		double r = sqrt(pow(f[i],2) + pow(f[i+nv], 2));
		
		if (r < R)
		{			
			double x_pin_nearest = -(R) + (round((f[i] + R)/a) * a);
			double y_pin_nearest = -(R) + (round((f[i+nv] + R)/a) * a);
			double r2_pin_nearest = pow((x_pin_nearest - f[i]),2) + pow((y_pin_nearest - f[i+nv]),2);
			double r_pin_nearest = sqrt(r2_pin_nearest);
			
			if(r_pin_nearest > Xi)
			{
				unpinned_count += 1;	
			}
		}
	}

	return unpinned_count;
}
