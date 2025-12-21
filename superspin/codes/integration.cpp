#include <iostream>
#include <vector>
#include <functional> // For bind function
#include <omp.h>
#include <boost/numeric/odeint.hpp>
#include "parameters.h"
#include "support.h"
#include "node.h"

using namespace std;
using namespace boost::numeric::odeint;

// Defines required for eom_bh
#define xi f[i]
#define yi f[i+nv]
#define ri2 ((f[i]*f[i])+(f[i+nv]*f[i+nv]))
#define rj2 ((f[j]*f[j])+(f[j+nv]*f[j+nv]))
#define dxidt dfdt[i]
#define dyidt dfdt[i+nv]

// Defines required for eom_exact
#define xij (f[i] - f[j])
#define yij (f[i+nv] - f[j+nv])
#define rij sqrt((pow((f[i] - f[j]),2) + pow((f[i+nv] - f[j+nv]),2)))
#define rij2  (pow((f[i] - f[j]),2) + pow((f[i+nv] - f[j+nv]),2))
#define xij_img (f[i] - f_image[j])
#define yij_img (f[i+nv] - f_image[j+nv])
#define rij2_img  (pow((f[i] - f_image[j]),2) + pow((f[i+nv] - f_image[j+nv]),2))

// Stepper definition
runge_kutta_cash_karp54<vector<double>> rkck54;

// Global variables for the scope of this file to store forces calculated using Barnes-Hut
vector<vector<double>> force_bh(nv, {0,0}); // Required for quadtree_and_forces() and eom_bh()
int threads_supported = std::thread::hardware_concurrency(); // Threads supported by the system
int chunk_size = nv / threads_supported;

// Barnes-Hut procedure
// Create quadtree and calculate forces on all particles due to all others
void quadtree_and_forces(vector<double> &f, vector<double> &f_image, vector<double> &K)
{
	// Create vortices and images
	vector<vector<double>> vortices(nv, {0,0,0}); // States and circulations combined into vortices // {{x1, y1, k1}, {x2, y2, k2}...} // Size nv x 3
	vector<vector<double>> images(nv, {0,0,0}); // States and circulations combined into vortices // {{x1, y1, k1}, {x2, y2, k2}...} // Size nv x 3
	state_to_vortices(f, K, vortices); // Convert states and circulations to vortices
	state_to_vortices(f_image, K, images); // Convert image states and circulations to images

	// Create quadtree
	int level = 0;
	vector<double> anchor = {0,0};
	node system_real(size_bh_real, level, anchor); // A node of real vortices with parameteres of level 0 (root) and anchor {0,0} (centre at origin)
	node system_image(size_bh_image, level, anchor); // A node of image vortices with parameteres of level 0 (root) and anchor {0,0} (centre at origin)

	for (int i = 0; i < nv; i++)
	{
		// Insert vortex only if it exists
		if (vortices[i][2] != 0)
		{
			system_real.insert(vortices[i]);
			system_image.insert(images[i]);
		}
	}
	
	// Calculate forces and store in an array
	// Scheduling the parallel loop to target 100% CPU Usage
	# pragma omp parallel for schedule(static, chunk_size)
	for (int i=0; i<nv; i++)
	{
		// Traverse tree and calculate force only if vortex exists
		if (vortices[i][2] != 0)
		{
			vector<double> force_bh_real(2,0.0), force_bh_image(2,0.0);
			double force_bh_x = 0, force_bh_y = 0;
			
			force_bh_real = system_real.compute_force(vortices[i], theta_bh_real);
			force_bh_image = system_image.compute_force(vortices[i], theta_bh_image);
			
			// Minus in the below equation since image vortices act opposite to real ones
			force_bh_x =  (force_bh_real[0] - force_bh_image[0]); 
			force_bh_y =  (force_bh_real[1] - force_bh_image[1]);
			
			// Saving the x and y components of the forces on vortex i due to all other vortices (real and image)
			force_bh[i] = {force_bh_x, force_bh_y}; // force_bh is a vector of vectors
		}
		
		// Force is zero otherwise
		else
		{ 
			force_bh[i] = {0, 0};
		}

	}
}

// Equations of motion - BH
void eom_bh(const vector<double> &f, vector<double> &dfdt , double t, vector<vector<double>> pinning_vector)
{	
	vector<double> dfdt_temporary(2*nv, 0.0);
	dfdt = dfdt_temporary;

	// Evaluating the terms of the EOM
	# pragma omp parallel for schedule(static, chunk_size)
	for (int i=0; i<nv; i++)
	{	
		// Don't do the math if the vortex doesn't exist
		if (ri2 >= R*R) 
		{
			dxidt = 0;
			dyidt = 0;
		}
		
		else
		{

		// Rotation
		dxidt =   (omega_c) * yi;
		dyidt = - (omega_c) * xi;

		// Vortex-vortex interactions using BH
		dxidt += force_bh[i][0];
		dyidt += force_bh[i][1];
						
		// Pinning
		// Only the pin closest to the vortex is considered to simplify calculations	
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

		dxidt += ( (pinning_vector[nx][ny])  * exp(- r2_pin_nearest/(2*pow(Xi,2))) * (yi - y_pin_nearest));
		dyidt += - ( (pinning_vector[nx][ny])  * exp(- r2_pin_nearest/(2*pow(Xi,2))) * (xi - x_pin_nearest));
		
		//dxidt += ( (pinning_vector[nx][ny])  * exp(- r2_pin_nearest/(2*pow(trap_separation/2,2))) * (yi - y_pin_nearest));
		//dyidt += - ( (pinning_vector[nx][ny])  * exp(- r2_pin_nearest/(2*pow(trap_separation/2,2))) * (xi - x_pin_nearest));

		// Dissipation
		double	vel_x = dxidt;
		double	vel_y = dyidt;
		
		dxidt = cos(phi) * vel_x + sin(phi) * vel_y; 
		dyidt = - sin(phi) * vel_x + cos(phi) * vel_y;
		}
	}
}

// Adaptive integration where the maximum step size is capped by del_t - BH
void adaptive_integrate_bh(vector<double> &f, vector<double> &f_image, vector<double> &K, vector<vector<double>> &pinning_vector, double &t)
{
	vector<double> f_err(2*nv, 0.0);	
	vector<double> f_temp = f;

	// Binding eom with required additional arguments to suit the needs of odeint
	// std::ref(var) passes the latest value of the variable
	auto bound_eom_bh = std::bind(eom_bh, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::ref(pinning_vector));
	// One step of integration with default timestep del_t
	quadtree_and_forces(f, f_image, K); // Updates force_bh
	rkck54.do_step(bound_eom_bh , f , t , del_t, f_err);

	// To make all entries of f_err positive
	for(int i=0; i<f_err.size(); i++)
	{
		if(f_err[i] < 0)
		{
			f_err[i] = (-1) * f_err[i];
		}
	}
	
	// Maximum error due to previous step of integration
	double maxerr = *max_element(f_err.begin(),f_err.end()); 

	// If error is beyond tolerance, correct stepsize and integrate
	if(maxerr >= error_tolerance)
	{	
		f = f_temp; // Rewrite old state
		
		double correction = pow((error_tolerance/maxerr), 0.2); // Correction factor for time step
				
		double low10 = pow(10,floor(log10(correction))); // To bring it to the nearest lower 10^x
		
		// Conditions for obtaining optimal correction factor
		if(correction/low10 < 2)
		{
			correction = 1 * low10;
		}
		
		else if( correction/low10 >= 2 && correction/low10 < 5)
		{
			correction = 2 * low10;
		}
		
		else if( correction/low10 >=5)
		{
			correction = 5 * low10;
		}
		
		// Setting the required size of a timestep and the number of such steps required
		double del_t_temp = correction * del_t;
		int n_steps = 1/correction;

		for(int i = 0; i<n_steps; i++)
		{
			// One step of integration with reduced timestep
//DEBUG		//quadtree_and_forces(f, f_image, K);
			rkck54.do_step(bound_eom_bh , f , t , del_t_temp, f_err);
			updates(f, f_image, K);
		}
	}

	// Else update image state and circulation
	else
	{
		updates(f, f_image, K);
	}

	// At the end of the routine, no matter what, the total time elapsed due to integration is del_t
	// The increment in time is performed in the calling block
}

// Exact procedure
// Equations of motion - Exact
void eom_exact(const vector<double> &f, vector<double> &dfdt , double t, vector<double> f_image, vector<vector<double>> pinning_vector)
{	
	vector<double> dfdt_temporary(2*nv, 0.0);
	dfdt = dfdt_temporary;

	// Evaluating the terms of the EOM
	# pragma omp parallel for
	for (int i=0; i<nv; i++)
	{	
		// Don't do the math if the vortex doesn't exist
		if (ri2 >= R*R) 
		{
			dxidt = 0;
			dyidt = 0;
		}
		
		else
		{

		// Rotation
		dxidt =   (omega_c) * yi;
		dyidt = - (omega_c) * xi;

		// Vortex-vortex interactions using exact O(N^2) calculations
		for (int j=0; j<nv; j++)
		{
			if (rj2 < R*R) // Do the math only if the other vortex doesn't exist
			{
				if (i!=j)
				{
					// Other vortices
					dxidt += - yij / rij2;
					dyidt +=   xij / rij2;
				}
				
				// Image vortices
				dxidt += yij_img / rij2_img;
				dyidt += -xij_img / rij2_img;
			}
		}
						
		// Pinning
		// Only the pin closest to the vortex is considered to simplify calculations
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

		dxidt += ( (pinning_vector[nx][ny])  * exp(- r2_pin_nearest/(2*pow(Xi,2))) * (yi - y_pin_nearest)); // Differential equations for x - Part 4 (Pinning)
		dyidt += - ( (pinning_vector[nx][ny])  * exp(- r2_pin_nearest/(2*pow(Xi,2))) * (xi - x_pin_nearest)); // Differential equations for y - Part 4 (Pinning)
		
		// Dissipation
		double	vel_x = dxidt;
		double	vel_y = dyidt;
		
		dxidt = cos(phi) * vel_x + sin(phi) * vel_y; 
		dyidt = - sin(phi) * vel_x + cos(phi) * vel_y;
		}
	}
}

// Adaptive integration where the maximum step size is capped by del_t - Exact
void adaptive_integrate_exact(vector<double> &f, vector<double> &f_image, vector<double> &K, vector<vector<double>> &pinning_vector, double &t)
{
	vector<double> f_err(2*nv, 0.0);	
	vector<double> f_temp = f;

	// Binding eom with required additional arguments to suit the needs of odeint
	// std::ref(var) passes the latest value of the variable
	auto bound_eom_exact = std::bind(eom_exact, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::ref(f_image), std::ref(pinning_vector));
	// One step of integration with default timestep del_t
	rkck54.do_step(bound_eom_exact , f , t , del_t, f_err);

	// To make all entries of f_err positive
	for(int i=0; i<f_err.size(); i++)
	{
		if(f_err[i] < 0)
		{
			f_err[i] = (-1) * f_err[i];
		}
	}
	
	// Maximum error due to previous step of integration
	double maxerr = *max_element(f_err.begin(),f_err.end()); 

	// If error is beyond tolerance, correct stepsize and integrate
	if(maxerr >= error_tolerance)
	{	
		f = f_temp; // Rewrite old state
		
		double correction = pow((error_tolerance/maxerr), 0.2); // Correction factor for time step
				
		double low10 = pow(10,floor(log10(correction))); // To bring it to the nearest lower 10^x
		
		// Conditions for obtaining optimal correction factor
		if(correction/low10 < 2)
		{
			correction = 1 * low10;
		}
		
		else if( correction/low10 >= 2 && correction/low10 < 5)
		{
			correction = 2 * low10;
		}
		
		else if( correction/low10 >=5)
		{
			correction = 5 * low10;
		}
		
		// Setting the required size of a timestep and the number of such steps required
		double del_t_temp = correction * del_t;
		int n_steps = 1/correction;

		for(int i = 0; i<n_steps; i++)
		{
			// One step of integration with reduced timestep
			rkck54.do_step(bound_eom_exact , f , t , del_t_temp, f_err);
			updates(f, f_image, K);
		}
	}

	// Else update image state and circulation
	else
	{
		updates(f, f_image, K);
	}

	// At the end of the routine, no matter what, the total time elapsed due to integration is del_t
	// The increment in time is performed in the calling block
}

