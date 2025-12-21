#include <vector>
#include <cmath>
#include <algorithm> // For sort and shuffle functions
#include <random>    // For random_device and mt19937
#include <utility> // For pair function
#include "utility.h"
#include "parameters.h"

using namespace std;

// For all triggers
// Initialization of trigger times
vector<double> triggers_init()
{
	vector<double> times_trig(number_triggers, 0);

	// Generate trigger times
	for (int i=0; i<number_triggers; i++)
	{
		times_trig[i] = randnumb01() * dynamics_runtime;
	}
	
	// Sort in ascending order
	sort(times_trig.begin(), times_trig.end()); // Needs 'algorithm' header

	// Ensure that the triggers are separated by at least the duration of the triggers
	for (int i=1; i<number_triggers; i++)
	{
		if (times_trig[i]-times_trig[i-1] < duration_triggers)
		{
			times_trig[i] += duration_triggers;
		}
	}

	times_trig.push_back(dynamics_runtime+5); // To ensure that the trigger is not activated for all times between the last random trigger time and the simulation runtime

	return times_trig;
}

// For stresswaves
// Reduce pinning strength across the star
pair<int, int> pin_reduce(vector<vector<double>> &pinning_vector)
{
	for (double x = -R; x <= R; x = x+a)
	{  	
		for (double y = -R; y <= R; y = y+a)
		{  
			double dist_pin = sqrt(pow(x,2)+pow(y,2)); 

			if (dist_pin<R)
			{
				int nx = round((R+x)/a);
				int ny = round((R+y)/a);

				pinning_vector[nx][ny] = stresswave_threshold * pinning_vector[nx][ny];
			}
		}
	}

	return make_pair(0,0);
}

// For sectorial
// Check if pinning site falls within triggered sector
int sector_check(double x, double y, int numb_sector)
{
  double pin_angle;

  if (x == 0)
  {
    x = 0.01; // To avoid any funny errors in atan2 function
  }
  if (y == 0)
  {
    y = 0.01; // To avoid any funny errors in atan2 function
  }
  if (y>0)
  {
    pin_angle = atan2(y,x) * 180 / pi;
  }
  if (y<0)
  {
    pin_angle = 360 + (atan2(y, x) * 180 / pi);
  }

  double angle_start =  numb_sector * (360/sector_count);
  double angle_end = angle_start + 360/sector_count;
  int check_value;

  if (pin_angle>angle_start && pin_angle<angle_end)
  {
    check_value = 1;
  }
  else
  {
    check_value = 0;
  }

  return check_value;
}

// For sectorial
// Turn off pinning for randomly chosen pinning sites in a randomly chosen sector
pair<int, int> pin_off(vector<vector<double>> &pinning_vector)
{
	int numboff = 0; // The number of pins switched off for a given trigger
	int sector_id = randnumb01() * sector_count; // To randomly choose a sector number between 0 and 7 - Each sector has an opening angle of 45 degrees

	for (double x = -R; x <= R; x = x+a)
	{  	
		for (double y = -R; y <= R; y = y+a)
		{  
			double dist_pin = sqrt(pow(x,2)+pow(y,2)); 

			if (dist_pin<R)
			{
				int nx = round((R+x)/a);
				int ny = round((R+y)/a);

				if (dist_pin < r_annulus)
				{
					if (sector_check(x, y, sector_id) == 1 && (sectorial_region == "INNER" || sectorial_region == "FULLSECTOR" ))
					{
						if (randnumb01() <= 0.5) // To unpin with a probability of half
						{
							pinning_vector[nx][ny] = 0;
							numboff = numboff + 1;
						}
					}
				}

				else if (dist_pin >= r_annulus)
				{

					if (sector_check(x, y, sector_id) == 1 && (sectorial_region == "OUTER" || sectorial_region == "FULLSECTOR" ))
					{
						if (randnumb01() <= 0.5) // To unpin with a probability of half
						{
							pinning_vector[nx][ny] = 0;
							numboff = numboff + 1;
						}
					}		
				}

			}

		}
	}

	return make_pair(numboff, sector_id);
}

// For trapwise -- Switch off the pinning within number_trapwise nummber of traps in the star
// Create a list of rounded coordinates to label traps
vector<vector<int>> list_of_traps()
{
	vector<vector<int>> trap_list;

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
					// Define coordinate
					vector<int> coordinate;
					coordinate = {int(round(x/trap_size)),int(round(y/trap_size))};

					// If the trap list is not empty
					if (trap_list.size() != 0)
					{
						// Search for the entry
						auto it = find(trap_list.begin(), trap_list.end(), coordinate);
						// If the entry is present
						if (it != trap_list.end())
						{
							// Do nothing
						}
						else
						{
							// Push to list the rounded coordinates of the traps 
							trap_list.push_back(coordinate);
						}
					}
					else
					{
							// Push to list the rounded coordinates of the traps 
							trap_list.push_back(coordinate);
					}
				}
				else
				{
					// Do nothing
				}
			}
		}
	}

	return trap_list;
}

// Switch off the appropriate number of traps 
pair<int, int> traps_off(vector<vector<double>> &pinning_vector)
{
	// Create a list of traps
	vector<vector<int>> trap_list = list_of_traps();

	// Restrict the list to only those traps within the correct radial range
	vector<vector<int>> trap_list_temp;
	
	for (int i = 0; i < trap_list.size(); i++)
	{
		vector<int> coordinate = trap_list[i];
		double trap_dist = sqrt((coordinate[0] * coordinate[0]) + (coordinate[1] * coordinate[1]));
		if(trap_dist > r_trapwise_in && trap_dist < r_trapwise_out)
		{
			trap_list_temp.push_back(coordinate);
		}
	}

	trap_list = trap_list_temp;

	// Shuffle list
	std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(trap_list.begin(), trap_list.end(), gen);

	// For the appropritate number of traps
	for (int i = 0; i < number_trapwise; i++)
	{
		// Return and remove the last element from the shuffled list
		vector<int> trap_coordinate = trap_list.back();
		trap_list.pop_back();

		// Set the pinning of the chosen trap to zero
		int trap_x = trap_coordinate[0];
		int trap_y = trap_coordinate[1];

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
					
					// And is within the chosen trap
					if (int(round(x/trap_size)) == trap_x && int(round(y/trap_size)) == trap_y)
					{
						pinning_vector[nx][ny] = 0;
					}
					else
					{
						// Do nothing
					}
				}
			}
		}
	}

	return make_pair(0,0);
}


// For all triggers
// Check for trigger at every timestep
double time_start_trig = 0;
pair<int, int> triggers_check(double t, vector<vector<double>> &pinning_vector, int &count_trig, int &state_trig, vector<double> &times_trig)
{
	pair<int, int> return_info;
	return_info = make_pair(0,0);

	if (trigger_type != "OFF")
	{
		// Copy pinning
		vector<vector<double>> pinning_vector_copy = pinning_vector;

		// Reduce pinning temprarily
		if (t > times_trig[count_trig] && state_trig == 0)
		{
			time_start_trig = t;
			state_trig = 1;

			if (trigger_type == "STRESSWAVE")
			{
				return_info = pin_reduce(pinning_vector);
			}
			else if (trigger_type == "SECTORIAL")
			{
				return_info = pin_off(pinning_vector);
			}
			else if (trigger_type == "TRAPWISE")
			{
				return_info = traps_off(pinning_vector);
			}	
		}

		// Restore original pinning
		if (state_trig == 1)
		{
			if (t > time_start_trig + duration_triggers)
			{
				state_trig = 0;
				pinning_vector = pinning_vector_copy;
				count_trig = count_trig + 1;
			}
		}
	}

	// Return information of how many sites were switched off and in which sector
	// Applicable for sectorial
	return return_info;
}
