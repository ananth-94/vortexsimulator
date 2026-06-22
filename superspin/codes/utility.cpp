// Generate a random number between 0 and 1
#include <random>
#include <ctime>

using namespace std;

// Derive a number between 0 and 1 from a uniform distribution
double randnumb01() 
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dist(0, 1);	
	
	return dist(gen);
 } 

// Current computer date and time calculation
char* calc_time()
{
	// Current date and time retrieved from the computer
	time_t now = time(0);
	
	// Convert now to string form
	char* dt = ctime(&now);
	
	return dt;
}
