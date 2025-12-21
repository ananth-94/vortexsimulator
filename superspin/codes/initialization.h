#ifndef INITIALIZATION_H
#define INITIALIZATION_H

std::vector<double> initialize_state_real(std::string condition);
std::vector<double> initialize_state_image(vector<double> &f);
void create_output_files(int run_id);
double kiss_fix(std::vector<double> &f);

#endif
