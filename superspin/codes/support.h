#ifndef SUPPORT_H
#define SUPPORT_H

void state_to_vortices(std::vector<double> &f, std::vector<double> &K, std::vector<std::vector<double>> &vortices);
double H_calculate(vector<double> &f);
void updates(std::vector<double> &f, std::vector<double> &f_image, std::vector<double> &K);
double superfluid_rotation(std::vector<double> &f, std::vector<double> &K, double kiss);
int out_counter(std::vector<double> &f);
int unpinned_counter(std::vector<double> &f);

#endif
