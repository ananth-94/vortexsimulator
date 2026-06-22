#ifndef INTEGRATION_H
#define INTEGRATION_H

void adaptive_integrate_bh(std::vector<double> &f, std::vector<double> &f_image, std::vector<double> &K, std::vector<std::vector<double>> &pinning_vector, double &t);
void adaptive_integrate_exact(std::vector<double> &f, std::vector<double> &f_image, std::vector<double> &K, std::vector<std::vector<double>> &pinning_vector, double &t);

#endif
