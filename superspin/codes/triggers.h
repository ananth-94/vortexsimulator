#ifndef TRIGGERS_H
#define TRIGGERS_H

std::vector<double> triggers_init();
std::pair<int, int> triggers_check(double t, std::vector<std::vector<double>> &pinning_vector, int &count_trig, int &state_trig, std::vector<double> &times_trig);

#endif
