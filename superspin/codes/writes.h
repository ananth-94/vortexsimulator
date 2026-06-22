#ifndef WRITES_H
#define WRITES_H

void write_progress(double t, int &progress, const double runtime, const string prefix);
void write_info_start(std::vector<double> &f);
void write_data_post_stabilization(std::vector<double> &tvalues, std::vector<double> &f, std::vector<double> &Hvalues);
void write_info_mid(std::vector<double> &f, double kiss);
void write_info_end(std::vector<double> &f);
void write_data_post_dynamics(std::vector<double> &t_vector, std::vector<double> &t_vector_write, std::vector<std::vector<double>> &f_vector, std::vector<double> &omega_c_vector, std::vector<double> &omega_s_vector,
	std::vector<int> &number_off_vector, std::vector<int> &sector_id_vector, std::vector<double> &trigger_times, std::vector<int> &out_vector, std::vector<int> &unpinned_vector, int run_id);

#endif
