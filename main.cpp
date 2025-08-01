#include <iostream>
#include <vector>
#include "./potential/dark_abelian_higgs.h"

int main(int argc, char *argv[]) {
    DarkAbelianHiggs benchmark_2(1.0, 6e-3, 0.75);


    std::vector<double> T_values;
    std::vector<double> nucleation_rates;

    double T_min = 0.01;
    double T_max = benchmark_2.T_crit();

    double dT = (T_max - T_min) / 100; // 100 steps

    for (int i = 0; i < 100; ++i) {
        double T = T_min + i * dT;
        if (T > T_max)
            break; // Stop if T exceeds maximum temperature
        T_values.push_back(T);
    }

    for (auto T: T_values) {
        std::printf("T = %.4f, s3/T = %.4f, s4 = %.4f\n", T, benchmark_2.s3(T)/T, benchmark_2.s4(T));
    }


    return 0;
}
