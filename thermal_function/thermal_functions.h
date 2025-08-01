#ifndef THERMAL_FUNCTIONS_H
#define THERMAL_FUNCTIONS_H

#include <cmath>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/zeta.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/bessel.hpp>

static const double PI = boost::math::constants::pi<double>();
static const double EULER_GAMMA = boost::math::constants::euler<double>();

// Low-temperature expansion y^2 << 1

double jb_low(double y2, int N = 10);
double jb_prime_low(double y2, int N = 10);
double jb_2prime_low(double y2, int N = 10);

// High-temperature expansion y^2 >> 1

double jb_high(double y2, int N = 10);
double jb_prime_high(double y2, int N = 10);
double jb_2prime_high(double y2, int N = 10);

// Select appropriate regime

double jb(double y2, int N = 10);
double jb_prime(double y2, int N = 10);
double jb_2prime(double y2, int N = 10);

// Finite-T effective potential corrections

double V1_finiteT(double T, double m2, int N = 10);
double V1_prime_finiteT(double T, double m2, int N = 10);
double V1_2prime_finiteT(double T, double m2, int N = 10);

#endif // THERMAL_FUNCTIONS_H