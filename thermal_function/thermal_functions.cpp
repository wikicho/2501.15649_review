#include "thermal_functions.h"


// Low-temperature expansion y^2 << 1

double jb_low(double y2, int N) {
    double aB = std::exp(1.5 - 2.0 * EULER_GAMMA + 2.0 * std::log(4.0 * PI));
    double series = 0.0;
    if (std::abs(y2) <= 1e-6) {
        return -2.16465; // approximate constant
    }
    for (int l = 1; l <= N; ++l) {
        double term = std::pow(-1.0, l)
            * boost::math::zeta(2.0 * l + 1.0)
            / std::tgamma(l + 2)
            * std::tgamma(l + 0.5)
            * std::pow(y2 / (4.0 * PI * PI), l + 2);
        series += term;
    }
    double result;
    if (y2 < 0.0) {
        result = -std::pow(PI,4)/45.0
               + PI*PI/12.0 * y2
               - (1.0/32.0) * y2*y2 * std::log(-y2 / aB)
               - 2.0 * std::pow(PI,3.5) * series;
    } else {
        result = -std::pow(PI,4)/45.0
               + PI*PI/12.0 * y2
               - PI/6.0 * std::pow(y2,1.5)
               - (1.0/32.0) * y2*y2 * std::log(y2 / aB)
               - 2.0 * std::pow(PI,3.5) * series;
    }
    return result;
}

double jf_low(double y2, int N = 10)
{
    return 0;
}

// Derivative

double jb_prime_low(double y2, int N) {
    double aB = std::exp(1.5 - 2.0 * EULER_GAMMA + 2.0 * std::log(4.0 * PI));
    double series = 0.0;
    if (std::abs(y2) <= 1e-6) {
        return PI*PI/12.0;
    }
    for (int l = 1; l <= N; ++l) {
        double term = std::pow(-1.0, l)
            * boost::math::zeta(2.0 * l + 1.0)
            / std::tgamma(l + 2)
            * std::tgamma(l + 0.5)
            * (l + 2)
            * std::pow(y2, l + 1)
            / std::pow(4.0 * PI*PI, l + 2);
        series += term;
    }
    double result;
    if (y2 < 0.0) {
        result = PI*PI/12.0 - y2/32.0 - (1.0/16.0) * y2 * std::log(-y2/aB);
    } else {
        result = PI*PI/12.0 - PI/4.0 * std::sqrt(y2)
               - y2/32.0 - (1.0/16.0) * y2 * std::log(y2/aB);
    }
    result -= 2.0 * std::pow(PI,3.5) * series;
    return result;
}

// Second derivative

double jb_2prime_low(double y2, int N) {
    double aB = std::exp(1.5 - 2.0 * EULER_GAMMA + 2.0 * std::log(4.0 * PI));
    double series = 0.0;
    for (int l = 1; l <= N; ++l) {
        double inner = (1 + l) * (2 + l) * std::pow(y2, l);
        double term = std::pow(-1.0, l)
            * std::pow(4.0, -2.0 - l)
            * std::pow(PI, -2.0 * (2 + l))
            * inner
            * std::tgamma(l + 0.5)
            * boost::math::zeta(1.0 + 2.0 * l)
            / std::tgamma(l + 2);
        series += term;
    }
    double result;
    if (y2 < 0.0) {
        result = 3.0/32.0 - (1.0/16.0) * std::log(-y2/aB);
    } else {
        result = 3.0/32.0 - 0.125 * PI / std::sqrt(y2)
               - (1.0/16.0) * std::log(y2/aB);
    }
    result -= 2.0 * std::pow(PI,3.5) * series;
    return result;
}

// High-temperature expansion y^2 >> 1

double jb_high(double y2, int n) {
    double series = 0.0;
    if (y2 < 0.0) {
        for (int k = 1; k <= n; ++k) {
            series += 0.5 * PI * boost::math::cyl_neumann(2, k * std::sqrt(-y2)) / (k*k);
        }
    } else {
        for (int k = 1; k <= n; ++k) {
            series += boost::math::cyl_bessel_k(2, k * std::sqrt(y2)) / (k*k);
        }
    }
    return -y2 * series;
}

double jb_prime_high(double y2, int n) {
    double series = 0.0;
    if (y2 < 0.0) {
        for (int k = 1; k <= n; ++k) {
            series += -0.25 * PI * std::sqrt(-y2)
                   * boost::math::cyl_neumann(1, k * std::sqrt(-y2))
                   / k;
        }
    } else {
        for (int k = 1; k <= n; ++k) {
            series += 0.5 * std::sqrt(y2)
                   * boost::math::cyl_bessel_k(1, k * std::sqrt(y2))
                   / k;
        }
    }
    return series;
}

double jb_2prime_high(double y2, int n) {
    double series = 0.0;
    if (y2 < 0.0) {
        for (int k = 1; k <= n; ++k) {
            series += 0.125 * PI
                   * boost::math::cyl_neumann(0, k * std::sqrt(-y2));
        }
    } else {
        for (int k = 1; k <= n; ++k) {
            series += -0.25 * boost::math::cyl_bessel_k(0, k * std::sqrt(y2));
        }
    }
    return series;
}

// Select appropriate regime

double jb(double y2, int N) {
    if (y2 < -0.279909) return jb_high(y2, N);
    else if (y2 < 0.25905) return jb_low(y2, N);
    else return jb_high(y2, N);
}

double jb_prime(double y2, int N) {
    if (y2 < -0.155735) return jb_prime_high(y2, N);
    else if (y2 < 0.268101) return jb_prime_low(y2, N);
    else return jb_prime_high(y2, N);
}

double jb_2prime(double y2, int N) {
    if (y2 < -0.0724474) return jb_2prime_high(y2, N);
    else if (y2 < 0.0285167) return jb_2prime_low(y2, N);
    else return jb_2prime_high(y2, N);
}

// Finite-T effective potential corrections

double V1_finiteT(double T, double m2, int N) {
    double y2 = m2 / (T*T);
    return std::pow(T,4) * jb(y2, N) / (2.0 * PI*PI);
}

// Derivative of finite-T potential

double V1_prime_finiteT(double T, double m2, int N) {
    double y2 = m2 / (T*T);
    return 2.0 * std::pow(T,3) * jb(y2, N) / (PI*PI)
         - std::pow(T,3) / (PI*PI) * jb_prime(y2, N) * y2;
}

// Second derivative of finite-T potential

double V1_2prime_finiteT(double T, double m2, int N) {
    double y2 = m2 / (T*T);
    double aB = std::exp(1.5 - 2.0 * EULER_GAMMA + 2.0 * std::log(4.0*PI));
    if (std::abs(y2) < 0.01) {
        double jb2;
        if (y2 < 0.0) {
            jb2 = 0.03125 * y2*y2 * (3.0 - 2.0 * std::log(-y2 / aB));
        } else {
            jb2 = 0.03125 * y2*y2 * (3.0 - 2.0 * std::log(y2 / aB))
                - 0.125 * PI * std::sqrt(y2) * y2;
        }
        return 6.0 * std::pow(T,2) / (PI*PI) * jb(y2, N)
             - 5.0 * std::pow(T,2) / (PI*PI) * jb_prime(y2, N) * y2
             + 2.0 * std::pow(T,2) / (PI*PI) * jb2;
    } else {
        return 6.0 * std::pow(T,2) / (PI*PI) * jb(y2, N)
             - 5.0 * std::pow(T,2) / (PI*PI) * jb_prime(y2, N) * y2
             + 2.0 * std::pow(T,2) / (PI*PI) * jb_2prime(y2, N) * y2*y2;
    }
}

