#pragma once

#include <boost/math/constants/constants.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/quadrature/gauss.hpp>
#include <boost/math/quadrature/tanh_sinh.hpp>
#include <boost/math/quadrature/trapezoidal.hpp>

#include <boost/math/tools/roots.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

// compute bounce action for a scalar field theory in 3d at finite temperature T
class BounceSolverS3
{
    using Potential = std::function<double(double)>;      // V(phi)
    using PotentialPrime = std::function<double(double)>; // dV/dphi

private:
    double phi_metamin_;
    double phi0_;
    Potential V_;
    PotentialPrime dV_;

    double qd_;
    double phiT_;

    double calculate_qd_();                 // quartic coefficient
    double V_t(double phi) const;           // tunneling potential
    double V_t_prime(double phi) const;     // first derivative of tunneling potential
    double find_top() const;                // find the top of the potential

public:
    BounceSolverS3(double phi_metamin, double phi0, Potential V, PotentialPrime dV)
    {
        if (phi_metamin >= phi0)
        {
            throw std::invalid_argument("phi_metamin must be less than phi0");
        }
        phi_metamin_ = phi_metamin;
        phi0_ = phi0;
        V_ = V;
        dV_ = dV;
        phiT_ = find_top();
        qd_ = calculate_qd_();
    }

    double bounce_action() const;
};

class BounceSolverS4
{

    using Potential = std::function<double(double)>;      // V(phi,T)
    using PotentialPrime = std::function<double(double)>; // dV/dphi

private:
    double phi_metamin_;
    double phi0_;
    double qd_;
    double phiT_;

    Potential V_;
    PotentialPrime dV_;

    double calculate_qd_() const;       // quartic coefficient
    double V_t(double phi) const;       // tunneling potential
    double V_t_prime(double phi) const; // first derivative of tunneling potential
    double find_top() const;

public:
    BounceSolverS4(double phi_metamin, double phi0, Potential V, PotentialPrime dV)
    {
        if (phi_metamin >= phi0)
        {
            throw std::invalid_argument("phi_metamin must be less than phi0");
        }
        phi_metamin_ = phi_metamin;
        phi0_ = phi0;
        V_ = V;
        dV_ = dV;
        phiT_ = find_top();
        qd_ = calculate_qd_();
    }

    double bounce_action() const;
};