#ifndef BOUNCE_SOLVER_DIM4_H
#define BOUNCE_SOLVER_DIM4_H

#include <boost/math/tools/roots.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

class BounceSolverS4
{
    using Potential = std::function<double(double)>;      // V(phi,T)
    using PotentialPrime = std::function<double(double)>; // dV/dphi
private:
    double phi_metamin_;
    double phi_true_;
    double phiT_;
    double phi_cross_;
    double action_;
    double phi0_min_;

    Potential V_;
    PotentialPrime dV_;

    std::vector<double> phi_vec_;
    std::vector<double> V_vec_;
    std::vector<double> dV_vec_;
    std::vector<double> actions_vec_;
    std::vector<double> Vt_vec_;
    std::vector<double> dVt_vec_;


    double calculate_qd_(double phi0_) const;       // quartic coefficient
    double V_t(double phi, double phi0_, double qd_) const;       // tunneling potential
    double V_t_prime(double phi, double phi0_, double qd_) const; // first derivative of tunneling potential
    double find_top() const;   // find maximum of
    double find_cross() const; // find V(0) = V(phi)
    double calculate_action(double phi0_) const;


public:
    BounceSolverS4(double phi_metamin, double phi_true, Potential V, PotentialPrime dV, double dphi)
    {
        phi_metamin_ = phi_metamin;
        phi_true_ = phi_true;
        V_ = V;
        dV_ = dV;

        for (double phi = phi_metamin_; phi < phi_true_; phi += dphi ) {
            phi_vec_.push_back(phi);
            V_vec_.push_back(V(phi));
            dV_vec_.push_back(dphi);
        }

        phi_cross_ = find_cross();

        phiT_ = find_top();

        for (double phi = phi_cross_; phi < 2 * phi_cross_; phi += dphi) {
            actions_vec_.push_back(calculate_action(phi));
        }

        // find minimum
        auto iter_action = std::min_element(actions_vec_.begin(), actions_vec_.end());
        action_ = *iter_action;
        std::size_t idx = std::distance(actions_vec_.begin(), iter_action);
        phi0_min_ = actions_vec_[idx - 1];
    }

    double bounce_action() const;
};


#endif //SUPERCOOL_FOPT_BOUNCE_SOLVER_DIM4_H
