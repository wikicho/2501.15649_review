// DarkAbelianHiggs.h
#ifndef DARK_ABELIAN_HIGGS_H
#define DARK_ABELIAN_HIGGS_H

#include <vector>
#include <map>
#include <cmath>
#include <stdexcept>
#include <functional>
#include <memory>
#include <string>
#include <mutex>
#include <shared_mutex>
#include <omp.h>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/differentiation/finite_difference.hpp>
#include "../standard_cosmology/cosmology.h"
#include "../thermal_function/thermal_functions.h"
#include "../bounce_solver/bounce_solver_dim3.h"
#include "../util/csv_importer.h"

class DarkAbelianHiggs
{
public:
    DarkAbelianHiggs(double v0,
                     double lambda_phi,
                     double g_d,
                     const std::string &dof_file = "data/sm_eff_data.csv");

    // Masses and derivatives
    double mPhi2(double phi) const;
    double mGB2(double phi) const;
    double mZd2(double phi) const;
    double dmPhi2_dphi(double phi) const;
    double dmGB2_dphi(double phi) const;
    double dmZd2_dphi(double phi) const;
    double d2mPhi2_dphi2(double phi) const;
    double d2mGB2_dphi2(double phi) const;
    double d2mZd2_dphi2(double phi) const;

    // Thermal masses for daisy resummation
    double thermal_mass_phi(double T) const;
    double thermal_mass_GB(double T) const;
    double thermal_mass_ZD(double T) const;

    // Potentials
    double V0(double phi) const;
    double V_CW(double phi) const;
    double V1T(double phi, double T) const;
    double V1_Daisy(double phi, double T) const;
    double Vtot(double phi, double T, bool include_daisy = true) const;
    double Veff(double phi, double T) const;

    // Temperature derivatives
    double dV1T_dT(double phi, double T) const;
    double d2V1T_dT2(double phi, double T) const;

    // Field derivatives
    double dV_dphi(double phi, double T) const;
    double d2V_dphi2(double phi, double T) const;

    // Phase transition parameters
    double cs2(double T) const;
    double w_density(double phi, double T) const;
    double pressure(double phi, double T) const;
    double energy_density(double phi, double T) const;
    double pseudo_trace(double phi, double T) const;
    double alpha(const double T) const;

    // time-temperature relation
    double mathcal_J(double T) const;

    double findTcrit(double T_max, double T_min, double dT, double tol) const; // Critical temperatures
    
    double findTn(double T_min, double dT, double tol) const;
    double findTp(double T_min) const;
    double findTcomplete(double T_min) const;

    double T_crit() const; // Getter for critical temperature

    // Cosmological rates & scales
    double nucleation_rate(double T) const;
    double hubble_rate(double T) const;
    double scale_factor(double T_f, double T_i) const;
    double volume_bubble(double T_f, double T_i) const;
    double false_vacuum_fraction(double T) const;
    double false_vacuum_fraction_integrand(double T, double Tp) const;

    // Derived kinetics
    double radius_init(double T_perc) const;
    double radius_sep(double T_perc) const;
    double gamma_star(double T_perc) const;

    double gamma_eq(double T_perc) const;
    double v_w(double T_perc) const;
    double kappa_collision(double T_perc) const;
    double kappa_soundwave(double T_perc) const;
    double alpha_eq(double T_perc) const;
    double alpha_inf(double T_perc) const;
    double alpha_eff(double T_perc) const;
    double rms_fluid_velocity(double T_perc) const;
    double omega_soundwave(double f, double T_perc) const;

    double s3(double T) const {
        return bounce_action_dim3(T);
    }
    double s4(double T) const {
        return bounce_action_dim4(T);
    }

private:
    // Model parameters
    double v0, lambda_phi, g_d, mu_eff;
    double mPhi2_vev, mGB2_vev, mZd2_vev;

    // Bounce solver


    // Standard cosmology
    std::unique_ptr<Universe> sm_cosmology;

    // Degrees of freedom and c-factors
    std::vector<double> dof, c_factors;

    // Critical and nucleation temperatures
    double T_crit_, T_nuc, T_perc, T_comp;

    // Action memorization
    mutable std::map<double,double> action_dim3_cache_;
    mutable std::map<double,double> action_dim4_cache_;
    mutable std::shared_mutex action_dim3_cache_mutex_;
    mutable std::shared_mutex action_dim4_cache_mutex_;

    // memorization of bounce actions
    double bounce_action_dim3(double T) const;
    double bounce_action_dim4(double T) const;
    double compute_bounce_action_dim3(double T) const;
    double compute_bounce_action_dim4(double T) const;
    std::vector<double> bounce_action_bulk_dim3(const std::vector<double> &Ts) const;
    std::vector<double> bounce_action_bulk_dim4(const std::vector<double> &Ts) const;
};

#endif // DARK_ABELIAN_HIGGS_H