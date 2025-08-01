#include "dark_abelian_higgs.h"

constexpr double Mpl = 2.435323204e18;

DarkAbelianHiggs::DarkAbelianHiggs(double v0_, double lambda_phi_, double g_d_, const std::string &dof_file)
    : v0(v0_), lambda_phi(lambda_phi_), g_d(g_d_), mu_eff(std::sqrt(lambda_phi) * v0), dof{1.0, 1.0, 3.0},
      c_factors{1.5, 1.5, 1.5} {
    mPhi2_vev = -mu_eff * mu_eff + 3 * lambda_phi * v0 * v0;
    mGB2_vev = -mu_eff * mu_eff + lambda_phi * v0 * v0;
    mZd2_vev = g_d * g_d * v0 * v0;

    sm_cosmology = std::make_unique<Universe>();

    // find the critical temperature
    T_crit_ = findTcrit(1, 1e-3, 1e-6, 1e-9);
    if (std::isnan(T_crit_) || std::isinf(T_crit_)) {
        std::cerr << "Error: Critical temperature is not defined. Check the parameters." << std::endl;
        exit(1);
    }

    double T_min = 10e-3;
    const double dT = 1e-4;
    const unsigned int n_steps = static_cast<unsigned int>((T_crit_ - T_min) / dT);

    // action memorization to avoid recomputation

    // Temperature values for action memorization
    std::vector<double> T_action;

    for (unsigned int i = 0; i < n_steps; ++i) {
        double T = T_min + i * dT;
        if (T > T_crit_)
            break; // Stop if T exceeds critical temperature
        T_action.push_back(T);
    }

    bounce_action_bulk_dim3(T_action);
    bounce_action_bulk_dim4(T_action);
}

double DarkAbelianHiggs::findTcrit(double T_max, double T_min, double dT, double tol) const {
    // Define the function whose root (zero) we seek:
    auto f = [this](double T) {
        return Veff(0.0, T) - Veff(v0, T);
    };

    // Evaluate at the endpoints to bracket the root:
    double f_low = f(T_min);
    double f_high = f(T_max);
    if (f_low * f_high > 0.0) {
        throw std::runtime_error(
            "findTcrit: function has same sign at T_min and T_max — no guaranteed root");
    }

    // Build a Boost bit-precision tolerance (here ~1e-9 relative)
    auto boost_tol = boost::math::tools::eps_tolerance<double>(std::numeric_limits<double>::digits - 2);
    boost::uintmax_t max_iter = 50;

    // Run TOMS748
    auto sol = boost::math::tools::toms748_solve(
        f,
        T_min, T_max,
        f_low, f_high,
        boost_tol,
        max_iter);

    // Return midpoint of the final bracket
    return 0.5 * (sol.first + sol.second);
}


// Mass and derivative implementations
double DarkAbelianHiggs::mPhi2(double phi) const {
    return -mu_eff * mu_eff + 3 * lambda_phi * phi * phi;
}

double DarkAbelianHiggs::mGB2(double phi) const {
    return -mu_eff * mu_eff + lambda_phi * phi * phi;
}

double DarkAbelianHiggs::mZd2(double phi) const {
    return g_d * g_d * phi * phi;
}

double DarkAbelianHiggs::dmPhi2_dphi(double phi) const {
    return 6 * lambda_phi * phi;
}

double DarkAbelianHiggs::dmGB2_dphi(double phi) const {
    return 2 * lambda_phi * phi;
}

double DarkAbelianHiggs::dmZd2_dphi(double phi) const {
    return 2 * g_d * g_d * phi;
}

double DarkAbelianHiggs::d2mPhi2_dphi2(double) const {
    return 6 * lambda_phi;
}

double DarkAbelianHiggs::d2mGB2_dphi2(double) const {
    return 2 * lambda_phi;
}

double DarkAbelianHiggs::d2mZd2_dphi2(double) const {
    return 2 * g_d * g_d;
}

// Thermal masses
double DarkAbelianHiggs::thermal_mass_phi(double T) const {
    return 0.5 * (lambda_phi + 0.5 * g_d * g_d) * T * T;
}

double DarkAbelianHiggs::thermal_mass_GB(double T) const {
    return thermal_mass_phi(T);
}

double DarkAbelianHiggs::thermal_mass_ZD(double T) const {
    return g_d * g_d * T * T / 3.0;
}

// Potentials
double DarkAbelianHiggs::V0(double phi) const {
    return -0.5 * mu_eff * mu_eff * phi * phi + 0.25 * lambda_phi * phi * phi * phi * phi;
}

double DarkAbelianHiggs::V_CW(double phi) const {
    double a = dof[0] * (std::pow(mPhi2(phi), 2) * (std::log(std::abs(mPhi2(phi) / mPhi2_vev) + 1e-150) - c_factors[0])
                         + 2 * mPhi2_vev * mPhi2(phi));
    double b = dof[1] * std::pow(mGB2(phi), 2) * (std::log(std::abs(mGB2(phi) / mPhi2_vev) + 1e-150) - c_factors[1]);
    double c = dof[2] * (std::pow(mZd2(phi), 2) * (std::log(std::abs(mZd2(phi) / mZd2_vev) + 1e-150) - c_factors[2]) + 2
                         * mZd2_vev * mZd2(phi));
    return (a + b + c) / (64 * M_PI * M_PI);
}

double DarkAbelianHiggs::V1T(double phi, double T) const {
    return dof[0] * V1_finiteT(T, mPhi2(phi)) + dof[1] * V1_finiteT(T, mGB2(phi)) + dof[2] * V1_finiteT(T, mZd2(phi));
}

double F(double m2, double T) {
    if (m2 < 0.0)
        return 0.0; // our prescription
    double x = std::sqrt(m2) / T;

    if (x < 1e-3) {
        return 1.0;
    } else {
        return 0.5 * x * x * boost::math::cyl_bessel_k(2, x);
    }
}

double DarkAbelianHiggs::V1_Daisy(double phi, double T) const {
    double m2p = mPhi2(phi) + thermal_mass_phi(T) * F(std::max(0.0, mPhi2(phi)), T);
    double m2g = mGB2(phi) + thermal_mass_GB(T) * F(std::max(0.0, mGB2(phi)), T);
    double m2z = mZd2(phi) + thermal_mass_ZD(T) * F(std::max(0.0, mZd2(phi)), T);

    double front_factor = -T / (12 * M_PI);

    double con = 0.0;

    double phi_contrib = std::pow(std::max(0.0, mPhi2(phi)), 1.5) - std::pow(std::max(0.0, m2p), 1.5);
    double gb_contrib = std::pow(std::max(0.0, mGB2(phi)), 1.5) - std::pow(std::max(0.0, m2g), 1.5);
    double zd_contrib = std::pow(std::max(0.0, mZd2(phi)), 1.5) - std::pow(std::max(0.0, m2z), 1.5);
    con = phi_contrib + gb_contrib + zd_contrib;

    return front_factor * con;
}

double DarkAbelianHiggs::Vtot(double phi, double T, bool include_daisy) const {
    double y = V0(phi) + V_CW(phi);
    if (T > 0)
        y += V1T(phi, T);
    if (include_daisy && T > 0)
        y += V1_Daisy(phi, T);
    return y;
}

double DarkAbelianHiggs::Veff(double phi, double T) const {
    return Vtot(phi, T, true) - Vtot(0.0, T, true);
}

double DarkAbelianHiggs::dV_dphi(double phi, double T) const {
    // Tree-level derivative
    double dV_tree = -mu_eff * mu_eff * phi + lambda_phi * phi * phi * phi;

    // Coleman-Weinberg one-loop derivative
    double loop = 0.0;
    // dark Higgs contribution
    loop += dof[0] * 2 * (std::log(std::abs(mPhi2(phi) / mPhi2_vev) + 1e-150) * mPhi2(phi) * dmPhi2_dphi(phi) +
                          mPhi2_vev * dmPhi2_dphi(phi) - mPhi2(phi) * dmPhi2_dphi(phi));
    // Goldstone boson
    loop += dof[1] * 2 * (std::log(std::abs(mGB2(phi) / mPhi2_vev) + 1e-150) * mGB2(phi) * dmGB2_dphi(phi) - mGB2(phi) *
                          dmGB2_dphi(phi));
    // dark gauge boson
    loop += dof[2] * 2 * (std::log(std::abs(mZd2(phi) / mZd2_vev) + 1e-150) * mZd2(phi) * dmZd2_dphi(phi) + mZd2_vev *
                          dmZd2_dphi(phi) - mZd2(phi) * dmZd2_dphi(phi));
    loop /= (64 * M_PI * M_PI);

    if (T == 0.0) {
        return dV_tree + loop;
    }

    // Thermal contributions (finite T derivative)
    double thermal = 0.0;
    thermal += dof[0] * std::pow(T, 4) * jb_prime(mPhi2(phi) / (T * T)) * 0.5 / (M_PI * M_PI) * dmPhi2_dphi(phi) / (T * T);
    thermal += dof[1] * std::pow(T, 4) * jb_prime(mGB2(phi) / (T * T)) * 0.5 / (M_PI * M_PI) * dmGB2_dphi(phi) / (T * T);
    thermal += dof[2] * 3 * std::pow(T, 4) * jb_prime(mZd2(phi) / (T * T)) * 0.5 / (M_PI * M_PI) * dmZd2_dphi(phi) / (
        T * T);

    // daisy resummation contributions
    double daisy_contribution = 0.0;
    auto daisy_constT = [this, T](double phi) {
        return this->V1_Daisy(phi, T);
    };

    daisy_contribution = boost::math::differentiation::finite_difference_derivative(daisy_constT, phi);

    return dV_tree + loop + thermal + daisy_contribution;

}

double DarkAbelianHiggs::d2V_dphi2(double phi, double T) const {
    // Tree-level second derivative
    double tree = -mu_eff * mu_eff + 3 * lambda_phi * phi * phi;
    // One-loop zero-T piece
    double loop = 0.0;
    // dark Higgs
    loop += dof[0] * 2 * (std::log(std::abs(mPhi2(phi) / mPhi2_vev) + 1e-150) * dmPhi2_dphi(phi) * dmPhi2_dphi(phi) +
                          std::log(std::abs(mPhi2(phi) / mPhi2_vev) + 1e-100) * mPhi2(phi) * d2mPhi2_dphi2(phi) -
                          mPhi2(phi) * d2mPhi2_dphi2(phi) + mPhi2_vev * d2mPhi2_dphi2(phi));
    // Goldstone boson
    loop += dof[1] * 2 * (std::log(std::abs(mGB2(phi) / mPhi2_vev) + 1e-150) * dmGB2_dphi(phi) * dmGB2_dphi(phi) +
                          std::log(std::abs(mGB2(phi) / mPhi2_vev) + 1e-100) * mGB2(phi) * d2mGB2_dphi2(phi) - mGB2(phi)
                          * d2mGB2_dphi2(phi));
    // Dark gauge boson
    loop += dof[2] * 2 * (std::log(std::abs(mZd2(phi) / mZd2_vev) + 1e-150) * dmZd2_dphi(phi) * dmZd2_dphi(phi) +
                          std::log(std::abs(mZd2(phi) / mZd2_vev) + 1e-100) * mZd2(phi) * d2mZd2_dphi2(phi) - mZd2(phi)
                          * d2mZd2_dphi2(phi) + mZd2_vev * d2mZd2_dphi2(phi));
    loop /= (64 * M_PI * M_PI);
    if (T == 0.0) {
        return tree + loop;
    }
    // Finite-T contributions
    double thermal = 0.0;
    thermal += dof[0] * jb_2prime(mPhi2(phi) / (T * T)) * 0.5 / (M_PI * M_PI) * dmPhi2_dphi(phi) * dmPhi2_dphi(phi);
    thermal += dof[1] * jb_2prime(mGB2(phi) / (T * T)) * 0.5 / (M_PI * M_PI) * dmGB2_dphi(phi) * dmGB2_dphi(phi);
    thermal += dof[2] * jb_2prime(mZd2(phi) / (T * T)) * 0.5 / (M_PI * M_PI) * dmZd2_dphi(phi) * dmZd2_dphi(phi);
    thermal += dof[0] * T * T * jb_prime(mPhi2(phi) / (T * T)) * 0.5 / (M_PI * M_PI) * d2mPhi2_dphi2(phi);
    thermal += dof[1] * T * T * jb_prime(mGB2(phi) / (T * T)) * 0.5 / (M_PI * M_PI) * d2mGB2_dphi2(phi);
    thermal += dof[2] * T * T * jb_prime(mZd2(phi) / (T * T)) * 0.5 / (M_PI * M_PI) * d2mZd2_dphi2(phi);

    // daisy resummation contributions
    double daisy_contribution = 0.0;
    auto daisy_constT = [this, T](double phi) {
        return this->V1_Daisy(phi, T);
    };

    daisy_contribution = boost::math::differentiation::finite_difference_derivative<decltype(daisy_constT), double, 2>(
        daisy_constT, phi);

    return tree + loop + thermal + daisy_contribution;
}

double DarkAbelianHiggs::dV1T_dT(double phi, double T) const {
    double thermal_contribution = dof[0] * V1_prime_finiteT(T, mPhi2(phi)) + dof[1] * V1_prime_finiteT(T, mGB2(phi)) +
                                  dof[2] * V1_prime_finiteT(T, mZd2(phi));


    // daisy resummation contributions
    double daisy_contribution = 0.0;
    auto daisy_const_phi = [this, phi](double T) {
        return this->V1_Daisy(phi, T);
    };

    daisy_contribution = boost::math::differentiation::finite_difference_derivative(daisy_const_phi, T);

    return thermal_contribution + daisy_contribution;
}

double DarkAbelianHiggs::d2V1T_dT2(double phi, double T) const {
    double thermal_contribution = dof[0] * V1_2prime_finiteT(T, mPhi2(phi)) + dof[1] * V1_2prime_finiteT(T, mGB2(phi)) +
                                  dof[2] * V1_2prime_finiteT(T, mZd2(phi));

    // daisy resummation contributions
    double daisy_contribution = 0.0;
    auto daisy_const_phi = [this, phi](double T) {
        return this->V1_Daisy(phi, T);
    };

    daisy_contribution = boost::math::differentiation::finite_difference_derivative(daisy_const_phi, T);

    return thermal_contribution + daisy_contribution;
}

// Speed of sound squared
double DarkAbelianHiggs::cs2(double T) const {
    // cs^2 = (dV1T/dT) / [T * (d2V1T/dT2)] evaluated at phi = v0 - epsilon
    double phi = v0 - 1e-15;
    double num = dV1T_dT(phi, T);
    double den = T * d2V1T_dT2(phi, T);
    return num / den;
}

// Enthalpy density of radiation
double DarkAbelianHiggs::w_density(double phi, double T) const {
    return -T * dV1T_dT(phi, T);
}

// Pressure from effective potential
double DarkAbelianHiggs::pressure(double phi, double T) const {
    // p = - Veff
    return -Veff(phi, T);
}

// Energy density from effective potential
double DarkAbelianHiggs::energy_density(double phi, double T) const {
    // rho = Veff - T * dV1T/dT
    return Veff(phi, T) - T * dV1T_dT(phi, T);
}

// Pseudo trace
double DarkAbelianHiggs::pseudo_trace(double phi, double T) const {
    // theta = 1/4 * (rho - p / cs2)
    double rho_val = energy_density(phi, T);
    double p_val = pressure(phi, T);
    double cs2_val = cs2(T);
    return 0.25 * (rho_val - p_val / cs2_val);
}

// Strength of the transition parameter alpha
double DarkAbelianHiggs::alpha(const double T) const {
    // pseudo trace at false vacuum (phi ~ 0)
    double theta_false = pseudo_trace(0, T);
    // pseudo trace at true vacuum (phi ~ v0)
    double theta_true = pseudo_trace(v0, T);
    // enthalpy density in false vacuum
    double w_false = w_density(0, T) + 4.0 / 3.0 * sm_cosmology->rho(T);
    // alpha = (4/3)*(theta_false - theta_true)/w_false
    return (4.0 / 3.0) * (theta_false - theta_true) / w_false;
}

double DarkAbelianHiggs::mathcal_J(double T) const {
    return 3.0 * dV1T_dT(0.0 + 1e-15, T) / d2V1T_dT2(0.0 + 1e-15, T);
}

// Scale factor between two temperatures: exp(integral dT / J(T)) using Boost quadrature
double DarkAbelianHiggs::scale_factor(double T_high, double T_low) const {
    auto f = [this](double T) {
        return 1.0 / this->mathcal_J(T);
    };
    boost::math::quadrature::gauss_kronrod<double, 15> integrator;
    double result = integrator.integrate(f, T_low, T_high, 5, 1e-6);
    return std::exp(result);
}

// Volume of a bubble between temperatures Tf and Ti using Boost quadrature
double DarkAbelianHiggs::volume_bubble(double T_formed, double T) const {
    auto f = [this, T](double Tp) {
        double sf = this->scale_factor(T, Tp);
        return sf / (this->mathcal_J(Tp) * this->hubble_rate(Tp));
    };
    boost::math::quadrature::gauss_kronrod<double, 15> integrator;
    double result = integrator.integrate(f, T, T_formed, 5, 1e-6);
    double wall_velocity = 1; // Assuming wall velocity is 1 for simplicity; adjust as needed
    return 4.0 * M_PI * std::pow(wall_velocity, 3) / 3.0 * std::pow(result, 3);
}

// False vacuum fraction at temperature T using Boost quadrature
double DarkAbelianHiggs::false_vacuum_fraction(double T) const {
    auto f = [this, T](double Tp) {
        double sf3 = std::pow(this->scale_factor(T, Tp), -3);
        return sf3 * this->nucleation_rate(Tp) * this->volume_bubble(Tp, T) / (
                   this->mathcal_J(Tp) * this->hubble_rate(Tp));
    };
    boost::math::quadrature::gauss_kronrod<double, 15> integrator;
    double result = integrator.integrate(f, T, T_crit_, 5, 1e-6);
    return std::exp(-result);
}

double DarkAbelianHiggs::false_vacuum_fraction_integrand(double T, double Tp) const {
    double sf3 = std::pow(scale_factor(T, Tp), -3);
    return sf3 * this->nucleation_rate(Tp) * this->volume_bubble(Tp, T) / (this->mathcal_J(Tp) * this->hubble_rate(Tp));
}

// Initial bubble radius
double DarkAbelianHiggs::radius_init(double T) const {
    double action;
    const double s3 = bounce_action_dim3(T);
    const double s4 = bounce_action_dim4(T);

    if (s4 > s3 / T) {
        action = s4;
    } else {
        action = s3;
    }

    if (action < 0)
        throw std::runtime_error("Negative bounce action");
    double numer = 3 * action / (4 * M_PI * (Veff(0, T) - Veff(v0, T)));
    return std::pow(numer, 1.0 / 3.0);
}

// Separation radius from number density
double DarkAbelianHiggs::radius_sep(double T_perc) const {
    auto f = [this, T_perc](double T) {
        return nucleation_rate(T) / hubble_rate(T) * false_vacuum_fraction(T) / mathcal_J(T) * std::pow(
                   scale_factor(T, T_perc), -3);
    };
    boost::math::quadrature::gauss_kronrod<double, 61> integrator;
    double nd = integrator.integrate(f, T_perc, T_crit_, 5, 1e-6);
    if (nd <= 0)
        throw std::runtime_error("Negative number density");
    return std::pow(nd, -1.0 / 3.0);
}

// Nucleation rate
double DarkAbelianHiggs::nucleation_rate(double T) const {
    const double s3 = bounce_action_dim3(T);
    const double s4 = bounce_action_dim4(T);

    if (s4 > s3 / T) {
        return std::exp(-s4) * std::pow(s4 / (2 * M_PI), 2);
    }
    return std::pow(T, 4) * std::pow(s3 / (2 * M_PI * T), 1.5) * std::exp(-s3 / T);
}

// Hubble rate
double DarkAbelianHiggs::hubble_rate(double T) const {
    double rho_rad = sm_cosmology->rho(T);
    double deltaV = Veff(0, T) - Veff(v0, T);
    return std::sqrt((rho_rad + deltaV) / 3.0) / Mpl;
}

// Lorentz factor at breakup: gamma_star = (2/3)*(radius_sep/radius_init)
double DarkAbelianHiggs::gamma_star(double T_perc) const {
    return 2.0 / 3.0 * radius_sep(T_perc) / radius_init(T_perc);
}

// Alpha at equilibrium
double DarkAbelianHiggs::alpha_eq(double T_perc) const {
    double term = std::sqrt(mZd2_vev) - std::sqrt(mZd2(0.0));
    double numerator = 4.0 * 3.0 * g_d * g_d * term * std::pow(T_perc, 3);
    double denom = 3.0 * w_density(0.0, T_perc);
    return numerator / denom;
}

// Alpha at infinite wall speed
double DarkAbelianHiggs::alpha_inf(double T_perc) const {
    double delta_m2 = dof[0] * (mPhi2_vev - mPhi2(0.0)) + dof[1] * (mGB2_vev - mGB2(0.0)) + dof[2] * (
                          mZd2_vev - mZd2(0.0));
    return delta_m2 * std::pow(T_perc, 2) / (w_density(0.0, T_perc) * 18.0);
}

// Lorentz factor at equilibrium
double DarkAbelianHiggs::gamma_eq(double T_perc) const {
    return (alpha(T_perc) - alpha_inf(T_perc)) / alpha_eq(T_perc);
}

// Wall velocity
double DarkAbelianHiggs::v_w(double T_perc) const {
    double ge = gamma_eq(T_perc);
    return std::sqrt(1.0 - 1.0 / (ge * ge));
}

// Collision efficiency
double DarkAbelianHiggs::kappa_collision(double T_perc) const {
    double gs = gamma_star(T_perc);
    double ge = gamma_eq(T_perc);
    double ai = alpha_inf(T_perc);
    double a = alpha(T_perc);
    if (gs > ge) {
        return ge / gs * (1.0 - ai / a * std::pow(ge / gs, 2));
    } else {
        return 1.0 - ai / a;
    }
}

// Sound-wave damping efficiency
double DarkAbelianHiggs::kappa_soundwave(double T_perc) const {
    double ae = alpha_eff(T_perc);
    double a = alpha(T_perc);
    return ae / a * ae / (0.73 + 0.083 * std::sqrt(ae) + ae);
}

// Effective alpha
double DarkAbelianHiggs::alpha_eff(double T_perc) const {
    return alpha(T_perc) * (1.0 - kappa_collision(T_perc));
}

// RMS fluid velocity
double DarkAbelianHiggs::rms_fluid_velocity(double T_perc) const {
    double ae = alpha_eff(T_perc);
    return std::sqrt(0.75 * ae / (1.0 + ae) * kappa_soundwave(T_perc));
}

// Sound-wave frequency spectrum
double DarkAbelianHiggs::omega_soundwave(double f, double T_perc) const {
    double f_sw = 3.4 / ((v_w(T_perc) - cs2(T_perc)) * radius_sep(T_perc));
    double HR = hubble_rate(T_perc) * radius_sep(T_perc);
    double HT_SW = std::min(1.0, hubble_rate(T_perc) * radius_sep(T_perc) / rms_fluid_velocity(T_perc));
    double ks = kappa_soundwave(T_perc);
    double a = alpha(T_perc);
    double factor = 0.38 * HR * HT_SW * std::pow(ks * a / (1.0 + a), 2);
    double x = f / f_sw;
    return factor * std::pow(x, 3) * std::pow(1.0 + 0.75 * x * x, -3.5);
}

double DarkAbelianHiggs::compute_bounce_action_dim3(double T) const {
    BounceSolverS3 bounce_solver_dim3(0, 0.9 * v0, [this, T](double phi) { return Veff(phi, T); },
                                      [this, T](double phi) { return dV_dphi(phi, T); });


    double action = bounce_solver_dim3.bounce_action();

    // Compute the bounce action using the solver
    if (action < 0 || std::isnan(action) || std::isinf(action)) {
        action = 0.0; // Set to zero to avoid NaN issues
    }

    return action;
}

// for the zero temperature action
double DarkAbelianHiggs::compute_bounce_action_dim4(double T) const {
    BounceSolverS4 bounce_solver_dim4(0, 0.9 * v0, [this, T](double phi) { return Veff(phi, T); },
                                      [this, T](double phi) { return dV_dphi(phi, T); });

    double action = bounce_solver_dim4.bounce_action();

    if (action < 0 || std::isnan(action) || std::isinf(action)) {
        action = 0.0;
    }

    return action;
}

double DarkAbelianHiggs::bounce_action_dim3(double T) const { {
        // 1) Try exact lookup under shared lock
        std::shared_lock lock(action_dim3_cache_mutex_);
        auto it = action_dim3_cache_.find(T);
        if (it != action_dim3_cache_.end())
            return it->second;

        // 2) If no exact match, try to bracket for linear interpolation
        auto it_hi = action_dim3_cache_.lower_bound(T);
        if (it_hi != action_dim3_cache_.begin() && it_hi != action_dim3_cache_.end()) {
            auto it_lo = std::prev(it_hi);

            double T1 = it_lo->first;
            double S1 = it_lo->second;
            double T2 = it_hi->first;
            double S2 = it_hi->second;

            // linear interpolation: S(T) ≈ S1 + (T–T1)/(T2–T1) * (S2–S1)
            double w = (T - T1) / (T2 - T1);
            return S1 + w * (S2 - S1);
        }
        // if T is outside [smallest key, largest key], fall through to exact compute
    }

    // 3) Pure cache‐miss: do the expensive solve
    double S_exact = compute_bounce_action_dim3(T); {
        // 4) Store the exact result under exclusive lock
        std::unique_lock lock(action_dim3_cache_mutex_);
        action_dim3_cache_[T] = S_exact;
    }
    return S_exact;
}

double DarkAbelianHiggs::bounce_action_dim4(double T) const { {
        // 1) Try exact lookup under shared lock
        std::shared_lock lock(action_dim4_cache_mutex_);
        auto it = action_dim4_cache_.find(T);
        if (it != action_dim4_cache_.end())
            return it->second;

        // 2) If no exact match, try to bracket for linear interpolation
        auto it_hi = action_dim4_cache_.lower_bound(T);
        if (it_hi != action_dim4_cache_.begin() && it_hi != action_dim4_cache_.end()) {
            auto it_lo = std::prev(it_hi);

            double T1 = it_lo->first;
            double S1 = it_lo->second;
            double T2 = it_hi->first;
            double S2 = it_hi->second;

            // linear interpolation: S(T) ≈ S1 + (T–T1)/(T2–T1) * (S2–S1)
            double w = (T - T1) / (T2 - T1);
            return S1 + w * (S2 - S1);
        }
        // if T is outside [smallest key, largest key], fall through to exact compute
    }

    // 3) Pure cache‐miss: do the expensive solve
    double S_exact = compute_bounce_action_dim4(T); {
        // 4) Store the exact result under exclusive lock
        std::unique_lock lock(action_dim4_cache_mutex_);
        action_dim4_cache_[T] = S_exact;
    }
    return S_exact;
}

std::vector<double> DarkAbelianHiggs::bounce_action_bulk_dim3(const std::vector<double> &Ts) const {
    std::vector<double> results(Ts.size());

#pragma omp parallel for schedule(dynamic, 8)
    for (std::size_t i = 0; i < Ts.size(); ++i) {
        // each thread safely calls compute_bounce_action()
        results[i] = this->bounce_action_dim3(Ts[i]);
    }

    return results;
}

std::vector<double> DarkAbelianHiggs::bounce_action_bulk_dim4(const std::vector<double> &Ts) const {
    std::vector<double> results(Ts.size());
#pragma omp parallel for schedule(dynamic, 8)
    for (std::size_t i = 0; i < Ts.size(); ++i) {
        results[i] = this->bounce_action_dim4(Ts[i]);
    }
    return results;
}

double DarkAbelianHiggs::T_crit() const {
    return T_crit_;
}
