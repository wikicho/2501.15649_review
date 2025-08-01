#include "bounce_solver_dim3.h"

double BounceSolverS3::find_top() const
{
    int n_scan = 400;
    double tol = 1e-8;

    std::vector<double> grid(n_scan + 1), deriv(n_scan + 1);
    double h = (phi0_ - phi_metamin_) / n_scan;
    for (int i = 0; i <= n_scan; ++i)
    {
        double phi = phi_metamin_ + i * h;
        grid[i] = phi;
        deriv[i] = dV_(phi);
    }

    // 2. Locate sign‐change intervals
    std::vector<std::pair<double, double>> brackets;
    for (int i = 0; i < n_scan; ++i)
    {
        if (deriv[i] * deriv[i + 1] < 0.0)
        {
            brackets.emplace_back(grid[i], grid[i + 1]);
        }
    }

    if (brackets.empty())
    {
        // no turning point
        return std::numeric_limits<double>::quiet_NaN();
    }

    // 3. Refine each bracket with a robust solver
    std::vector<double> roots;
    for (auto &br : brackets)
    {
        double lo = br.first, hi = br.second;
        auto f = [&](double x)
        { return dV_(x); };

        // Use TOMS748, which doesn’t throw but returns a pair<lower,upper>
        boost::uintmax_t max_iter = 50;
        auto result = boost::math::tools::toms748_solve(f, lo, hi, [tol](double l, double u)
                                                        { return std::abs(u - l) <= tol; }, max_iter);
        // take midpoint of the final bracket
        double root = (result.first + result.second) * 0.5;
        roots.push_back(root);
    }

    // 4. Pick the root with highest V
    return *std::max_element(
        roots.begin(), roots.end(),
        [&](double a, double b)
        { return V_(a) < V_(b); });
}

double BounceSolverS3::calculate_qd_()
{
    double V0 = V_(phi0_) - V_(phi_metamin_);
    double V0p = dV_(phi0_);

    double Q_T = phiT_ * phiT_ * std::pow(phiT_ - phi0_, 2);
    double Q_T_Prime = 2 * (phi0_ - 2 * phiT_) * (phi0_ - phiT_) * phiT_;
    double Q_T_2Prime = 2 * (phi0_ * phi0_ - 6 * phi0_ * phiT_ + 6 * phiT_ * phiT_);

    // Compute the cubic term at phiT_
    double Vt3_phiT_ = V0 * phiT_ / phi0_;
    Vt3_phiT_ += ((2 * phi0_) * V0p - 3 * V0) / (3 * phi0_ * phi0_) * phiT_ * (phiT_ - phi0_);
    Vt3_phiT_ += ((2 * phi0_) * V0p - 6 * V0) / (3 * std::pow(phi0_, 3)) * phiT_ * std::pow(phiT_ - phi0_, 2);
    // first derivative of the cubic term
    double Vt3_prime_phiT_ = 6 * V0 * phiT_ / (phi0_ * phi0_) - 6 * phiT_ * phiT_ * V0 / std::pow(phi0_, 3) - 4 * phiT_ * V0p / (3 * phi0_) + 2 * V0p * phiT_ * phiT_ / (phi0_ * phi0_);
    // second derivative of the cubic term
    double Vt3_2prime_phiT_ = 6 * V0 / (phi0_ * phi0_) - 12 * phiT_ * V0 / std::pow(phi0_, 3) - 4 * V0p / (3 * phi0_) + 4 * V0p * phiT_ / (phi0_ * phi0_);

    double VT = V_(phiT_) - V_(phi_metamin_);

    double ad = 3 * Q_T_Prime * Q_T_Prime - 4 * Q_T * Q_T_2Prime;
    double bd = -2 * Vt3_2prime_phiT_ * Q_T + 3 * Vt3_prime_phiT_ * Q_T_Prime - 2 * Q_T_2Prime * (Vt3_phiT_ - VT);
    double cd = 3 * Vt3_prime_phiT_ * Vt3_prime_phiT_ - 4 * Vt3_2prime_phiT_ * (Vt3_phiT_ - VT);

    double disc = bd * bd - ad * cd;
    if (disc < 0)
    {
        return 0;
    }

    return (-bd - std::sqrt(disc)) / ad;
}

double BounceSolverS3::V_t(double phi) const
{
    double V0 = V_(phi0_) - V_(phi_metamin_);
    double V0p = dV_(phi0_);

    double vt1 = V0 * phi / phi0_;
    double vt2 = ((2 * phi0_) * V0p - 3 * V0) / (3 * phi0_ * phi0_) * phi * (phi - phi0_);
    double vt3 = ((2 * phi0_) * V0p - 6 * V0) / (3 * std::pow(phi0_, 3)) * phi * std::pow((phi - phi0_), 2);
    double vt4 = qd_ * phi * phi * std::pow((phi - phi0_), 2);

    return vt1 + vt2 + vt3 + vt4;
}

double BounceSolverS3::V_t_prime(double phi) const
{
    double V0 = V_(phi0_) - V_(phi_metamin_);
    double V0p = dV_(phi0_);

    double vt3_prime = -6 * phi * phi * V0 / std::pow(phi0_, 3) + 6 * V0 * phi / (phi0_ * phi0_) + 2 * phi * phi * V0p / (phi0_ * phi0_) - 4 * phi * V0p / (3 * phi0_);

    double vt4_prime = 2.0 * qd_ * phi * (2.0 * phi * phi - 3.0 * phi * phi0_ + phi0_ * phi0_);

    return vt3_prime + vt4_prime;
}

double BounceSolverS3::bounce_action() const
{
    double front_factor = 15.0849 * M_PI;
    // integrand definition
    std::function<double(double)> integrand = [this](double phi)
    {
        if (phi < phi_metamin_ + 1e-9) {
            return 0.0;
        }
        else if (phi >= 1 - 1e-9) {
            return 0.0;
        }
        else
        {
            double V_exact = this->V_(phi) - this->V_(phi_metamin_);
            double V_approx = this->V_t(phi);
            double Vt_prime = this->V_t_prime(phi);
            return std::pow(V_exact - V_approx, 1.5) / std::pow(Vt_prime, 2);
        }
    };

    boost::math::quadrature::gauss_kronrod<double, 15> integrator;
    double integral_result = integrator.integrate(integrand, phi_metamin_, phi0_);

    return front_factor * integral_result;
}
