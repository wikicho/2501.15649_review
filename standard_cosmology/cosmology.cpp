#include "cosmology.h"

using namespace boost::math::double_constants;

Universe::Universe()
{
    CSVImporter importer("../data/sm_eff_data.csv", ',');
    CSVImporter g_eff_data("../data/g_eff_data.csv", ',');
    CSVImporter gp_eff_data("../data/gp_eff_data.csv", ',');
    CSVImporter h_eff_data("../data/h_eff_data.csv", ',');
    CSVImporter i_eff_data("../data/i_eff_data.csv", ',');

    for (size_t i = 0; i < importer.numRows(); ++i)
    {
        vec_T_SM.push_back(importer.getRow(i)[0]);
        vec_g_eff_SM.push_back(importer.getRow(i)[1]);
        vec_h_eff_SM.push_back(importer.getRow(i)[2]);
    }

    for (size_t i = 0; i < g_eff_data.numRows(); ++i)
    {
        vec_x.push_back(g_eff_data.getRow(i)[0]);
        vec_g_b.push_back(g_eff_data.getRow(i)[1]);
        vec_g_f.push_back(g_eff_data.getRow(i)[2]);
        vec_g_diff_b.push_back(g_eff_data.getRow(i)[3]);
        vec_g_diff_f.push_back(g_eff_data.getRow(i)[4]);
    }

    for (size_t i = 0; i < gp_eff_data.numRows(); ++i)
    {
        vec_x_p.push_back(gp_eff_data.getRow(i)[0]);
        vec_gp_b.push_back(gp_eff_data.getRow(i)[1]);
        vec_gp_f.push_back(gp_eff_data.getRow(i)[2]);
        vec_gp_diff_b.push_back(gp_eff_data.getRow(i)[3]);
        vec_gp_diff_f.push_back(gp_eff_data.getRow(i)[4]);
    }

    for (size_t i = 0; i < h_eff_data.numRows(); ++i)
    {
        vec_x_h.push_back(h_eff_data.getRow(i)[0]);
        vec_h_b.push_back(h_eff_data.getRow(i)[1]);
        vec_h_f.push_back(h_eff_data.getRow(i)[2]);
        vec_h_diff_b.push_back(h_eff_data.getRow(i)[3]);
        vec_h_diff_f.push_back(h_eff_data.getRow(i)[4]);
    }

    for (size_t i = 0; i < i_eff_data.numRows(); ++i)
    {
        vec_x_i.push_back(i_eff_data.getRow(i)[0]);
        vec_i_b.push_back(i_eff_data.getRow(i)[1]);
        vec_i_f.push_back(i_eff_data.getRow(i)[2]);
        vec_i_diff_b.push_back(i_eff_data.getRow(i)[3]);
        vec_i_diff_f.push_back(i_eff_data.getRow(i)[4]);
    }

    for (size_t i = 0; i < vec_T_SM.size(); i++)
    {
        map_g_eff_SM[vec_T_SM[i]] = vec_g_eff_SM[i];
        map_h_eff_SM[vec_T_SM[i]] = vec_h_eff_SM[i];
    }

    for (size_t i = 0; i < vec_x.size(); i++)
    {
        map_g_b[vec_x[i]] = vec_g_b[i];
        map_g_f[vec_x[i]] = vec_g_f[i];
        map_g_diff_b[vec_x[i]] = vec_g_diff_b[i];
        map_g_diff_f[vec_x[i]] = vec_g_diff_f[i];
    }

    for (size_t i = 0; i < vec_x_p.size(); i++)
    {
        map_gp_b[vec_x_p[i]] = vec_gp_b[i];
        map_gp_f[vec_x_p[i]] = vec_gp_f[i];
        map_gp_diff_b[vec_x_p[i]] = vec_gp_diff_b[i];
        map_gp_diff_f[vec_x_p[i]] = vec_gp_diff_f[i];
    }

    for (size_t i = 0; i < vec_x_h.size(); i++)
    {
        map_h_b[vec_x_h[i]] = vec_h_b[i];
        map_h_f[vec_x_h[i]] = vec_h_f[i];
        map_h_diff_b[vec_x_h[i]] = vec_h_diff_b[i];
        map_h_diff_f[vec_x_h[i]] = vec_h_diff_f[i];
    }

    for (size_t i = 0; i < vec_x_i.size(); i++)
    {
        map_i_b[vec_x_i[i]] = vec_i_b[i];
        map_i_f[vec_x_i[i]] = vec_i_f[i];
        map_i_diff_b[vec_x_i[i]] = vec_i_diff_b[i];
        map_i_diff_f[vec_x_i[i]] = vec_i_diff_f[i];
    }
}

double Universe::H(double T)
{
    return std::sqrt(0.3333333333 / std::pow(constant::M_Pl, 2) * rho(T)); // in GeV
}

double Universe::rho(double T)
{
    return std::pow(pi, 2) / 30.0 * g_eff(T) * std::pow(T, 4);
}

double Universe::s(double T)
{
    return 2 * std::pow(pi, 2) / 45.0 * h_eff(T) * std::pow(T, 3);
}

double Universe::g_eff(double T) const
{
    if (T < vec_T_SM[0])
    {
        return vec_g_eff_SM[0];
    }

    else if (T > vec_T_SM.back())
    {
        return vec_g_eff_SM.back();
    }

    else
    {
        auto it = map_g_eff_SM.lower_bound(T);
        double T_h = it->first;
        double value_h = it->second;

        it--;

        double T_l = it->first;
        double value_l = it->second;

        double result = (value_h - value_l) / (T_h - T_l) * (T - T_l) + value_l;

        return result;
    }
}

double Universe::h_eff(double T) const
{
    if (T < vec_T_SM[0])
    {
        return vec_h_eff_SM[0];
    }

    else if (T > vec_T_SM.back())
    {
        return vec_h_eff_SM.back();
    }

    else
    {
        auto it = map_h_eff_SM.lower_bound(T);
        double T_h = it->first;
        double value_h = it->second;

        it--;

        double T_l = it->first;
        double value_l = it->second;

        double result = (value_h - value_l) / (T_h - T_l) * (T - T_l) + value_l;

        return result;
    }
}

// need to be fixed
double Universe::g_eff_diff(double T)
{
    double h = T * 1e-11;
    double result = (g_eff(T + h) - g_eff(T - h)) / (2 * h);
    if (std::abs(result) < 1e-30)
    {
        return 0;
    }

    return result;
}

// need to be fixed
double Universe::h_eff_diff(double T)
{
    double h = T * 1e-11;
    return (h_eff(T + h) - h_eff(T - h)) / (2 * h);
}

double Universe::g_boson(double T, double g, double m) const
{
    double x = m / T;

    if (x <= 1e-3)
    {
        return g;
    }

    else if (x > 1000)
    {
        return 0;
    }

    else
    {
        auto it = map_g_b.lower_bound(x);
        double x_h = it->first;
        double value_h = it->second;

        it--;

        double x_l = it->first;
        double value_l = it->second;

        double result = (value_h - value_l) / (x_h - x_l) * (x - x_l) + value_l;

        return g * result;
    }
}

double Universe::g_fermion(double T, double g, double m) const
{
    double x = m / T;

    if (x < 1e-3)
    {
        return 0.875 * g;
    }

    else if (x > 1000)
    {
        return 0;
    }

    else
    {
        auto it = map_g_f.lower_bound(x);
        double x_h = it->first;
        double value_h = it->second;

        it--;

        double x_l = it->first;
        double value_l = it->second;

        double result = (value_h - value_l) / (x_h - x_l) * (x - x_l) + value_l;

        return g * result;
    }
}

double Universe::g_boson_diff(double T, double g, double m)
{
    double x = m / T;

    if (x < 1e-3)
    {
        return 0;
    }

    else if (x > 1000)
    {
        return 0;
    }

    else
    {
        auto it = map_g_diff_b.lower_bound(x);
        double x_h = it->first;
        double value_h = it->second;

        it--;

        double x_l = it->first;
        double value_l = it->second;

        double result = (value_h - value_l) / (x_h - x_l) * (x - x_l) + value_l;

        return g * 1 / m * result;
    }
}

double Universe::g_fermion_diff(double T, double g, double m)
{
    double x = m / T;

    if (x < 1e-3)
    {
        return 0;
    }

    else if (x > 1000)
    {
        return 0;
    }

    else
    {
        auto it = map_g_diff_f.lower_bound(x);
        double x_h = it->first;
        double value_h = it->second;

        it--;

        double x_l = it->first;
        double value_l = it->second;

        double result = (value_h - value_l) / (x_h - x_l) * (x - x_l) + value_l;

        return g * 1 / m * result;
    }
}

double Universe::gp_boson(double T, double g, double m)
{
    double x = m / T;

    if (x <= 1e-3)
    {
        return g;
    }

    else if (x > 1000)
    {
        return 0;
    }

    else
    {
        auto it = map_gp_b.lower_bound(x);
        double x_h = it->first;
        double value_h = it->second;

        it--;

        double x_l = it->first;
        double value_l = it->second;

        double result = (value_h - value_l) / (x_h - x_l) * (x - x_l) + value_l;

        return g * result;
    }
}

double Universe::gp_fermion(double T, double g, double m)
{
    double x = m / T;

    if (x < 1e-3)
    {
        return 0.875 * g;
    }

    else if (x > 1000)
    {
        return 0;
    }

    else
    {
        auto it = map_gp_f.lower_bound(x);
        double x_h = it->first;
        double value_h = it->second;

        it--;

        double x_l = it->first;
        double value_l = it->second;

        double result = (value_h - value_l) / (x_h - x_l) * (x - x_l) + value_l;

        return g * result;
    }
}

double Universe::gp_boson_diff(double T, double g, double m)
{
    double x = m / T;

    if (x < 1e-3)
    {
        return 0;
    }

    else if (x > 1000)
    {
        return 0;
    }

    else
    {
        auto it = map_gp_diff_b.lower_bound(x);
        double x_h = it->first;
        double value_h = it->second;

        it--;

        double x_l = it->first;
        double value_l = it->second;

        double result = (value_h - value_l) / (x_h - x_l) * (x - x_l) + value_l;

        return g * 1 / m * result;
    }
}

double Universe::gp_fermion_diff(double T, double g, double m)
{
    double x = m / T;

    if (x < 1e-3)
    {
        return 0;
    }

    else if (x > 1000)
    {
        return 0;
    }

    else
    {
        auto it = map_gp_diff_f.lower_bound(x);
        double x_h = it->first;
        double value_h = it->second;

        it--;

        double x_l = it->first;
        double value_l = it->second;

        double result = (value_h - value_l) / (x_h - x_l) * (x - x_l) + value_l;

        return g * 1 / m * result;
    }
}

double Universe::h_boson(double T, double g, double m)
{
    double x = m / T;

    if (x <= 1e-3)
    {
        return g;
    }

    else if (x > 1000)
    {
        return 0;
    }

    else
    {
        auto it = map_h_b.lower_bound(x);
        double x_h = it->first;
        double value_h = it->second;

        it--;

        double x_l = it->first;
        double value_l = it->second;

        double result = (value_h - value_l) / (x_h - x_l) * (x - x_l) + value_l;

        return g * result;
    }
}

double Universe::h_fermion(double T, double g, double m)
{
    double x = m / T;

    if (x <= 1e-3)
    {
        return 0.875 * g;
    }

    else if (x > 1000)
    {
        return 0;
    }

    else
    {
        auto it = map_h_f.lower_bound(x);
        double x_h = it->first;
        double value_h = it->second;

        it--;

        double x_l = it->first;
        double value_l = it->second;

        double result = (value_h - value_l) / (x_h - x_l) * (x - x_l) + value_l;

        return g * result;
    }
}

double Universe::h_boson_diff(double T, double g, double m)
{
    double x = m / T;

    if (x < 1e-3)
    {
        return 0;
    }

    else if (x > 1000)
    {
        return 0;
    }

    else
    {
        auto it = map_h_diff_b.lower_bound(x);
        double x_h = it->first;
        double value_h = it->second;

        it--;

        double x_l = it->first;
        double value_l = it->second;

        double result = (value_h - value_l) / (x_h - x_l) * (x - x_l) + value_l;

        return g * 1 / m * result;
    }
}
double Universe::h_fermion_diff(double T, double g, double m)
{
    double x = m / T;

    if (x < 1e-3)
    {
        return 0;
    }

    else if (x > 1000)
    {
        return 0;
    }

    else
    {
        auto it = map_h_diff_f.lower_bound(x);
        double x_h = it->first;
        double value_h = it->second;

        it--;

        double x_l = it->first;
        double value_l = it->second;

        double result = (value_h - value_l) / (x_h - x_l) * (x - x_l) + value_l;

        return g * 1 / m * result;
    }
}

double Universe::n_eq_boson(double T, double g, double m)
{
    double x = m / T;

    if (x <= 1e-3)
    {
        return g * std::pow(T, 3) * zeta_three * std::pow(one_div_pi, 2);
    }

    else if (x > 1000)
    {
        return 0;
    }

    else
    {
        auto it = map_i_b.lower_bound(x);
        double x_h = it->first;
        double value_h = it->second;

        it--;

        double x_l = it->first;
        double value_l = it->second;

        double result = (value_h - value_l) / (x_h - x_l) * (x - x_l) + value_l;

        return g * std::pow(one_div_pi, 2) * zeta_three * result * std::pow(T, 3);
    }
}

double Universe::n_eq_fermion(double T, double g, double m)
{
    double x = m / T;

    if (x < 1e-3)
    {
        return g * std::pow(one_div_pi, 2) * std::pow(T, 3) * 0.75 * zeta_three;
    }

    else if (x > 1000)
    {
        return 0;
    }

    else
    {
        auto it = map_i_f.lower_bound(x);
        double x_h = it->first;
        double value_h = it->second;

        it--;

        double x_l = it->first;
        double value_l = it->second;

        double result = (value_h - value_l) / (x_h - x_l) * (x - x_l) + value_l;

        return g * std::pow(one_div_pi, 2) * zeta_three * result * std::pow(T, 3);
    }
}

double Universe::n_eq_boson_diff(double T, double g, double m)
{
    double x = m / T;

    if (x < 1e-3)
    {
        return g * std::pow(one_div_pi, 2) * zeta_three * 3 * std::pow(T, 2);
    }

    else if (x > 1000)
    {
        return g * std::pow(one_div_pi, 2) * zeta_three * 3 * std::pow(T, 2);
        ;
    }

    else
    {
        auto it = map_i_diff_b.lower_bound(x);
        double x_h = it->first;
        double value_h = it->second;

        it--;

        double x_l = it->first;
        double value_l = it->second;

        double result = (value_h - value_l) / (x_h - x_l) * (x - x_l) + value_l;

        return g * std::pow(one_div_pi, 2) * zeta_three * (3 * std::pow(T, 2) + result / m * std::pow(T, 3));
    }
}

double Universe::n_eq_fermion_diff(double T, double g, double m)
{
    double x = m / T;

    if (x < 1e-3)
    {
        return g * std::pow(one_div_pi, 2) * 3 * std::pow(T, 2) * 0.75 * zeta_three;
    }

    else if (x > 1000)
    {
        return g * std::pow(one_div_pi, 2) * 3 * std::pow(T, 2) * 0.75 * zeta_three;
    }

    else
    {
        auto it = map_i_diff_f.lower_bound(x);
        double x_h = it->first;
        double value_h = it->second;

        it--;

        double x_l = it->first;
        double value_l = it->second;

        double result = (value_h - value_l) / (x_h - x_l) * (x - x_l) + value_l;

        return g * std::pow(one_div_pi, 2) * zeta_three * (0.75 * 3 * std::pow(T, 2) + result / m * std::pow(T, 3));
    }
}

double Universe::rho_eq_boson(double T, double g, double m)
{
    return std::pow(pi, 2) / 30.0 * g_boson(T, g, m) * std::pow(T, 4);
}

double Universe::rho_eq_fermion(double T, double g, double m)
{
    return std::pow(pi, 2) / 30.0 * g_fermion(T, g, m) * std::pow(T, 4);
}

double Universe::rho_eq_boson_diff(double T, double g, double m)
{
    return std::pow(pi, 2) / 30.0 * (g_boson_diff(T, g, m) * std::pow(T, 4) + 4 * g_boson(T, g, m) * std::pow(T, 3));
}

double Universe::rho_eq_fermion_diff(double T, double g, double m)
{
    return std::pow(pi, 2) / 30.0 * (g_fermion_diff(T, g, m) * std::pow(T, 4) + 4 * g_fermion(T, g, m) * std::pow(T, 3));
}
// namespace standard_cosmology

double Universe::p_eq_boson(double T, double g, double m)
{
    return std::pow(pi, 2) / 90.0 * gp_boson(T, g, m) * std::pow(T, 4);
}

double Universe::p_eq_fermion(double T, double g, double m)
{
    return std::pow(pi, 2) / 90.0 * gp_fermion(T, g, m) * std::pow(T, 4);
}

double Universe::n_eq_boson_T3(double T, double g, double m)
{
    double x = m / T;

    if (x < 1e-3)
    {
        return g * std::pow(one_div_pi, 2) * 0.75 * zeta_three;
    }

    else if (x > 1000)
    {
        return 0;
    }

    else
    {
        auto it = map_i_b.lower_bound(x);
        double x_h = it->first;
        double value_h = it->second;

        it--;

        double x_l = it->first;
        double value_l = it->second;

        double result = (value_h - value_l) / (x_h - x_l) * (x - x_l) + value_l;

        return g * std::pow(one_div_pi, 2) * zeta_three * result;
    }
}

double Universe::n_eq_fermion_T3(double T, double g, double m)
{
    double x = m / T;

    if (x < 1e-3)
    {
        return g * std::pow(one_div_pi, 2) * 0.75 * zeta_three;
    }

    else if (x > 1000)
    {
        return 0;
    }

    else
    {
        auto it = map_i_f.lower_bound(x);
        double x_h = it->first;
        double value_h = it->second;

        it--;

        double x_l = it->first;
        double value_l = it->second;

        double result = (value_h - value_l) / (x_h - x_l) * (x - x_l) + value_l;

        return g * std::pow(one_div_pi, 2) * zeta_three * result;
    }
}

double Universe::n_MB(double T, double g, double m)
{
    double x = m / T;

    if ((x > 1e-3) && (x < 1000))
    {
        return g * 0.5 * std::pow(one_div_pi, 2) * std::pow(T, 3) * std::pow(x, 2) * sf::bessel_k2(x);
    }

    else if (x < 1e-3)
    {
        return g * std::pow(one_div_pi, 2) * std::pow(T, 3);
    }

    else
    {
        return 0;
    }
}

double Universe::n_MB_T3(double T, double g, double m)
{
    double x = m / T;

    if ((x > 1e-3) && (x < 1000))
    {
        return 0.5 * std::pow(one_div_pi, 2) * std::pow(x, 2) * sf::bessel_k2(x);
    }

    else if (x < 1e-3)
    {
        return std::pow(one_div_pi, 2);
    }

    else
    {
        return 0;
    }
}

double Universe::rho_MB(double T, double g, double m)
{
    double x = m / T;

    if ((x > 1e-3) && (x < 1000))
    {
        return 0.5 * g * std::pow(one_div_pi, 2) * std::pow(T, 4) * std::pow(x, 2) * (x * sf::bessel_k1(x) + 3 * sf::bessel_k2(x));
    }

    else if (x < 1e-3)
    {
        return 3 * g * std::pow(one_div_pi, 2) * std::pow(T, 4);
    }

    else
    {
        return 0;
    }
}

double Universe::T_nu(double T)
{
    if (T >= 1e-2)
    {
        return T;
    }

    else
    {
        double result = (g_eff(T) - 2 - g_fermion(T, 4, constant::m_e)) / (0.875 * 3 * 2);

        result = std::pow(result, 0.25) * T;

        return result;
    }
}

// for one neutrino species
double Universe::rho_nu(double T)
{
    return 0.875 * std::pow(pi, 2) / 30.0 * 2 * std::pow(T_nu(T), 4);
}