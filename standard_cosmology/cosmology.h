#ifndef COSMOLOGY_H_
#define COSMOLOGY_H_

#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <boost/math/constants/constants.hpp>
#include <boost/math/quadrature/tanh_sinh.hpp>
#include <boost/format.hpp>
#include "constant.hpp"
#include "../util/csv_importer.h"
#include "../math_personal/sf_bessel_k.h"

// Universe without beyond standard model particles
class Universe
{
private:
    std::vector<double> vec_T_SM;
    std::vector<double> vec_g_eff_SM;
    std::vector<double> vec_h_eff_SM;

    std::map<double, double> map_g_eff_SM;
    std::map<double, double> map_h_eff_SM;

    std::vector<double> vec_x;
    std::vector<double> vec_g_b;
    std::vector<double> vec_g_f;
    std::vector<double> vec_g_diff_b;
    std::vector<double> vec_g_diff_f;

    std::vector<double> vec_x_p;
    std::vector<double> vec_gp_b;
    std::vector<double> vec_gp_f;
    std::vector<double> vec_gp_diff_b;
    std::vector<double> vec_gp_diff_f;

    std::vector<double> vec_x_h;
    std::vector<double> vec_h_b;
    std::vector<double> vec_h_f;
    std::vector<double> vec_h_diff_b;
    std::vector<double> vec_h_diff_f;

    std::vector<double> vec_x_i;
    std::vector<double> vec_i_b;
    std::vector<double> vec_i_f;
    std::vector<double> vec_i_diff_b;
    std::vector<double> vec_i_diff_f;

    // map for interpolation of g_eff (effective degree of freedom) (from energy density)
    std::map<double, double> map_g_b;
    std::map<double, double> map_g_f;

    // map for interpolation of d g_eff/dT
    std::map<double, double> map_g_diff_b;
    std::map<double, double> map_g_diff_f;

    // map for interpolation of gp_eff
    std::map<double, double> map_gp_b;
    std::map<double, double> map_gp_f;

    // map for interpolation of d gp_eff/dT
    std::map<double, double> map_gp_diff_b;
    std::map<double, double> map_gp_diff_f;

    // map for interpolation of h_eff (entropy density)
    std::map<double, double> map_h_b;
    std::map<double, double> map_h_f;

    // map for interpolation of d h_eff/dT (entropy density)
    std::map<double, double> map_h_diff_b;
    std::map<double, double> map_h_diff_f;

    // map for interpolation of i_eff (number density dof)
    std::map<double, double> map_i_b;
    std::map<double, double> map_i_f;

    // map for interpolation of d i_eff/dT (number density dof)
    std::map<double, double> map_i_diff_b;
    std::map<double, double> map_i_diff_f;


public:
    Universe();

    double H(double T);

    double g_eff(double T) const;

    double h_eff(double T) const;

    double g_eff_diff(double T);

    double h_eff_diff(double T);

    double rho(double T);

    double s(double T);

    double g_boson(double T, double g, double m) const;

    double g_fermion(double T, double g, double m) const;

    double g_boson_diff(double T, double g, double m);

    double g_fermion_diff(double T, double g, double m);

    double gp_boson(double T, double g, double m);

    double gp_fermion(double T, double g, double m);

    double gp_boson_diff(double T, double g, double m);

    double gp_fermion_diff(double T, double g, double m);

    double h_boson(double T, double g, double m);

    double h_fermion(double T, double g, double m);

    double h_boson_diff(double T, double g, double m);

    double h_fermion_diff(double T, double g, double m);

    double n_MB(double T, double g, double m);

    double n_eq_boson(double T, double g, double m);

    double n_eq_fermion(double T, double g, double m);

    double n_MB_T3(double T, double g, double m);

    double n_eq_fermion_T3(double T, double g, double m);

    double n_eq_boson_T3(double T, double g, double m);

    double n_eq_boson_diff(double T, double g, double m);

    double n_eq_fermion_diff(double T, double g, double m);

    double rho_MB(double T, double g, double m);

    double rho_eq_boson(double T, double g, double m);

    double rho_eq_fermion(double T, double g, double m);

    double rho_eq_boson_diff(double T, double g, double m);

    double rho_eq_fermion_diff(double T, double g, double m);

    double p_eq_boson(double T, double g, double m);

    double p_eq_fermion(double T, double g, double m);

    double T_nu(double T);

    double rho_nu(double T);
};

#endif