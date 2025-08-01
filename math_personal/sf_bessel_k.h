#ifndef SF_BESSEL_K_HPP_
#define SF_BESSEL_K_HPP_

#pragma once

#include <cmath>
#include <boost/math/constants/constants.hpp>

namespace sf
{
    using namespace boost::math::double_constants;

    /**
     * @brief Calculate Modified Bessel K2.
     *
     * @return double
     */
    double bessel_k1(double x);

    /**
     * @brief Calculate Modified Bessel K2.
     *
     * @return double
     */
    double bessel_k2(double x);
}

#endif
