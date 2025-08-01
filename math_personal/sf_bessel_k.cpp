#include "sf_bessel_k.h"

namespace sf
{
    /**
     * @brief Calculate Modified Bessel K2.
     *
     * @return double
     */
    double bessel_k1(double x)
    {
        if (x >= 705)
        {
            return 0;
        }
        else if (x < 0)
        {
            return std::nan("complex infinity or complex");
        }
        else
        {
            return 1 / x * std::exp(-x) * std::sqrt(1 + pi / 2 * x);
        }
    }

    /**
     * @brief Calculate Modified Bessel K2.
     *
     * @return double
     */
    double bessel_k2(double x)
    {
        if (x >= 705)
        {
            return 0;
        }

        else if (x < 0)
        {
            return std::nan("complex infinity or complex");
        }

        else
        {
            return (1.0 / (x * x)) * ((15.0 / 8.0) + x) * std::sqrt(1.0 + (pi / 2.0) * x) * std::exp(-x);
        }
    }
}