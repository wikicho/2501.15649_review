#include "sf_trig.h"

namespace sf
{
    double sech(double x)
    {
        return 1.0 / std::cosh(x);
    }
}
