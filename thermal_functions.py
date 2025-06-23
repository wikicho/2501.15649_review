import numpy as np
from scipy.special import kn, yn, zeta, gamma, factorial

# ------------------------------------------------------------------
# high - temperature expansion y^2 << 1
# ------------------------------------------------------------------


def jb_low(y2: float, N: int = 10) -> float:

    pi = np.pi
    aB = np.exp(3/2 - 2*np.euler_gamma + 2*np.log(4*pi))
    series = 0.0

    if abs(y2) <= 1e-6:
        return -2.16465  # -np.pi**4 / 45

    for l in range(1, N+1):
        term = ((-1)**l * zeta(2*l + 1) / factorial(l + 1) *
                gamma(l + 0.5) * (y2 / (4 * pi**2))**(l + 2))
        series += term

    # if y2 < 0, we use pure imaginary argument
    if y2 < 0:
        # If y2 < 0, the result may be complex. Return only the real part.
        result = (-pi**4/45 + pi**2/12 * y2 - (1/32) * y2**2 * np.log(-y2 / aB) - 2 * pi**(7/2) * series)
    else:
        result = (-pi**4/45 + pi**2/12 * y2 - pi/6 * np.power(y2, 3/2) -
              (1/32) * y2**2 * np.log(y2 / aB) - 2 * pi**(7/2) * series)
    
    # Return the real part of the result
    return result


def jb_prime_low(y2: float, N: int = 10) -> float:

    pi = np.pi
    aB = np.exp(3/2 - 2*np.euler_gamma + 2*np.log(4*pi))
    series = 0.0

    if abs(y2) <= 1e-6:
        return np.pi**2 / 12

    for l in range(1, N+1):
        term = ((-1)**l * zeta(2*l + 1) / factorial(l + 1) *
                gamma(l + 0.5) * (l+2) * y2**(l + 1) / (4 * pi**2)**(l + 2))
        series += term

    if y2 < 0:
        result = pi**2 / 12.0 - y2/32.0 - (1/16) * y2 * np.log(-y2/aB)
    else:
        result = pi**2 / 12.0 - pi / 4.0 * \
            np.sqrt(y2) - y2/32.0 - (1/16) * y2 * np.log(y2/aB)

    result = result - 2 * pi**(7/2) * series

    return result


def jb_2prime_low(y2: float, N: int = 10) -> float:

    pi = np.pi
    aB = np.exp(3/2 - 2*np.euler_gamma + 2*np.log(4*pi))
    series = 0.0

    for l in range(1, N+1): 
        inner_term = (1 + l) * (2 + l) * y2**l
        term = (-1)**l * 4.0 ** (-2.0 - l) * pi ** (-2.0 * (2 + l)) * inner_term * \
            gamma(0.5 + l) * zeta(1 + 2 * l, 1) / factorial(1 + l, exact=False)
        series += term

    if y2 < 0:
        result = 3.0/32.0 - (1/16) * np.log(-y2/aB)

    else:
        result = 3.0/32.0 - 0.125 * pi / np.sqrt(y2) - (1/16) * np.log(y2/aB)

    result = result - 2 * pi**(7/2) * series

    return result


# ------------------------------------------------------------------
# low - temperature expansion y^2 >> 1
# ------------------------------------------------------------------

def jb_high(y2: float, n: int = 10) -> float:

    pi = np.pi

    series = 0.0

    if y2 < 0:
        for k in range(1, n+1):
            series += 0.5 * pi * yn(2, k * np.sqrt(-y2)) / (k * k)
    else:
        for k in range(1, n+1):
            series += kn(2, k * np.sqrt(y2)) / (k * k)

    return -y2 * series


def jb_prime_high(y2: float, n: int = 10) -> float:
    """
    Derivative of the low-temperature expansion.
    """
    pi = np.pi
    series = 0.0

    if y2 < 0:
        for k in range(1, n+1):
            series += -0.25 * pi * np.sqrt(-y2) * yn(1, k * np.sqrt(-y2)) / k
    else:
        for k in range(1, n+1):
            series += 0.5 * np.sqrt(y2) * kn(1, k * np.sqrt(y2)) / k

    return series


def jb_2prime_high(y2: float, n: int = 10) -> float:
    """
    Second derivative of the low-temperature expansion.
    """
    pi = np.pi
    series = 0.0

    if y2 < 0:
        for k in range(1, n+1):
            series += 0.125 * pi * yn(0, k * np.sqrt(-y2))
    else:
        for k in range(1, n+1):
            series += -0.25 * kn(0, k * np.sqrt(y2))

    return series

# ------------------------------------------------------------------

# Main function to test the approximations

def jb(y2: float, N: int = 10) -> float:
    """
    Main function to compute the Bessel function approximation.
    It chooses the appropriate approximation based on the value of y2.
    """
    if y2 < -0.279909:
        return jb_high(y2, N)
    elif y2 < 0:
        return jb_low(y2, N)
    elif y2 < 0.25905:
        return jb_low(y2, N)
    else:
        return jb_high(y2, N)
    
def jb_prime(y2: float, N: int = 10) -> float:
    """
    Main function to compute the first derivative of the Bessel function approximation.
    It chooses the appropriate approximation based on the value of y2.
    """
    if y2 < -0.155735:
        return jb_prime_high(y2, N)
    elif y2 < 0:
        return jb_prime_low(y2, N)
    elif y2 < 0.268101:
        return jb_prime_low(y2, N)
    else:
        return jb_prime_high(y2, N)
def jb_2prime(y2: float, N: int = 10) -> float:
    """
    Main function to compute the second derivative of the Bessel function approximation.
    It chooses the appropriate approximation based on the value of y2.
    """
    if y2 < -0.0724474:
        return jb_2prime_high(y2, N)
    elif y2 < 0:
        return jb_2prime_low(y2, N)
    elif y2 < 0.0285167:
        return jb_2prime_low(y2, N)
    else:
        return jb_2prime_high(y2, N)
    
# ------------------------------------------------------------------

def V1_finiteT(T: float, m2: float,  N: int = 10) -> float:
    """
    Computes the finite temperature correction to the volume.
    """
    y2 = m2 / (T**2)

    return T ** 4 * jb(y2, N) / (2 * np.pi**2)

def V1_prime_finiteT(T: float, m2: float, N: int = 10) -> float:
    
    y2 = m2 / (T**2)

    return 2 * T**3 * jb(y2, N) / (np.pi**2) - T**3 / (np.pi**2) * jb_prime(y2, N) * y2

def V1_2prime_finiteT(T: float, m2: float, N: int = 10) -> float:
    
    pi = np.pi
    y2 = m2 / (T**2)
    aB = np.exp(3/2 - 2*np.euler_gamma + 2*np.log(4*pi))


    if np.abs(y2) < 0.01 and y2 < 0:
        jb_2prime_value_y22 = 0.03125 * y2**2 * (3.0 - 2 * np.log(np.abs(y2/aB + 1e-100)))
        return 6 * T**2 / (np.pi**2) * jb(y2, N)  - 5 * T**2 / (np.pi**2) * jb_prime(y2, N) * y2 + 2 * T**2 / (np.pi**2) * jb_2prime_value_y22
    elif np.abs(y2) < 0.01 and y2 >= 0:
        jb_2prime_value_y22 = 0.03125 * y2**2 * (3.0 - 2 * np.log(y2/aB + 1e-100)) - 0.125 * pi * np.sqrt(y2) * y2
        return 6 * T**2 / (np.pi**2) * jb(y2, N)  - 5 * T**2 / (np.pi**2) * jb_prime(y2, N) * y2 + 2 * T**2 / (np.pi**2) * jb_2prime_value_y22
    else:
        return 6 * T**2 / (np.pi**2) * jb(y2, N)  - 5 * T**2 / (np.pi**2) * jb_prime(y2, N) * y2 + 2 * T**2 / (np.pi**2) * jb_2prime(y2, N) * y2**2
        



# ------------------------------------------------------------------
# Daisy corrections
def daisy_correction(T: float, m2: float, N: int = 10) -> float:
    """
    Computes the daisy correction to the mass.
    """
    y2 = m2 / (T**2)
    return T**2 * jb(y2, N) / (4 * np.pi**2)
def daisy_correction_prime(T: float, m2: float, N: int = 10) -> float:
    """
    Computes the first derivative of the daisy correction to the mass.
    """
    y2 = m2 / (T**2)
    return T * jb(y2, N) / (2 * np.pi**2) - T / (2 * np.pi**2) * jb_prime(y2, N) * y2
def daisy_correction_2prime(T: float, m2: float, N: int = 10) -> float:
    """
    Computes the second derivative of the daisy correction to the mass.
    """
    y2 = m2 / (T**2)
    return jb(y2, N) / (2 * np.pi**2) - T / (np.pi**2) * jb_prime(y2, N) * y2 - T / (4 * np.pi**2) * jb_2prime(y2, N) * y2**2


# ------------------------------------------------------------------
def cutoff(y2):
    """
    Compute the cutoff for the effective potential.
    This is a placeholder function and should be replaced with
    the actual implementation of the cutoff.0
    """

    if y2 < 0:
        return np.nan()
    elif y2 <= 1e-6:
        return 1
    else:
        return 0.5 * y2 * kn(2,np.sqrt(y2)) 

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    # Example usage
    if len(sys.argv) > 1:
        y = float(sys.argv[1])
    else:
        y = float(input("Enter the value of y: "))

    print("Jb_low(x):", jb_low(y))
    print("Jb_prime_low(x):", jb_prime_low(y))
    print("Jb_2prime_low(x):", jb_2prime_low(y))
    print("Jb_high(x):", jb_high(y))
    print("Jb_prime_high(x):", jb_prime_high(y))
    print("Jb_2prime_high(x):", jb_2prime_high(y))

    x = np.logspace(-2, 1, 100)
    m2 = -0.006
    y= [V1_2prime_finiteT(t, m2) for t in x]
    plt.plot(x, y)
    plt.xlabel('T')
    plt.ylabel('V1_2prime_finiteT(T, m)')
    plt.title('Bessel Function Approximations')
    plt.xscale('log')
    plt.yscale('symlog')
    plt.legend()
    plt.grid()
    plt.show()
