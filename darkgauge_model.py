import numpy as np
import matplotlib.pyplot as plt
from thermal_functions import V1_finiteT, V1_prime_finiteT, V1_2prime_finiteT, cutoff
from findbounce import single_field_bounce
from scipy.optimize import minimize
from scipy.integrate import quad, solve_ivp
from scipy.special import kn
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import CubicSpline   

pi = np.pi

# reduced planck mass
Mpl = 2.435323204e18


class DarkAbelianHiggs:
    def __init__(self, v0=1.0, lambda_phi=6e-3, g_d=0.75):
        """
        Initialize the dark abelian Higgs model.

        Parameters:
          v0 : float
            Vacuum expectation value of the dark Higgs field in GeV.
          lambda_phi : float
            The quartic coupling of the dark Higgs field.
          g_d : float
            The coupling of the dark Higgs to the dark gauge boson.
          inputScale : float
            The scale at which the initial parameters are defined.
        """
        # One real degree of freedom (after gauge fixing)
        self.Ndim = 1

        # Set the tree-level potential parameters.
        self.lambda_phi = lambda_phi
        self.v0 = v0
        self.mu_eff = np.sqrt(self.lambda_phi) * self.v0

        # Set the gauge coupling.
        self.g_d = g_d

        # set the mass of the particles squared at the vev
        self.mPhi2_vev = -self.mu_eff**2 + 3 * self.lambda_phi * v0**2
        self.mGB2_vev = -self.mu_eff**2 + self.lambda_phi * v0**2
        self.mZd2_vev = self.g_d**2 * v0**2

        # degrees of freedom
        self.dof = np.array([1.0, 1.0, 3.0])
        #
        self.c = np.array([1.5, 1.5, 1.5])

        # Define the renormalization scales.
        self.renormScaleSq = v0

        self.T_crit = None  # Critical temperature for the phase transition.
        self.T_nucleation = None  # Nucleation temperature
        self.T_percolation = None  # Percolation temperature
        self.T_complete = None  # Complete percolation temperature.

    def mPhi2(self, X):
        """
        Compute the field-dependent mass-squared of the dark Higgs.
        """
        return -self.mu_eff**2 + 3 * self.lambda_phi * X**2

    def mGB2(self, X):
        """
        Compute the field-dependent mass-squared of the Goldstone boson.
        """
        return -self.mu_eff**2 + self.lambda_phi * X**2

    def mZd2(self, X):
        """
        Compute the field-dependent mass-squared of the dark gauge boson.
        """
        return self.g_d**2 * X**2

    def thermal_mass_sqaured_phi(self, T):
        '''
        Compute the debye mass for the phi 
        '''
        return 0.5 * (self.lambda_phi + 0.5 * self.g_d**2) * T**2

    def thermal_mass_sqaured_GB(self, T):

        return 0.5 * (self.lambda_phi + 0.5 * self.g_d**2) * T**2

    def thermal_mass_squared_ZD(self, T):

        return self.g_d**2 * T**2 / 3.0

    def V0(self, X):
        """
        Tree-level potential:

        V0 = -0.5 * mu_eff**2 * phi^2 + 0.25 * lam_eff * phi^4.
        """
        phi = X
        return -0.5 * self.mu_eff**2 * phi**2 + 0.25 * self.lambda_phi * phi**4

    def V1(self, X):
        """
        One-loop zero-T CW potential
        """
        term_phi = self.dof[0] * (
            self.mPhi2(X) ** 2
            * (np.log(np.abs(self.mPhi2(X) / self.mPhi2_vev) + 1e-100) - self.c[0])
            + 2 * self.mPhi2_vev * self.mPhi2(X)
        )
        term_GB = self.dof[1] * (
            self.mGB2(X) ** 2
            * (np.log(np.abs(self.mGB2(X) / self.mPhi2_vev) + 1e-100) - self.c[1])
        )
        term_ZD = self.dof[2] * (
            self.mZd2(X) ** 2
            * (np.log(np.abs(self.mZd2(X) / self.mZd2_vev) + 1e-100) - self.c[2])
            + 2 * self.mZd2_vev * self.mZd2(X)
        )

        return (term_phi + term_GB + term_ZD) / (64 * np.pi**2)

    def V1T(self, X, T):
        """
        One-loop finite-temperature effective potential.

        Parameters
        ----------
        bosons : tuple
            A tuple containing the field-dependent mass-squared matrix,
            degrees of freedom, and c-factors for the bosons.
        fermions : tuple
            A tuple containing the field-dependent mass-squared matrix,
            degrees of freedom, and c-factors for the fermions.
        T : float or array_like
            The temperature at which to evaluate the potential.
        include_radiation : bool, optional
            If False, this will drop all field-independent radiation
            terms from the effective potential. Useful for calculating
            differences or derivatives.
        """
        if np.isscalar(X):
            return (
                self.dof[0] * V1_finiteT(T, self.mPhi2(X))
                + self.dof[1] * V1_finiteT(T, self.mGB2(X))
                + self.dof[2] * V1_finiteT(T, self.mZd2(X))
            )

        result = np.zeros_like(X)

        for i in range(len(X)):
            result[i] += self.dof[0] * V1_finiteT(T, self.mPhi2(X[i]))
            result[i] += self.dof[1] * V1_finiteT(T, self.mGB2(X[i]))
            result[i] += self.dof[2] * V1_finiteT(T, self.mZd2(X[i]))

        return result

    def V1_Daisy(self, X, T):
        '''
        One-loop finite-temperature effective potential with daisy resummation.
        Parameters
        ----------
        X : array_like
            Field value(s).
            Either a single point (with length `Ndim`), or an array of points.
        T : float or array_like
            The temperature at which to evaluate the potential.        
        -------
        '''

        mPhi2 = self.mPhi2(X)
        mGB2 = self.mGB2(X)
        mZd2 = self.mZd2(X)

        if np.isscalar(X):
            if mPhi2 < 0:
                if mPhi2 + self.thermal_mass_sqaured_phi(T) < 0:
                    mphi2_contribution = 0
                else:
                    mphi2_contribution = (
                        mPhi2 + self.thermal_mass_sqaured_phi(T))**(1.5)
            else:
                mphi2_contribution = mPhi2**(1.5) - \
                    (self.thermal_mass_sqaured_phi(T) + mPhi2)**(1.5)

            if mGB2 < 0:
                if mGB2 + self.thermal_mass_sqaured_GB(T) < 0:
                    mGB2_contribution = 0
                else:
                    mGB2_contribution = (
                        mGB2 + self.thermal_mass_sqaured_GB(T))**(1.5)
            else:
                mGB2_contribution = mGB2**(1.5) - \
                    (self.thermal_mass_sqaured_GB(T) + mGB2)**(1.5)

            if mZd2 < 0:
                if mZd2 + self.thermal_mass_squared_ZD(T) < 0:
                    mZd2_contribution = 0
                else:
                    mZd2_contribution = (
                        mZd2 + self.thermal_mass_squared_ZD(T))**(1.5)
            else:
                mZd2_contribution = mZd2**(1.5) - \
                    (self.thermal_mass_squared_ZD(T) + mZd2)**(1.5)

            result = mphi2_contribution + mGB2_contribution + mZd2_contribution

            return T / (12 * np.pi) * result

        else:
            result = np.zeros_like(X)

            for i in range(len(X)):
                if mPhi2[i] < 0:
                    if mPhi2[i] + self.thermal_mass_sqaured_phi(T) < 0:
                        mphi2_contribution = 0
                    else:
                        mphi2_contribution = (
                            mPhi2[i] + self.thermal_mass_sqaured_phi(T))**(1.5)
                else:
                    mphi2_contribution = mPhi2[i]**(1.5) - (
                        self.thermal_mass_sqaured_phi(T) + mPhi2[i])**(1.5)

                if mGB2[i] < 0:
                    if mGB2[i] + self.thermal_mass_sqaured_GB(T) < 0:
                        mGB2_contribution = 0
                    else:
                        mGB2_contribution = (
                            mGB2[i] + self.thermal_mass_sqaured_GB(T))**(1.5)
                else:
                    mGB2_contribution = mGB2[i]**(1.5) - (
                        self.thermal_mass_sqaured_GB(T) + mGB2[i])**(1.5)

                if mZd2[i] < 0:
                    if mZd2[i] + self.thermal_mass_squared_ZD(T) < 0:
                        mZd2_contribution = 0
                    else:
                        mZd2_contribution = (
                            mZd2[i] + self.thermal_mass_squared_ZD(T))**(1.5)
                else:
                    mZd2_contribution = mZd2[i]**(1.5) - (
                        self.thermal_mass_squared_ZD(T) * cutoff(mZd2[i]/T**2) + mZd2[i])**(1.5)

                result[i] += (mphi2_contribution +
                              mGB2_contribution + mZd2_contribution)

            return T / (12 * np.pi) * result

    def dV1T_dT(self, X, T):
        """
        Derivative of the one-loop finite-temperature effective potential
        with respect to temperature.
        """

        if np.isscalar(X):
            return (
                self.dof[0] * V1_prime_finiteT(T, self.mPhi2(X))
                + self.dof[1] * V1_prime_finiteT(T, self.mGB2(X))
                + self.dof[2] * V1_prime_finiteT(T, self.mZd2(X))
            )

        result = np.zeros_like(X)

        # Calculate the derivative for each field.
        for i in range(len(X)):
            result[i] += self.dof[0] * V1_prime_finiteT(T, self.mPhi2(X[i]))
            result[i] += self.dof[1] * V1_prime_finiteT(T, self.mGB2(X[i]))
            result[i] += self.dof[2] * V1_prime_finiteT(T, self.mZd2(X[i]))

        return result

    def d2V1T_dT2(self, X, T):
        """
        Second derivative of the one-loop finite-temperature effective potential
        with respect to temperature.
        """
        if np.isscalar(X):
            return (
                self.dof[0] * V1_2prime_finiteT(T, self.mPhi2(X))
                + self.dof[1] * V1_2prime_finiteT(T, self.mGB2(X))
                + self.dof[2] * V1_2prime_finiteT(T, self.mZd2(X))
            )

        result = np.zeros_like(X)

        # Calculate the second derivative for each field.
        for i in range(len(X)):
            result[i] += self.dof[0] * V1_2prime_finiteT(T, self.mPhi2(X[i]))
            result[i] += self.dof[1] * V1_2prime_finiteT(T, self.mGB2(X[i]))
            result[i] += self.dof[2] * V1_2prime_finiteT(T, self.mZd2(X[i]))

        # Return the result as an array.
        return result

    def Vtot(self, X, T, include_radiation=False):
        """
        The total finite temperature effective potential.

        Parameters
        ----------
        X : array_like
            Field value(s).
            Either a single point (with length `Ndim`), or an array of points.
        T : float or array_like
            The temperature. The shapes of `X` and `T`
            should be such that ``X.shape[:-1]`` and ``T.shape`` are
            broadcastable (that is, ``X[...,0]*T`` is a valid operation).
        include_radiation : bool, optional
            If False, this will drop all field-independent radiation
            terms from the effective potential. Useful for calculating
            differences or derivatives.
        """

        # evaluate tree‐level
        y0 = self.V0(X)  # shape (N,1) if X was 2‐d
        # squeeze off last axis so y is (N,) or () for a scalar
        y = np.squeeze(y0)

        # now add 1‐loop pieces, both return (N,)
        y = y + self.V1(X)
        if T > 0:
            y = y + self.V1T(X, T)

        return y

    def cs2(self, T):
        """
        Compute the speed of sound squared.
        """

        return self.dV1T_dT(self.v0-1e-15, T) / (T * self.d2V1T_dT2(self.v0-1e-15, T))

    def w_density(self, X, T):
        """
        Compute the enthalpy density of the radiation.
        """
        return -T * self.dV1T_dT(X, T)

    def p(self, X, T):
        """
        Compute the pressure from the effective potential.
        """
        return -self.Vtot(X, T)

    def rho(self, X, T):
        """
        Compute the energy density from the effective potential.
        """
        return self.Vtot(X, T) - T * self.dV1T_dT(X, T)

    def pseudo_trace(self, X, T):
        """
        Compute the pseudo trace
        """

        return 0.25 * (self.rho(X, T) - self.p(X, T) / self.cs2(T))

    def alpha(self, T_percolation):
        """
        Compute the strength of the transition parameter alpha
        """
        # Get the field-dependent mass-squared matrix.

        thetabar_false = self.pseudo_trace(1e-9, T_percolation)

        # pseudo trace at true vacuum
        thetabar_true = self.pseudo_trace(
            self.v0-1e-9, T_percolation)  # pseudo trace at true vacuum

        # enthalpy density at false vacuum
        w_density_false = self.w_density(1e-9, T_percolation)

        # Calculate the alpha parameter.
        alpha = 4.0 / 3.0 * (thetabar_false - thetabar_true) / w_density_false

        return alpha

    def findTCrit(self, T_max=0.5, T_min=1e-4, dT=1e-3, tol=3e-6):
        """
        Find the critical temperature for the phase transition.
        """
        phi = np.linspace(0.5 * self.v0, self.v0, 1000)
        # from v_0/2 to v_0, calculate the effective potential difference \delta V = V(phi, T) - V(0, T) < TOL, break the loop else decrease T by dT.
        for Ti in np.arange(T_max, T_min, -dT):
            # Calculate the effective potential difference at the critical temperature.
            dV = self.Vtot(phi, Ti) - self.Vtot(0, Ti)
            # Check if the effective potential difference is within the tolerance.
            if np.any(np.abs(dV) < tol):
                return Ti

    def findBounce(self, T):
        '''
        return the bounce solution and action at temperature T.
        '''
        sol, action = single_field_bounce(
            self.Vtot, self.dV1T_dT, self.d2V1T_dT2, T, v0=self.v0, tol=1e-6)

        sol_interpolated = CubicSpline(sol['r'], sol['phi'])

        r_max = sol['r'][-1]  # Maximum radius from the bounce solution.

        # Check if the bounce solution is valid.
        if sol is None or action is None:
            print(f"No bounce solution found at T = {T} GeV.")
            return None, None

        return action, sol, r_max

    def mathcal_J(self, T):
        """
        Compute the integral J(T) for the bounce action.
        This function is a placeholder and does not implement the actual integral logic.
        """
        return 3.0 * self.dV1T_dT(0, T) / (self.d2V1T_dT2(0, T))

    def scale_factor(self, T_f, T_i):
        """ Compute the scale factor between two temperatures
        """
        # integrate the mathcal_J from T_f to T_i
        integral, _ = quad(self.mathcal_J, T_f, T_i)
        return np.exp(integral)

    def volume_bubble(self, T_f, T_i):
        """
        Compute the volume of the bubble at temperature T.
        """
        # Get the bounce solution.

        # define integrand function
        def integrand(T):
            return self.scale_factor(T_f, T) / (self.mathcal_J(T) * self.hubble_rate(T))

        result, err = quad(self.integrand(T), T_f, T_i)

        v_w = 0  # not implemented yet, placeholder for bubble wall velocity

        return 4 * pi * v_w**3 / 3.0 * result**3

    def false_vacuum_fraction(self, T):
        """
        Compute the fraction of the universe in the false vacuum at temperature T.
        """

        # define the integrand function as a lambda function
        def integrand(T_prime):
            return self.scale_factor(T, T_prime)**(-3) * self.nucleation_rate(T_prime) * self.volume_bubble(T, T_prime) / (self.mathcal_J(T_prime) * self.hubble_rate(T_prime))

        result, err = quad(integrand, T, self.T_crit)

        return np.exp(-result)

    def findTp(self):
        """
        Find the percolation temperature T_p.
        This function is a placeholder and does not implement the actual percolation logic.
        """
        # find false_vacuum_fraction(T_p) = 0.71, using a root-finding method.
        def func(T):
            return self.false_vacuum_fraction(T) - 0.71

        # Use a numerical method to find the root.
        T_p = minimize(func, x0=0.1, bounds=[(1e-4, 0.5)]).x[0]

        return T_p

    def findTcomplete(self):
        """
        Find the complete percolation temperature T_complete.
        This function is a placeholder and does not implement the actual percolation logic.
        """
        # find false_vacuum_fraction(T_complete) = 0.01, using a root-finding method.
        def func(T):
            return self.false_vacuum_fraction(T) - 0.01

        # Use a numerical method to find the root.
        T_complete = minimize(func, x0=0.1, bounds=[(1e-4, 0.5)]).x[0]

        return T_complete

    def radius_init(self):
        """
        Return the bounce radius.
        """

        sol, action, r_max = self.findBounce(0.02254)

        # integrate the bounce solution to find the radius

        def integrand(r):
            return 4 * pi * r**2 * self.Vtot(sol(r), 0.02254)

        result, err = quad(integrand, 0, r_max)

        return (3 * result/(4*pi * (self.Vtot(0) - self.Vtot(self.v0))))**(1/3)

    def radius_sep(self):
        """
        Return the bounce radius.
        """
        return 0.0

    def nucleation_rate(self, T):
        """
        Compute the nucleation rate at temperature T.
        """
        # Find the bounce solution.
        action, sol = self.findBounce(T)
        if not sol:
            return 0.0

        # Calculate the nucleation rate.
        return T**4 * (action / (2 * pi * T))**(3/2) * np.exp(-action/T)

    def hubble_rate(self, T):
        """
        Compute the Hubble rate at temperature T.
        """
        rho_radiation = pi**2 / 30 * self.g_eff(T) * T**4

        delta_v = self.Vtot(0, T) - self.Vtot(self.v0-1e-9, T)

        return np.sqrt((rho_radiation + delta_v) / 3.0) / Mpl

    def g_eff(self, T):
        """
        Compute the effective number of relativistic degrees of freedom at temperature T.
        """
        return 0

    def gamma_star(self):
        return 2.0/3.0 * self.radius_sep() / self.radius_init()


if __name__ == "__main__":

    tick_font_size = 22
    label_font_size = 24
    fig_size = (9, 6.75)

    plt.rc("axes", facecolor="white")
    plt.rc("savefig", facecolor="white")
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", weight="normal")
    plt.rc("hatch", color="b")
    plt.rc("xtick", labelsize=tick_font_size)
    plt.rc("ytick", labelsize=tick_font_size)
    plt.rc("axes", labelsize=label_font_size)
    plt.rc("lines", linewidth=2.5)
    plt.rc("legend", framealpha=1.0)
    plt.rc("legend", edgecolor="black")
    plt.rc("axes", grid=True)
    plt.rc("savefig", pad_inches=0.1)
    plt.rc("figure", figsize=fig_size)

    # Example usage
    model_bp2 = DarkAbelianHiggs(v0=1.0, lambda_phi=6e-3, g_d=0.75)

    phi_values = np.linspace(0, 1.5, 1000)
    v = np.zeros_like(phi_values)

    print(model_bp2.Vtot(0, 0.02254))

    plt.figure(figsize=fig_size)
    # main potential curves
    T1 = 0.22799999999999976
    T2 = 22.54e-3
    T3 = 0.5
    y1 = model_bp2.Vtot(phi_values, T1) - model_bp2.Vtot(v, T1)
    y2 = model_bp2.Vtot(phi_values, T2) - model_bp2.Vtot(v, T2)
    y3 = model_bp2.Vtot(phi_values, T3) - model_bp2.Vtot(v, T3)
    plt.plot(phi_values, y1, label=r"$T = T_{\mathrm{crit}}$",)
    plt.plot(phi_values, y2, label=r"$T = T_P$",)
    plt.plot(phi_values, y3, label=r"$T = 500~\mathrm{MeV}$",)

    plt.axvline(x=1, color="k", linestyle="--", lw=2.5)
    plt.xlabel(r"$\langle \phi \rangle$")
    plt.ylabel(r"$V_{\mathrm{eff}}(\phi) - V_\mathrm{eff}(0)$")
    plt.xlim(0, 1.5)
    plt.ylim(-1e-3, 5e-3)
    plt.legend(loc=1, title=r"$v_\phi = 1~\mathrm{GeV}, \lambda = 0.006, g_d = 0.75$", prop={
               "size": 14})
    # create inset showing zoom around phi in [0,0.03]
    ax = plt.gca()
    ax_inset = inset_axes(ax, width="25%", height="25%",
                          loc="upper left", borderpad=2)
    ax_inset.plot(phi_values, y1, color='C0')
    ax_inset.plot(phi_values, y2, color='C1')
    # ax_inset.plot(phi_values, y3, color='C2')
    ax_inset.set_xlim(1e-3, 5e-2)
    # adjust y-limits if needed
    ax_inset.set_ylim(0, 2.e-8)
    # ax_inset.set_xticklabels(labels=["0", "0.025", "0.05"], fontsize=10)
    # ax_inset.set_yticklabels(labels=["0", r"$10^{-8}$", "2"], fontsize=10)
    plt.savefig("./figures/potential_bp2.pdf", bbox_inches="tight")

    # Calculate the strength of the transition parameter alpha at T=22.54 MeV,
    # which is the critical temperature for the phase transition.

    # T_crit = model.findTCrit(T_max=0.5, T_min=1e-3, dT=1e-3, tol=3e-6)
    # print("Critical temperature for the phase transition:", T_crit)

    # sound of speed
    T = np.logspace(-2, 0, 1000)
    cs2 = np.array([model_bp2.cs2(Ti) for Ti in T]) / (1/3)

    plt.figure(figsize=fig_size)
    plt.plot(T, cs2, label=r"$c_s^2(T)$")
    plt.xlabel(r"$T$ [GeV]")
    plt.ylabel(r"$c_{s,t}^2(T)/(1/3)$")
    plt.xscale("log")
    plt.yscale("linear")
    plt.xlim(0.01, 1)
    plt.ylim(0.8, 1)
    plt.tight_layout()
    plt.axvline(x=0.02254, color="k", linestyle="--", lw=2.5)
    # plt.axhline(y=model.cs2(0.02254)/(1/3), color="k", linestyle="--", lw=2.5)
    plt.legend(loc=0)
    plt.savefig("./figures/sound_of_speed_bp2.pdf", bbox_inches="tight")

    alpha = model_bp2.alpha(22.54e-3)
    print("strength of the transition parameter alpha at T=22.54 MeV: %.2e" % alpha)
