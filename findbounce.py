import numpy as np
from scipy import optimize, special, interpolate
from scipy.integrate import simpson, quad, solve_ivp
from collections import namedtuple


class IntegrationError(Exception):
    """
    Used to indicate an integration error, primarily in :func:`rkqs`.
    """
    pass


class SingleFieldInstanton:
    """
    This class will calculate properties of an instanton with a single scalar
    Field without gravity using the overshoot/undershoot method.

    Most users will probably be primarily interested in the functions
    :func:`findProfile` and :func:`findAction`.

    Note
    ----
    When the bubble is thin-walled (due to nearly degenerate minima), an
    approximate solution is found to the equations of motion and integration
    starts close to the wall itself (instead of always starting at the center
    of the bubble). This way the overshoot/undershoot method runs just as fast
    for extremely thin-walled bubbles as it does for thick-walled bubbles.

    Parameters
    ----------
    phi_absMin : float
        The field value at the stable vacuum to which the ins tanton
        tunnels. Nowhere in the code is it *required* that there actually be a
        minimum at `phi_absMin`, but the :func:`findProfile` function will only
        use initial conditions between `phi_absMin` and `phi_metaMin`, and the
        code is optimized for thin-walled bubbles when the center of the
        instanton is close to `phi_absMin`.
    phi_metaMin : float
        The field value in the metastable vacuum.
    V : callable
        The potential function. It should take as its single parameter the field
        value `phi`.
    dV, d2V : callable, optional
        The potential's first and second derivatives. If not None, these
        override the methods :func:`dV` and :func:`d2V`.
    phi_eps : float, optional
        A small value used to calculate derivatives (if not overriden by
        the user) and in the function :func:`dV_from_absMin`. The input should
        be unitless; it is later rescaled by ``abs(phi_absMin - phi_metaMin)``.
    dimension : int or float, optional
        The coefficient for the friction term in the ODE. This is also
        the number of spacetime dimensions minus 1.
    phi_bar : float, optional
        The field value at the edge of the barrier. If `None`, it is found by
        :func:`findBarrierLocation`.
    rscale : float, optional
        The approximate radial scale of the instanton. If `None` it is found by
        :func:`findRScale`.
    """

    def __init__(self, phi_absMin, phi_metaMin, V, dV, d2V,
                 phi_eps=1e-9, dimension=2, phi_bar=None, rscale=None):
        self.phi_absMin = phi_absMin
        self.phi_metaMin = phi_metaMin
        self.V = V
        self.dV = dV
        self.d2V = d2V

        self.dimension = dimension
        self.phi_eps = phi_eps

        self.phi_bar = phi_bar
        self.rscale = rscale

    def findBarrierLocation(self):
        R"""
        Find edge of the potential barrier.

        Returns
        -------
        phi_barrier : float
            The value such that `V(phi_barrier) = V(phi_metaMin)`
        """
        phi_tol = abs(self.phi_metaMin - self.phi_absMin) * 1e-12
        V_phimeta = self.V(self.phi_metaMin)
        phi1 = self.phi_metaMin
        phi2 = self.phi_absMin
        phi0 = 0.5 * (phi1+phi2)

        # Do a very simple binary search to narrow down on the right answer.
        while abs(phi1-phi2) > phi_tol:
            V0 = self.V(phi0)
            if V0 > V_phimeta:
                phi1 = phi0
            else:
                phi2 = phi0
            phi0 = 0.5 * (phi1+phi2)
        return phi0

    _initialConditions_rval = namedtuple(
        "initialConditions_rval", "r0 phi dphi")

    def initialConditions(self, delta_phi0, rmin, delta_phi_cutoff):
        R"""
        Finds the initial conditions for integration.

        The instanton equations of motion are singular at `r=0`, so we
        need to start the integration at some larger radius. This
        function finds the value `r0` such that `phi(r0) = phi_cutoff`.
        If there is no such value, it returns the intial conditions at `rmin`.

        Parameters
        ----------
        delta_phi0 : float
            `delta_phi0 = phi(r=0) - phi_absMin`
        rmin : float
            The smallest acceptable radius at which to start integration.
        delta_phi_cutoff : float
            The desired value for `phi(r0)`.
            `delta_phi_cutoff = phi(r0) - phi_absMin`.

        Returns
        -------
        r0, phi, dphi : float
            The initial radius and the field and its derivative at that radius.

        Notes
        -----
        The field values are calculated using :func:`exactSolution`.
        """
        phi0 = self.phi_absMin + delta_phi0
        dV = self.dV_from_absMin(delta_phi0)
        d2V = self.d2V(phi0)
        phi_r0, dphi_r0 = self.exactSolution(rmin, phi0, dV, d2V)
        if abs(phi_r0 - self.phi_absMin) > abs(delta_phi_cutoff):
            # The initial conditions at rmin work. Stop here.
            return self._initialConditions_rval(rmin, phi_r0, dphi_r0)
        if np.sign(dphi_r0) != np.sign(delta_phi0):
            # The field is evolving in the wrong direction.
            # Increasing r0 won't increase |delta_phi_r0|/
            return rmin, phi_r0, dphi_r0

        # Find the smallest r0 such that delta_phi_r0 > delta_phi_cutoff
        r = rmin
        while np.isfinite(r):
            rlast = r
            r *= 10
            phi, dphi = self.exactSolution(r, phi0, dV, d2V)
            if abs(phi - self.phi_absMin) > abs(delta_phi_cutoff):
                break

        # Now find where phi - self.phi_absMin = delta_phi_cutoff exactly

        def deltaPhiDiff(r_):
            p = self.exactSolution(r_, phi0, dV, d2V)[0]
            return abs(p - self.phi_absMin) - abs(delta_phi_cutoff)

        r0 = optimize.brentq(deltaPhiDiff, rlast, r, disp=False)
        phi_r0, dphi_r0 = self.exactSolution(r0, phi0, dV, d2V)
        return self._initialConditions_rval(r0, phi_r0, dphi_r0)

    _exactSolution_rval = namedtuple("exactSolution_rval", "phi dphi")

    def exactSolution(self, r, phi0, dV, d2V):
        R"""
        Find `phi(r)` given `phi(r=0)`, assuming a quadratic potential.

        Parameters
        ----------
        r : float
            The radius at which the solution should be calculated.
        phi0 : float
            The field at `r=0`.
        dV, d2V : float
            The potential's first and second derivatives evaluated at `phi0`.

        Returns
        -------
        phi, dphi : float
            The field and its derivative evaluated at `r`.

        Notes
        -----

        If the potential at the point :math:`\phi_0` is a simple quadratic, the
        solution to the instanton equation of motion can be determined exactly.
        The non-singular solution to

        .. math::
          \frac{d^2\phi}{dr^2} + \frac{\alpha}{r}\frac{d\phi}{dr} =
          V'(\phi_0) + V''(\phi_0) (\phi-\phi_0)

        is

        .. math::
          \phi(r)-\phi_0 = \frac{V'}{V''}\left[
          \Gamma(\nu+1)\left(\frac{\beta r}{2}\right)^{-\nu} I_\nu(\beta r) - 1
          \right]

        where :math:`\nu = \frac{\alpha-1}{2}`, :math:`I_\nu` is the modified
        Bessel function, and :math:`\beta^2 = V''(\phi_0) > 0`. If instead
        :math:`-\beta^2 = V''(\phi_0) < 0`, the solution is the same but with
        :math:`I_\nu \rightarrow J_\nu`.

        """
        beta = np.sqrt(abs(d2V))
        beta_r = beta*r
        nu = 0.5 * (self.alpha - 1)
        gamma = special.gamma  # Gamma function
        iv, jv = special.iv, special.jv  # (modified) Bessel function
        if beta_r < 1e-2:
            # Use a small-r approximation for the Bessel function.
            s = +1 if d2V > 0 else -1
            phi = 0.0
            dphi = 0.0
            for k in range(1, 4):
                _ = (0.5*beta_r)**(2*k-2) * s**k / (gamma(k+1)*gamma(k+1+nu))
                phi += _
                dphi += _ * (2*k)
            phi *= 0.25 * gamma(nu+1) * r**2 * dV * s
            dphi *= 0.25 * gamma(nu+1) * r * dV * s
            phi += phi0
        elif d2V > 0:
            import warnings
            # If beta_r is very large, this will throw off overflow and divide
            # by zero errors in iv(). It will return np.inf though, which is
            # what we want. Just ignore the warnings.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                phi = (gamma(nu+1)*(0.5*beta_r)**-
                       nu * iv(nu, beta_r)-1) * dV/d2V
                dphi = -nu*((0.5*beta_r)**-nu / r) * iv(nu, beta_r)
                dphi += (0.5*beta_r)**-nu * 0.5*beta \
                    * (iv(nu-1, beta_r)+iv(nu+1, beta_r))
                dphi *= gamma(nu+1) * dV/d2V
                phi += phi0
        else:
            phi = (gamma(nu+1)*(0.5*beta_r)**-nu * jv(nu, beta_r) - 1) * dV/d2V
            dphi = -nu*((0.5*beta_r)**-nu / r) * jv(nu, beta_r)
            dphi += (0.5*beta_r)**-nu * 0.5*beta \
                * (jv(nu-1, beta_r)-jv(nu+1, beta_r))
            dphi *= gamma(nu+1) * dV/d2V
            phi += phi0
        return self._exactSolution_rval(phi, dphi)

    def equationOfMotion(self, y, r):
        """
        Used to integrate the bubble wall.
        """
        return np.array([y[1], self.dV(y[0])-self.dimension*y[1]/r])

    _integrateProfile_rval = namedtuple(
        "integrateProfile_rval", "r y convergence_type")

    def integrateProfile(self, r0, y0, dr0, epsfrac, epsabs, drmin, rmax, *eqn_args):
        R"""
        Integrate the bubble wall equation:

        .. math::
          \frac{d^2\phi}{dr^2} + \frac{\dimension}{r}\frac{d\phi}{dr} =
          \frac{dV}{d\phi}.

        The integration will stop when it either overshoots or undershoots
        the false vacuum minimum, or when it converges upon the false vacuum
        minimum.

        Parameters
        ----------
        r0 : float
            The starting radius for the integration.
        y0 : array_like
            The starting values [phi(r0), dphi(r0)].
        dr0 : float
            The starting integration stepsize.
        epsfrac, epsabs : float
            The error tolerances used for integration. This is fed into
            :func:`helper_functions.rkqs` and is used to test for convergence.
        drmin : float
            The minimum allowed value of `dr` before raising an error.
        rmax : float
            The maximum allowed value of `r-r0` before raising an error.
        eqn_args : tuple
            Extra arguments to pass to :func:`equationOfMotion`. Useful for
            subclasses.

        Returns
        -------
        r : float
            The final radius.
        y : array_like
            The final field values [phi, dphi]
        convergence_type : str
            Either 'overshoot', 'undershoot', or 'converged'.

        Raises
        ------
        helper_functions.IntegrationError
        """
        # build the ODE

        def fun(r, y):
            return self.equationOfMotion(y, r, *eqn_args)

        # set sign for overshoot/undershoot direction
        ysign = np.sign(y0[0] - self.phi_metaMin)

        # event: converged to false vacuum within tolerance
        def event_converged(r, y):
            return np.linalg.norm([y[0] - self.phi_metaMin, y[1]]) - 3*epsabs
        event_converged.terminal = True
        event_converged.direction = 0

        # event: overshoot (φ crosses φ_metaMin)
        def event_overshoot(r, y):
            return y[0] - self.phi_metaMin
        event_overshoot.terminal = True
        event_overshoot.direction = -ysign

        # event: undershoot (φ′ crosses zero)
        def event_undershoot(r, y):
            return y[1]
        event_undershoot.terminal = True
        event_undershoot.direction = ysign

        sol = solve_ivp(
            fun,
            (r0, r0 + rmax),
            y0,
            rtol=epsfrac,
            atol=epsabs,
            max_step=dr0,
            events=(event_converged, event_overshoot, event_undershoot)
        )

        # figure out which event fired first
        t_e, y_e = sol.t_events, sol.y_events
        if t_e[0].size:
            r_final, y_final, conv = t_e[0][0], y_e[0][:, 0], "converged"
        elif t_e[1].size:
            r_final, y_final, conv = t_e[1][0], y_e[1][:, 0], "overshoot"
        elif t_e[2].size:
            r_final, y_final, conv = t_e[2][0], y_e[2][:, 0], "undershoot"
        else:
            raise IntegrationError("integration reached rmax without event")

        return self._integrateProfile_rval(r_final, y_final, conv)

    def findAction(self, profile):
        R"""
        Calculate the Euclidean action for the instanton:

        .. math::
          S = \int [(d\phi/dr)^2 + V(\phi)] r^\alpha dr d\Omega_\alpha

        Arguments
        ---------
        profile
            Output from :func:`findProfile()`.

        Returns
        -------
        float
            The Euclidean action.
        """
        r, phi, dphi = profile.R, profile.Phi, profile.dPhi
        # Find the area of an n-sphere (alpha=n):
        d = self.alpha+1  # Number of dimensions in the integration
        area = r**self.alpha * 2*np.pi**(d*.5)/special.gamma(d*.5)
        # And integrate the profile
        integrand = 0.5 * dphi**2 + self.V(phi) - self.V(self.phi_metaMin)
        integrand *= area
        S = simpson(integrand, x=r)
        # Find the bulk term in the bubble interior
        volume = r[0]**d * np.pi**(d*.5)/special.gamma(d*.5 + 1)
        S += volume * (self.V(phi[0]) - self.V(self.phi_metaMin))
        return S
