from types import SimpleNamespace
from copy import deepcopy
import numpy as np
import matplotlib
import matplotlib as mpl

try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def _maybe_show(show: bool = True):
    """Only call plt.show() on interactive backends."""
    if not show:
        return
    backend = matplotlib.get_backend().lower()
    if "agg" not in backend:
        plt.show()


class PreannouncedDeficit_OLG_Original:
    """
    Original deterministic anticipated-deficit OLG-NK self-financing model.
    Kept only for the mu=0 nesting tests.
    """

    def __init__(self, verbose=True, eps0: float = 0.01):
        self.verbose = verbose
        self.eps0 = float(eps0)
        self.par = SimpleNamespace()
        self.eqsys = SimpleNamespace()
        self.sol = SimpleNamespace()
        self.sol_all = SimpleNamespace()
        self.set_parameters()
        self.allocate()

    def set_parameters(self):
        par = self.par
        par.beta = 0.998
        par.omega = 0.865
        par.sigma = 1.0
        par.tau_y = 1.0 / 3.0
        par.tau_d = 0.026
        par.kappa = 0.0062
        par.Dbar = 1.04
        par.alpha_y = 0.0
        par.alpha_pi = 0.0
        par.T = 500
        par.announce_t = 0
        par.implement_t = 8
        par.lag_max = 80
        par.x_plot_max = 40
        par.horizon_x_max = 80

    def allocate(self):
        self.sol_all.results = []

    def model_coefficients(self):
        par = self.par
        eqsys = self.eqsys

        Delta = 1.0 - par.omega * (1.0 - par.tau_d)
        if abs(Delta) < 1e-14:
            raise ValueError("Denominator too close to zero.")

        eqsys.Delta = Delta
        eqsys.X_d = (
            (1.0 - par.beta * par.omega)
            * (1.0 - par.tau_d)
            * (1.0 - par.omega)
            / Delta
        )
        eqsys.X_y = 1.0 - par.tau_y * (1.0 - par.omega) / Delta
        eqsys.X_r = (
            par.sigma * par.beta * par.omega / (1.0 - par.beta * par.omega)
            - par.Dbar * par.beta * (1.0 - par.omega) / Delta
        )

    def _deficit_path_from_announcement(self, announce_t: int, implement_t: int):
        par = self.par

        if implement_t < announce_t:
            raise ValueError("implement_t must be >= announce_t.")

        lag = implement_t - announce_t
        n = par.T + 1

        e = np.zeros(n)
        if lag <= par.T:
            e[lag] = self.eps0

        S_e = np.zeros(n)
        for t in range(n):
            if t <= lag:
                S_e[t] = self.eps0 * (par.beta * par.omega) ** (lag - t)

        return e, S_e, lag

    def build_system(self, announce_t=None, implement_t=None):
        par = self.par
        eqsys = self.eqsys

        if announce_t is None:
            announce_t = par.announce_t
        if implement_t is None:
            implement_t = par.implement_t

        self.model_coefficients()
        e, S_e, lag = self._deficit_path_from_announcement(announce_t, implement_t)

        n = par.T + 1
        N = 5 * n
        offsets = {"y": 0, "pi": n, "d": 2 * n, "Sy": 3 * n, "Sr": 4 * n}

        def idx(block, t):
            return offsets[block] + t

        A = np.zeros((N, N))
        b = np.zeros(N)
        row = 0

        X_d, X_y, X_r = eqsys.X_d, eqsys.X_y, eqsys.X_r

        for t in range(n):
            A[row, idx("y", t)] = 1.0
            A[row, idx("d", t)] = -X_d
            A[row, idx("Sy", t)] = -(1.0 - par.beta * par.omega) * X_y
            A[row, idx("Sr", t)] = +(1.0 - par.beta * par.omega) * X_r
            b[row] = X_d * S_e[t]
            row += 1

        for t in range(n):
            A[row, idx("Sy", t)] = 1.0
            A[row, idx("y", t)] = -1.0
            if t < par.T:
                A[row, idx("Sy", t + 1)] = -par.beta * par.omega
            row += 1

        for t in range(n):
            A[row, idx("Sr", t)] = 1.0
            A[row, idx("y", t)] = -par.alpha_y
            A[row, idx("pi", t)] = -par.alpha_pi
            if t < par.T:
                A[row, idx("Sr", t + 1)] = -par.beta * par.omega
            row += 1

        for t in range(n):
            A[row, idx("pi", t)] = 1.0
            A[row, idx("y", t)] = -par.kappa
            if t < par.T:
                A[row, idx("pi", t + 1)] = -par.beta
            row += 1

        for t in range(par.T):
            A[row, idx("d", t + 1)] = 1.0
            A[row, idx("d", t)] = -(1.0 - par.tau_d) / par.beta
            A[row, idx("y", t)] = par.tau_y / par.beta - par.Dbar * par.alpha_y
            A[row, idx("pi", t)] = -par.Dbar * par.alpha_pi
            b[row] = (1.0 - par.tau_d) / par.beta * e[t]
            row += 1

        A[row, idx("d", 0)] = 1.0
        A[row, idx("pi", 0)] = par.Dbar
        row += 1

        if row != N:
            raise RuntimeError(f"Equation count mismatch: row={row}, N={N}")

        eqsys.A = A
        eqsys.b = b
        eqsys.offsets = offsets
        eqsys.e = e
        eqsys.S_e = S_e
        eqsys.lag = lag

    def solve_model(self, announce_t=None, implement_t=None):
        par = self.par

        if announce_t is None:
            announce_t = par.announce_t
        if implement_t is None:
            implement_t = par.implement_t

        self.build_system(announce_t=announce_t, implement_t=implement_t)
        x = np.linalg.solve(self.eqsys.A, self.eqsys.b)

        n = par.T + 1
        off = self.eqsys.offsets

        self.sol = SimpleNamespace()
        self.sol.announce_t = int(announce_t)
        self.sol.implement_t = int(implement_t)
        self.sol.lag = int(self.eqsys.lag)

        self.sol.y = x[off["y"]: off["y"] + n]
        self.sol.pi = x[off["pi"]: off["pi"] + n]
        self.sol.d = x[off["d"]: off["d"] + n]
        self.sol.Sy = x[off["Sy"]: off["Sy"] + n]
        self.sol.Sr = x[off["Sr"]: off["Sr"] + n]

        self.sol.r = par.alpha_y * self.sol.y + par.alpha_pi * self.sol.pi

        exp_pi_next = np.empty_like(self.sol.pi)
        exp_pi_next[:-1] = self.sol.pi[1:]
        exp_pi_next[-1] = 0.0
        self.sol.i_nom = self.sol.r + exp_pi_next

        self.sol.e = self.eqsys.e.copy()
        self.sol.S_e = self.eqsys.S_e.copy()
        self.sol.t = np.arange(n)

        self.sol.debt_end = np.empty_like(self.sol.d)
        self.sol.debt_end[:-1] = self.sol.d[1:]
        self.sol.debt_end[-1] = np.nan

        self.sol.q = self.sol.d + self.sol.e
        self.sol.eps0 = self.eps0

        self.compute_financing_share()
        return self.sol

    def compute_financing_share(self):
        par = self.par
        sol = self.sol
        k = np.arange(len(sol.y))

        tax_base_gain = par.tau_y * np.sum((par.beta ** k) * sol.y)
        debt_erosion_gain = par.Dbar * sol.pi[0]
        servicing_cost = par.Dbar * np.sum((par.beta ** (k + 1)) * sol.r)
        pv_deficit = np.sum((par.beta ** k) * sol.e)

        denom = pv_deficit + servicing_cost
        if abs(denom) < 1e-14:
            raise ValueError("Self-financing denominator too close to zero.")

        fiscal_adjustment = par.tau_d * np.sum((par.beta ** k) * (sol.d + sol.e))

        sol.tax_base_gain = float(tax_base_gain)
        sol.debt_erosion_gain = float(debt_erosion_gain)
        sol.servicing_cost = float(servicing_cost)
        sol.pv_deficit = float(pv_deficit)
        sol.sf_denominator = float(denom)

        sol.share_tax_base = float(tax_base_gain / denom)
        sol.share_debt_erosion = float(debt_erosion_gain / denom)
        sol.share_actual = float((tax_base_gain + debt_erosion_gain) / denom)

        sol.fiscal_adjustment = float(fiscal_adjustment)
        sol.residual_need = float(fiscal_adjustment / denom)

        sol.budget_identity_error = float(
            denom - fiscal_adjustment - tax_base_gain - debt_erosion_gain
        )

        sol.share_actual_over_eps0 = float((tax_base_gain + debt_erosion_gain) / self.eps0)


class PreannouncedDeficit_OLG:
    """
    Preannounced-deficit OLG-NK model with HtM share mu.
    """

    def __init__(self, verbose=True, eps0: float = 0.01):
        self.verbose = verbose
        self.eps0 = float(eps0)
        self.par = SimpleNamespace()
        self.eqsys = SimpleNamespace()
        self.sol = SimpleNamespace()
        self.sol_all = SimpleNamespace()
        self.set_parameters()
        self.allocate()

    def set_parameters(self):
        par = self.par

        par.beta = 0.998
        par.omega = 0.865
        par.sigma = 1.0
        par.tau_y = 1.0 / 3.0
        par.tau_d = 0.026
        par.kappa = 0.0062
        par.Dbar = 1.04

        par.mu = 0.0

        par.alpha_y = 0.0
        par.alpha_pi = 0.0

        par.T = 500
        par.announce_t = 0
        par.implement_t = 8
        par.lag_max = 80
        par.x_plot_max = 40
        par.horizon_x_max = 80

    def allocate(self):
        self.sol_all.results = []
        self.sol_all.implement_t_list = []
        self.sol_all.share_actual_list = []
        self.sol_all.share_tax_base_list = []
        self.sol_all.share_debt_erosion_list = []
        self.sol_all.convergence_spread_list = []

    def model_coefficients(self):
        par = self.par
        eqsys = self.eqsys

        beta, omega = par.beta, par.omega
        tau_y, tau_d = par.tau_y, par.tau_d
        sigma, Dbar = par.sigma, par.Dbar
        mu = par.mu

        Delta = 1.0 - omega * (1.0 - tau_d)
        if abs(Delta) < 1e-14:
            raise ValueError("Denominator too close to zero.")

        eqsys.Delta = Delta

        eqsys.X_d_olg = (
            (1.0 - beta * omega)
            * (1.0 - tau_d)
            * (1.0 - omega)
            / Delta
        )

        eqsys.C_y_olg = (
            (1.0 - beta * omega)
            * (1.0 - tau_y * (1.0 - omega) / Delta)
        )

        eqsys.C_r_olg = (
            (1.0 - beta * omega)
            * (
                sigma * beta * omega / (1.0 - beta * omega)
                - Dbar * beta * (1.0 - omega) / Delta
            )
        )

        eqsys.mu_left = 1.0 - mu * (1.0 - tau_y)

        # Correct HtM debt loading: savers hold all public debt.
        eqsys.B_d = (
            (1.0 - mu) * eqsys.X_d_olg
            + mu * ((1.0 - beta * omega) - tau_d)
        )

        eqsys.B_y = (1.0 - mu) * eqsys.C_y_olg
        eqsys.B_r = (1.0 - mu) * eqsys.C_r_olg

        eqsys.B_e_current = mu * (1.0 - tau_d)
        eqsys.B_e_news = (1.0 - mu) * eqsys.X_d_olg

    def reduced_htm_dis_coefficients(self, tau_d=None, mu=None):
        par = self.par

        tau_d_use = par.tau_d if tau_d is None else float(tau_d)
        mu_use = par.mu if mu is None else float(mu)

        beta, omega = par.beta, par.omega
        tau_y, sigma, Dbar = par.tau_y, par.sigma, par.Dbar

        Delta = 1.0 - omega * (1.0 - tau_d_use)
        if abs(Delta) < 1e-14:
            raise ValueError("Denominator too close to zero.")

        X_d_olg = (
            (1.0 - beta * omega)
            * (1.0 - tau_d_use)
            * (1.0 - omega)
            / Delta
        )

        C_y_olg = (
            (1.0 - beta * omega)
            * (1.0 - tau_y * (1.0 - omega) / Delta)
        )

        C_r_olg = (
            (1.0 - beta * omega)
            * (
                sigma * beta * omega / (1.0 - beta * omega)
                - Dbar * beta * (1.0 - omega) / Delta
            )
        )

        mu_left = 1.0 - mu_use * (1.0 - tau_y)

        B_d = (
            (1.0 - mu_use) * X_d_olg
            + mu_use * ((1.0 - beta * omega) - tau_d_use)
        )
        B_y = (1.0 - mu_use) * C_y_olg
        B_r = (1.0 - mu_use) * C_r_olg

        reduced_denom = mu_left - B_y
        if abs(reduced_denom) < 1e-14:
            raise ValueError("Reduced HtM DIS denominator too close to zero.")

        X_d_mu = B_d / reduced_denom
        X_y_mu = B_y / reduced_denom
        X_r_mu = B_r / reduced_denom

        return {
            "Delta": float(Delta),
            "X_d_mu": float(X_d_mu),
            "X_y_mu": float(X_y_mu),
            "X_r_mu": float(X_r_mu),
            "reduced_denom": float(reduced_denom),
            "mu_left": float(mu_left),
            "B_d": float(B_d),
            "B_y": float(B_y),
            "B_r": float(B_r),
        }

    def homogeneous_system_matrix(self, tau_d=None):
        par = self.par
        tau_d_use = par.tau_d if tau_d is None else float(tau_d)

        red = self.reduced_htm_dis_coefficients(tau_d=tau_d_use, mu=par.mu)

        beta, omega = par.beta, par.omega
        tau_y, kappa = par.tau_y, par.kappa
        Dbar = par.Dbar
        alpha_y, alpha_pi = par.alpha_y, par.alpha_pi

        X_d_mu = red["X_d_mu"]
        X_y_mu = red["X_y_mu"]
        X_r_mu = red["X_r_mu"]

        denom_y = beta * omega * (1.0 + X_y_mu)
        if abs(denom_y) < 1e-14:
            raise ValueError("AD-row denominator too close to zero.")

        m_dd = (1.0 - tau_d_use) / beta
        m_dy = Dbar * alpha_y - tau_y / beta
        m_dpi = Dbar * alpha_pi

        m_yd = -X_d_mu * (1.0 - omega * (1.0 - tau_d_use)) / denom_y

        m_yy = (
            1.0
            - omega * X_d_mu * tau_y
            + (beta * omega * X_d_mu * Dbar + X_r_mu) * alpha_y
        ) / denom_y

        m_ypi = (
            (beta * omega * X_d_mu * Dbar + X_r_mu) * alpha_pi
        ) / denom_y

        A = np.array(
            [
                [m_dd, m_dy, m_dpi],
                [m_yd, m_yy, m_ypi],
                [0.0, -kappa / beta, 1.0 / beta],
            ],
            dtype=float,
        )

        return A

    def check_determinacy(self, tau_d=None, tol=1e-9):
        tau_d_use = self.par.tau_d if tau_d is None else float(tau_d)

        try:
            A = self.homogeneous_system_matrix(tau_d=tau_d_use)
            eigvals = np.linalg.eigvals(A)
            mod = np.abs(eigvals)

            n_stable = int(np.sum(mod < 1.0 - tol))
            n_unstable = int(np.sum(mod > 1.0 + tol))
            n_boundary = int(len(eigvals) - n_stable - n_unstable)

            is_unique_bounded = (
                n_stable == 1
                and n_unstable == 2
                and n_boundary == 0
            )

            return {
                "tau_d": float(tau_d_use),
                "A": A,
                "eigvals": eigvals,
                "n_stable": n_stable,
                "n_unstable": n_unstable,
                "n_boundary": n_boundary,
                "is_unique_bounded": bool(is_unique_bounded),
            }

        except Exception as exc:
            return {
                "tau_d": float(tau_d_use),
                "A": None,
                "eigvals": np.array([]),
                "n_stable": 0,
                "n_unstable": 0,
                "n_boundary": 3,
                "is_unique_bounded": False,
                "error": str(exc),
            }

    def _require_determinate(self, tau_d=None, tol=1e-9):
        info = self.check_determinacy(tau_d=tau_d, tol=tol)
        if not info["is_unique_bounded"]:
            raise RuntimeError(
                f"No unique bounded equilibrium from BK matrix for tau_d={info['tau_d']:.6f}. "
                f"stable={info.get('n_stable', 'NA')}, "
                f"unstable={info.get('n_unstable', 'NA')}, "
                f"boundary={info.get('n_boundary', 'NA')}, "
                f"eigvals={info.get('eigvals', [])}"
            )
        return info

    def _deficit_path_from_announcement(self, announce_t: int, implement_t: int):
        par = self.par

        if implement_t < announce_t:
            raise ValueError("implement_t must be >= announce_t.")

        lag = implement_t - announce_t
        n = par.T + 1

        e = np.zeros(n)
        if lag <= par.T:
            e[lag] = self.eps0

        S_e = np.zeros(n)
        for t in range(n):
            if t <= lag:
                S_e[t] = self.eps0 * (par.beta * par.omega) ** (lag - t)

        return e, S_e, lag

    def build_system(self, announce_t=None, implement_t=None):
        par = self.par
        eqsys = self.eqsys

        if announce_t is None:
            announce_t = par.announce_t
        if implement_t is None:
            implement_t = par.implement_t

        self.model_coefficients()
        e, S_e, lag = self._deficit_path_from_announcement(announce_t, implement_t)

        n = par.T + 1
        N = 5 * n
        offsets = {"y": 0, "pi": n, "d": 2 * n, "Sy": 3 * n, "Sr": 4 * n}

        def idx(block, t):
            return offsets[block] + t

        A = np.zeros((N, N))
        b = np.zeros(N)
        row = 0

        for t in range(n):
            A[row, idx("y", t)] = eqsys.mu_left
            A[row, idx("d", t)] = -eqsys.B_d
            A[row, idx("Sy", t)] = -eqsys.B_y
            A[row, idx("Sr", t)] = +eqsys.B_r
            b[row] = eqsys.B_e_current * e[t] + eqsys.B_e_news * S_e[t]
            row += 1

        for t in range(n):
            A[row, idx("Sy", t)] = 1.0
            A[row, idx("y", t)] = -1.0
            if t < par.T:
                A[row, idx("Sy", t + 1)] = -par.beta * par.omega
            row += 1

        for t in range(n):
            A[row, idx("Sr", t)] = 1.0
            A[row, idx("y", t)] = -par.alpha_y
            A[row, idx("pi", t)] = -par.alpha_pi
            if t < par.T:
                A[row, idx("Sr", t + 1)] = -par.beta * par.omega
            row += 1

        for t in range(n):
            A[row, idx("pi", t)] = 1.0
            A[row, idx("y", t)] = -par.kappa
            if t < par.T:
                A[row, idx("pi", t + 1)] = -par.beta
            row += 1

        for t in range(par.T):
            A[row, idx("d", t + 1)] = 1.0
            A[row, idx("d", t)] = -(1.0 - par.tau_d) / par.beta
            A[row, idx("y", t)] = par.tau_y / par.beta - par.Dbar * par.alpha_y
            A[row, idx("pi", t)] = -par.Dbar * par.alpha_pi
            b[row] = (1.0 - par.tau_d) / par.beta * e[t]
            row += 1

        A[row, idx("d", 0)] = 1.0
        A[row, idx("pi", 0)] = par.Dbar
        row += 1

        if row != N:
            raise RuntimeError(f"Equation count mismatch: row={row}, N={N}")

        eqsys.A = A
        eqsys.b = b
        eqsys.offsets = offsets
        eqsys.e = e
        eqsys.S_e = S_e
        eqsys.lag = lag

    def solve_model(self, announce_t=None, implement_t=None, check_determinacy_first=True):
        par = self.par

        if announce_t is None:
            announce_t = par.announce_t
        if implement_t is None:
            implement_t = par.implement_t

        if check_determinacy_first:
            self._require_determinate(tau_d=par.tau_d)

        self.build_system(announce_t=announce_t, implement_t=implement_t)
        x = np.linalg.solve(self.eqsys.A, self.eqsys.b)

        n = par.T + 1
        off = self.eqsys.offsets

        self.sol = SimpleNamespace()
        self.sol.announce_t = int(announce_t)
        self.sol.implement_t = int(implement_t)
        self.sol.lag = int(self.eqsys.lag)

        self.sol.y = x[off["y"]: off["y"] + n]
        self.sol.pi = x[off["pi"]: off["pi"] + n]
        self.sol.d = x[off["d"]: off["d"] + n]
        self.sol.Sy = x[off["Sy"]: off["Sy"] + n]
        self.sol.Sr = x[off["Sr"]: off["Sr"] + n]

        self.sol.r = par.alpha_y * self.sol.y + par.alpha_pi * self.sol.pi

        exp_pi_next = np.empty_like(self.sol.pi)
        exp_pi_next[:-1] = self.sol.pi[1:]
        exp_pi_next[-1] = 0.0
        self.sol.i_nom = self.sol.r + exp_pi_next

        self.sol.e = self.eqsys.e.copy()
        self.sol.S_e = self.eqsys.S_e.copy()
        self.sol.t = np.arange(n)

        self.sol.debt_end = np.empty_like(self.sol.d)
        self.sol.debt_end[:-1] = self.sol.d[1:]
        self.sol.debt_end[-1] = np.nan

        self.sol.q = self.sol.d + self.sol.e
        self.sol.eps0 = self.eps0

        self.compute_financing_share()
        return self.sol

    def compute_financing_share(self):
        par = self.par
        sol = self.sol
        k = np.arange(len(sol.y))

        tax_base_gain = par.tau_y * np.sum((par.beta ** k) * sol.y)
        debt_erosion_gain = par.Dbar * sol.pi[0]
        servicing_cost = par.Dbar * np.sum((par.beta ** (k + 1)) * sol.r)
        pv_deficit = np.sum((par.beta ** k) * sol.e)

        denom = pv_deficit + servicing_cost
        if abs(denom) < 1e-14:
            raise ValueError("Self-financing denominator too close to zero.")

        fiscal_adjustment = par.tau_d * np.sum((par.beta ** k) * (sol.d + sol.e))

        sol.tax_base_gain = float(tax_base_gain)
        sol.debt_erosion_gain = float(debt_erosion_gain)
        sol.servicing_cost = float(servicing_cost)
        sol.pv_deficit = float(pv_deficit)
        sol.sf_denominator = float(denom)

        sol.share_tax_base = float(tax_base_gain / denom)
        sol.share_debt_erosion = float(debt_erosion_gain / denom)
        sol.share_actual = float((tax_base_gain + debt_erosion_gain) / denom)

        sol.fiscal_adjustment = float(fiscal_adjustment)
        sol.residual_need = float(fiscal_adjustment / denom)

        sol.budget_identity_error = float(
            denom - fiscal_adjustment - tax_base_gain - debt_erosion_gain
        )

        sol.share_actual_over_eps0 = float((tax_base_gain + debt_erosion_gain) / self.eps0)

    def convergence_check(self, announce_t=None, implement_t=None, T_list=(160, 240, 360)):
        old_T = self.par.T
        shares = []

        for T in T_list:
            self.par.T = int(T)
            self.solve_model(
                announce_t=announce_t,
                implement_t=implement_t,
                check_determinacy_first=True,
            )
            shares.append(self.sol.share_actual)

        self.par.T = old_T
        self.solve_model(
            announce_t=announce_t,
            implement_t=implement_t,
            check_determinacy_first=True,
        )

        return {
            "T_list": list(T_list),
            "shares": shares,
            "spread": float(np.max(shares) - np.min(shares)),
        }

    def solve_implement_sweep(self, announce_t=None, implement_t_grid=None, do_convergence_check=True):
        par = self.par

        if announce_t is None:
            announce_t = par.announce_t
        if implement_t_grid is None:
            implement_t_grid = np.arange(announce_t, par.lag_max + 1)

        det = self.check_determinacy(tau_d=par.tau_d)
        if not det["is_unique_bounded"]:
            if self.verbose:
                print(f"Skipping implement sweep: no unique bounded equilibrium for tau_d={par.tau_d:.6f}")
            return

        self.allocate()

        for implement_t in implement_t_grid:
            self.solve_model(
                announce_t=announce_t,
                implement_t=int(implement_t),
                check_determinacy_first=False,
            )

            res = deepcopy(self.sol)

            if do_convergence_check:
                conv = self.convergence_check(
                    announce_t=announce_t,
                    implement_t=int(implement_t),
                )
                res.convergence_spread = conv["spread"]
            else:
                res.convergence_spread = np.nan

            self.sol_all.results.append(res)
            self.sol_all.implement_t_list.append(int(implement_t))
            self.sol_all.share_actual_list.append(res.share_actual)
            self.sol_all.share_tax_base_list.append(res.share_tax_base)
            self.sol_all.share_debt_erosion_list.append(res.share_debt_erosion)
            self.sol_all.convergence_spread_list.append(res.convergence_spread)

    # ---------------------------------------------------------------------
    # Plotting helpers
    # ---------------------------------------------------------------------

    def style_colors(self):
        return {
            "line_list": np.array(
                [
                    [0.97, 0.67, 0.78],
                    [0.80, 0.20, 0.52],
                    [0.55, 0.05, 0.30],
                ]
            ),
            "finance_price": np.array([0.80, 0.20, 0.52]),
            "finance_tax": np.array([0.97, 0.67, 0.78]),
        }

    def _get_line_colors(self, n):
        base = self.style_colors()["line_list"]
        if n <= len(base):
            return base[:n]

        idx = np.linspace(0, len(base) - 1, n)
        return np.vstack(
            [np.interp(idx, np.arange(len(base)), base[:, c]) for c in range(3)]
        ).T

    def _sweep_total_share_given_tau(self, tau_d, announce_t=None, implement_t_grid=None):
        par = self.par
        old_tau_d = par.tau_d

        if announce_t is None:
            announce_t = par.announce_t
        if implement_t_grid is None:
            implement_t_grid = np.arange(announce_t, par.lag_max + 1)

        det = self.check_determinacy(tau_d=tau_d)
        if not det["is_unique_bounded"]:
            return np.asarray(implement_t_grid, dtype=float), np.full(len(implement_t_grid), np.nan)

        vals = []

        try:
            par.tau_d = float(tau_d)
            for implement_t in implement_t_grid:
                self.solve_model(
                    announce_t=announce_t,
                    implement_t=int(implement_t),
                    check_determinacy_first=False,
                )
                vals.append(self.sol.share_actual)
        finally:
            par.tau_d = old_tau_d

        return np.asarray(implement_t_grid, dtype=float), np.asarray(vals, dtype=float)

    def _compute_self_financing_curve_tau(self, announce_t=None, implement_t=None, tau_d_grid=None):
        par = self.par

        if announce_t is None:
            announce_t = par.announce_t
        if implement_t is None:
            implement_t = par.implement_t

        if tau_d_grid is None:
            tau_d_grid = np.sort(
                np.concatenate(
                    (
                        np.linspace(0.0, 1.0, 301),
                        np.array([0.001, 0.026, 0.086]),
                    )
                )
            )

        tau_d_grid = np.asarray(tau_d_grid, dtype=float)

        old_tau_d = par.tau_d

        valid_mask = np.zeros(len(tau_d_grid), dtype=bool)
        share_total = np.full(len(tau_d_grid), np.nan)
        share_tax = np.full(len(tau_d_grid), np.nan)
        share_price = np.full(len(tau_d_grid), np.nan)
        share_old_eps0_norm = np.full(len(tau_d_grid), np.nan)

        try:
            for i, tau_d in enumerate(tau_d_grid):
                det = self.check_determinacy(tau_d=tau_d)
                if not det["is_unique_bounded"]:
                    valid_mask[i] = False
                    continue

                par.tau_d = float(tau_d)

                self.solve_model(
                    announce_t=announce_t,
                    implement_t=implement_t,
                    check_determinacy_first=False,
                )

                share_total[i] = self.sol.share_actual
                share_tax[i] = self.sol.share_tax_base
                share_price[i] = self.sol.share_debt_erosion
                share_old_eps0_norm[i] = self.sol.share_actual_over_eps0
                valid_mask[i] = np.isfinite(self.sol.share_actual)

        finally:
            par.tau_d = old_tau_d

        return {
            "tau_d_grid": tau_d_grid,
            "valid_mask": valid_mask,
            "share_total": share_total,
            "share_tax": share_tax,
            "share_price": share_price,
            "share_old_eps0_norm": share_old_eps0_norm,
        }

    def _shade_invalid_tau_regions(self, ax, tau_d_grid, valid_mask, label=None):
        tau_d_grid = np.asarray(tau_d_grid, dtype=float)
        invalid_mask = ~np.asarray(valid_mask, dtype=bool)

        if not np.any(invalid_mask):
            return

        starts = np.where(invalid_mask & ~np.r_[False, invalid_mask[:-1]])[0]
        ends = np.where(invalid_mask & ~np.r_[invalid_mask[1:], False])[0]

        first_patch = True
        for s, e in zip(starts, ends):
            x0 = 0.0 if s == 0 else 0.5 * (tau_d_grid[s - 1] + tau_d_grid[s])
            x1 = 1.0 if e == len(tau_d_grid) - 1 else 0.5 * (tau_d_grid[e] + tau_d_grid[e + 1])

            ax.axvspan(
                x0,
                x1,
                facecolor="grey",
                alpha=0.55,
                zorder=0,
                label=label if first_patch else None,
            )
            first_patch = False

    def _get_ylim(self, ylims, key):
        if ylims is None:
            return None
        if not isinstance(ylims, dict):
            raise TypeError("ylims must be a dictionary or None.")
        return ylims.get(key, None)

    def _apply_ylim(self, ax, ylim, key_name=""):
        if ylim is None:
            return

        if not isinstance(ylim, (tuple, list)):
            raise TypeError(
                f"ylim for '{key_name}' must be a tuple/list like (0, None), not {ylim}."
            )

        if len(ylim) != 2:
            raise ValueError(
                f"ylim for '{key_name}' must have length 2, e.g. (0, None)."
            )

        ymin, ymax = ylim
        ax.set_ylim(bottom=ymin, top=ymax)

    # ---------------------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------------------

    def plot_eps0_irfs(
        self,
        announce_t=None,
        implement_t=None,
        tau_d_list=None,
        tau_d_grid=None,
        truncation_T=None,
        x_plot_max=None,
        savepath=None,
        ylim0=True,
        show=True,
        ylims=None,
    ):
        par = self.par
        colors = self.style_colors()

        if announce_t is None:
            announce_t = par.announce_t
        if implement_t is None:
            implement_t = par.implement_t
        if tau_d_list is None:
            tau_d_list = [0.001, 0.026, 0.086]
        if truncation_T is None:
            truncation_T = par.T
        if x_plot_max is None:
            x_plot_max = par.x_plot_max
        if tau_d_grid is None:
            tau_d_grid = np.sort(
                np.concatenate((np.linspace(0.0, 1.0, 301), np.array(tau_d_list)))
            )

        old_T = par.T
        old_tau_d = par.tau_d

        try:
            par.T = int(truncation_T)
            line_colors = self._get_line_colors(len(tau_d_list))

            selected_results = []
            selected_colors = []
            missing_tau = []

            for color, tau_d in zip(line_colors, tau_d_list):
                det = self.check_determinacy(tau_d=tau_d)
                if not det["is_unique_bounded"]:
                    missing_tau.append(float(tau_d))
                    continue

                par.tau_d = float(tau_d)

                self.solve_model(
                    announce_t=announce_t,
                    implement_t=implement_t,
                    check_determinacy_first=False,
                )

                res = deepcopy(self.sol)
                res.tau_d = float(tau_d)

                selected_results.append(res)
                selected_colors.append(color)

            if len(selected_results) == 0:
                raise RuntimeError("None of the requested tau_d values has a unique bounded equilibrium.")

            if missing_tau and self.verbose:
                print(f"Skipping undetermined tau_d values in IRF plot: {missing_tau}")

            sf = self._compute_self_financing_curve_tau(
                announce_t=announce_t,
                implement_t=implement_t,
                tau_d_grid=tau_d_grid,
            )

            with mpl.rc_context(
                {
                    "font.family": "Times New Roman",
                    "font.size": 13,
                    "axes.titlesize": 17,
                    "axes.labelsize": 14,
                    "legend.fontsize": 12,
                    "xtick.labelsize": 12,
                    "ytick.labelsize": 12,
                    "mathtext.fontset": "stix",
                }
            ):
                fig, axes = plt.subplots(
                    2,
                    2,
                    figsize=(10, 8),
                    sharex=False,
                    constrained_layout=False,
                )
                fig.subplots_adjust(
                    left=0.10,
                    right=0.85,
                    top=0.92,
                    bottom=0.12,
                    wspace=0.36,
                    hspace=0.34,
                )

                ax = axes[0, 0]
                ax_output = ax
                for color, res in zip(selected_colors, selected_results):
                    ax.plot(res.t, res.y, lw=2.8, color=color)

                ax.axvline(implement_t, color="black", ls="--", alpha=0.8, linewidth=1.2)
                ax.set_title(r"Output $y_t$")
                ax.set_xlabel(r"$t$")
                ax.set_ylabel(r"%")
                ax.set_xlim(0, min(int(x_plot_max), int(selected_results[0].t[-1])))
                ax.grid(True, alpha=0.25)

                ax = axes[0, 1]
                ax_debt = ax
                for color, res in zip(selected_colors, selected_results):
                    ax.plot(res.t, res.debt_end, lw=2.8, color=color)

                ax.axvline(implement_t, color="black", ls="--", alpha=0.8, linewidth=1.2)
                ax.set_title(r"Public debt $d_{t+1}$")
                ax.set_xlabel(r"$t$")
                ax.set_xlim(0, min(int(x_plot_max), int(selected_results[0].t[-1])))
                ax.grid(True, alpha=0.25)

                ax = axes[1, 0]
                ax_rates_left = ax
                ax_r = ax.twinx()
                ax_rates_right = ax_r

                for color, res in zip(selected_colors, selected_results):
                    ax.plot(res.t, res.pi, lw=2.5, color=color, ls="--")
                    ax.plot(res.t, res.i_nom, lw=2.5, color=color, ls=":")
                    ax_r.plot(res.t, res.r, lw=2.5, color=color, ls="-")

                ax.axvline(implement_t, color="black", ls="--", alpha=0.8, linewidth=1.2)
                ax.set_title(r"Inflation and interest rates")
                ax.set_xlabel(r"$t$")
                ax.set_ylabel(r"$\pi_t,\ i_t$")
                ax_r.set_ylabel(r"$r_t$")
                ax.set_xlim(0, min(int(x_plot_max), int(selected_results[0].t[-1])))
                ax.grid(True, alpha=0.25)

                style_handles = [
                    Line2D([0], [0], color="black", lw=2.5, ls="--", label=r"$\pi_t$"),
                    Line2D([0], [0], color="black", lw=2.5, ls=":", label=r"$i_t$"),
                    Line2D([0], [0], color="black", lw=2.5, ls="-", label=r"$r_t$"),
                ]
                ax.legend(handles=style_handles, loc="upper right", frameon=True)

                ax = axes[1, 1]
                ax_self_financing = ax

                self._shade_invalid_tau_regions(
                    ax=ax,
                    tau_d_grid=sf["tau_d_grid"],
                    valid_mask=sf["valid_mask"],
                    label="No unique bounded eq.",
                )

                tau_valid = sf["tau_d_grid"][sf["valid_mask"]]
                nu_price_valid = sf["share_price"][sf["valid_mask"]]
                nu_total_valid = sf["share_total"][sf["valid_mask"]]

                if tau_valid.size > 0:
                    ax.fill_between(
                        tau_valid,
                        0.0,
                        nu_price_valid,
                        color=colors["finance_price"],
                        alpha=0.95,
                        label="Date-0 Inflation",
                        zorder=2,
                    )
                    ax.fill_between(
                        tau_valid,
                        nu_price_valid,
                        nu_total_valid,
                        color=colors["finance_tax"],
                        alpha=0.95,
                        label="Tax Base",
                        zorder=3,
                    )
                    ax.plot(
                        tau_valid,
                        nu_total_valid,
                        color="black",
                        lw=1.4,
                        alpha=0.7,
                        zorder=4,
                    )

                for color, tau0 in zip(selected_colors, [r.tau_d for r in selected_results]):
                    idx = np.argmin(np.abs(sf["tau_d_grid"] - tau0))
                    if sf["valid_mask"][idx]:
                        ax.scatter(
                            sf["tau_d_grid"][idx],
                            sf["share_total"][idx],
                            s=55,
                            color=color,
                            edgecolor="black",
                            linewidth=0.8,
                            zorder=5,
                        )

                ax.axhline(1.0, color="black", ls="--", alpha=0.45, linewidth=1.2)

                ax.set_title(r"Self-financing share $\nu$")
                ax.set_xlabel(r"$\tau_d$")
                ax.set_xlim(0.0, 1.0)
                ax.grid(True, alpha=0.25)

                legend_handles = [
                    Patch(facecolor=colors["finance_price"], edgecolor="none", label="Date-0 Inflation"),
                    Patch(facecolor=colors["finance_tax"], edgecolor="none", label="Tax Base"),
                    Patch(facecolor="grey", alpha=0.55, edgecolor="none", label="No unique bounded eq."),
                    Line2D([0], [0], color="black", ls="--", alpha=0.45, label=r"$\nu=1$"),
                ]
                ax.legend(handles=legend_handles, loc="upper right", frameon=True)

                if ylim0:
                    for ax0 in [ax_output, ax_debt]:
                        ymin, ymax = ax0.get_ylim()
                        ax0.set_ylim(0.0, ymax)

                self._apply_ylim(ax_output, self._get_ylim(ylims, "fig1_output"), "fig1_output")
                self._apply_ylim(ax_debt, self._get_ylim(ylims, "fig1_debt"), "fig1_debt")
                self._apply_ylim(ax_rates_left, self._get_ylim(ylims, "fig1_rates_left"), "fig1_rates_left")
                self._apply_ylim(ax_rates_right, self._get_ylim(ylims, "fig1_rates_right"), "fig1_rates_right")
                self._apply_ylim(
                    ax_self_financing,
                    self._get_ylim(ylims, "fig1_self_financing"),
                    "fig1_self_financing",
                )

                tmax = min(int(x_plot_max), int(selected_results[0].t[-1]))
                xticks = np.arange(0, tmax + 1, 10)

                for ax0 in [axes[0, 0], axes[0, 1], axes[1, 0]]:
                    ax0.set_xticks(xticks)
                    ax0.tick_params(axis="x", labelbottom=True)

                tau_handles = [
                    Line2D([0], [0], color=color, lw=2.8, label=rf"{res.tau_d:.3f}")
                    for color, res in zip(selected_colors, selected_results)
                ]

                fig.legend(
                    handles=tau_handles,
                    loc="center left",
                    bbox_to_anchor=(0.88, 0.50),
                    frameon=True,
                    title=r"$\tau_d$",
                    borderaxespad=0.0,
                )

                fig.align_ylabels([axes[0, 0], axes[1, 0]])

                if savepath is not None:
                    fig.savefig(savepath, dpi=600, bbox_inches="tight")
                    if self.verbose:
                        print(f"saved to: {savepath}")

                _maybe_show(show)
                plt.close(fig)

        finally:
            par.T = old_T
            par.tau_d = old_tau_d

    def plot_implementation_delay_sweep(
        self,
        announce_t=None,
        tau_d_list=None,
        implement_t_grid=None,
        truncation_T=None,
        figsize=(9, 7),
        savepath=None,
        show=True,
        ylims=None,
    ):
        par = self.par

        if announce_t is None:
            announce_t = par.announce_t
        if tau_d_list is None:
            tau_d_list = [0.001, 0.026, 0.086]
        if implement_t_grid is None:
            implement_t_grid = np.arange(announce_t, par.horizon_x_max + 1)
        if truncation_T is None:
            truncation_T = par.T

        old_T = par.T
        old_tau_d = par.tau_d

        try:
            par.T = int(truncation_T)
            line_colors = self._get_line_colors(len(tau_d_list))

            with mpl.rc_context(
                {
                    "font.family": "Times New Roman",
                    "font.size": 13,
                    "axes.titlesize": 17,
                    "axes.labelsize": 14,
                    "legend.fontsize": 12,
                    "xtick.labelsize": 12,
                    "ytick.labelsize": 12,
                    "mathtext.fontset": "stix",
                }
            ):
                fig, ax = plt.subplots(figsize=figsize)

                plotted = []

                for color, tau_d in zip(line_colors, tau_d_list):
                    det = self.check_determinacy(tau_d=tau_d)

                    if not det["is_unique_bounded"]:
                        if self.verbose:
                            print(
                                f"Skipping tau_d={tau_d:.6f} in delay sweep: "
                                "no unique bounded equilibrium."
                            )
                        continue

                    s_grid, nu_grid = self._sweep_total_share_given_tau(
                        tau_d=float(tau_d),
                        announce_t=announce_t,
                        implement_t_grid=implement_t_grid,
                    )

                    ax.plot(s_grid, nu_grid, lw=2.8, color=color)
                    plotted.append((color, tau_d))

                if len(plotted) == 0:
                    raise RuntimeError("No determinate tau_d values available for implementation-delay sweep.")

                ax.axhline(1.0, color="black", ls="--", alpha=0.5, linewidth=1.2)

                ax.set_title(r"Self-financing share $\nu(s)$")
                ax.set_xlabel(r"Implementation date $s$")
                ax.set_ylabel(r"$\nu(s)$")
                ax.set_xlim(float(np.min(implement_t_grid)), float(np.max(implement_t_grid)))
                ax.grid(True, alpha=0.25)

                tau_handles = [
                    Line2D([0], [0], color=color, lw=2.8, label=rf"$\tau_d={tau:.3f}$")
                    for color, tau in plotted
                ]
                ax.legend(handles=tau_handles, loc="best", frameon=True)

                ymin, ymax = ax.get_ylim()
                ax.set_ylim(min(ymin, 0.0), max(ymax, 1.05))

                self._apply_ylim(
                    ax,
                    self._get_ylim(ylims, "delay_sweep"),
                    "delay_sweep",
                )

                if savepath is not None:
                    fig.savefig(savepath, dpi=600, bbox_inches="tight")
                    if self.verbose:
                        print(f"saved to: {savepath}")

                _maybe_show(show)
                plt.close(fig)

        finally:
            par.T = old_T
            par.tau_d = old_tau_d

    # ---------------------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------------------

    def summary(self, announce_t=None, implement_t=None, truncation_T=None):
        par = self.par

        if announce_t is None:
            announce_t = par.announce_t
        if implement_t is None:
            implement_t = par.implement_t
        if truncation_T is None:
            truncation_T = par.T

        det = self.check_determinacy(tau_d=par.tau_d)

        print("--- Preannounced-deficit OLG model with HtM share ---")
        print(f"announcement date: {announce_t}")
        print(f"implementation date: {implement_t}")
        print(f"lag: {implement_t - announce_t}")
        print(f"shock size eps0: {self.eps0:.6f}")
        print(f"mu: {par.mu:.6f}")
        print(f"tau_d: {par.tau_d:.6f}")
        print(
            f"determinacy: {det['is_unique_bounded']} "
            f"(stable={det['n_stable']}, unstable={det['n_unstable']}, boundary={det['n_boundary']})"
        )
        print(f"BK eigenvalues: {np.array2string(det['eigvals'], precision=6)}")

        if not det["is_unique_bounded"]:
            print("Path not solved because the homogeneous HtM system is not uniquely bounded.")
            return

        old_T = par.T

        try:
            par.T = int(truncation_T)

            self.solve_model(
                announce_t=announce_t,
                implement_t=implement_t,
                check_determinacy_first=False,
            )

            conv = self.convergence_check(
                announce_t=announce_t,
                implement_t=implement_t,
            )

            print(f"calculation horizon T: {par.T}")
            print(f"PV deficit:           {self.sol.pv_deficit:.8f}")
            print(f"servicing cost:       {self.sol.servicing_cost:.8f}")
            print(f"SF denominator:       {self.sol.sf_denominator:.8f}")
            print(f"share financed:       {self.sol.share_actual:.8f}")
            print(f"  tax-base share:     {self.sol.share_tax_base:.8f}")
            print(f"  debt-erosion share: {self.sol.share_debt_erosion:.8f}")
            print(f"residual need:        {self.sol.residual_need:.8f}")
            print(f"budget identity err:  {self.sol.budget_identity_error:.3e}")
            print(f"old eps0-normalized:  {self.sol.share_actual_over_eps0:.8f}")
            print(f"convergence spread:   {conv['spread']:.3e}")

        finally:
            par.T = old_T

    def determinacy_table(self, tau_d_list, tol=1e-9):
        rows = []

        for tau_d in tau_d_list:
            det = self.check_determinacy(tau_d=tau_d, tol=tol)
            rows.append(
                {
                    "tau_d": float(tau_d),
                    "unique_bounded": det["is_unique_bounded"],
                    "stable": det["n_stable"],
                    "unstable": det["n_unstable"],
                    "boundary": det["n_boundary"],
                    "eigvals": det["eigvals"],
                }
            )

        return rows

    def run(
        self,
        truncation_T=None,
        x_plot_max=None,
        tau_d_list=None,
        savepath_main=None,
        savepath_delay=None,
        show=True,
        ylims=None,
    ):
        if tau_d_list is None:
            tau_d_list = [0.001, 0.026, 0.086]

        self.summary(truncation_T=truncation_T)

        self.plot_eps0_irfs(
            tau_d_list=tau_d_list,
            truncation_T=truncation_T,
            x_plot_max=x_plot_max,
            savepath=savepath_main,
            ylim0=True,
            show=show,
            ylims=ylims,
        )

        self.plot_implementation_delay_sweep(
            tau_d_list=tau_d_list,
            truncation_T=truncation_T,
            savepath=savepath_delay,
            show=show,
            ylims=ylims,
        )


def quick_self_test():
    print("\n--- quick self test ---")

    delays = [0, 1, 5, 20, 40]
    tau_list = [0.001, 0.026, 0.086]

    print("\nmu=0 nesting test against original finite-path system:")

    for tau_d in tau_list:
        maxdiff_y = 0.0
        maxdiff_pi = 0.0
        maxdiff_d = 0.0
        maxdiff_q = 0.0
        maxdiff_nu = 0.0
        max_budg_err = 0.0

        for implement_t in delays:
            old = PreannouncedDeficit_OLG_Original(verbose=False, eps0=1.0)
            new = PreannouncedDeficit_OLG(verbose=False, eps0=1.0)

            for mdl in (old, new):
                mdl.par.beta = 0.99**0.25
                mdl.par.omega = 0.865
                mdl.par.sigma = 1.0
                mdl.par.tau_y = 1.0 / 3.0
                mdl.par.tau_d = tau_d
                mdl.par.kappa = 0.0062
                mdl.par.Dbar = 1.04
                mdl.par.alpha_y = 0.5
                mdl.par.alpha_pi = 0.5
                mdl.par.T = 220
                mdl.par.announce_t = 0
                mdl.par.implement_t = implement_t

            new.par.mu = 0.0

            det = new.check_determinacy(tau_d=tau_d)
            if not det["is_unique_bounded"]:
                print(f"  tau_d={tau_d:.3f}: skipped because BK not unique.")
                continue

            old.solve_model()
            new.solve_model(check_determinacy_first=False)

            maxdiff_y = max(maxdiff_y, float(np.max(np.abs(old.sol.y - new.sol.y))))
            maxdiff_pi = max(maxdiff_pi, float(np.max(np.abs(old.sol.pi - new.sol.pi))))
            maxdiff_d = max(maxdiff_d, float(np.max(np.abs(old.sol.d - new.sol.d))))
            maxdiff_q = max(maxdiff_q, float(np.max(np.abs(old.sol.q - new.sol.q))))
            maxdiff_nu = max(maxdiff_nu, abs(old.sol.share_actual - new.sol.share_actual))
            max_budg_err = max(
                max_budg_err,
                abs(old.sol.budget_identity_error),
                abs(new.sol.budget_identity_error),
            )

        print(
            f"  tau_d={tau_d:.3f} | "
            f"max|dy|={maxdiff_y:.3e}, "
            f"max|dpi|={maxdiff_pi:.3e}, "
            f"max|dd|={maxdiff_d:.3e}, "
            f"max|dq|={maxdiff_q:.3e}, "
            f"max|dnu|={maxdiff_nu:.3e}, "
            f"max budget err={max_budg_err:.3e}"
        )

    print("\nactive-rule determinacy table:")

    m = PreannouncedDeficit_OLG(verbose=False, eps0=1.0)
    p = m.par

    p.mu = 0.0
    p.beta = 0.99**0.25
    p.omega = 0.865
    p.sigma = 1.0
    p.tau_y = 1.0 / 3.0
    p.kappa = 0.0062
    p.Dbar = 1.04
    p.alpha_y = 0.5
    p.alpha_pi = 0.5

    for row in m.determinacy_table([0.0, 0.004, 0.026, 0.085, 0.1, 0.3, 0.5, 1.0]):
        eig_short = np.array2string(row["eigvals"], precision=4)
        print(
            f"  tau_d={row['tau_d']:.3f}, "
            f"unique={row['unique_bounded']}, "
            f"stable={row['stable']}, "
            f"unstable={row['unstable']}, "
            f"boundary={row['boundary']}, "
            f"eigvals={eig_short}"
        )

    print("\nself-financing curve smoke test:")

    p.tau_d = 0.026
    p.T = 250
    p.announce_t = 0
    p.implement_t = 20

    sf = m._compute_self_financing_curve_tau(
        announce_t=0,
        implement_t=20,
        tau_d_grid=np.array([0.0, 0.004, 0.026, 0.085, 0.1, 0.3, 0.5, 1.0]),
    )

    print("  tau_d      =", sf["tau_d_grid"])
    print("  valid_mask =", sf["valid_mask"])
    print("  total_nu   =", np.round(sf["share_total"], 6))
    print("  old_eps0   =", np.round(sf["share_old_eps0_norm"], 6))

    print("\nmu sensitivity smoke test:")

    for mu in [0.0, 0.001, 0.05, 0.10]:
        m2 = PreannouncedDeficit_OLG(verbose=False, eps0=1.0)
        p2 = m2.par

        p2.mu = mu
        p2.beta = 0.99**0.25
        p2.omega = 0.865
        p2.sigma = 1.0
        p2.tau_y = 1.0 / 3.0
        p2.tau_d = 0.026
        p2.kappa = 0.0062
        p2.Dbar = 1.04
        p2.alpha_y = 0.5
        p2.alpha_pi = 0.5
        p2.T = 300
        p2.announce_t = 0
        p2.implement_t = 20

        det = m2.check_determinacy(tau_d=p2.tau_d)
        if det["is_unique_bounded"]:
            m2.solve_model(check_determinacy_first=False)
            print(
                f"  mu={mu:.3f}: "
                f"nu={m2.sol.share_actual:.6f}, "
                f"tax={m2.sol.share_tax_base:.6f}, "
                f"price={m2.sol.share_debt_erosion:.6f}, "
                f"budget_err={m2.sol.budget_identity_error:.2e}"
            )
        else:
            print(f"  mu={mu:.3f}: no unique bounded equilibrium")


if __name__ == "__main__":
    quick_self_test()

    model = PreannouncedDeficit_OLG(verbose=True, eps0=1.0)
    par = model.par

    par.mu = 0.073
    par.beta = 0.99**0.25
    par.omega = 0.865
    par.sigma = 1.0
    par.tau_y = 1.0 / 3.0
    par.tau_d = 0.026
    par.kappa = 0.0062
    par.Dbar = 1.04

    # Active real-rate rule:
    # r_t = 0.08*y_t + 1.04*pi_t
    par.alpha_y = 0.08
    par.alpha_pi = 1.04

    par.T = 500
    par.announce_t = 0
    par.implement_t = 20
    par.horizon_x_max = 80 # sweep horizon

    # ================================================================
    # Y-LIMIT CONTROL PANEL
    # ================================================================
    # Use None to keep automatic matplotlib limits.
    # Use (0, None) for lower bound only.
    # Use (0, 0.2) for exact lower and upper bounds.
    #
    # Figure 1 / 2x2 IRF figure:
    #   "fig1_output"          -> output y_t
    #   "fig1_debt"            -> public debt d_{t+1}
    #   "fig1_rates_left"      -> left axis: pi_t and i_t
    #   "fig1_rates_right"     -> right axis: r_t
    #   "fig1_self_financing"  -> self-financing panel
    #
    # Standalone tau_d self-financing plot:
    #   "self_financing_vs_tau"
    #
    # Implementation-delay plot:
    #   "delay_sweep"
    # ================================================================

    YLIMS = {
        "fig1_output": (None,None),
        "fig1_debt": (None, None),
        "fig1_rates_left": None,
        "fig1_rates_right": None,

        # Figure 1 self-financing subplot
        "fig1_self_financing": (0, 1.0),

        # Implementation-delay plot
        "delay_sweep": (0, None),
        # "delay_sweep": (0, 0.2),
    }

    model.run(
        truncation_T=600,
        x_plot_max=40,
        tau_d_list=[0.004, 0.026, 0.085],
        # tau_d_list=[0.0, 0.1, 0.3, 0.5, 1.0],
        savepath_main="panel_active_delay20.png",
        savepath_delay="delay_active_delay20.png",
        show=True,
        ylims=YLIMS,
    )

    # model.plot_self_financing_vs_tau(
    #     announce_t=0,
    #     implement_t=20,
    #     selected_tau_d=[0.0, 0.004, 0.026, 0.085, 0.1, 0.3, 0.5, 1.0],
    #     savepath="preannounced_self_financing_vs_tau_mu_active_checked.png",
    #     show=True,
    #     ylims=YLIMS,
    # )