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
    """Original model, kept only for the mu=0 nesting test."""

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
        self.sol_all.implement_t_list = []
        self.sol_all.share_actual_list = []
        self.sol_all.share_tax_base_list = []
        self.sol_all.share_debt_erosion_list = []
        self.sol_all.convergence_spread_list = []

    def model_coefficients(self):
        par = self.par
        eqsys = self.eqsys
        Delta = 1.0 - par.omega * (1.0 - par.tau_d)
        if abs(Delta) < 1e-14:
            raise ValueError("Denominator too close to zero.")
        eqsys.Delta = Delta
        eqsys.X_d = (1.0 - par.beta * par.omega) * (1.0 - par.tau_d) * (1.0 - par.omega) / Delta
        eqsys.X_y = 1.0 - par.tau_y * (1.0 - par.omega) / Delta
        eqsys.X_r = (
            par.sigma * par.beta * par.omega / (1.0 - par.beta * par.omega)
            - par.Dbar * par.beta * (1.0 - par.omega) / Delta
        )

    def _deficit_path_from_announcement(self, announce_t: int, implement_t: int):
        par = self.par
        T = par.T
        if implement_t < announce_t:
            raise ValueError("implement_t must be >= announce_t.")
        lag = implement_t - announce_t
        n = T + 1
        e = np.zeros(n)
        if lag <= T:
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
        sol.tax_base_gain = float(tax_base_gain)
        sol.debt_erosion_gain = float(debt_erosion_gain)
        sol.servicing_cost = float(servicing_cost)
        sol.pv_deficit = float(np.sum((par.beta ** k) * sol.e))
        sol.share_tax_base = float(tax_base_gain / self.eps0)
        sol.share_debt_erosion = float(debt_erosion_gain / self.eps0)
        sol.share_actual = float((tax_base_gain + debt_erosion_gain) / self.eps0)
        sol.residual_need = float(1.0 - sol.share_actual + servicing_cost / self.eps0)


class PreannouncedDeficit_OLG:
    """OLG-NK model with a share mu of hand-to-mouth / spender households."""

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
        Delta = 1.0 - par.omega * (1.0 - par.tau_d)
        if abs(Delta) < 1e-14:
            raise ValueError("Denominator too close to zero.")
        eqsys.Delta = Delta
        eqsys.X_d_olg = (1.0 - par.beta * par.omega) * (1.0 - par.tau_d) * (1.0 - par.omega) / Delta
        eqsys.C_y_olg = (1.0 - par.beta * par.omega) * (1.0 - par.tau_y * (1.0 - par.omega) / Delta)
        eqsys.C_r_olg = (1.0 - par.beta * par.omega) * (
            par.sigma * par.beta * par.omega / (1.0 - par.beta * par.omega)
            - par.Dbar * par.beta * (1.0 - par.omega) / Delta
        )
        eqsys.mu_left = 1.0 - par.mu * (1.0 - par.tau_y)
        eqsys.mu_d = -par.mu * par.tau_d
        eqsys.mu_e0 = par.mu * (1.0 - par.tau_d)

    def _deficit_path_from_announcement(self, announce_t: int, implement_t: int):
        par = self.par
        T = par.T
        if implement_t < announce_t:
            raise ValueError("implement_t must be >= announce_t.")
        lag = implement_t - announce_t
        n = T + 1
        e = np.zeros(n)
        if lag <= T:
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
            A[row, idx("d", t)] = -(eqsys.mu_d + (1.0 - par.mu) * eqsys.X_d_olg)
            A[row, idx("Sy", t)] = -(1.0 - par.mu) * eqsys.C_y_olg
            A[row, idx("Sr", t)] = +(1.0 - par.mu) * eqsys.C_r_olg
            b[row] = eqsys.mu_e0 * e[t] + (1.0 - par.mu) * eqsys.X_d_olg * S_e[t]
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
        sol.tax_base_gain = float(tax_base_gain)
        sol.debt_erosion_gain = float(debt_erosion_gain)
        sol.servicing_cost = float(servicing_cost)
        sol.pv_deficit = float(np.sum((par.beta ** k) * sol.e))
        sol.share_tax_base = float(tax_base_gain / self.eps0)
        sol.share_debt_erosion = float(debt_erosion_gain / self.eps0)
        sol.share_actual = float((tax_base_gain + debt_erosion_gain) / self.eps0)
        sol.residual_need = float(1.0 - sol.share_actual + servicing_cost / self.eps0)

    def convergence_check(self, announce_t=None, implement_t=None, T_list=(160, 240, 360)):
        old_T = self.par.T
        shares = []
        for T in T_list:
            self.par.T = int(T)
            self.solve_model(announce_t=announce_t, implement_t=implement_t)
            shares.append(self.sol.share_actual)
        self.par.T = old_T
        self.solve_model(announce_t=announce_t, implement_t=implement_t)
        return {"T_list": list(T_list), "shares": shares, "spread": float(np.max(shares) - np.min(shares))}

    def solve_implement_sweep(self, announce_t=None, implement_t_grid=None, do_convergence_check=True):
        par = self.par
        if announce_t is None:
            announce_t = par.announce_t
        if implement_t_grid is None:
            implement_t_grid = np.arange(announce_t, par.lag_max + 1)
        self.allocate()
        for implement_t in implement_t_grid:
            self.solve_model(announce_t=announce_t, implement_t=int(implement_t))
            res = deepcopy(self.sol)
            if do_convergence_check:
                conv = self.convergence_check(announce_t=announce_t, implement_t=int(implement_t))
                res.convergence_spread = conv["spread"]
            else:
                res.convergence_spread = np.nan
            self.sol_all.results.append(res)
            self.sol_all.implement_t_list.append(int(implement_t))
            self.sol_all.share_actual_list.append(res.share_actual)
            self.sol_all.share_tax_base_list.append(res.share_tax_base)
            self.sol_all.share_debt_erosion_list.append(res.share_debt_erosion)
            self.sol_all.convergence_spread_list.append(res.convergence_spread)

    def style_colors(self):
        return {
            "line_list": np.array(
                [
                    [0.99, 0.85, 0.90],
                    [0.97, 0.67, 0.78],
                    [0.93, 0.44, 0.64],
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
        return np.vstack([
            np.interp(idx, np.arange(len(base)), base[:, c]) for c in range(3)
        ]).T

    def _sweep_total_share_given_tau(self, tau_d, announce_t=None, implement_t_grid=None):
        par = self.par
        old_tau_d = par.tau_d
        if announce_t is None:
            announce_t = par.announce_t
        if implement_t_grid is None:
            implement_t_grid = np.arange(announce_t, par.lag_max + 1)
        vals = []
        try:
            par.tau_d = float(tau_d)
            for implement_t in implement_t_grid:
                self.solve_model(announce_t=announce_t, implement_t=int(implement_t))
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
            tau_d_grid = np.sort(np.concatenate((np.linspace(0.0, 1.0, 301), np.array([0.001, 0.026, 0.086]))))
        tau_d_grid = np.asarray(tau_d_grid, dtype=float)
        old_tau_d = par.tau_d
        valid_mask = np.zeros(len(tau_d_grid), dtype=bool)
        share_total = np.full(len(tau_d_grid), np.nan)
        share_tax = np.full(len(tau_d_grid), np.nan)
        share_price = np.full(len(tau_d_grid), np.nan)
        try:
            for i, tau_d in enumerate(tau_d_grid):
                try:
                    par.tau_d = float(tau_d)
                    self.solve_model(announce_t=announce_t, implement_t=implement_t)
                    share_total[i] = self.sol.share_actual
                    share_tax[i] = self.sol.share_tax_base
                    share_price[i] = self.sol.share_debt_erosion
                    valid_mask[i] = np.isfinite(self.sol.share_actual)
                except Exception:
                    valid_mask[i] = False
        finally:
            par.tau_d = old_tau_d
        return {"tau_d_grid": tau_d_grid, "valid_mask": valid_mask, "share_total": share_total, "share_tax": share_tax, "share_price": share_price}

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
            ax.axvspan(x0, x1, facecolor="grey", alpha=0.55, zorder=0, label=label if first_patch else None)
            first_patch = False

    def plot_self_financing_vs_tau(
        self,
        announce_t=None,
        implement_t=None,
        tau_d_grid=None,
        selected_tau_d=None,
        figsize=(6.4, 4.8),
        savepath=None,
        show=True,
    ):
        par = self.par
        colors = self.style_colors()
        if announce_t is None:
            announce_t = par.announce_t
        if implement_t is None:
            implement_t = par.implement_t
        if selected_tau_d is None:
            selected_tau_d = [0.001, 0.026, 0.086]
        out = self._compute_self_financing_curve_tau(announce_t=announce_t, implement_t=implement_t, tau_d_grid=tau_d_grid)
        tau = out["tau_d_grid"]
        valid = out["valid_mask"]
        nu_total = out["share_total"]
        nu_price = out["share_price"]
        line_colors = self._get_line_colors(len(selected_tau_d))
        with mpl.rc_context({
            "font.family": "Times New Roman",
            "font.size": 13,
            "axes.titlesize": 17,
            "axes.labelsize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "mathtext.fontset": "stix",
        }):
            fig, ax = plt.subplots(figsize=figsize)
            self._shade_invalid_tau_regions(ax=ax, tau_d_grid=tau, valid_mask=valid, label="No valid solve")
            tau_valid = tau[valid]
            nu_price_valid = nu_price[valid]
            nu_total_valid = nu_total[valid]
            if tau_valid.size > 0:
                ax.fill_between(tau_valid, 0.0, nu_price_valid, color=colors["finance_price"], alpha=0.95, label="Date-0 Inflation", zorder=2)
                ax.fill_between(tau_valid, nu_price_valid, nu_total_valid, color=colors["finance_tax"], alpha=0.95, label="Tax Base", zorder=3)
                ax.plot(tau_valid, nu_total_valid, color="black", lw=1.4, alpha=0.7, zorder=4)
            for color, tau0 in zip(line_colors, selected_tau_d):
                idx = np.argmin(np.abs(tau - tau0))
                if valid[idx]:
                    ax.scatter(tau[idx], nu_total[idx], s=55, color=color, edgecolor="black", linewidth=0.8, zorder=5)
            ax.set_title(rf"Self-financing share $\nu$ at horizon $s={implement_t}$")
            ax.set_xlabel(r"$\tau_d$")
            ax.set_xlim(0.0, 1.0)
            ax.grid(True, alpha=0.25)
            legend_handles = [
                Patch(facecolor=colors["finance_price"], edgecolor="none", label="Date-0 Inflation"),
                Patch(facecolor=colors["finance_tax"], edgecolor="none", label="Tax Base"),
                Patch(facecolor="grey", alpha=0.55, edgecolor="none", label="No valid solve"),
            ]
            ax.legend(handles=legend_handles, loc="upper right", frameon=True)
            if savepath is not None:
                fig.savefig(savepath, dpi=200, bbox_inches="tight")
                if self.verbose:
                    print(f"saved to: {savepath}")
            _maybe_show(show)
            plt.close(fig)

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
            tau_d_grid = np.sort(np.concatenate((np.linspace(0.0, 1.0, 301), np.array(tau_d_list))))
        old_T = par.T
        old_tau_d = par.tau_d
        try:
            par.T = int(truncation_T)
            line_colors = self._get_line_colors(len(tau_d_list))
            selected_results = []
            for tau_d in tau_d_list:
                par.tau_d = float(tau_d)
                self.solve_model(announce_t=announce_t, implement_t=implement_t)
                res = deepcopy(self.sol)
                res.tau_d = float(tau_d)
                selected_results.append(res)
            sf = self._compute_self_financing_curve_tau(announce_t=announce_t, implement_t=implement_t, tau_d_grid=tau_d_grid)
            with mpl.rc_context({
                "font.family": "Times New Roman",
                "font.size": 13,
                "axes.titlesize": 17,
                "axes.labelsize": 14,
                "legend.fontsize": 12,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "mathtext.fontset": "stix",
            }):
                fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=False, constrained_layout=False)
                fig.subplots_adjust(left=0.10, right=0.85, top=0.92, bottom=0.12, wspace=0.36, hspace=0.34)
                ax = axes[0, 0]
                for color, res in zip(line_colors, selected_results):
                    ax.plot(res.t, res.y, lw=2.8, color=color)
                ax.axvline(implement_t, color="black", ls="--", alpha=0.8, linewidth=1.2)
                ax.set_title(r"Output $y_t$")
                ax.set_xlabel(r"$t$")
                ax.set_ylabel(r"%")
                ax.set_xlim(0, min(int(x_plot_max), int(selected_results[0].t[-1])))
                ax.grid(True, alpha=0.25)

                ax = axes[0, 1]
                for color, res in zip(line_colors, selected_results):
                    debt_to_plot = res.debt_end if hasattr(res, "debt_end") else np.r_[res.d[1:], np.nan]
                    ax.plot(res.t, debt_to_plot, lw=2.8, color=color)
                ax.axvline(implement_t, color="black", ls="--", alpha=0.8, linewidth=1.2)
                ax.set_title(r"Public debt $d_{t+1}$")
                ax.set_xlabel(r"$t$")
                ax.set_xlim(0, min(int(x_plot_max), int(selected_results[0].t[-1])))
                ax.grid(True, alpha=0.25)

                ax = axes[1, 0]
                ax_r = ax.twinx()
                for color, res in zip(line_colors, selected_results):
                    ax.plot(res.t, res.pi,    lw=2.5, color=color, ls="--")
                    ax.plot(res.t, res.i_nom, lw=2.5, color=color, ls=":")
                    ax_r.plot(res.t, res.r,   lw=2.5, color=color, ls="-")
                ax.axvline(implement_t, color="black", ls="--", alpha=0.8, linewidth=1.2)
                ax.set_title(r"Inflation and interest rates")
                ax.set_xlabel(r"$t$")
                ax.set_ylabel(r"$\pi_t,\ i_t$")
                ax_r.set_ylabel(r"$r_t$")
                ax.set_xlim(0, min(int(x_plot_max), int(selected_results[0].t[-1])))
                ax.grid(True, alpha=0.25)
                style_handles = [
                    Line2D([0], [0], color="black", lw=2.5, ls="--", label=r"$\pi_t$"),
                    Line2D([0], [0], color="black", lw=2.5, ls=":",  label=r"$i_t$"),
                    Line2D([0], [0], color="black", lw=2.5, ls="-",  label=r"$r_t$"),
                ]
                ax.legend(handles=style_handles, loc="upper right", frameon=True)

                ax = axes[1, 1]
                self._shade_invalid_tau_regions(ax=ax, tau_d_grid=sf["tau_d_grid"], valid_mask=sf["valid_mask"], label="No valid solve")
                tau_valid = sf["tau_d_grid"][sf["valid_mask"]]
                nu_price_valid = sf["share_price"][sf["valid_mask"]]
                nu_total_valid = sf["share_total"][sf["valid_mask"]]
                if tau_valid.size > 0:
                    ax.fill_between(tau_valid, 0.0, nu_price_valid, color=colors["finance_price"], alpha=0.95, label="Date-0 Inflation", zorder=2)
                    ax.fill_between(tau_valid, nu_price_valid, nu_total_valid, color=colors["finance_tax"], alpha=0.95, label="Tax Base", zorder=3)
                    ax.plot(tau_valid, nu_total_valid, color="black", lw=1.4, alpha=0.7, zorder=4)
                for color, tau0 in zip(line_colors, tau_d_list):
                    idx = np.argmin(np.abs(sf["tau_d_grid"] - tau0))
                    if sf["valid_mask"][idx]:
                        ax.scatter(sf["tau_d_grid"][idx], sf["share_total"][idx], s=55, color=color, edgecolor="black", linewidth=0.8, zorder=5)
                ax.set_title(r"Self-financing share $\nu$")
                ax.set_xlabel(r"$\tau_d$")
                ax.set_xlim(0.0, 1.0)
                ax.grid(True, alpha=0.25)
                legend_handles = [
                    Patch(facecolor=colors["finance_price"], edgecolor="none", label="Date-0 Inflation"),
                    Patch(facecolor=colors["finance_tax"], edgecolor="none", label="Tax Base"),
                    Patch(facecolor="grey", alpha=0.55, edgecolor="none", label="No valid solve"),
                ]
                ax.legend(handles=legend_handles, loc="upper right", frameon=True)

                if ylim0:
                    for ax0 in [axes[0, 0], axes[0, 1], axes[1, 1]]:
                        ymin, ymax = ax0.get_ylim()
                        ax0.set_ylim(0.0, ymax)
                tmax = min(int(x_plot_max), int(selected_results[0].t[-1]))
                xticks = np.arange(0, tmax + 1, 10)
                for ax0 in [axes[0, 0], axes[0, 1], axes[1, 0]]:
                    ax0.set_xticks(xticks)
                    ax0.tick_params(axis="x", labelbottom=True)
                tau_handles = [Line2D([0], [0], color=color, lw=2.8, label=rf"{res.tau_d:.3f}") for color, res in zip(line_colors, selected_results)]
                fig.legend(handles=tau_handles, loc="center left", bbox_to_anchor=(0.88, 0.50), frameon=True, title=r"$\tau_d$", borderaxespad=0.0)
                fig.align_ylabels([axes[0, 0], axes[1, 0]])
                if savepath is not None:
                    fig.savefig(savepath, dpi=200, bbox_inches="tight")
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
    ):
        """Standalone implementation-delay figure in the same style as the reference code."""
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

            with mpl.rc_context({
                "font.family": "Times New Roman",
                "font.size": 13,
                "axes.titlesize": 17,
                "axes.labelsize": 14,
                "legend.fontsize": 12,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "mathtext.fontset": "stix",
            }):
                fig, ax = plt.subplots(figsize=figsize)

                for color, tau_d in zip(line_colors, tau_d_list):
                    s_grid, nu_grid = self._sweep_total_share_given_tau(
                        tau_d=float(tau_d),
                        announce_t=announce_t,
                        implement_t_grid=implement_t_grid,
                    )
                    ax.plot(s_grid, nu_grid, lw=2.8, color=color)

                ax.axhline(1.0, color="black", ls="--", alpha=0.5, linewidth=1.2)
                ax.set_title(r"Self-financing share $\nu(s)$")
                ax.set_xlabel(r"Implementation date $s$")
                ax.set_ylabel(r"$\nu(s)$")
                ax.set_xlim(float(np.min(implement_t_grid)), float(np.max(implement_t_grid)))
                ax.grid(True, alpha=0.25)

                tau_handles = [
                    Line2D([0], [0], color=color, lw=2.8, label=rf"$\tau_d={tau:.3f}$")
                    for color, tau in zip(line_colors, tau_d_list)
                ]
                ax.legend(handles=tau_handles, loc="best", frameon=True)

                if savepath is not None:
                    fig.savefig(savepath, dpi=200, bbox_inches="tight")
                    if self.verbose:
                        print(f"saved to: {savepath}")

                _maybe_show(show)
                plt.close(fig)

        finally:
            par.T = old_T
            par.tau_d = old_tau_d

    def summary(self, announce_t=None, implement_t=None, truncation_T=None):
        par = self.par
        if announce_t is None:
            announce_t = par.announce_t
        if implement_t is None:
            implement_t = par.implement_t
        if truncation_T is None:
            truncation_T = par.T
        old_T = par.T
        try:
            par.T = int(truncation_T)
            self.solve_model(announce_t=announce_t, implement_t=implement_t)
            conv = self.convergence_check(announce_t=announce_t, implement_t=implement_t)
            print("--- Preannounced-deficit OLG model with HtM share ---")
            print(f"announcement date: {announce_t}")
            print(f"implementation date: {implement_t}")
            print(f"lag: {implement_t - announce_t}")
            print(f"shock size eps0: {self.eps0:.6f}")
            print(f"mu: {par.mu:.6f}")
            print(f"tau_d (baseline paths): {par.tau_d:.6f}")
            print(f"calculation horizon T: {par.T}")
            print(f"share financed:       {self.sol.share_actual:.8f}")
            print(f"  tax-base share:     {self.sol.share_tax_base:.8f}")
            print(f"  debt-erosion share: {self.sol.share_debt_erosion:.8f}")
            print(f"residual need:        {self.sol.residual_need:.8f}")
            print(f"convergence spread:   {conv['spread']:.3e}")
        finally:
            par.T = old_T

    def run(
        self,
        truncation_T=None,
        x_plot_max=None,
        tau_d_list=None,
        savepath_main=None,
        savepath_delay=None,
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
        )

        self.plot_implementation_delay_sweep(
            tau_d_list=tau_d_list,
            truncation_T=truncation_T,
            savepath=savepath_delay,
        )


def quick_self_test():
    print("\n--- quick self test ---")
    T_test = 120
    impl_test = 25
    old = PreannouncedDeficit_OLG_Original(verbose=False, eps0=0.01)
    new = PreannouncedDeficit_OLG(verbose=False, eps0=0.01)
    for mdl in (old, new):
        mdl.par.beta = 0.998
        mdl.par.omega = 0.865
        mdl.par.sigma = 1.0
        mdl.par.tau_y = 1.0 / 3.0
        mdl.par.tau_d = 0.026
        mdl.par.kappa = 0.0062
        mdl.par.Dbar = 1.04
        mdl.par.alpha_y = 0.0
        mdl.par.alpha_pi = 0.0
        mdl.par.T = T_test
        mdl.par.announce_t = 0
        mdl.par.implement_t = impl_test
    new.par.mu = 0.0
    old.solve_model()
    new.solve_model()
    print("mu = 0 nesting test:")
    print(f"  max |y_old - y_new|   = {np.max(np.abs(old.sol.y  - new.sol.y )):.3e}")
    print(f"  max |pi_old - pi_new| = {np.max(np.abs(old.sol.pi - new.sol.pi)):.3e}")
    print(f"  max |d_old - d_new|   = {np.max(np.abs(old.sol.d  - new.sol.d )):.3e}")
    print("\npreannounced transfer, implementation at t=25:")
    for mu in [0.0, 0.2, 0.5]:
        m = PreannouncedDeficit_OLG(verbose=False, eps0=0.01)
        p = m.par
        p.mu = mu
        p.beta = 0.998
        p.omega = 0.865
        p.sigma = 1.0
        p.tau_y = 1.0 / 3.0
        p.tau_d = 0.026
        p.kappa = 0.0062
        p.Dbar = 1.04
        p.alpha_y = 0.0
        p.alpha_pi = 0.0
        p.T = T_test
        p.announce_t = 0
        p.implement_t = 25
        m.solve_model()
        print(f"  mu={mu:.1f} | y0={m.sol.y[0]: .8f} | y25={m.sol.y[25]: .8f} | nu={m.sol.share_actual: .8f}")
    print("\nsurprise transfer, implementation at t=0:")
    for mu in [0.0, 0.2, 0.5]:
        m = PreannouncedDeficit_OLG(verbose=False, eps0=0.01)
        p = m.par
        p.mu = mu
        p.beta = 0.998
        p.omega = 0.865
        p.sigma = 1.0
        p.tau_y = 1.0 / 3.0
        p.tau_d = 0.026
        p.kappa = 0.0062
        p.Dbar = 1.04
        p.alpha_y = 0.0
        p.alpha_pi = 0.0
        p.T = T_test
        p.announce_t = 0
        p.implement_t = 0
        m.solve_model()
        print(f"  mu={mu:.1f} | y0={m.sol.y[0]: .8f} | nu={m.sol.share_actual: .8f}")
    print("\nself-financing curve test:")
    m = PreannouncedDeficit_OLG(verbose=False, eps0=0.01)
    p = m.par
    p.mu = 0.073
    p.beta = 0.99**0.25
    p.omega = 0.865
    p.sigma = 1.0
    p.tau_y = 1.0 / 3.0
    p.tau_d = 0.026
    p.kappa = 0.0062
    p.Dbar = 1.04
    p.alpha_y = 0.0
    p.alpha_pi = 0.0
    p.T = 180
    p.announce_t = 0
    p.implement_t = 0
    sf = m._compute_self_financing_curve_tau(announce_t=0, implement_t=0, tau_d_grid=np.array([0.001, 0.026, 0.086]))
    print("  valid_mask =", sf["valid_mask"])
    print("  total_nu   =", np.round(sf["share_total"], 6))


if __name__ == "__main__":
    quick_self_test()
    model = PreannouncedDeficit_OLG(verbose=True, eps0=1.00)
    par = model.par
    par.mu = 0.073
    par.beta = 0.99**0.25
    par.omega = 0.865
    par.sigma = 1.0
    par.tau_y = 1.0 / 3.0
    par.tau_d = 0.026
    par.kappa = 0.0062
    par.Dbar = 1.04
    par.alpha_y = 0.5
    par.alpha_pi = 0.5
    par.T = 500
    par.announce_t = 0
    par.implement_t = 20
    par.x_plot_max = 30
    par.horizon_x_max = 40
    model.run(
        truncation_T=500,
        x_plot_max=50,
        tau_d_list=[0.0, 0.1,0.3,0.5,1.0],
        savepath_main="preannounced_deficit_multi_tau_controls_mu_active.png",
        savepath_delay="preannounced_delay_sweep_mu_active.png",
    )
    model.plot_self_financing_vs_tau(announce_t=0, implement_t=0, selected_tau_d=[0.0, 0.001, 0.026, 0.086,0.1], savepath="preannounced_self_financing_vs_tau_mu_active.png")
