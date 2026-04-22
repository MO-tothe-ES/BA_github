from types import SimpleNamespace
from copy import deepcopy
import numpy as np
import matplotlib

# Use TkAgg locally. If you run headless, switch to "Agg".
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class PreannouncedDeficit_OLG_Original:
    """
    Original deterministic anticipated-deficit OLG-NK self-financing model.
    Kept here only for the mu=0 nesting test.
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

        # 1) Aggregate demand
        for t in range(n):
            A[row, idx("y", t)] = 1.0
            A[row, idx("d", t)] = -X_d
            A[row, idx("Sy", t)] = -(1.0 - par.beta * par.omega) * X_y
            A[row, idx("Sr", t)] = +(1.0 - par.beta * par.omega) * X_r
            b[row] = X_d * S_e[t]
            row += 1

        # 2) Sy recursion: Sy_t = y_t + beta*omega*Sy_{t+1}
        for t in range(n):
            A[row, idx("Sy", t)] = 1.0
            A[row, idx("y", t)] = -1.0
            if t < par.T:
                A[row, idx("Sy", t + 1)] = -par.beta * par.omega
            row += 1

        # 3) Sr recursion with r_t = alpha_y y_t + alpha_pi pi_t
        for t in range(n):
            A[row, idx("Sr", t)] = 1.0
            A[row, idx("y", t)] = -par.alpha_y
            A[row, idx("pi", t)] = -par.alpha_pi
            if t < par.T:
                A[row, idx("Sr", t + 1)] = -par.beta * par.omega
            row += 1

        # 4) NKPC: pi_t = kappa y_t + beta pi_{t+1}
        for t in range(n):
            A[row, idx("pi", t)] = 1.0
            A[row, idx("y", t)] = -par.kappa
            if t < par.T:
                A[row, idx("pi", t + 1)] = -par.beta
            row += 1

        # 5) Debt law under perfect foresight after announcement
        for t in range(par.T):
            A[row, idx("d", t + 1)] = 1.0
            A[row, idx("d", t)] = -(1.0 - par.tau_d) / par.beta
            A[row, idx("y", t)] = par.tau_y / par.beta - par.Dbar * par.alpha_y
            A[row, idx("pi", t)] = -par.Dbar * par.alpha_pi
            b[row] = (1.0 - par.tau_d) / par.beta * e[t]
            row += 1

        # 6) Initial debt jump at announcement: d_0 = -Dbar * pi_0
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
        self.sol.e = self.eqsys.e.copy()
        self.sol.S_e = self.eqsys.S_e.copy()
        self.sol.t = np.arange(n)
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

        sol.tax_base_gain = float(tax_base_gain)
        sol.debt_erosion_gain = float(debt_erosion_gain)
        sol.servicing_cost = float(servicing_cost)
        sol.pv_deficit = float(pv_deficit)

        sol.share_tax_base = float(tax_base_gain / self.eps0)
        sol.share_debt_erosion = float(debt_erosion_gain / self.eps0)
        sol.share_actual = float((tax_base_gain + debt_erosion_gain) / self.eps0)
        sol.residual_need = float(1.0 - sol.share_actual + servicing_cost / self.eps0)


class PreannouncedDeficit_OLG:
    """
    Deterministic anticipated-deficit OLG-NK self-financing model
    extended with a share mu of hand-to-mouth / spender households.

    The extension is the smallest possible:
        c_t = mu * (y_t - t_t) + (1-mu) * c_t^OLG

    where c_t^OLG is exactly the old saver / OLG block.

    mu = 0 nests the original code exactly.
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

        # Baseline close to thesis calibration
        par.beta = 0.998
        par.omega = 0.865
        par.sigma = 1.0
        par.tau_y = 1.0 / 3.0
        par.tau_d = 0.026
        par.kappa = 0.0062
        par.Dbar = 1.04

        # NEW: share of hand-to-mouth / spenders
        par.mu = 0.0

        # Real-rate rule: r_t = alpha_y * y_t + alpha_pi * pi_t
        par.alpha_y = 0.0
        par.alpha_pi = 0.0

        # Simulation controls
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

        # --- Saver / OLG block: exactly the old coefficients ---
        eqsys.X_d_olg = (1.0 - par.beta * par.omega) * (1.0 - par.tau_d) * (1.0 - par.omega) / Delta

        eqsys.C_y_olg = (1.0 - par.beta * par.omega) * (
            1.0 - par.tau_y * (1.0 - par.omega) / Delta
        )

        eqsys.C_r_olg = (1.0 - par.beta * par.omega) * (
            par.sigma * par.beta * par.omega / (1.0 - par.beta * par.omega)
            - par.Dbar * par.beta * (1.0 - par.omega) / Delta
        )

        # --- HtM / spender block ---
        # c_H,t = y_t - t_t = (1-tau_y)*y_t - tau_d*d_t + (1-tau_d)*e_t
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

        # 1) Aggregate demand with HtM share mu
        #
        # y_t = mu*(y_t - t_t) + (1-mu)*c_t^OLG
        #
        # c_t^OLG = X_d_olg*(d_t + S_e[t]) + C_y_olg*Sy_t - C_r_olg*Sr_t
        # y_t - t_t = (1-tau_y)*y_t - tau_d*d_t + (1-tau_d)*e_t
        #
        # Rearranging gives:
        for t in range(n):
            A[row, idx("y", t)] = eqsys.mu_left
            A[row, idx("d", t)] = -(eqsys.mu_d + (1.0 - par.mu) * eqsys.X_d_olg)
            A[row, idx("Sy", t)] = -(1.0 - par.mu) * eqsys.C_y_olg
            A[row, idx("Sr", t)] = +(1.0 - par.mu) * eqsys.C_r_olg

            b[row] = eqsys.mu_e0 * e[t] + (1.0 - par.mu) * eqsys.X_d_olg * S_e[t]
            row += 1

        # 2) Sy recursion: Sy_t = y_t + beta*omega*Sy_{t+1}
        for t in range(n):
            A[row, idx("Sy", t)] = 1.0
            A[row, idx("y", t)] = -1.0
            if t < par.T:
                A[row, idx("Sy", t + 1)] = -par.beta * par.omega
            row += 1

        # 3) Sr recursion with r_t = alpha_y y_t + alpha_pi pi_t
        for t in range(n):
            A[row, idx("Sr", t)] = 1.0
            A[row, idx("y", t)] = -par.alpha_y
            A[row, idx("pi", t)] = -par.alpha_pi
            if t < par.T:
                A[row, idx("Sr", t + 1)] = -par.beta * par.omega
            row += 1

        # 4) NKPC: pi_t = kappa y_t + beta pi_{t+1}
        for t in range(n):
            A[row, idx("pi", t)] = 1.0
            A[row, idx("y", t)] = -par.kappa
            if t < par.T:
                A[row, idx("pi", t + 1)] = -par.beta
            row += 1

        # 5) Debt law under perfect foresight after announcement
        for t in range(par.T):
            A[row, idx("d", t + 1)] = 1.0
            A[row, idx("d", t)] = -(1.0 - par.tau_d) / par.beta
            A[row, idx("y", t)] = par.tau_y / par.beta - par.Dbar * par.alpha_y
            A[row, idx("pi", t)] = -par.Dbar * par.alpha_pi
            b[row] = (1.0 - par.tau_d) / par.beta * e[t]
            row += 1

        # 6) Initial debt jump at announcement: d_0 = -Dbar * pi_0
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
        self.sol.e = self.eqsys.e.copy()
        self.sol.S_e = self.eqsys.S_e.copy()
        self.sol.t = np.arange(n)
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

        sol.tax_base_gain = float(tax_base_gain)
        sol.debt_erosion_gain = float(debt_erosion_gain)
        sol.servicing_cost = float(servicing_cost)
        sol.pv_deficit = float(pv_deficit)

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

        spread = float(np.max(shares) - np.min(shares))
        return {"T_list": list(T_list), "shares": shares, "spread": spread}

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
                    [0.78, 0.88, 0.96],
                    [0.55, 0.75, 0.90],
                    [0.30, 0.55, 0.80],
                    [0.15, 0.30, 0.55],
                    [0.05, 0.10, 0.20],
                ]
            ),
            "finance_price": np.array([0.20, 0.45, 0.85]),
            "finance_tax": np.array([0.40, 0.60, 0.80]),
        }

    def set_plot_style(self):
        plt.rcParams.update(
            {
                "font.size": 14,
                "axes.titlesize": 21,
                "axes.labelsize": 16,
                "legend.fontsize": 15,
                "xtick.labelsize": 14,
                "ytick.labelsize": 14,
            }
        )

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

    def plot_1x4(
        self,
        announce_t=None,
        implement_t=None,
        tau_d_list=None,
        truncation_T=None,
        x_plot_max=None,
        horizon_x_max=None,
        savepath=None,
        show=True,
    ):
        """
        1x4 figure:
          1) output IRFs across tau_d
          2) debt IRFs across tau_d
          3) inflation and real-rate IRFs across tau_d
          4) total self-financing share versus implementation horizon across tau_d
        """
        par = self.par
        colors = self.style_colors()
        self.set_plot_style()

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
        if horizon_x_max is None:
            horizon_x_max = par.horizon_x_max

        old_T = par.T
        old_tau_d = par.tau_d

        try:
            par.T = int(truncation_T)
            implement_t_grid = np.arange(announce_t, int(horizon_x_max) + 1)

            line_colors = colors["line_list"][: len(tau_d_list)]

            # Solve path IRFs for each tau_d at the chosen implementation date
            irf_results = []
            for tau_d in tau_d_list:
                par.tau_d = float(tau_d)
                self.solve_model(announce_t=announce_t, implement_t=implement_t)
                res = deepcopy(self.sol)
                res.tau_d = float(tau_d)
                irf_results.append(res)

            fig, axes = plt.subplots(1, 4, figsize=(24, 5.8), constrained_layout=False)
            fig.subplots_adjust(right=0.84, wspace=0.30)

            # Panel 1: Output IRFs
            ax = axes[0]
            for color, res in zip(line_colors, irf_results):
                ax.plot(res.t, res.y, lw=3, color=color)
            ax.axvline(implement_t, color="black", ls="--", alpha=0.8, linewidth=1.2)
            ax.set_title(r"Output $y_t$")
            ax.set_xlabel(r"$t$")
            ax.set_xlim(0, min(int(x_plot_max), irf_results[0].t[-1]))
            ax.grid(True, alpha=0.25)

            # Panel 2: Debt IRFs
            ax = axes[1]
            for color, res in zip(line_colors, irf_results):
                ax.plot(res.t, res.d, lw=3, color=color)
            ax.axvline(implement_t, color="black", ls="--", alpha=0.8, linewidth=1.2)
            ax.set_title(r"Gov't Debt $d_t$")
            ax.set_xlabel(r"$t$")
            ax.set_xlim(0, min(int(x_plot_max), irf_results[0].t[-1]))
            ax.grid(True, alpha=0.25)

            # Panel 3: Inflation and real rate IRFs
            ax = axes[2]
            for color, res in zip(line_colors, irf_results):
                ax.plot(res.t, res.pi, lw=2.8, color=color, ls="--")
                ax.plot(res.t, res.r, lw=2.8, color=color, ls="-")
            ax.axvline(implement_t, color="black", ls="--", alpha=0.8, linewidth=1.2)
            ax.set_title(r"Rates and Inflation: $r_t,\ \pi_t$")
            ax.set_xlabel(r"$t$")
            ax.set_xlim(0, min(int(x_plot_max), irf_results[0].t[-1]))
            ax.grid(True, alpha=0.25)
            style_handles = [
                Line2D([0], [0], color="black", lw=2.8, ls="-", label=r"$r_t$"),
                Line2D([0], [0], color="black", lw=2.8, ls="--", label=r"$\pi_t$"),
            ]
            ax.legend(handles=style_handles, loc="upper right", frameon=True)

            # Panel 4: Horizon plot, total self-financing only, sweeping tau_d
            ax = axes[3]
            for color, tau_d in zip(line_colors, tau_d_list):
                s_grid, nu_grid = self._sweep_total_share_given_tau(
                    tau_d=float(tau_d), announce_t=announce_t, implement_t_grid=implement_t_grid
                )
                ax.plot(s_grid, nu_grid, lw=3, color=color)

            ax.axhline(1.0, color="black", ls="--", alpha=0.5, linewidth=1.2)
            ax.axvline(implement_t, color="black", ls=":", alpha=0.6, linewidth=1.2)
            ax.set_title(r"Self-Financing Share $\nu$")
            ax.set_xlabel(r"Implementation date $s$")
            ax.set_ylabel(r"$\nu(s)$")
            ax.set_xlim(float(np.min(implement_t_grid)), float(np.max(implement_t_grid)))
            ax.grid(True, alpha=0.25)

            fig.legend(
                handles=[
                    Line2D([0], [0], color=color, lw=3, label=rf"$\tau_d={tau:.1f}$")
                    for color, tau in zip(line_colors, tau_d_list)
                ],
                loc="center left",
                bbox_to_anchor=(0.86, 0.5),
                frameon=True,
                title=r"($\tau_d$)",
                borderaxespad=0.0,
            )

            fig.suptitle(
                rf"Announcement at $t={announce_t}$, implementation at $t={implement_t}$, "
                rf"calculation horizon $T={truncation_T}$, $\mu={par.mu:.2f}$",
                fontsize=16,
            )

            if savepath is not None:
                fig.savefig(savepath, dpi=200, bbox_inches="tight")
                if self.verbose:
                    print(f"saved to: {savepath}")

            if show:
                plt.show()
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

    def run(self, truncation_T=None, x_plot_max=None, horizon_x_max=None, savepath=None):
        self.summary(truncation_T=truncation_T)
        self.plot_1x4(
            tau_d_list=[0.001, 0.026, 0.086],
            truncation_T=truncation_T,
            x_plot_max=x_plot_max,
            horizon_x_max=horizon_x_max,
            savepath=savepath,
        )


def quick_self_test():
    """
    Small local test block.

    1) checks mu=0 nests the original code exactly
    2) prints some intuitive comparative statics for mu > 0

    Keep T modest for speed in the test.
    """
    print("\n--- quick self test ---")

    T_test = 120
    impl_test = 25

    # mu = 0 nesting test
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

    # intuition check: preannounced shock
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
        print(
            f"  mu={mu:.1f} | y0={m.sol.y[0]: .8f} | "
            f"y25={m.sol.y[25]: .8f} | nu={m.sol.share_actual: .8f}"
        )

    # intuition check: surprise shock
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
        print(
            f"  mu={mu:.1f} | y0={m.sol.y[0]: .8f} | "
            f"nu={m.sol.share_actual: .8f}"
        )


if __name__ == "__main__":
    # Run the quick test first
    quick_self_test()

    # Then run your preferred plotting experiment
    model = PreannouncedDeficit_OLG(verbose=True, eps0=0.01)

    par = model.par
    par.mu = 0.073
    par.beta = 0.99**0.25
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
    par.implement_t = 0
    par.x_plot_max = 30
    par.horizon_x_max = 50

    model.run(
        truncation_T=500,
        x_plot_max=50,
        horizon_x_max=50,
        savepath="preannounced_deficit_multi_tau_controls_mu.png",
    )