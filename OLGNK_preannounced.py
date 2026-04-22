from types import SimpleNamespace
from copy import deepcopy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


class AnnouncedTransferOLG:
    """
    Announced-transfer extension of the user's OLG-NK model.

    Numerical solution method
    -------------------------
    For a deterministic transfer path announced at t=0, the post-announcement
    equilibrium is solved as one stacked linear system in the unknown time paths

        {y_t}_{t=0}^H, {pi_t}_{t=0}^H, {d_t}_{t=0}^{H+1}

    rather than by recursive affine policy rules.

    For a fixed implementation delay s, the code:
    1. builds the known transfer path e_t,
    2. stacks the equilibrium conditions period by period,
    3. imposes terminal conditions y_{H+1}=0 and pi_{H+1}=0,
    4. imposes the date-0 debt jump condition d_0 = -Dbar * pi_0,
    5. solves the full sparse-looking but dense-stored linear system A x = b.

    This is less numerically efficient than the backward-recursive method, but it
    is much closer to the model equations and easier to explain in a thesis.
    """
    def __init__(self, verbose=True, eps0: float = 1.0):
        self.verbose = verbose
        self.eps0 = float(eps0)

        self.par = SimpleNamespace()
        self.eqsys = SimpleNamespace()
        self.sol_all = SimpleNamespace()

        self.set_parameters()
        self.allocate()

    def set_parameters(self):
        par = self.par

        par.beta  = 0.99**0.25
        par.omega = 0.865
        par.tau_y = 1.0 / 3.0
        par.sigma = 1.0
        par.kappa = 0.0062
        par.Dbar  = 1.04
        par.psi   = 0.5 + 1.0 / par.beta
        par.phi   = 0.5 - par.kappa / par.beta
        par.tau_d = None

        # plotting horizon
        par.T = 50

        # truncation horizon used for backward induction
        # choose this large enough so the terminal steady-state condition is a very
        # accurate approximation to the infinite-horizon bounded solution
        par.solve_horizon = 500

        # default max delay used in delay sweeps
        par.max_delay = 80

    def allocate(self):
        self.sol_all.results = []
        self.sol_all.tau_list = []
        self.sol_all.delay_list = []
        self.sol_all.nu_total_list = []
        self.sol_all.nu_tax_list = []
        self.sol_all.nu_price_list = []

    # =========================
    # Model coefficients
    # =========================
    def eqsys_matrix_elements(self):
        par = self.par
        eqsys = self.eqsys

        beta, omega = par.beta, par.omega
        tau_y, tau_d = par.tau_y, par.tau_d
        sigma, kappa = par.sigma, par.kappa
        Dbar, psi, phi = par.Dbar, par.psi, par.phi

        denom = 1.0 - omega * (1.0 - tau_d)
        if abs(denom) < 1e-14:
            raise ValueError("Denominator too close to zero.")

        eqsys.X_d = (1.0 - beta * omega) * (1.0 - tau_d) * (1.0 - omega) / denom
        eqsys.X_y = 1.0 - tau_y * (1.0 - omega) / denom
        eqsys.X_r = sigma * beta * omega / (1.0 - beta * omega) - Dbar * beta * (1.0 - omega) / denom

        eqsys.r_y = phi + kappa / beta
        eqsys.r_pi = psi - 1.0 / beta

        # debt law: d_{t+1} = ...
        eqsys.m_dd  = (1.0 - tau_d) / beta
        eqsys.m_dy  = Dbar * eqsys.r_y - tau_y / beta
        eqsys.m_dpi = Dbar * eqsys.r_pi
        eqsys.m_de  = (1.0 - tau_d) / beta

        X_d, X_y, X_r = eqsys.X_d, eqsys.X_y, eqsys.X_r

        # recursive DIS with current realized transfer e_t = n_t^(0)
        eqsys.m_yd = -X_d * (1.0 - omega * (1.0 - tau_d)) / (beta * omega)
        eqsys.m_ye = -X_d * (1.0 - omega * (1.0 - tau_d)) / (beta * omega)
        eqsys.m_yy = (
            1.0
            - (1.0 - beta * omega) * (X_y - X_r * eqsys.r_y)
            + beta * omega * X_d * eqsys.m_dy
        ) / (beta * omega)
        eqsys.m_ypi = (
            ((1.0 - beta * omega) * X_r + beta * omega * X_d * Dbar) * eqsys.r_pi
        ) / (beta * omega)

        # NKPC
        eqsys.m_piy = -kappa / beta
        eqsys.m_pipi = 1.0 / beta

    def system_matrix_given_policy(self, psi: float, phi: float, tau_d: float) -> np.ndarray:
        """
        Time-invariant 3x3 matrix for the homogeneous system (e_t = 0).
        This is used for determinacy checks and matches the s=0 baseline case.
        """
        par = self.par
        beta, omega = par.beta, par.omega
        tau_y = par.tau_y
        sigma, kappa = par.sigma, par.kappa
        Dbar = par.Dbar

        denom = 1.0 - omega * (1.0 - tau_d)
        if abs(denom) < 1e-14:
            raise ValueError("Denominator too close to zero.")

        X_d = (1.0 - beta * omega) * (1.0 - tau_d) * (1.0 - omega) / denom
        X_y = 1.0 - tau_y * (1.0 - omega) / denom
        X_r = sigma * beta * omega / (1.0 - beta * omega) - Dbar * beta * (1.0 - omega) / denom

        r_y = phi + kappa / beta
        r_pi = psi - 1.0 / beta

        m_dd  = (1.0 - tau_d) / beta
        m_dy  = Dbar * r_y - tau_y / beta
        m_dpi = Dbar * r_pi

        m_yd = -X_d * (1.0 - omega * (1.0 - tau_d)) / (beta * omega)
        m_yy = (
            1.0
            - (1.0 - beta * omega) * (X_y - X_r * r_y)
            + beta * omega * X_d * m_dy
        ) / (beta * omega)
        m_ypi = (
            ((1.0 - beta * omega) * X_r + beta * omega * X_d * Dbar) * r_pi
        ) / (beta * omega)

        return np.array([
            [m_dd,  m_dy,         m_dpi],
            [m_yd,  m_yy,         m_ypi],
            [0.0,  -kappa / beta, 1.0 / beta],
        ], dtype=float)

    def exists_unique_bounded_equilibrium(self, tol=1e-9):
        vals = np.linalg.eigvals(
            self.system_matrix_given_policy(
                psi=float(self.par.psi),
                phi=float(self.par.phi),
                tau_d=float(self.par.tau_d),
            )
        )
        n_stable = int(np.sum(np.abs(vals) < 1.0 - tol))
        n_unstable = int(np.sum(np.abs(vals) > 1.0 + tol))
        return n_stable == 1 and n_unstable == 2

    # =========================
    # Announced transfer path
    # =========================
    def build_transfer_path(self, delay: int, horizon: int = None):
        """
        One-off transfer e_t announced at t=0 and implemented at t=delay.
        """
        if horizon is None:
            horizon = self.par.solve_horizon

        e_path = np.zeros(horizon + 1)
        if 0 <= delay <= horizon:
            e_path[delay] = self.eps0
        return e_path

    def _stack_indices(self, horizon: int):
        """
        Indices for the stacked unknown vector

            x = [y_0,...,y_H, pi_0,...,pi_H, d_0,...,d_{H+1}]'
        """
        n = horizon + 1
        off_y = 0
        off_pi = n
        off_d = 2 * n

        def iy(t):
            return off_y + t

        def ipi(t):
            return off_pi + t

        def idd(t):
            return off_d + t

        N = 3 * n + 1
        return SimpleNamespace(n=n, N=N, iy=iy, ipi=ipi, idd=idd)

    def build_stacked_system(self, e_path):
        """
        Build the full linear system A x = b for the announced-transfer experiment.

        Unknown vector:
            x = [y_0,...,y_H, pi_0,...,pi_H, d_0,...,d_{H+1}]'

        Equations:
            1. debt law for t=0,...,H
            2. recursive DIS / output transition for t=0,...,H
               with terminal condition y_{H+1}=0 used in the last equation
            3. NKPC for t=0,...,H
               with terminal condition pi_{H+1}=0 used in the last equation
            4. date-0 debt jump condition d_0 = -Dbar * pi_0

        """
        par = self.par
        eq = self.eqsys
        H = len(e_path) - 1
        idx = self._stack_indices(H)

        A = np.zeros((idx.N, idx.N), dtype=float)
        b = np.zeros(idx.N, dtype=float)
        row = 0

        # 1) Debt law: d_{t+1} - m_dd d_t - m_dy y_t - m_dpi pi_t = m_de e_t
        for t in range(H + 1):
            A[row, idx.idd(t + 1)] = 1.0
            A[row, idx.idd(t)] = -eq.m_dd
            A[row, idx.iy(t)] = -eq.m_dy
            A[row, idx.ipi(t)] = -eq.m_dpi
            b[row] = eq.m_de * e_path[t]
            row += 1

        # 2) Recursive DIS / output transition
        #    y_{t+1} - m_yd d_t - m_yy y_t - m_ypi pi_t = m_ye e_t
        # For t=H, use y_{H+1}=0.
        for t in range(H + 1):
            A[row, idx.idd(t)] = -eq.m_yd
            A[row, idx.iy(t)] = -eq.m_yy
            A[row, idx.ipi(t)] = -eq.m_ypi
            if t < H:
                A[row, idx.iy(t + 1)] = 1.0
                b[row] = eq.m_ye * e_path[t]
            else:
                b[row] = eq.m_ye * e_path[t]
            row += 1

        # 3) NKPC: pi_t - kappa y_t - beta pi_{t+1} = 0
        # For t=H, use pi_{H+1}=0.
        for t in range(H + 1):
            A[row, idx.ipi(t)] = 1.0
            A[row, idx.iy(t)] = -par.kappa
            if t < H:
                A[row, idx.ipi(t + 1)] = -par.beta
            row += 1

        # 4) Initial debt jump from predetermined nominal debt
        #    d_0 + Dbar * pi_0 = 0
        A[row, idx.idd(0)] = 1.0
        A[row, idx.ipi(0)] = par.Dbar
        row += 1

        if row != idx.N:
            raise RuntimeError(f"Equation count mismatch: row={row}, N={idx.N}")

        return A, b, idx

    def solve_given_delay(self, delay: int, horizon: int = None):
        """
        Solve the deterministic announced-transfer experiment for a fixed delay
        by stacking all equilibrium conditions into one linear system.
        """
        if horizon is None:
            horizon = self.par.solve_horizon

        self.eqsys_matrix_elements()
        e_path = self.build_transfer_path(delay=delay, horizon=horizon)
        A, b, idx = self.build_stacked_system(e_path)
        x = np.linalg.solve(A, b)

        H = horizon
        par = self.par
        eq = self.eqsys

        y = x[idx.iy(0): idx.iy(H) + 1].copy()
        pi = x[idx.ipi(0): idx.ipi(H) + 1].copy()
        d = x[idx.idd(0): idx.idd(H + 1) + 1].copy()
        d0 = float(d[0])
        debt_end = d[1:].copy()

        r = eq.r_y * y + eq.r_pi * pi
        i_nom = par.psi * pi + par.phi * y

        # announced-shock self-financing object
        k = np.arange(H + 1)
        pv_e = float(np.sum((par.beta ** k) * e_path))
        pv_y = float(np.sum((par.beta ** k) * y))
        pv_r = float(np.sum((par.beta ** (k + 1)) * r))

        tax_gain = par.tau_y * pv_y
        price_gain = par.Dbar * pi[0]
        denom = pv_e + par.Dbar * pv_r

        sol = SimpleNamespace()
        sol.delay = int(delay)
        sol.t = np.arange(H + 1)
        sol.e = e_path.copy()
        sol.d = d[:-1].copy()
        sol.debt_end = debt_end.copy()
        sol.y = y.copy()
        sol.pi = pi.copy()
        sol.r = r.copy()
        sol.i_nom = i_nom.copy()
        sol.d0 = d0

        # Keep the stacked objects for transparency / thesis explanation
        sol.system_matrix = A
        sol.system_rhs = b
        sol.solution_vector = x

        sol.pv_e = pv_e
        sol.pv_y = pv_y
        sol.pv_r = pv_r

        sol.tax_gain = float(tax_gain)
        sol.price_gain = float(price_gain)
        sol.denominator = float(denom)

        sol.nu_tax = float(tax_gain / denom)
        sol.nu_price = float(price_gain / denom)
        sol.nu_total = float((tax_gain + price_gain) / denom)

        # full tax rule
        sol.tax_rule = par.tau_y * sol.y + par.tau_d * (sol.d + sol.e) - sol.e

        # tax rule excluding the direct transfer term -e_t
        sol.tax_no_direct_transfer = par.tau_y * sol.y + par.tau_d * (sol.d + sol.e)

        # optional decomposition
        sol.tax_base_part = par.tau_y * sol.y
        sol.debt_feedback_part = par.tau_d * (sol.d + sol.e)

        return sol

    # =========================
    # Sweeps
    # =========================
    def solve_tau_sweep_fixed_delay(self, delay: int, tau_d_grid=None, horizon: int = None):
        if tau_d_grid is None:
            tau_d_grid = np.sort(
                np.concatenate((
                    np.linspace(0.0, 1.0, 301),
                    np.array([0.085, 0.026, 0.004])
                ))
            )
        if horizon is None:
            horizon = self.par.solve_horizon

        self.allocate()
        old_tau = self.par.tau_d

        try:
            for tau_d in tau_d_grid:
                self.par.tau_d = float(tau_d)
                try:
                    if self.exists_unique_bounded_equilibrium():
                        sol = self.solve_given_delay(delay=delay, horizon=horizon)
                        sol.tau_d = float(tau_d)

                        # keep only plotted horizon for convenience in plotting
                        T_plot = self.par.T
                        sol.t = sol.t[:T_plot + 1]
                        sol.y = sol.y[:T_plot + 1]
                        sol.pi = sol.pi[:T_plot + 1]
                        sol.r = sol.r[:T_plot + 1]
                        sol.i_nom = sol.i_nom[:T_plot + 1]
                        sol.debt_end = sol.debt_end[:T_plot + 1]
                        sol.d = sol.d[:T_plot + 1]
                        sol.e = sol.e[:T_plot + 1]
                        sol.tax_no_direct_transfer = sol.tax_no_direct_transfer[:T_plot + 1]
                        sol.tax_base_part = sol.tax_base_part[:T_plot + 1]
                        sol.debt_feedback_part = sol.debt_feedback_part[:T_plot + 1]

                        self.sol_all.results.append(deepcopy(sol))
                        self.sol_all.tau_list.append(float(tau_d))
                    else:
                        continue
                except Exception:
                    continue
        finally:
            self.par.tau_d = old_tau

    def sweep_delay_given_tau(self, tau_d: float, delay_grid=None, horizon: int = None):
        if delay_grid is None:
            delay_grid = np.arange(0, self.par.max_delay + 1)
        if horizon is None:
            horizon = self.par.solve_horizon

        delay_grid = np.asarray(delay_grid, dtype=float)
        old_tau = self.par.tau_d
        try:
            self.par.tau_d = float(tau_d)
            is_determinate = bool(self.exists_unique_bounded_equilibrium())

            if not is_determinate:
                nan_vec = np.full(len(delay_grid), np.nan, dtype=float)
                return (
                    delay_grid.copy(),
                    nan_vec.copy(),
                    nan_vec.copy(),
                    nan_vec.copy(),
                    False,
                )

            delay_list = []
            nu_total_list = []
            nu_tax_list = []
            nu_price_list = []

            for delay in delay_grid:
                sol = self.solve_given_delay(delay=int(delay), horizon=horizon)
                delay_list.append(int(delay))
                nu_total_list.append(sol.nu_total)
                nu_tax_list.append(sol.nu_tax)
                nu_price_list.append(sol.nu_price)

            return (
                np.asarray(delay_list, dtype=float),
                np.asarray(nu_total_list, dtype=float),
                np.asarray(nu_tax_list, dtype=float),
                np.asarray(nu_price_list, dtype=float),
                True,
            )
        finally:
            self.par.tau_d = old_tau

    # =========================
    # Style helpers
    # =========================
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
        return plt.cm.PuRd(np.linspace(0.35, 0.90, n))

    def _shade_invalid_tau_regions(self, ax, tau_d_grid, valid_tau, label=None):
        tau_d_grid = np.asarray(tau_d_grid, dtype=float)
        valid_tau = np.asarray(valid_tau, dtype=float)

        valid_mask = np.array([
            np.any(np.isclose(tau, valid_tau, atol=1e-10))
            for tau in tau_d_grid
        ], dtype=bool)

        invalid_mask = ~valid_mask
        if not np.any(invalid_mask):
            return

        starts = np.where(invalid_mask & ~np.r_[False, invalid_mask[:-1]])[0]
        ends   = np.where(invalid_mask & ~np.r_[invalid_mask[1:], False])[0]

        first_patch = True
        for s, e in zip(starts, ends):
            x0 = 0.0 if s == 0 else 0.5 * (tau_d_grid[s - 1] + tau_d_grid[s])
            x1 = 1.0 if e == len(tau_d_grid) - 1 else 0.5 * (tau_d_grid[e] + tau_d_grid[e + 1])

            ax.axvspan(
                x0, x1,
                facecolor="grey",
                alpha=0.55,
                zorder=0,
                label=label if first_patch else None
            )
            first_patch = False

    # =========================
    # Plotting
    # =========================
    def plot_announced_irfs(
        self,
        delay=0,
        selected_tau_d=(0.0, 0.1, 0.3, 0.5, 1.0),
        tau_d_grid=None,
        figsize=(10, 8),
        savepath=None,
        ylim0=False,
        horizon=None,
    ):
        """
        Same 2x2 layout and visual style as the user's baseline plot,
        but for a fixed implementation delay.
        """
        if horizon is None:
            horizon = self.par.solve_horizon

        if tau_d_grid is None:
            tau_d_grid = np.sort(
                np.concatenate((
                    np.linspace(0.0, 1.0, 301),
                    np.array([0.085, 0.026, 0.004])
                ))
            )

        self.solve_tau_sweep_fixed_delay(delay=delay, tau_d_grid=tau_d_grid, horizon=horizon)

        if len(self.sol_all.results) == 0:
            raise RuntimeError("No determinate tau_d values were found, so nothing can be plotted.")

        selected_results = []
        missing_tau = []

        for tau_d in selected_tau_d:
            found = False
            for sol in self.sol_all.results:
                if abs(sol.tau_d - tau_d) < 1e-9:
                    selected_results.append(sol)
                    found = True
                    break
            if not found:
                missing_tau.append(tau_d)

        if len(selected_results) == 0:
            raise RuntimeError("None of the selected tau_d values produced a unique bounded equilibrium.")

        if len(missing_tau) > 0 and self.verbose:
            print(f"Warning: no unique bounded equilibrium for tau_d = {missing_tau}")

        results_sorted = sorted(self.sol_all.results, key=lambda x: x.tau_d)
        tau_sorted = np.array([res.tau_d for res in results_sorted])
        nu_price = np.array([res.nu_price for res in results_sorted])
        nu_total = np.array([res.nu_total for res in results_sorted])

        line_colors = self._get_line_colors(len(selected_results))
        colors = self.style_colors()

        with mpl.rc_context({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 13,
            "axes.titlesize": 17,
            "axes.labelsize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "mathtext.fontset": "stix",
        }):
            fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=False, constrained_layout=False)
            fig.subplots_adjust(
                left=0.1,
                right=0.85,
                top=0.92,
                bottom=0.12,
                wspace=0.36,
                hspace=0.34,
            )

            # Panel 1: Output
            ax = axes[0, 0]
            for color, res in zip(line_colors, selected_results):
                ax.plot(res.t, res.y, lw=2.8, color=color)
            if delay > 0:
                ax.axvline(delay, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
            ax.set_title(r"Output $y_t$")
            ax.set_xlabel(r"t")
            ax.set_ylabel(r"%")
            ax.set_xlim(0, selected_results[0].t[-1])
            ax.grid(True, alpha=0.25)

            # Panel 2: End-of-period debt
            ax = axes[0, 1]
            for color, res in zip(line_colors, selected_results):
                ax.plot(res.t, res.debt_end, lw=2.8, color=color)
            if delay > 0:
                ax.axvline(delay, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
            ax.set_title(r"Public debt $d_{t+1}$")
            ax.set_xlabel(r"t")
            ax.set_ylabel(r"")
            ax.set_xlim(0, selected_results[0].t[-1])
            ax.grid(True, alpha=0.25)
            ax.set_ylim(None,None)

            # Panel 3: Inflation / nominal / real rate
            ax = axes[1, 0]
            ax_r = ax.twinx()

            for color, res in zip(line_colors, selected_results):
                ax.plot(res.t, res.pi,    lw=2.5, color=color, ls="--")
                ax.plot(res.t, res.i_nom, lw=2.5, color=color, ls=":")
                ax_r.plot(res.t, res.r,   lw=2.5, color=color, ls="-")

            if delay > 0:
                ax.axvline(delay, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
            ax.set_title(r"Inflation and interest rates")
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(r"$\pi_t,\ i_t$")
            ax_r.set_ylabel(r"$r_t$")
            ax.set_xlim(0, selected_results[0].t[-1])
            ax.grid(True, alpha=0.25)

            style_handles = [
                Line2D([0], [0], color="black", lw=2.5, ls="--", label=r"$\pi_t$"),
                Line2D([0], [0], color="black", lw=2.5, ls=":",  label=r"$i_t$"),
                Line2D([0], [0], color="black", lw=2.5, ls="-",  label=r"$r_t$"),
            ]
            ax.legend(handles=style_handles, loc="upper right", frameon=True)

            # Panel 4: Self-financing as function of tau_d at fixed delay
            ax = axes[1, 1]
            self._shade_invalid_tau_regions(
                ax=ax,
                tau_d_grid=tau_d_grid,
                valid_tau=tau_sorted,
                label=r"no unique eq.",
            )

            ax.fill_between(
                tau_sorted, 0.0, nu_price,
                color=colors["finance_price"],
                alpha=0.95,
                label="Date-0 Inflation",
                zorder=2,
            )
            ax.fill_between(
                tau_sorted, nu_price, nu_total,
                color=colors["finance_tax"],
                alpha=0.95,
                label="Tax Base",
                zorder=3,
            )
            ax.plot(tau_sorted, nu_total, color="black", lw=1.4, alpha=0.7, zorder=4)
            
            for color, res in zip(line_colors, selected_results):
                ax.scatter(
                    res.tau_d,
                    res.nu_total,
                    s=55,
                    color=color,
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=5,
                )

            ax.set_title(rf"Self-financing share $\nu$ (delay $s={delay}$)")
            ax.set_xlabel(r"$\tau_d$")
            ax.set_ylabel(r"")
            ax.set_xlim(0.0, 1.0)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right", frameon=True)

            # ticks
            tmax = int(selected_results[0].t[-1])
            xticks = np.arange(0, tmax + 1, 10)

            for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
                ax.set_xticks(xticks)
                ax.tick_params(axis="x", labelbottom=True)

            if ylim0:
                for ax in [axes[0, 0], axes[0, 1]]:
                    ax.set_ylim(0, None)

            tau_handles = [
                Line2D([0], [0], color=color, lw=2.8, label=rf"{res.tau_d:.3f}")
                for color, res in zip(line_colors, selected_results)
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
                fig.savefig(savepath, dpi=200, bbox_inches="tight")
                if self.verbose:
                    print(f"saved to: {savepath}")

            plt.show()
            plt.close(fig)

    def plot_self_financing_vs_delay(
        self,
        selected_tau_d=(0.0, 0.1, 0.3, 0.5, 1.0),
        delay_grid=None,
        figsize=(9.5, 6.0),
        savepath=None,
        horizon=None,
        show_components=False,
    ):
        """
        Plot self-financing share nu as a function of the implementation delay s,
        for one or several values of tau_d.

        If some selected tau_d values do not admit a unique bounded equilibrium,
        the figure is still produced. Those values are not plotted as curves and are
        instead reported with a grey "no unique equilibrium" legend entry.
        If all selected tau_d values are indeterminate, the whole panel is shaded grey.
        """
        if delay_grid is None:
            delay_grid = np.arange(0, self.par.max_delay + 1)
        if horizon is None:
            horizon = self.par.solve_horizon

        delay_grid = np.asarray(delay_grid, dtype=float)
        line_colors = self._get_line_colors(len(selected_tau_d))

        with mpl.rc_context({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 13,
            "axes.titlesize": 17,
            "axes.labelsize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "mathtext.fontset": "stix",
        }):
            fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

            invalid_tau = []
            valid_handles = []

            for color, tau_d in zip(line_colors, selected_tau_d):
                s_grid, nu_total, nu_tax, nu_price, is_determinate = self.sweep_delay_given_tau(
                    tau_d=float(tau_d),
                    delay_grid=delay_grid,
                    horizon=horizon,
                )

                if is_determinate:
                    line = ax.plot(s_grid, nu_total, lw=2.8, color=color, label=rf"$\tau_d={tau_d:.3f}$")[0]
                    valid_handles.append(line)

                    if show_components:
                        ax.plot(s_grid, nu_tax, lw=1.8, color=color, ls="--", alpha=0.9)
                        ax.plot(s_grid, nu_price, lw=1.8, color=color, ls=":", alpha=0.9)
                else:
                    invalid_tau.append(float(tau_d))

            if len(valid_handles) == 0 and len(invalid_tau) > 0:
                ax.axvspan(
                    float(np.min(delay_grid)),
                    float(np.max(delay_grid)),
                    facecolor="grey",
                    alpha=0.35,
                    zorder=0,
                )

            ax.set_title(r"Self-financing share $\nu$ by implementation delay")
            ax.set_xlabel(r"Delay $s$")
            ax.set_ylabel(r"$\nu(s)$")
            ax.set_xlim(float(np.min(delay_grid)), float(np.max(delay_grid)))
            ax.grid(True, alpha=0.25)

            legend_handles = list(valid_handles)
            if len(invalid_tau) > 0:
                invalid_label = r"no unique eq.: " + ", ".join([f"{tau:.3f}" for tau in invalid_tau])
                legend_handles.append(
                    Patch(facecolor="grey", edgecolor="grey", alpha=0.55, label=invalid_label)
                )
            if len(legend_handles) > 0:
                ax.legend(handles=legend_handles, loc="best", frameon=True)

            if show_components:
                style_handles = [
                    Line2D([0], [0], color="black", lw=2.0, ls="-", label=r"Total"),
                    Line2D([0], [0], color="black", lw=2.0, ls="--", label=r"Tax base"),
                    Line2D([0], [0], color="black", lw=2.0, ls=":", label=r"Date-0 inflation"),
                ]
                ax2 = ax.twinx()
                ax2.set_yticks([])
                ax2.legend(handles=style_handles, loc="upper right", frameon=True)

            if savepath is not None:
                fig.savefig(savepath, dpi=200, bbox_inches="tight")
                if self.verbose:
                    print(f"saved to: {savepath}")

            plt.show()
            plt.close(fig)

    # =========================
    # Nesting test
    # =========================
    def compare_to_baseline_s0(self, baseline_model, tau_d_values=(0.085, 0.026, 0.004), horizon=None):
        """
        Compare the announced-transfer solver at s=0 against the user's baseline code.

        Returns a list of dictionaries with max absolute differences.
        """
        if horizon is None:
            horizon = self.par.solve_horizon

        out = []
        for tau_d in tau_d_values:
            # announced model
            self.par.tau_d = float(tau_d)
            sol_a = self.solve_given_delay(delay=0, horizon=horizon)

            # baseline model
            baseline_model.par.tau_d = float(tau_d)
            baseline_model.eqsys_matrix_elements()
            baseline_model.system_matrix()
            baseline_model.solve_unique_bounded_eq()
            baseline_model.compute_irf()
            sol_b = baseline_model.sol

            T = min(len(sol_b.y), len(sol_a.y))

            out.append({
                "tau_d": float(tau_d),
                "max_abs_y": float(np.max(np.abs(sol_b.y[:T] - sol_a.y[:T]))),
                "max_abs_pi": float(np.max(np.abs(sol_b.pi[:T] - sol_a.pi[:T]))),
                "max_abs_r": float(np.max(np.abs(sol_b.r[:T] - sol_a.r[:T]))),
                "max_abs_debt_end": float(np.max(np.abs(sol_b.debt_end[:T] - sol_a.debt_end[:T]))),
                "abs_diff_nu_total": float(abs(sol_b.nu_total_from_params - sol_a.nu_total)),
                "abs_diff_nu_price": float(abs(sol_b.nu_price_from_params - sol_a.nu_price)),
            })
        return out

    def plot_tax_rule(
        self,
        delay=0,
        selected_tau_d=(0.0, 0.1, 0.3, 0.5, 1.0),
        tau_d_grid=None,
        figsize=(9.5, 6.0),
        savepath=None,
        horizon=None,
    ):
        """
        Plot the tax-rule path

            t_t = tau_y y_t + tau_d (d_t + e_t) - e_t

        for a fixed implementation delay and one or several values of tau_d.

        Note:
        This is the tax variable t_t in the model (deviation from steady state,
        scaled by steady-state output), not a literal statutory tax rate.
        """
        if horizon is None:
            horizon = self.par.solve_horizon

        if tau_d_grid is None:
            tau_d_grid = np.sort(
                np.concatenate((
                    np.linspace(0.0, 1.0, 301),
                    np.array([0.085, 0.026, 0.004])
                ))
            )

        self.solve_tau_sweep_fixed_delay(delay=delay, tau_d_grid=tau_d_grid, horizon=horizon)

        if len(self.sol_all.results) == 0:
            raise RuntimeError("No determinate tau_d values were found, so nothing can be plotted.")

        selected_results = []
        missing_tau = []

        for tau_d in selected_tau_d:
            found = False
            for sol in self.sol_all.results:
                if abs(sol.tau_d - tau_d) < 1e-9:
                    selected_results.append(sol)
                    found = True
                    break
            if not found:
                missing_tau.append(tau_d)

        if len(selected_results) == 0:
            raise RuntimeError("None of the selected tau_d values produced a unique bounded equilibrium.")

        if len(missing_tau) > 0 and self.verbose:
            print(f"Warning: no unique bounded equilibrium for tau_d = {missing_tau}")

        line_colors = self._get_line_colors(len(selected_results))

        with mpl.rc_context({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 13,
            "axes.titlesize": 17,
            "axes.labelsize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "mathtext.fontset": "stix",
        }):
            fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

            for color, res in zip(line_colors, selected_results):
                ax.plot(res.t, res.tax_no_direct_transfer, lw=2.8, color=color, label=rf"$\tau_d={res.tau_d:.3f}$")

            if delay > 0:
                ax.axvline(delay, color="black", linestyle="--", linewidth=1.0, alpha=0.8)

            ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
            ax.set_title(r"Tax rule $t_t$")
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(r"$t_t$")
            ax.set_xlim(0, selected_results[0].t[-1])
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", frameon=True)
            

            tmax = int(selected_results[0].t[-1])
            xticks = np.arange(0, tmax + 1, 10)
            ax.set_xticks(xticks)

            if savepath is not None:
                fig.savefig(savepath, dpi=200, bbox_inches="tight")
                if self.verbose:
                    print(f"saved to: {savepath}")

            plt.show()
            plt.close(fig)


model = AnnouncedTransferOLG()
par = model.par

par.beta  = 0.99**0.25
par.omega = 0.865
par.tau_y = 1.0 / 3.0
par.sigma = 1.0
par.kappa = 0.0062
par.Dbar  = 1.04
par.psi   = 0.5 + 1.0 / par.beta
par.phi   = 0.5 - par.kappa / par.beta
par.tau_d = None

# plotting horizon
par.T = 40

# truncation horizon used for backward induction
# choose this large enough so the terminal steady-state condition is a very
# accurate approximation to the infinite-horizon bounded solution
par.solve_horizon = 500

# max delay used in delay sweeps
par.max_delay = 80

# model.plot_tax_rule(delay=20)
model.plot_announced_irfs(delay=20)
model.plot_self_financing_vs_delay()
