from types import SimpleNamespace
from copy import deepcopy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.font_manager import FontProperties

class Taud_taylor_OLG:

    def __init__(self, verbose=True, eps0: float = 1.0):
        self.verbose = verbose

        self.par = SimpleNamespace()
        self.eqsys = SimpleNamespace()
        # self.sol = SimpleNamespace()
        self.IRFres = SimpleNamespace()

        self.set_parameters()
        self.eps0 = eps0
        self.allocate()


    def set_parameters(self):
        par = self.par

        par.beta  = 0.99**0.25
        par.omega = 0.86
        par.tau_y = 1.0 / 3.0
        par.sigma = 1.0
        par.kappa = 0.0062
        par.Dbar  = 1.04
        par.psi   = 1.1 # 1.0 / par.beta 
        par.phi   = 0.1 -par.kappa / par.beta
        par.tau_d = None
        par.T = 50


    def allocate(self):
        self.sol_all = SimpleNamespace()
        self.sol_all.outputvalues = []
        self.sol_all.chi_list = []
        self.sol_all.tau_d_list = []

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

        X_d, X_y, X_r = eqsys.X_d, eqsys.X_y, eqsys.X_r

        eqsys.m_qq  = (1 - tau_d) / beta
        eqsys.m_qy  = (phi + kappa / beta) * Dbar - tau_y / beta
        eqsys.m_qpi = (psi - 1 / beta) * Dbar

        eqsys.m_yq  = -X_d * (1 - omega * (1 - tau_d)) / (beta * omega)
        eqsys.m_yy  = (1 / (beta * omega)) * (
            1 - (1 - beta * omega) * (X_y - X_r * (phi + kappa / beta))
            + beta * omega * X_d * ((phi + kappa / beta) * Dbar - tau_y / beta)
        )
        eqsys.m_ypi = (psi - 1 / beta) * (1 / (beta * omega)) * (
            (1 - beta * omega) * X_r + beta * omega * X_d * Dbar
        )

        eqsys.m_piq  = 0.0
        eqsys.m_piy  = -kappa / beta
        eqsys.m_pipi = 1 / beta

    def system_matrix(self):
        e = self.eqsys
        e.A = np.array([
            [e.m_qq,  e.m_qy,  e.m_qpi],
            [e.m_yq,  e.m_yy,  e.m_ypi],
            [e.m_piq, e.m_piy, e.m_pipi]
        ], dtype=np.complex128)


    def system_matrix_given_policy(self, psi: float, phi: float, tau_d: float) -> np.ndarray:
        """
        Same matrix as your model, but evaluated at arbitrary (psi, phi, tau_d),
        so the determinacy plot is internally consistent with the solved model.
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

        a_q  = (1.0 - tau_d) / beta
        a_y  = -tau_y / beta + Dbar * r_y
        a_pi = Dbar * r_pi

        m21 = -X_d * (1.0 - omega * (1.0 - tau_d)) / (beta * omega)

        m22 = (
            1.0
            - (1.0 - beta * omega) * X_y
            + (1.0 - beta * omega) * X_r * r_y
            + beta * omega * X_d * a_y
        ) / (beta * omega)

        m23 = (
            ((1.0 - beta * omega) * X_r + beta * omega * X_d * Dbar) * r_pi
        ) / (beta * omega)

        return np.array([
            [a_q,  a_y,          a_pi],
            [m21,  m22,          m23],
            [0.0, -kappa / beta, 1.0 / beta],
        ], dtype=float)

    def solve_unique_bounded_eq(self, tol=1e-9):
        A = self.eqsys.A
        sol = self.sol
        par = self.par
        eps0 = self.eps0

        eigvals, eigvecs = np.linalg.eig(A)
        sol.all_eigenvalues = eigvals

        stable_idx   = [i for i, val in enumerate(eigvals) if abs(val) < 1.0 - tol]
        unstable_idx = [i for i, val in enumerate(eigvals) if abs(val) > 1.0 + tol]

        if len(stable_idx) != 1 or len(unstable_idx) != 2:
            raise RuntimeError(f"No unique bounded equilibrium. Eigenvalues: {eigvals}")

        idx = stable_idx[0]
        lambda_s = eigvals[idx]
        if abs(lambda_s.imag) > 1e-8:
            raise RuntimeError(f"Stable root is not numerically real: {lambda_s}")

        v_s = eigvecs[:, idx]
        if abs(v_s[0]) < 1e-12:
            raise RuntimeError("Stable eigenvector first element too close to zero.")

        v_s = v_s / v_s[0]
        if np.max(np.abs(v_s.imag)) > 1e-8:
            raise RuntimeError(f"Stable eigenvector is not numerically real: {v_s}")

        v_s = np.real(v_s)
        lambda_s = float(np.real(lambda_s))

        sol.lambda_s = lambda_s
        sol.v_s = v_s
        sol.chi = float(v_s[1])
        sol.eta = float(v_s[2])
        sol.q0 = float(eps0 / (1.0 + par.Dbar * sol.eta))
        sol.residual = float(np.max(np.abs(A @ v_s - lambda_s * v_s)))

    def compute_irf(self):
        par = self.par
        sol = self.sol
        eps0 = self.eps0
        T = par.T

        q0, lambda_s, v_s = sol.q0, sol.lambda_s, sol.v_s

        q_full = np.full(T + 2, np.nan)
        for t in range(T + 2):
            q_full[t] = q0 * (lambda_s ** t)

        sol.q = q_full[:-1]
        sol.debt_end = q_full[1:T+2]
        sol.t = np.arange(T + 1)

        sol.y = v_s[1] * sol.q
        sol.pi = v_s[2] * sol.q
        sol.i_nom = par.psi * sol.pi + par.phi * sol.y

        r_loading = (par.psi - 1.0 / par.beta) * v_s[2] + (par.phi + par.kappa / par.beta) * v_s[1]
        sol.r = r_loading * sol.q


        denom = par.tau_d + (par.tau_y+par.kappa*par.Dbar)*v_s[1]
        sol.nu_base_from_params = (
            (par.tau_y * v_s[1] )  / denom
        )

        sol.nu_price_from_params = (
            (par.Dbar * par.kappa * v_s[1]) /
            denom
        )
        sol.nu_total_from_params = (
            (par.tau_y + par.Dbar * par.kappa)*v_s[1] /
            denom
        )

    def exists_unique_bounded_equilibrium(self, tol=1e-9):
        vals = np.linalg.eigvals(self.eqsys.A)
        n_stable = sum(abs(v) < 1.0 - tol for v in vals)
        n_unstable = sum(abs(v) > 1.0 + tol for v in vals)
        return n_stable == 1 and n_unstable == 2

    def compute_tau_sweep(self):
        par = self.par
        self.sol_all.outputvalues = []

        tau_d_grid = np.sort(
                        np.concatenate((
                            np.linspace(0.0, 1.0, 301),
                            np.array([0.085, 0.026, 0.004]) # the specific estimated fiscal adjustment values 
                        ))
                    )
        for tau_d in tau_d_grid:
            par.tau_d = tau_d
            self.sol = SimpleNamespace()

            self.eqsys_matrix_elements()
            self.system_matrix()

            if self.exists_unique_bounded_equilibrium():
                self.solve_unique_bounded_eq()
                self.compute_irf()
                self.sol.tau_d = tau_d
                self.sol_all.outputvalues.append(deepcopy(self.sol))
                self.sol_all.chi_list.append(deepcopy(self.sol.chi))
                self.sol_all.tau_d_list.append(deepcopy(self.sol.tau_d))



    def style_colors(self):
        return {
            # "line_list": np.array(
            #     [
            #         [0.78, 0.88, 0.96],  # very light blue
            #         [0.55, 0.75, 0.90],  # light blue
            #         [0.30, 0.55, 0.80],  # medium blue
            #         [0.15, 0.30, 0.55],  # deep blue
            #         [0.05, 0.10, 0.20],  # very dark blue
            #     ]
            # ),
            # "finance_price": np.array([0.20, 0.45, 0.85]),  # strong blue
            # "finance_tax": np.array([0.40, 0.60, 0.80]),    # muted blue-gray

            "line_list": np.array(
                [
                    [0.99, 0.85, 0.90],
                    [0.97, 0.67, 0.78],
                    [0.93, 0.44, 0.64],
                    [0.80, 0.20, 0.52],
                    [0.70, 0.10, 0.40],
                    [0.55, 0.05, 0.30],
                    [0.45, 0.01, 0.20]
                ]
            ),
            "finance_price": np.array([0.80, 0.20, 0.52]),
            "finance_tax": np.array([0.97, 0.67, 0.78]),
        }


    def classify_point_real_phi(self, phi_real: float, tau_d: float, psi: float = None, tol: float = 1e-9) -> int:
        MULTIPLE, NONE, UNIQUE, BOUNDARY, OUTSIDE = 0, 1, 2, 3, 4

        if tau_d < 0.0 or tau_d > 1.0:
            return OUTSIDE

        par = self.par

        if psi is None:
            psi = par.psi   # <-- use the model's current psi by default

        # keep your current "real-rate-rule" x-axis convention
        phi_nom = phi_real - par.kappa / par.beta

        try:
            eigvals = np.linalg.eigvals(
                self.system_matrix_given_policy(psi=float(psi), phi=float(phi_nom), tau_d=float(tau_d))
            )
        except ValueError:
            return BOUNDARY

        mod = np.abs(eigvals)

        if np.any(np.isclose(mod, 1.0, atol=tol)):
            return BOUNDARY

        n_stable = int(np.sum(mod < 1.0 - tol))
        n_unstable = int(np.sum(mod > 1.0 + tol))

        if n_stable == 1 and n_unstable == 2:
            return UNIQUE
        if n_stable > 1:
            return MULTIPLE
        if n_stable == 0:
            return NONE
        return BOUNDARY


    def region_grid_real_phi(self, phi_real_grid, tau_grid, psi: float = None):
        Z = np.empty((len(tau_grid), len(phi_real_grid)), dtype=int)
        for i, tau_d in enumerate(tau_grid):
            for j, phi_real in enumerate(phi_real_grid):
                Z[i, j] = self.classify_point_real_phi(phi_real, tau_d, psi=psi)
        return Z


    def draw_determinacy_panel(
        self,
        ax,
        selected_tau_d,
        selected_results,
        line_colors,
        phi_real_min=None,
        phi_real_max=None,
        tau_min=0.0,
        tau_max=1.0,
        n_phi=201,
        n_tau=201,
    ):
        MULTIPLE, NONE, UNIQUE, BOUNDARY, OUTSIDE = 0, 1, 2, 3, 4
        par = self.par

        # current calibrated point in your existing x-axis convention
        phi_real_selected = par.phi + par.kappa / par.beta
        psi_selected = par.psi

        # center the x-range around the selected calibration unless you override it
        if phi_real_min is None:
            phi_real_min = phi_real_selected - 0.25
        if phi_real_max is None:
            phi_real_max = phi_real_selected + 0.25

        phi_real_grid = np.linspace(phi_real_min, phi_real_max, n_phi)
        tau_grid = np.linspace(tau_min, tau_max, n_tau)

        # <-- crucial fix: classify with the actual psi used elsewhere
        Z = self.region_grid_real_phi(phi_real_grid, tau_grid, psi=psi_selected)

        colors = [
            "#1D4ED8",  # multiple bounded equilibria
            "#9CA3AF",  # no bounded equilibrium
            "#93C5FD",  # unique bounded equilibrium
            "#6B7280",  # boundary
            "white",    # outside
        ]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

        extent = [phi_real_grid[0], phi_real_grid[-1], tau_grid[0], tau_grid[-1]]
        ax.imshow(
            Z,
            origin="lower",
            extent=extent,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
        )

        ax.axvline(phi_real_selected, color="black", linestyle="--", linewidth=1.0, alpha=0.8)

        for tau in selected_tau_d:
            matched_color = None
            for color, res in zip(line_colors, selected_results):
                if abs(res.tau_d - tau) < 1e-9:
                    matched_color = color
                    break

            if matched_color is not None:
                ax.scatter(
                    phi_real_selected, tau,
                    s=85,
                    color=matched_color,
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=6
                )
            else:
                ax.scatter(
                    phi_real_selected, tau,
                    s=85,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=1.0,
                    zorder=6
                )
                ax.scatter(
                    phi_real_selected, tau,
                    s=60,
                    color="black",
                    marker="x",
                    linewidths=1.4,
                    zorder=7
                )

        ax.set_title(rf"Determinacy regions")
        ax.set_xlabel(r"$\alpha_y$")   # more honest label for your current setup
        ax.set_ylabel(r"$\tau_d$")
        ax.set_xlim(phi_real_min, phi_real_max)
        ax.set_ylim(tau_min, tau_max)
        ax.grid(False)

        legend_handles = [
            Patch(facecolor=colors[UNIQUE], edgecolor="black", label="Unique"),
            Patch(facecolor=colors[MULTIPLE], edgecolor="black", label="Multiple"),
            Patch(facecolor=colors[NONE], edgecolor="black", label="None"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                markeredgecolor="black", markersize=8, label=r"Chosen $\tau_d$"),
        ]
        ax.legend(handles=legend_handles, loc="upper left", frameon=True, fontsize=12)



    # helpers for plotting
    def _get_line_colors(self, n):
        base = self.style_colors()["line_list"]
        if n <= len(base):
            return base[:n]
        return plt.cm.Blues(np.linspace(0.35, 0.90, n))

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

    def plot_eps0_irfs(
        self,
        selected_tau_d=[0.0, 0.1, 0.3, 0.5, 1.0],
        tau_d_grid=None,
        figsize=(10, 8),
        savepath=None,
        ylim0=True
    ):

        sol_all = self.sol_all
        par = self.par
        colors = self.style_colors()

        if len(sol_all.outputvalues) == 0:
            raise RuntimeError("No determinate tau_d values were found, so nothing can be plotted.")

        if tau_d_grid is None:
            tau_d_grid = np.sort(
                np.concatenate((
                    np.linspace(0.0, 1.0, 301),
                    np.array([0.085, 0.026, 0.004])
                ))
            )

        selected_results = []
        missing_tau = []

        for tau_d in selected_tau_d:
            found = False
            for sol in sol_all.outputvalues:
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

        results_sorted = sorted(sol_all.outputvalues, key=lambda x: x.tau_d)
        tau_sorted = np.array([res.tau_d for res in results_sorted])
        nu_price = np.array([res.nu_price_from_params for res in results_sorted])
        nu_total = np.array([res.nu_total_from_params for res in results_sorted])

        line_colors = self._get_line_colors(len(selected_results))

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
            fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=False, constrained_layout=False)
            fig.subplots_adjust(
                left=0.1,
                right=0.85,
                top=0.92,
                bottom=0.12,
                wspace=0.36,
                hspace=0.34,
            )

            # ---------- Panel 1: Output ----------
            ax = axes[0, 0]
            for color, res in zip(line_colors, selected_results):
                ax.plot(res.t, res.y, lw=2.8, color=color)

            ax.set_title(r"Output $y_t$")
            ax.set_xlabel(r"t")
            ax.set_ylabel(r"%")
            ax.set_xlim(0, selected_results[0].t[-1])
            ax.grid(True, alpha=0.25)

            # ---------- Panel 2: End-of-period debt ----------
            ax = axes[0, 1]
            for color, res in zip(line_colors, selected_results):
                ax.plot(res.t, res.debt_end, lw=2.8, color=color)

            ax.set_title(r"Public debt $d_{t+1}$")
            ax.set_xlabel(r"t")
            ax.set_ylabel(r"")
            ax.set_xlim(0, selected_results[0].t[-1])
            ax.grid(True, alpha=0.25)

            # ---------- Panel 3: Inflation / nominal / real rate ----------
            ax = axes[1, 0]
            ax_r = ax.twinx()

            for color, res in zip(line_colors, selected_results):
                ax.plot(res.t, res.pi,    lw=2.5, color=color, ls="--")
                ax.plot(res.t, res.i_nom, lw=2.5, color=color, ls=":")
                ax_r.plot(res.t, res.r,   lw=2.5, color=color, ls="-")

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

            # ---------- Panel 4: Self-financing ----------
            ax = axes[1, 1]

            self._shade_invalid_tau_regions(
                ax=ax,
                tau_d_grid=tau_d_grid,
                valid_tau=tau_sorted,
                label=r"No unique eq.",
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
                    res.nu_total_from_params,
                    s=55,
                    color=color,
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=5,
                )

            ax.set_title(r"Self-financing share $\nu$")
            ax.set_xlabel(r"$\tau_d$")
            ax.set_ylabel(r"")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, None)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right", frameon=True)

            # ---------- t ticks only on IRF panels ----------
            tmax = int(selected_results[0].t[-1])
            xticks = np.arange(0, tmax + 1, 10)

            for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
                ax.set_xticks(xticks)
                ax.tick_params(axis="x", labelbottom=True)

            if ylim0:
                for ax in [axes[0, 0], axes[0, 1]]:
                    ax.set_ylim(0, None)

            # ---------- Figure-level tau_d legend on right ----------
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
                print(f"saved to: {savepath}")

            plt.show()
            fig.savefig(self.figname, dpi=600, bbox_inches="tight")
            plt.close(fig)

    def plot_self_financing_grid(self, psi_list, phi_list, tau_d_grid=None, figsize=(14, 10), savepath=None):
        """
        Plot only the self-financing graph on a 3x3 grid.

        Rows = psi_list (length 3)
        Columns = phi_list (length 3)

        Each panel shows:
        - blue area: Date-0 Inflation
        - darker area: Tax Base
        - grey region: tau_d values with no unique bounded equilibrium / no nu

        Example:
            model.plot_self_financing_grid(
                psi_list=[1.0 / model.par.beta, 1.2, 1.5],
                phi_list=[-model.par.kappa / model.par.beta, 0.0, 0.1]
            )
        """
        if len(psi_list) != 3:
            raise ValueError("psi_list must have length 3.")
        if len(phi_list) != 3:
            raise ValueError("phi_list must have length 3.")

        colors = self.style_colors()
        par = self.par

        if tau_d_grid is None:
            tau_d_grid = np.sort(
                np.concatenate((
                    np.linspace(0.0, 1.0, 301),
                    np.array([0.085, 0.026, 0.004])
                ))
            )
        tau_d_grid = np.asarray(tau_d_grid, dtype=float)

        old_psi = par.psi
        old_phi = par.phi
        old_tau_d = par.tau_d
        old_sol = getattr(self, "sol", None)

        plt.rcParams.update(
            {
                "font.size": 13,
                "axes.titlesize": 16,
                "axes.labelsize": 14,
                "legend.fontsize": 12,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
            }
        )

        fig, axes = plt.subplots(3, 3, figsize=figsize, sharex=True, sharey=True, constrained_layout=True)

        try:
            for i, psi in enumerate(psi_list):
                for j, phi in enumerate(phi_list):
                    ax = axes[i, j]

                    par.psi = float(psi)
                    par.phi = float(phi)

                    results = []
                    valid_mask = np.zeros(len(tau_d_grid), dtype=bool)

                    for k, tau_d in enumerate(tau_d_grid):
                        par.tau_d = float(tau_d)
                        self.sol = SimpleNamespace()

                        try:
                            self.eqsys_matrix_elements()
                            self.system_matrix()

                            if self.exists_unique_bounded_equilibrium():
                                self.solve_unique_bounded_eq()
                                self.compute_irf()
                                self.sol.tau_d = float(tau_d)
                                results.append(deepcopy(self.sol))
                                valid_mask[k] = True
                        except Exception:
                            pass

                    invalid_mask = ~valid_mask
                    if np.any(invalid_mask):
                        starts = np.where(invalid_mask & ~np.r_[False, invalid_mask[:-1]])[0]
                        ends   = np.where(invalid_mask & ~np.r_[invalid_mask[1:], False])[0]

                        for s, e in zip(starts, ends):
                            x0 = 0.0 if s == 0 else 0.5 * (tau_d_grid[s - 1] + tau_d_grid[s])
                            x1 = 1.0 if e == len(tau_d_grid) - 1 else 0.5 * (tau_d_grid[e] + tau_d_grid[e + 1])

                            ax.add_patch(
                                plt.Rectangle(
                                    (x0, 0.0),
                                    x1 - x0,
                                    1.0,
                                    facecolor="0.85",
                                    alpha=0.9,
                                    edgecolor=None,
                                    zorder=0
                                )
                            )

                    if len(results) > 0:
                        results = sorted(results, key=lambda x: x.tau_d)
                        tau_sorted = np.array([res.tau_d for res in results])
                        nu_price = np.array([res.nu_price_from_params for res in results])
                        nu_total = np.array([res.nu_total_from_params for res in results])

                        ax.fill_between(
                            tau_sorted, 0.0, nu_price,
                            color=colors["finance_price"],
                            alpha=0.95,
                            zorder=2
                        )
                        ax.fill_between(
                            tau_sorted, nu_price, nu_total,
                            color=colors["finance_tax"],
                            alpha=0.95,
                            zorder=3
                        )
                        ax.plot(
                            tau_sorted, nu_total,
                            color="black",
                            lw=1.3,
                            alpha=0.7,
                            zorder=4
                        )

                    ax.set_title(rf"$\psi={psi:.3f},\ \phi={phi:.3f}$")
                    ax.set_xlim(0.0, 1.0)
                    ax.set_ylim(0.0, 1.0)
                    ax.grid(True, alpha=0.25)

                    if i == 2:
                        ax.set_xlabel(r"$\tau_d$")
                    if j == 0:
                        ax.set_ylabel(r"Self-financing share $\nu$")

            legend_handles = [
                Patch(facecolor=colors["finance_price"], edgecolor="none", label="Date-0 Inflation"),
                Patch(facecolor=colors["finance_tax"], edgecolor="none", label="Tax Base"),
                Patch(facecolor="0.85", edgecolor="none", label=r"No unique bounded equilibrium / no $\nu$"),
            ]

            fig.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=3,
                frameon=True
            )

            if savepath is not None:
                fig.savefig(savepath, dpi=200, bbox_inches="tight")
                print(f"saved to: {savepath}")

            plt.show()
            plt.close(fig)

        finally:
            par.psi = old_psi
            par.phi = old_phi
            par.tau_d = old_tau_d
            self.sol = old_sol

    def plot_chis(self):
        import numpy as np
        import matplotlib.pyplot as plt

        chi_list = np.asarray(self.sol_all.chi_list, dtype=float)
        tau_d_list = np.asarray(self.sol_all.tau_d_list, dtype=float)

        # Sort in case tau_d_list is not already increasing
        order = np.argsort(tau_d_list)
        tau_d_list = tau_d_list[order]
        chi_list = chi_list[order]

        # Numerical derivative d chi / d tau_d
        chi_num_derivative = np.gradient(chi_list, tau_d_list)

        # Quantity whose sign determines monotonicity of nu
        cond_val = tau_d_list * chi_num_derivative - chi_list
        condition_holds = cond_val < 0

        fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

        # Top panel: chi and chi'
        axes[0].plot(tau_d_list, chi_list, label=r'$\chi(\tau_d)$', linewidth=2)
        # axes[0].plot(tau_d_list, chi_num_derivative, label=r"$\chi'(\tau_d)$", linewidth=2)
        axes[0].set_ylabel('Value')
        axes[0].set_title(r'$\chi(\tau_d)$ and numerical derivative')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Bottom panel: tau_d * chi' - chi
        axes[1].axhline(0.0, color='black', linewidth=1)
        axes[1].plot(
            tau_d_list,
            cond_val,
            label=r'$\tau_d \chi^\prime(\tau_d)-\chi(\tau_d)$',
            linewidth=2
        )

        # Shade where the condition holds
        axes[1].fill_between(
            tau_d_list,
            cond_val,
            0,
            where=condition_holds,
            interpolate=True,
            alpha=0.3,
            label=r'Condition holds: $\tau_d \chi^\prime(\tau_d) < \chi(\tau_d)$'
        )

        axes[1].set_xlabel(r'$\tau_d$')
        axes[1].set_ylabel('Condition value')
        axes[1].set_title(r'Condition for $d\nu/d\tau_d < 0$')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


    def plot_determinacy_phi_psi(
        self,
        tau_d=0.0,
        phi_min=None,
        phi_max=None,
        psi_min=None,
        psi_max=None,
        n_phi=301,
        n_psi=301,
        show_reference_lines=True,
        show_current_point=True,
        figsize=(9, 7),
        savepath=None,
        tol=1e-9,
    ):
        """
        Plot determinacy regions in the (phi, psi) plane for a fixed tau_d.

        Colors:
        - light blue: unique bounded equilibrium
        - dark blue: multiple bounded equilibria
        - grey: no bounded equilibrium
        - dark grey: boundary (root near unit circle)

        Example:
            model.plot_determinacy_phi_psi(
                tau_d=0.0,
                phi_min=-0.5,
                phi_max=0.5,
                psi_min=0.8,
                psi_max=2.5
            )
        """
        par = self.par

        if phi_min is None:
            phi_min = -0.5
        if phi_max is None:
            phi_max = 0.5
        if psi_min is None:
            psi_min = 0.5
        if psi_max is None:
            psi_max = 2.5

        phi_grid = np.linspace(phi_min, phi_max, n_phi)
        psi_grid = np.linspace(psi_min, psi_max, n_psi)

        MULTIPLE, NONE, UNIQUE, BOUNDARY = 0, 1, 2, 3

        Z = np.empty((len(psi_grid), len(phi_grid)), dtype=int)

        for i, psi in enumerate(psi_grid):
            for j, phi in enumerate(phi_grid):
                try:
                    A = self.system_matrix_given_policy(
                        psi=float(psi),
                        phi=float(phi),
                        tau_d=float(tau_d)
                    )
                    eigvals = np.linalg.eigvals(A)
                    mod = np.abs(eigvals)

                    if np.any(np.isclose(mod, 1.0, atol=tol)):
                        Z[i, j] = BOUNDARY
                        continue

                    n_stable = int(np.sum(mod < 1.0 - tol))
                    n_unstable = int(np.sum(mod > 1.0 + tol))

                    if n_stable == 1 and n_unstable == 2:
                        Z[i, j] = UNIQUE
                    elif n_stable > 1:
                        Z[i, j] = MULTIPLE
                    elif n_stable == 0:
                        Z[i, j] = NONE
                    else:
                        Z[i, j] = BOUNDARY

                except Exception:
                    Z[i, j] = BOUNDARY

        colors = [
            "#1D4ED8",  # multiple
            "#9CA3AF",  # none
            "#93C5FD",  # unique
            "#6B7280",  # boundary
        ]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

        fig, ax = plt.subplots(figsize=figsize)

        ax.imshow(
            Z,
            origin="lower",
            extent=[phi_grid[0], phi_grid[-1], psi_grid[0], psi_grid[-1]],
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
        )

        if show_reference_lines:
            ax.axvline(-par.kappa / par.beta, color="black", linestyle="--", linewidth=1.5, alpha=0.9)
            ax.axhline(1.0 / par.beta, color="black", linestyle="--", linewidth=1.5, alpha=0.9)

        if show_current_point:
            ax.scatter(
                par.phi,
                par.psi,
                s=90,
                color="white",
                edgecolor="black",
                linewidth=1.0,
                zorder=5
            )

        legend_handles = [
            Patch(facecolor=colors[UNIQUE], edgecolor="black", label="Unique"),
            Patch(facecolor=colors[MULTIPLE], edgecolor="black", label="Multiple"),
            Patch(facecolor=colors[NONE], edgecolor="black", label="None"),
            Patch(facecolor=colors[BOUNDARY], edgecolor="black", label="Boundary"),
        ]

        if show_current_point:
            legend_handles.append(
                Line2D([0], [0], marker="o", color="w",
                    markerfacecolor="white", markeredgecolor="black",
                    markersize=8, label="Current calibration")
            )

        ax.legend(handles=legend_handles, loc="upper right", frameon=True)

        ax.set_title(rf"Determinacy regions for $\tau_d = {tau_d:.2f}$")
        ax.set_xlabel(r"$\phi$")
        ax.set_ylabel(r"$\psi$")
        ax.set_xlim(phi_min, phi_max)
        ax.set_ylim(psi_min, psi_max)
        ax.grid(False)

        if savepath is not None:
            fig.savefig(savepath, dpi=200, bbox_inches="tight")
            print(f"saved to: {savepath}")

        plt.show()
        plt.close(fig)

    def map_condition_failure_from_matrix(
        self,
        phi_grid,
        psi_grid,
        tau_d_grid,
        tol=1e-9,
        imag_tol=1e-8,
        atol=1e-10,
        make_plot=True,
    ):
        """
        For each (phi, psi) pair, use self.system_matrix_given_policy(psi, phi, tau_d)
        to compute chi(tau_d) from the stable eigenvector, then check whether there
        exists at least one tau_d such that

            tau_d * chi'(tau_d) - chi(tau_d) >= 0

        i.e. the monotonicity condition fails.

        Parameters
        ----------
        phi_grid, psi_grid, tau_d_grid : array-like
            Grids for phi, psi, and tau_d.
        tol : float
            Tolerance for classifying stable/unstable roots.
        imag_tol : float
            Tolerance for accepting numerically real eigenvalues/eigenvectors.
        atol : float
            Numerical tolerance when testing the inequality.
        make_plot : bool
            If True, show a heatmap.

        Returns
        -------
        results : dict
            Contains boolean map, diagnostics, and per-gridpoint details.
        """

        phi_grid = np.asarray(phi_grid, dtype=float)
        psi_grid = np.asarray(psi_grid, dtype=float)
        tau_d_grid = np.asarray(tau_d_grid, dtype=float)

        nphi = len(phi_grid)
        npsi = len(psi_grid)

        failure_exists = np.zeros((nphi, npsi), dtype=bool)
        has_enough_points = np.zeros((nphi, npsi), dtype=bool)
        first_bad_tau = np.full((nphi, npsi), np.nan)
        max_condition_value = np.full((nphi, npsi), np.nan)

        details = {}

        for i, phi in enumerate(phi_grid):
            for j, psi in enumerate(psi_grid):

                tau_valid = []
                chi_valid = []

                for tau_d in tau_d_grid:
                    try:
                        A = self.system_matrix_given_policy(psi=float(psi), phi=float(phi), tau_d=float(tau_d))
                        eigvals, eigvecs = np.linalg.eig(A)
                    except Exception:
                        continue

                    stable_idx = [k for k, val in enumerate(eigvals) if abs(val) < 1.0 - tol]
                    unstable_idx = [k for k, val in enumerate(eigvals) if abs(val) > 1.0 + tol]

                    # Need exactly one stable and two unstable roots
                    if not (len(stable_idx) == 1 and len(unstable_idx) == 2):
                        continue

                    idx = stable_idx[0]
                    lam = eigvals[idx]
                    v = eigvecs[:, idx]

                    # Require numerically real stable root/eigenvector
                    if abs(lam.imag) > imag_tol:
                        continue
                    if abs(v[0]) < 1e-12:
                        continue

                    v = v / v[0]
                    if np.max(np.abs(v.imag)) > imag_tol:
                        continue

                    v = np.real(v)
                    chi = float(v[1])

                    if not np.isfinite(chi):
                        continue

                    tau_valid.append(float(tau_d))
                    chi_valid.append(chi)

                tau_valid = np.asarray(tau_valid, dtype=float)
                chi_valid = np.asarray(chi_valid, dtype=float)

                key = (float(phi), float(psi))

                if tau_valid.size < 2:
                    details[key] = {
                        "tau_d": tau_valid,
                        "chi": chi_valid,
                        "chi_prime": np.array([]),
                        "condition_value": np.array([]),
                        "bad_mask": np.array([], dtype=bool),
                    }
                    continue

                order = np.argsort(tau_valid)
                tau_valid = tau_valid[order]
                chi_valid = chi_valid[order]

                # Numerical derivative d chi / d tau_d
                edge_order = 2 if tau_valid.size >= 3 else 1
                chi_prime = np.gradient(chi_valid, tau_valid, edge_order=edge_order)

                # Condition fails when this is >= 0
                condition_value = tau_valid * chi_prime - chi_valid
                bad_mask = condition_value >= -atol

                has_enough_points[i, j] = True
                failure_exists[i, j] = np.any(bad_mask)
                max_condition_value[i, j] = np.max(condition_value)

                if np.any(bad_mask):
                    first_bad_tau[i, j] = tau_valid[np.argmax(bad_mask)]

                details[key] = {
                    "tau_d": tau_valid,
                    "chi": chi_valid,
                    "chi_prime": chi_prime,
                    "condition_value": condition_value,
                    "bad_mask": bad_mask,
                }

        results = {
            "failure_exists": failure_exists,
            "has_enough_points": has_enough_points,
            "first_bad_tau": first_bad_tau,
            "max_condition_value": max_condition_value,
            "phi_grid": phi_grid,
            "psi_grid": psi_grid,
            "details": details,
        }


        if make_plot:
            # NaN = not enough valid tau_d points
            # 0   = condition holds for all checked tau_d
            # 1   = at least one tau_d violates the condition
            plot_array = np.full((nphi, npsi), np.nan)
            plot_array[has_enough_points] = 0.0
            plot_array[failure_exists] = 1.0

            cmap = ListedColormap(["#93C5FD", "#6B7280"])
            cmap.set_bad(color="white")

            fig, ax = plt.subplots(figsize=(9, 7))
            masked = np.ma.masked_invalid(plot_array)

            ax.imshow(
                masked,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                extent=[psi_grid[0], psi_grid[-1], phi_grid[0], phi_grid[-1]],
            )

            # reference lines at psi = 1/beta and phi = -kappa/beta
            psi_ref = 1.0 / self.par.beta
            phi_ref = -self.par.kappa / self.par.beta

            ax.axvline(psi_ref, color="black", linestyle="--", linewidth=1.8, alpha=0.9)
            ax.axhline(phi_ref, color="black", linestyle="--", linewidth=1.8, alpha=0.9)

            legend_handles = [
                Patch(facecolor="#93C5FD", edgecolor="black",
                    label=r"$\tau_d \chi'(\tau_d) < \chi(\tau_d) \forall \tau_d$"),
                Patch(facecolor="#6B7280", edgecolor="black",
                    label=r"$\exists \tau_d : \tau_d \chi'(\tau_d) > \chi(\tau_d)$"),
                Patch(facecolor="white", edgecolor="black",
                    label="Indeterminate for all $\tau_d$"),
                Line2D([0], [0], color="black", linestyle="--", linewidth=1.8,
                    label=r"$\psi = 1/\beta$ and $\phi = -\kappa/\beta$")
            ]

            ax.legend(handles=legend_handles, loc="upper right", frameon=True)

            ax.set_xlabel(r"$\psi$")
            ax.set_ylabel(r"$\phi$")
            ax.set_title(
                r"Existence of $\tau_d$ such that "
                r"$\tau_d \chi'(\tau_d)-\chi(\tau_d)\geq 0$"
            )
            plt.tight_layout()
            plt.show()

    def decompose_persistence_at_tau(
        self,
        tau_d=0.10,
        alpha_y_val=0.5,
        alpha_pi_val=0.5,
        include_neutral=True,
        tol=1e-9,
    ):
        """
        Decompose the stable root lambda_s at a chosen tau_d.

        The decomposition is:
            lambda_s =
                (1 - tau_d)/beta
                - (tau_y/beta) * chi
                + Dbar * (alpha_y * chi + alpha_pi * eta)

        where along the stable path:
            y_t  = chi q_t
            pi_t = eta q_t
            r_t  = (alpha_y * chi + alpha_pi * eta) q_t

        The function prints and returns results for:
            - r_t = 0                      (optional)
            - r_t = alpha_y_val * y_t
            - r_t = alpha_pi_val * pi_t
            - r_t = alpha_y_val*y_t + alpha_pi_val*pi_t
        """

        par = self.par

        beta = par.beta
        tau_y = par.tau_y
        Dbar = par.Dbar
        kappa = par.kappa

        cases = []

        if include_neutral:
            cases.append({
                "name": r"$r_t=0$",
                "alpha_y": 0.0,
                "alpha_pi": 0.0,
            })

        cases += [
            {
                "name": rf"$r_t={alpha_y_val:.2f}y_t$",
                "alpha_y": alpha_y_val,
                "alpha_pi": 0.0,
            },
            {
                "name": rf"$r_t={alpha_pi_val:.2f}\pi_t$",
                "alpha_y": 0.0,
                "alpha_pi": alpha_pi_val,
            },
            {
                "name": rf"$r_t={alpha_y_val:.2f}y_t+{alpha_pi_val:.2f}\pi_t$",
                "alpha_y": alpha_y_val,
                "alpha_pi": alpha_pi_val,
            },
        ]

        rows = []

        for case in cases:
            alpha_y = case["alpha_y"]
            alpha_pi = case["alpha_pi"]

            # Convert desired real-rate feedbacks into your nominal-rule parameters:
            # alpha_y  = phi + kappa/beta
            # alpha_pi = psi - 1/beta
            phi_case = alpha_y - kappa / beta
            psi_case = alpha_pi + 1.0 / beta

            A = self.system_matrix_given_policy(
                psi=psi_case,
                phi=phi_case,
                tau_d=tau_d,
            )

            eigvals, eigvecs = np.linalg.eig(A)

            stable_idx = [i for i, val in enumerate(eigvals) if abs(val) < 1.0 - tol]
            unstable_idx = [i for i, val in enumerate(eigvals) if abs(val) > 1.0 + tol]

            if not (len(stable_idx) == 1 and len(unstable_idx) == 2):
                rows.append({
                    "case": case["name"],
                    "status": "No unique bounded equilibrium",
                    "lambda_s": np.nan,
                    "chi": np.nan,
                    "eta": np.nan,
                    "rollover": np.nan,
                    "tax_base": np.nan,
                    "real_rate": np.nan,
                    "reconstructed": np.nan,
                    "residual": np.nan,
                })
                continue

            idx = stable_idx[0]
            lambda_s = eigvals[idx]
            v_s = eigvecs[:, idx]

            if abs(lambda_s.imag) > 1e-8 or abs(v_s[0]) < 1e-12:
                rows.append({
                    "case": case["name"],
                    "status": "Complex / invalid stable root",
                    "lambda_s": np.nan,
                    "chi": np.nan,
                    "eta": np.nan,
                    "rollover": np.nan,
                    "tax_base": np.nan,
                    "real_rate": np.nan,
                    "reconstructed": np.nan,
                    "residual": np.nan,
                })
                continue

            v_s = v_s / v_s[0]

            if np.max(np.abs(v_s.imag)) > 1e-8:
                rows.append({
                    "case": case["name"],
                    "status": "Complex stable eigenvector",
                    "lambda_s": np.nan,
                    "chi": np.nan,
                    "eta": np.nan,
                    "rollover": np.nan,
                    "tax_base": np.nan,
                    "real_rate": np.nan,
                    "reconstructed": np.nan,
                    "residual": np.nan,
                })
                continue

            lambda_s = float(np.real(lambda_s))
            v_s = np.real(v_s)

            chi = float(v_s[1])
            eta = float(v_s[2])

            rollover_term = (1.0 - tau_d) / beta
            tax_base_term = -(tau_y / beta) * chi
            real_rate_loading = alpha_y * chi + alpha_pi * eta
            real_rate_term = Dbar * real_rate_loading

            reconstructed_lambda = rollover_term + tax_base_term + real_rate_term
            residual = lambda_s - reconstructed_lambda

            eta_from_nkpc = kappa * chi / (1.0 - beta * lambda_s)

            rows.append({
                "case": case["name"],
                "status": "OK",
                "lambda_s": lambda_s,
                "chi": chi,
                "eta": eta,
                "eta_from_nkpc": eta_from_nkpc,
                "alpha_y": alpha_y,
                "alpha_pi": alpha_pi,
                "r_over_q": real_rate_loading,
                "rollover": rollover_term,
                "tax_base": tax_base_term,
                "real_rate": real_rate_term,
                "reconstructed": reconstructed_lambda,
                "residual": residual,
            })

        # Pretty print
        print("\n" + "=" * 100)
        print(rf"Stable-root decomposition at tau_d = {tau_d:.4f}")
        print("=" * 100)

        header = (
            f"{'case':<30}"
            f"{'lambda':>10}"
            f"{'chi':>10}"
            f"{'eta':>10}"
            f"{'rollover':>12}"
            f"{'tax base':>12}"
            f"{'real rate':>12}"
            f"{'sum':>10}"
            f"{'resid':>10}"
        )
        print(header)
        print("-" * 100)

        for row in rows:
            if row["status"] != "OK":
                print(f"{row['case']:<30} {row['status']}")
                continue

            print(
                f"{row['case']:<30}"
                f"{row['lambda_s']:>10.4f}"
                f"{row['chi']:>10.4f}"
                f"{row['eta']:>10.4f}"
                f"{row['rollover']:>12.4f}"
                f"{row['tax_base']:>12.4f}"
                f"{row['real_rate']:>12.4f}"
                f"{row['reconstructed']:>10.4f}"
                f"{row['residual']:>10.2e}"
            )

        print("=" * 100)

        return rows


    def run(self):
        self.compute_tau_sweep()
        self.plot_eps0_irfs(selected_tau_d=[0.09,0.1,0.12,0.2,0.3])
        #self.plot_eps0_irfs(selected_tau_d=[0.1,0.11,0.13,0.17,0.2])
        # self.plot_chis()


model = Taud_taylor_OLG()

par = model.par
par.beta  = 0.99**0.25
par.omega = 0.75
par.tau_y = 1.0 / 3.0
par.sigma = 1.0
par.kappa = 0.05
par.Dbar  = 1.04
par.psi   = 0.0 + 1.0 / par.beta 
par.phi   = 0.0 - par.kappa / par.beta
par.tau_d = None
par.T = 30


model.figname = "skrald.png"
model.decompose_persistence_at_tau(tau_d=0.1)
#model.run()


# model.plot_self_financing_grid(
#     psi_list=[1.25, 1.5, 2.0],
#     phi_list=[-model.par.kappa / model.par.beta, 0.25, 0.35]
# )


a_y  = -model.par.kappa / model.par.beta
a_pi = 1 / model.par.beta
res = model.map_condition_failure_from_matrix(
    phi_grid=np.linspace(a_y-0.5, a_y+1.0, 300),
    psi_grid=np.linspace(a_pi-0.5, a_pi+1.0, 300),
    tau_d_grid=np.linspace(0.000, 1.0, 5000),
    make_plot=True,
)


# model.plot_determinacy_phi_psi(
#     tau_d=0.0,
#     phi_min=-50,
#     phi_max=50,
#     psi_min=-50,
#     psi_max=50
# )