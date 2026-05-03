from types import SimpleNamespace
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt


class Taud_taylor_OLG:

    def __init__(self, verbose=True, eps0: float = 1.0):
        self.verbose = verbose
        self.eps0 = eps0

        self.par = SimpleNamespace()
        self.eqsys = SimpleNamespace()
        self.IRFres = SimpleNamespace()

        self.set_parameters()
        self.allocate()

    def set_parameters(self):
        par = self.par

        par.beta  = 0.99**0.25
        par.omega = 0.75
        par.tau_y = 1.0 / 3.0
        par.sigma = 1.0
        par.kappa = 0.05
        par.Dbar  = 1.04

        # Real-rate-rule convention:
        # r_t = alpha_y y_t + alpha_pi pi_t
        alpha_y = 0.0
        alpha_pi = 0.0

        # Convert to nominal Taylor-rule parameters:
        # alpha_y  = phi + kappa / beta
        # alpha_pi = psi - 1 / beta
        par.psi = alpha_pi + 1.0 / par.beta
        par.phi = alpha_y - par.kappa / par.beta

        par.tau_d = None
        par.T = 30

    def allocate(self):
        self.sol_all = SimpleNamespace()
        self.sol_all.outputvalues = []
        self.sol_all.chi_list = []
        self.sol_all.tau_d_list = []
        self.sol_all.chi_sign_diagnostics = []

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
        eqsys.X_r = (
            sigma * beta * omega / (1.0 - beta * omega)
            - Dbar * beta * (1.0 - omega) / denom
        )

        X_d, X_y, X_r = eqsys.X_d, eqsys.X_y, eqsys.X_r

        eqsys.m_qq  = (1.0 - tau_d) / beta
        eqsys.m_qy  = (phi + kappa / beta) * Dbar - tau_y / beta
        eqsys.m_qpi = (psi - 1.0 / beta) * Dbar

        eqsys.m_yq  = -X_d * (1.0 - omega * (1.0 - tau_d)) / (beta * omega)

        eqsys.m_yy  = (1.0 / (beta * omega)) * (
            1.0
            - (1.0 - beta * omega) * (X_y - X_r * (phi + kappa / beta))
            + beta * omega * X_d * ((phi + kappa / beta) * Dbar - tau_y / beta)
        )

        eqsys.m_ypi = (psi - 1.0 / beta) * (1.0 / (beta * omega)) * (
            (1.0 - beta * omega) * X_r + beta * omega * X_d * Dbar
        )

        eqsys.m_piq  = 0.0
        eqsys.m_piy  = -kappa / beta
        eqsys.m_pipi = 1.0 / beta

    def system_matrix(self):
        e = self.eqsys
        e.A = np.array([
            [e.m_qq,  e.m_qy,  e.m_qpi],
            [e.m_yq,  e.m_yy,  e.m_ypi],
            [e.m_piq, e.m_piy, e.m_pipi],
        ], dtype=np.complex128)

    def system_matrix_given_policy(self, psi: float, phi: float, tau_d: float) -> np.ndarray:
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
        X_r = (
            sigma * beta * omega / (1.0 - beta * omega)
            - Dbar * beta * (1.0 - omega) / denom
        )

        alpha_y = phi + kappa / beta
        alpha_pi = psi - 1.0 / beta

        m11 = (1.0 - tau_d) / beta
        m12 = Dbar * alpha_y - tau_y / beta
        m13 = Dbar * alpha_pi

        m21 = -X_d * (1.0 - omega * (1.0 - tau_d)) / (beta * omega)

        m22 = (
            1.0
            - (1.0 - beta * omega) * X_y
            + (1.0 - beta * omega) * X_r * alpha_y
            + beta * omega * X_d * (Dbar * alpha_y - tau_y / beta)
        ) / (beta * omega)

        m23 = (
            ((1.0 - beta * omega) * X_r + beta * omega * X_d * Dbar) * alpha_pi
        ) / (beta * omega)

        return np.array([
            [m11,  m12,          m13],
            [m21,  m22,          m23],
            [0.0, -kappa / beta, 1.0 / beta],
        ], dtype=float)

    def exists_unique_bounded_equilibrium(self, tol=1e-9):
        vals = np.linalg.eigvals(self.eqsys.A)

        n_stable = int(np.sum(np.abs(vals) < 1.0 - tol))
        n_unstable = int(np.sum(np.abs(vals) > 1.0 + tol))

        return n_stable == 1 and n_unstable == 2

    def solve_unique_bounded_eq(self, tol=1e-9):
        A = self.eqsys.A
        sol = self.sol
        par = self.par
        eps0 = self.eps0

        eigvals, eigvecs = np.linalg.eig(A)

        sol.all_eigenvalues = eigvals

        stable_idx = [i for i, val in enumerate(eigvals) if abs(val) < 1.0 - tol]
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
        T = par.T

        q0 = sol.q0
        lambda_s = sol.lambda_s
        v_s = sol.v_s

        q_full = np.full(T + 2, np.nan)

        for t in range(T + 2):
            q_full[t] = q0 * (lambda_s ** t)

        sol.q = q_full[:-1]
        sol.debt_end = q_full[1:T + 2]
        sol.t = np.arange(T + 1)

        sol.y = v_s[1] * sol.q
        sol.pi = v_s[2] * sol.q
        sol.i_nom = par.psi * sol.pi + par.phi * sol.y

        alpha_y = par.phi + par.kappa / par.beta
        alpha_pi = par.psi - 1.0 / par.beta

        r_loading = alpha_y * v_s[1] + alpha_pi * v_s[2]
        sol.r = r_loading * sol.q

        # Neutral-rate style decomposition. Mainly for your old plots.
        denom = par.tau_d + (par.tau_y + par.kappa * par.Dbar) * v_s[1]

        if abs(denom) < 1e-12:
            sol.nu_base_from_params = np.nan
            sol.nu_price_from_params = np.nan
            sol.nu_total_from_params = np.nan
        else:
            sol.nu_base_from_params = (par.tau_y * v_s[1]) / denom
            sol.nu_price_from_params = (par.Dbar * par.kappa * v_s[1]) / denom
            sol.nu_total_from_params = (
                (par.tau_y + par.Dbar * par.kappa) * v_s[1] / denom
            )

    def chi_sign_diagnostics_from_solution(self, tol=1e-10):
        """
        Checks the exact two-case restriction for chi > 0.

        chi > 0 iff

            [tau_d - (1 - beta*lambda_s)]
            *
            [M(lambda_s) - tau_y/beta] > 0

        where

            M(lambda_s)
            =
            Dbar * [alpha_y + alpha_pi*kappa/(1-beta*lambda_s)]

        and

            alpha_y  = phi + kappa/beta
            alpha_pi = psi - 1/beta.
        """

        par = self.par
        sol = self.sol

        beta = par.beta
        tau_d = par.tau_d
        tau_y = par.tau_y
        kappa = par.kappa
        Dbar = par.Dbar
        lambda_s = sol.lambda_s

        alpha_y = par.phi + kappa / beta
        alpha_pi = par.psi - 1.0 / beta

        z = 1.0 - beta * lambda_s
        threshold = 1.0 - beta * lambda_s

        if z <= 0:
            return {
                "valid": False,
                "reason": "1 - beta*lambda_s <= 0",
                "tau_d": tau_d,
                "lambda_s": lambda_s,
            }

        M = Dbar * (alpha_y + alpha_pi * kappa / z)
        tau_y_over_beta = tau_y / beta

        numerator = lambda_s - (1.0 - tau_d) / beta
        denominator = M - tau_y_over_beta

        product_condition = (tau_d - threshold) * denominator

        if abs(denominator) < tol:
            chi_formula = np.nan
            chi_positive_formula = False
        else:
            chi_formula = numerator / denominator
            chi_positive_formula = chi_formula > 0.0

        if tau_d < threshold - tol:
            case = "Case 1: delayed fiscal adjustment"
            required_text = "tau_y / beta > M(lambda_s)"
            restriction_holds = tau_y_over_beta > M

        elif tau_d > threshold + tol:
            case = "Case 2: strong fiscal adjustment"
            required_text = "tau_y / beta < M(lambda_s)"
            restriction_holds = tau_y_over_beta < M

        else:
            case = "Boundary"
            required_text = "tau_d = 1 - beta*lambda_s"
            restriction_holds = abs(numerator) < tol

        return {
            "valid": True,
            "tau_d": tau_d,
            "lambda_s": lambda_s,
            "chi": sol.chi,
            "eta": sol.eta,

            "alpha_y": alpha_y,
            "alpha_pi": alpha_pi,
            "z": z,
            "threshold": threshold,
            "M": M,
            "tau_y_over_beta": tau_y_over_beta,

            "numerator": numerator,
            "denominator": denominator,
            "product_condition": product_condition,

            "chi_formula": chi_formula,
            "chi_positive_formula": chi_positive_formula,
            "chi_positive_eigenvector": sol.chi > 0.0,

            "case": case,
            "required_text": required_text,
            "restriction_holds": restriction_holds,
        }

    def compute_tau_sweep(self, tau_d_grid=None):
        par = self.par

        self.sol_all.outputvalues = []
        self.sol_all.chi_list = []
        self.sol_all.tau_d_list = []
        self.sol_all.chi_sign_diagnostics = []

        if tau_d_grid is None:
            tau_d_grid = np.sort(
                np.concatenate((
                    np.linspace(0.0, 1.0, 301),
                    np.array([0.085, 0.026, 0.004])
                ))
            )

        for tau_d in tau_d_grid:
            par.tau_d = float(tau_d)
            self.sol = SimpleNamespace()

            try:
                self.eqsys_matrix_elements()
                self.system_matrix()

                if self.exists_unique_bounded_equilibrium():
                    self.solve_unique_bounded_eq()
                    self.compute_irf()

                    self.sol.tau_d = float(tau_d)

                    diag = self.chi_sign_diagnostics_from_solution()
                    self.sol.chi_sign_diag = diag

                    self.sol_all.outputvalues.append(deepcopy(self.sol))
                    self.sol_all.chi_list.append(deepcopy(self.sol.chi))
                    self.sol_all.tau_d_list.append(deepcopy(self.sol.tau_d))
                    self.sol_all.chi_sign_diagnostics.append(deepcopy(diag))

            except Exception as e:
                if self.verbose:
                    pass

    def print_chi_sign_summary(self, selected_tau_d=None):
        if len(self.sol_all.chi_sign_diagnostics) == 0:
            raise RuntimeError("Run compute_tau_sweep() first.")

        diags = [d for d in self.sol_all.chi_sign_diagnostics if d.get("valid", False)]

        if selected_tau_d is not None:
            selected = []
            for tau in selected_tau_d:
                closest = min(diags, key=lambda d: abs(d["tau_d"] - tau))
                selected.append(closest)
            diags_to_print = selected
        else:
            diags_to_print = diags

        print("\n" + "=" * 130)
        print("Chi sign restriction check")
        print("=" * 130)
        print(
            f"{'tau_d':>8} "
            f"{'lambda_s':>10} "
            f"{'chi':>10} "
            f"{'threshold':>12} "
            f"{'M(lambda)':>12} "
            f"{'tau_y/beta':>12} "
            f"{'case':>34} "
            f"{'holds?':>8}"
        )
        print("-" * 130)

        for d in diags_to_print:
            print(
                f"{d['tau_d']:>8.4f} "
                f"{d['lambda_s']:>10.4f} "
                f"{d['chi']:>10.4f} "
                f"{d['threshold']:>12.4f} "
                f"{d['M']:>12.4f} "
                f"{d['tau_y_over_beta']:>12.4f} "
                f"{d['case']:>34} "
                f"{str(d['restriction_holds']):>8}"
            )

        print("=" * 130)

        all_holds = np.array([d["restriction_holds"] for d in diags], dtype=bool)
        all_chi_pos = np.array([d["chi"] > 0.0 for d in diags], dtype=bool)

        print(f"Number of determinate tau_d values: {len(diags)}")
        print(f"Share with chi > 0: {np.mean(all_chi_pos):.4f}")
        print(f"Share where restriction holds: {np.mean(all_holds):.4f}")
        print(f"Restriction matches chi > 0 everywhere? {np.all(all_holds == all_chi_pos)}")
        print("=" * 130)

    def plot_chi_sign_restrictions(self, figsize=(10, 9), savepath=None):
        if len(self.sol_all.chi_sign_diagnostics) == 0:
            raise RuntimeError("Run compute_tau_sweep() first.")

        diags = [d for d in self.sol_all.chi_sign_diagnostics if d.get("valid", False)]

        tau_d = np.array([d["tau_d"] for d in diags])
        lambda_s = np.array([d["lambda_s"] for d in diags])
        chi = np.array([d["chi"] for d in diags])
        threshold = np.array([d["threshold"] for d in diags])
        M = np.array([d["M"] for d in diags])
        tau_y_over_beta = np.array([d["tau_y_over_beta"] for d in diags])
        restriction_holds = np.array([d["restriction_holds"] for d in diags], dtype=bool)

        order = np.argsort(tau_d)

        tau_d = tau_d[order]
        lambda_s = lambda_s[order]
        chi = chi[order]
        threshold = threshold[order]
        M = M[order]
        tau_y_over_beta = tau_y_over_beta[order]
        restriction_holds = restriction_holds[order]

        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

        # Panel 1: chi
        ax = axes[0]
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.plot(tau_d, chi, linewidth=2.3, label=r"$\chi$")
        ax.fill_between(
            tau_d,
            chi,
            0.0,
            where=chi > 0.0,
            alpha=0.25,
            label=r"$\chi>0$"
        )
        ax.set_ylabel(r"$\chi$")
        ax.set_title(r"Output response $\chi$")
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.3)

        # Panel 2: case split
        ax = axes[1]
        ax.plot(tau_d, tau_d, linewidth=2.0, label=r"$\tau_d$")
        ax.plot(tau_d, threshold, linewidth=2.0, label=r"$1-\beta\lambda_s$")
        ax.fill_between(
            tau_d,
            tau_d,
            threshold,
            where=tau_d < threshold,
            alpha=0.20,
            label="Case 1"
        )
        ax.fill_between(
            tau_d,
            tau_d,
            threshold,
            where=tau_d > threshold,
            alpha=0.20,
            label="Case 2"
        )
        ax.set_ylabel("case split")
        ax.set_title(r"Case split: compare $\tau_d$ with $1-\beta\lambda_s$")
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.3)

        # Panel 3: force comparison
        ax = axes[2]
        ax.plot(tau_d, tau_y_over_beta, linewidth=2.0, label=r"$\tau_y/\beta$")
        ax.plot(tau_d, M, linewidth=2.0, label=r"$M(\lambda_s)$")

        ax.fill_between(
            tau_d,
            tau_y_over_beta,
            M,
            where=tau_y_over_beta > M,
            alpha=0.20,
            label="tax-base dominance"
        )
        ax.fill_between(
            tau_d,
            tau_y_over_beta,
            M,
            where=tau_y_over_beta < M,
            alpha=0.20,
            label="debt-service dominance"
        )

        ax.set_xlabel(r"$\tau_d$")
        ax.set_ylabel("forces")
        ax.set_title(r"Restriction: compare $\tau_y/\beta$ with $M(\lambda_s)$")
        ax.legend(frameon=True)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if savepath is not None:
            fig.savefig(savepath, dpi=300, bbox_inches="tight")
            print(f"saved to: {savepath}")

        plt.show()
        plt.close(fig)

        print("\n" + "=" * 90)
        print("Chi sign restriction summary")
        print("=" * 90)
        print(f"Number of determinate tau_d values: {len(tau_d)}")
        print(f"Share with chi > 0: {np.mean(chi > 0):.3f}")
        print(f"Share where restriction holds: {np.mean(restriction_holds):.3f}")
        print(f"Restriction matches chi > 0 everywhere? {np.all(restriction_holds == (chi > 0))}")
        print("=" * 90)

    def decompose_persistence_at_tau(
        self,
        tau_d=0.10,
        alpha_y_val=0.0,
        alpha_pi_val=0.0,
        tol=1e-9,
    ):
        par = self.par

        beta = par.beta
        tau_y = par.tau_y
        Dbar = par.Dbar
        kappa = par.kappa

        phi_case = alpha_y_val - kappa / beta
        psi_case = alpha_pi_val + 1.0 / beta

        A = self.system_matrix_given_policy(
            psi=psi_case,
            phi=phi_case,
            tau_d=tau_d,
        )

        eigvals, eigvecs = np.linalg.eig(A)

        stable_idx = [i for i, val in enumerate(eigvals) if abs(val) < 1.0 - tol]
        unstable_idx = [i for i, val in enumerate(eigvals) if abs(val) > 1.0 + tol]

        if not (len(stable_idx) == 1 and len(unstable_idx) == 2):
            print("No unique bounded equilibrium.")
            print("Eigenvalues:", eigvals)
            return None

        idx = stable_idx[0]
        lambda_s = eigvals[idx]
        v_s = eigvecs[:, idx]

        v_s = v_s / v_s[0]

        lambda_s = float(np.real(lambda_s))
        v_s = np.real(v_s)

        chi = float(v_s[1])
        eta = float(v_s[2])

        rollover_term = (1.0 - tau_d) / beta
        tax_base_term = -(tau_y / beta) * chi
        real_rate_loading = alpha_y_val * chi + alpha_pi_val * eta
        real_rate_term = Dbar * real_rate_loading
        reconstructed = rollover_term + tax_base_term + real_rate_term

        print("\n" + "=" * 90)
        print(f"Stable-root decomposition at tau_d = {tau_d:.4f}")
        print("=" * 90)
        print(f"lambda_s      = {lambda_s:.6f}")
        print(f"chi           = {chi:.6f}")
        print(f"eta           = {eta:.6f}")
        print(f"rollover      = {rollover_term:.6f}")
        print(f"tax base      = {tax_base_term:.6f}")
        print(f"real rate     = {real_rate_term:.6f}")
        print(f"sum           = {reconstructed:.6f}")
        print(f"residual      = {lambda_s - reconstructed:.2e}")
        print("=" * 90)

        return {
            "lambda_s": lambda_s,
            "chi": chi,
            "eta": eta,
            "rollover": rollover_term,
            "tax_base": tax_base_term,
            "real_rate": real_rate_term,
            "reconstructed": reconstructed,
            "residual": lambda_s - reconstructed,
        }


# ======================================================================
# RUN EXAMPLE
# ======================================================================

if __name__ == "__main__":

    model = Taud_taylor_OLG(verbose=True)

    par = model.par

    par.beta  = 0.99**0.25
    par.omega = 0.75
    par.tau_y = 1.0 / 3.0
    par.sigma = 1.0
    par.kappa = 0.05
    par.Dbar  = 1.04
    par.T = 30

    # ------------------------------------------------------------
    # Choose monetary policy here.
    #
    # Real-rate-rule notation:
    #     r_t = alpha_y y_t + alpha_pi pi_t
    #
    # Neutral real rate:
    #     alpha_y = 0
    #     alpha_pi = 0
    #
    # Active example:
    #     alpha_y = 0.08
    #     alpha_pi = 1.04
    # ------------------------------------------------------------

    alpha_y = 0.5
    alpha_pi = 0.5

    par.psi = alpha_pi + 1.0 / par.beta
    par.phi = alpha_y - par.kappa / par.beta

    # Sweep over tau_d
    tau_d_grid = np.linspace(0.0, 1.0, 1001)
    model.compute_tau_sweep(tau_d_grid=tau_d_grid)

    # Print selected values
    model.print_chi_sign_summary(
        selected_tau_d=[0.004, 0.026, 0.085, 0.1, 0.3, 0.5, 0.9]
    )

    # Plot the restrictions
    model.plot_chi_sign_restrictions()

    # Optional decomposition at one tau_d
    model.decompose_persistence_at_tau(
        tau_d=0.1,
        alpha_y_val=alpha_y,
        alpha_pi_val=alpha_pi,
    )