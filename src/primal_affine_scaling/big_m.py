import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from primal_affine_scaling.algorithm import PrimalAffineScaling


class BigMMethod:
    def __init__(
        self,
        A,
        b,
        c,
        M=1000,
        iter_max=1e3,
        epsilon=1e-8,
        rho=0.95,
        zero_tolerance=1e-8,
        scale_flag=True,
    ):
        """
        Initialize the Big-M Method to find an initial feasible solution.

        Parameters
        ----------
        - A: Constraint matrix.
        - b: Constraint vector.
        - c: Coefficient vector for the objective function.
        - M: Large positive value for the penalty term.
        """
        self.A_orig = A.astype(float)  # Ensure float type
        self.b = b.astype(float)
        self.c_orig = c.astype(float)
        self.M = float(M)  # Ensure M is float
        self.iter_max = iter_max
        self.epsilon = epsilon
        self.rho = rho
        self.zero_tolerance = zero_tolerance
        self.scale_flag = scale_flag

    def find_initial_feasible_point(self, x0):
        """
        Apply Big-M Method to find an initial feasible point.

        Parameters
        ----------
        - x0: Initial strictly positive point (not necessarily feasible).

        Returns
        -------
        - Feasible point if problem is feasible.
        - "Infeasible" if problem has no feasible solution.
        """
        x0 = x0.astype(float)

        r = self.b - self.A_orig @ x0

        A_extended = np.column_stack((self.A_orig, r))
        c_extended = np.append(self.c_orig, self.M)
        x_extended = np.append(x0, 1.0)

        result = self._solve_extended(A_extended, self.b, c_extended, x_extended)

        if isinstance(result, str):
            return result

        n = len(result)
        print(f"Final value of artificial variable: {result[n-1]}")
        if abs(result[n - 1]) > 1e-6:
            return "The original problem is infeasible."
        else:
            return result[:-1]

    def _solve_extended(self, A, b, c, x0):
        """
        Solve the extended problem using Primal Affine Scaling.
        """
        solver = PrimalAffineScaling(
            c,
            A,
            b,
            x0,
            iter_max=self.iter_max,
            epsilon=self.epsilon,
            rho=self.rho,
            zero_tolerance=self.zero_tolerance,
            scale_flag=self.scale_flag,
        )
        return solver.solve()

