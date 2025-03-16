import numpy as np


class PrimalAffineScaling:
    def __init__(
        self,
        c,
        A,
        b,
        x0,
        iter_max=1e3,
        epsilon=1e-6,
        rho=0.95,
        zero_tolerance=1e-8,
        scale_flag=True,
    ):
        """
        Initialize the Primal Affine Scaling Algorithm.

        Parameters
        ----------
        - c: Coefficient vector for the objective function (min c^T x).
        - A: Constraint matrix.
        - b: Constraint vector.
        - x0: Initial feasible point (must satisfy Ax = b and x > 0).
        - epsilon: Tolerance for optimality.
        - rho: Scaling factor for step size.
        - zero_tolerance: Tolerance for considering a value as zero.
        """
        self.c = c
        self.A = A
        self.b = b
        self.x = x0
        self.iter_max = iter_max
        self.epsilon = epsilon
        self.rho = rho
        self.zero_tolerance = zero_tolerance
        self.scale_flag = scale_flag
        self.n = 0

    def solve(self):
        """
        Solve the linear programming problem using the Primal Affine Scaling method.

        Returns
        -------
        - Optimal solution x* if found.
        - Message indicating if the problem is infeasible or unbounded.
        """
        print(f"Scaling flag: {self.scale_flag}")
        while True:
            if self.scale_flag:
                D = np.diag(self.x)
            else:
                D = np.identity(len(self.x))

            ADA_T = self.A @ D @ D @ self.A.T
            rhs = self.A @ D @ D @ self.c
            y = np.linalg.solve(ADA_T, rhs)
            for _ in range(10):  
                residual = rhs - ADA_T @ y
                dy = np.linalg.solve(ADA_T, residual)
                y += dy            

            gap_dual = abs(self.c.T @ self.x - self.b.T @ y)
            print(f"------ ITERATION: {self.n} ------")
            print(f"Objective: {self.c @ self.x}")
            print(f"Gap dual: {gap_dual}")
            if gap_dual / (1 + abs(self.c.T @ self.x)) < self.epsilon:
                return self.x

            z = self.c - self.A.T @ y
            Δx = -np.square(D) @ z

            if np.all(Δx >= 0) or np.linalg.norm(Δx) < 1e-14:
                return "The problem is unbounded."

            alpha_candidates = [-self.x[i] / Δx[i] for i in range(len(Δx)) if Δx[i] < 0]

            alpha = self.rho * min(alpha_candidates)

            self.x += alpha * Δx

            if np.any(self.x <= 0):
                return "The problem is infeasible."

            self.n += 1

            if self.n > self.iter_max:
                return "Maximum number of iterations reached"
