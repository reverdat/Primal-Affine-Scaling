from amplpy import AMPL, Environment
import numpy as np


def verify_with_ampl(c, A, b):
    ampl = AMPL(Environment(""))

    ampl.read("src/ampl/linear.mod")
    ampl.setOption("solver", "cplex")
    ampl.option["solver"] = "cplex"
    n_vars = len(c)
    n_constr = len(b)

    vars_set = [str(i + 1) for i in range(n_vars)]
    constr_set = [str(i + 1) for i in range(n_constr)]

    ampl.set["VARS"] = vars_set
    ampl.set["CONSTR"] = constr_set

    ampl.param["c"] = {str(i + 1): float(c[i]) for i in range(n_vars)}
    ampl.param["b"] = {str(i + 1): float(b[i]) for i in range(n_constr)}

    A_dict = {
        (str(i + 1), str(j + 1)): float(A[i, j])
        for i in range(A.shape[0])
        for j in range(A.shape[1])
    }
    ampl.param["A"] = A_dict

    ampl.solve()

    if ampl.solve_result != "solved":
        raise RuntimeError(f"Solver failed: {ampl.solve_message}")

    return np.array([ampl.get_value(f"x['{j+1}']") for j in range(n_vars)])