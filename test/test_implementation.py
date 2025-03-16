import pytest
import numpy as np

from src.primal_affine_scaling.big_m import BigMMethod
from src.primal_affine_scaling.algorithm import PrimalAffineScaling
from src.primal_affine_scaling.utils.compute_m import heuristic_m
from src.ampl.linear_ampl import verify_with_ampl


def _load_netlib(problem_name):
    import scipy.io as sio

    mat_file = sio.loadmat(f"data/mat/{problem_name}.mat")
    mat_data = mat_file["Problem"]
    bounds_data = mat_data[0, 0][5]

    A = mat_data[0, 0][2]
    b = mat_data[0, 0][3].reshape((A.shape[0],))
    c = bounds_data[0, 0]["c"].reshape((1, -1))[0]

    A_np = A.toarray()
    b_np = b
    c_np = c

    return A_np, b_np, c_np


# Test cases with different problem configurations
TEST_CASES = [
    # Feasible problem
    {
        "name": "feasible_problem",
        "A": np.array([[4, -2, 1, 0, 0], [3, 4, 0, -1, 0], [1, 1, 0, 0, 1]]),
        "b": np.array([5, 1, 2]),
        "c": np.array([-3, -2, 0, 0, 0]),
        "M": heuristic_m(np.array([-3, -2, 0, 0, 0])),
        "x0": np.array([0.5, 0.5, 4.0, 2.5, 1.0]),
        "iter_max": 1e3,
        "epsilon": 1e-8,
        "rho": 0.95,
        "zero_tolerance": 1e-8,
        "scale_flag": True,
        "expected_status": "solved",
    },
    # Infeasible problem
    {
        "name": "infeasible_problem",
        "A": np.array([[1, 1, 1, 0], [2, 2, 2, 0]]),
        "b": np.array([3, 9]),
        "c": np.array([1, 1, 0, 0]),
        "M": heuristic_m(np.array([1, 1, 0, 0])),
        "iter_max": 1e3,
        "x0": np.array([1.0, 1.0, 1.0, 1.0]),
        "epsilon": 1e-8,
        "rho": 0.95,
        "zero_tolerance": 1e-8,
        "scale_flag": True,
        "expected_status": "infeasible",
    },
    # Unbounded problem
    {
        "name": "unbounded_problem",
        "A": np.array([[1, -1, 1, 0], [1, 1, 0, 1]]),
        "b": np.array([2, 3]),
        "c": np.array([-1, -1, 0, 0]),
        "M": heuristic_m(np.array([-1, -1, 0, 0])),
        "iter_max": 1e3,
        "x0": np.array([1.0, 1.0, 1.0, 1.0]),
        "epsilon": 1e-8,
        "rho": 0.95,
        "zero_tolerance": 1e-8,
        "scale_flag": True,
        "expected_status": "unbounded",
    },
]


TEST_CASES_NO_SCALING = [
    # Feasible problem
    {
        "name": "feasible_problem",
        "A": np.array([[4, -2, 1, 0, 0], [3, 4, 0, -1, 0], [1, 1, 0, 0, 1]]),
        "b": np.array([5, 1, 2]),
        "c": np.array([-3, -2, 0, 0, 0]),
        "M": heuristic_m(np.array([-3, -2, 0, 0, 0])),
        "x0": np.array([0.5, 0.5, 4.0, 2.5, 1.0]),
        "iter_max": 1e3,
        "epsilon": 1e-8,
        "rho": 0.95,
        "zero_tolerance": 1e-8,
        "scale_flag": False,
        "expected_status": "solved",
    },
    # Infeasible problem
    {
        "name": "infeasible_problem",
        "A": np.array([[1, 1, 1, 0], [2, 2, 2, 0]]),
        "b": np.array([3, 9]),
        "c": np.array([1, 1, 0, 0]),
        "M": heuristic_m(np.array([1, 1, 0, 0])),
        "iter_max": 1e3,
        "x0": np.array([1.0, 1.0, 1.0, 1.0]),
        "epsilon": 1e-8,
        "rho": 0.95,
        "zero_tolerance": 1e-8,
        "scale_flag": False,
        "expected_status": "infeasible",
    },
    # Unbounded problem
    {
        "name": "unbounded_problem",
        "A": np.array([[1, -1, 1, 0], [1, 1, 0, 1]]),
        "b": np.array([2, 3]),
        "c": np.array([-1, -1, 0, 0]),
        "M": heuristic_m(np.array([-1, -1, 0, 0])),
        "iter_max": 1e3,
        "x0": np.array([1.0, 1.0, 1.0, 1.0]),
        "epsilon": 1e-8,
        "rho": 0.95,
        "zero_tolerance": 1e-8,
        "scale_flag": False,
        "expected_status": "unbounded",
    },
]
A_galenet, b_galenet, c_galenet = _load_netlib("lpi_galenet")
A_stocfor1, b_stocfor1, c_stocfor1 = _load_netlib("lp_stocfor1")
A_israel, b_israel, c_israel = _load_netlib("lp_israel")
A_sc205, b_sc205, c_sc205 = _load_netlib("lp_sc205")


NETLIB_TEST_CASES = [
    {
        "name": "lp_israel",
        "A": A_israel,
        "b": b_israel,
        "c": c_israel,
        "M": 10,
        "x0": np.ones(A_israel.shape[1]),
        "iter_max": 1e3,
        "epsilon": 1e-8,
        "rho": 0.95,
        "zero_tolerance": 1e-8,
        "scale_flag": True,
        "expected_status": "solved",
    },
    {
        "name": "lp_sc205",
        "A": A_sc205,
        "b": b_sc205,
        "c": c_sc205,
        "M": 100,
        "x0": np.ones(A_sc205.shape[1]),
        "iter_max": 1e3,
        "epsilon": 1e-6,
        "rho": 0.95,
        "zero_tolerance": 1e-6,
        "scale_flag": True,
        "expected_status": "solved",
    },

    {
        "name": "lpi_galenet",
        "A": A_galenet,
        "b": b_galenet,
        "c": c_galenet,
        "M": 100,
        "x0": np.ones(A_galenet.shape[1]),
        "iter_max": 1e3,
        "epsilon": 1e-6,
        "rho": 0.95,
        "zero_tolerance": 1e-6,
        "scale_flag": True,
        "expected_status": "solved",
    },
    {
        "name": "lp_stocfor1",
        "A": A_stocfor1,
        "b": b_stocfor1,
        "c": c_stocfor1,
        "M": 10,
        "x0": np.ones(A_stocfor1.shape[1]),
        "iter_max": 1e3,
        "epsilon": 1e-6,
        "rho": 0.95,
        "zero_tolerance": 1e-6,
        "scale_flag": True,
        "expected_status": "solved",
    }
]


def run_big_m_solver(case):
    """Run the Big-M solver pipeline"""
    print(case["M"])
    big_m = BigMMethod(
        case["A"],
        case["b"],
        case["c"],
        case["M"],
        case["iter_max"],
        case["epsilon"],
        case["rho"],
        case["zero_tolerance"],
        case["scale_flag"],
    )
    result = big_m.find_initial_feasible_point(case["x0"])

    return result


@pytest.mark.parametrize("case", TEST_CASES, ids=[tc["name"] for tc in TEST_CASES])
def test_lp_solution(case):
    """Main test function comparing Big-M and AMPL solutions"""
    custom_result = run_big_m_solver(case)

    if case["expected_status"] != "solved":
        assert case["expected_status"] in str(custom_result)
        return

    print(f"Our result: {custom_result}")

    # Run AMPL solver
    try:
        ampl_solution = verify_with_ampl(case["c"], case["A"], case["b"])
    except RuntimeError as e:
        if "infeasible" in str(e).lower():
            assert "infeasible" in str(custom_result)
            return
        elif "unbounded" in str(e).lower():
            assert "unbounded" in str(custom_result)
            return
        else:
            pytest.fail(f"Unexpected AMPL error: {e}")

    assert isinstance(
        custom_result, np.ndarray
    ), f"Expected solution array, got {type(custom_result)}"

    assert np.allclose(
        custom_result, ampl_solution, atol=1e-6
    ), "Solutions differ beyond tolerance"

    assert np.all(custom_result >= -1e-6), "Negative values in solution"
    assert np.allclose(
        case["A"] @ custom_result, case["b"], atol=1e-6
    ), "Constraints not satisfied"




@pytest.mark.parametrize("case", TEST_CASES_NO_SCALING, ids=[tc["name"] for tc in TEST_CASES_NO_SCALING])
def test_lp_solution_no_scaling(case):
    """Main test function comparing Big-M and AMPL solutions"""
    custom_result = run_big_m_solver(case)

    if case["expected_status"] != "solved":
        assert case["expected_status"] in str(custom_result)
        return

    print(f"Our result: {custom_result}")

    # Run AMPL solver
    try:
        ampl_solution = verify_with_ampl(case["c"], case["A"], case["b"])
    except RuntimeError as e:
        if "infeasible" in str(e).lower():
            assert "infeasible" in str(custom_result)
            return
        elif "unbounded" in str(e).lower():
            assert "unbounded" in str(custom_result)
            return
        else:
            pytest.fail(f"Unexpected AMPL error: {e}")

    assert isinstance(
        custom_result, np.ndarray
    ), f"Expected solution array, got {type(custom_result)}"

    assert np.allclose(
        custom_result, ampl_solution, atol=1e-6
    ), "Solutions differ beyond tolerance"

    assert np.all(custom_result >= -1e-6), "Negative values in solution"
    assert np.allclose(
        case["A"] @ custom_result, case["b"], atol=1e-6
    ), "Constraints not satisfied"


def test_objective_values():
    """Verify objective values match for feasible problems"""
    feasible_case = next(tc for tc in TEST_CASES if tc["name"] == "feasible_problem")

    custom_result = run_big_m_solver(feasible_case)
    custom_obj = feasible_case["c"] @ custom_result

    # Get AMPL solution
    ampl_solution = verify_with_ampl(
        feasible_case["c"], feasible_case["A"], feasible_case["b"]
    )
    ampl_obj = feasible_case["c"] @ ampl_solution

    assert (
        abs(custom_obj - ampl_obj) < 1e-6
    ), f"Objective values differ: {custom_obj} vs {ampl_obj}"


@pytest.mark.parametrize(
    "case", NETLIB_TEST_CASES, ids=[tc["name"] for tc in NETLIB_TEST_CASES]
)
def test_netlib_solution(case):
    """Main test function comparing Big-M and AMPL solutions
    for NETLIB problems
    """
    custom_result = run_big_m_solver(case)

    if case["expected_status"] != "solved":
        assert case["expected_status"] in str(custom_result)
        return

    print(f"Our result: {custom_result}")

    # Run AMPL solver
    try:
        ampl_solution = verify_with_ampl(case["c"], case["A"], case["b"])
    except RuntimeError as e:
        if "infeasible" in str(e).lower():
            assert "infeasible" in str(custom_result)
            return
        elif "unbounded" in str(e).lower():
            assert "unbounded" in str(custom_result)
            return
        else:
            pytest.fail(f"Unexpected AMPL error: {e}")

    assert isinstance(
        custom_result, np.ndarray
    ), f"Expected solution array, got {type(custom_result)}"

    assert np.allclose(
        custom_result, ampl_solution, atol=1e-6
    ), "Solutions differ beyond tolerance"

    assert np.all(custom_result >= -1e-6), "Negative values in solution"
    assert np.allclose(
        case["A"] @ custom_result, case["b"], atol=1e-6
    ), "Constraints not satisfied"
