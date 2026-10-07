"""Accepted-step callbacks preserve solver trajectories and feasible snapshots."""

import numpy as np
import pytest

from desc.backend import jnp
from desc.io import IOAble
from desc.objectives import FixParameters, ObjectiveFunction
from desc.objectives.objective_funs import _Objective
from desc.optimizable import Optimizable, optimizable_parameter
from desc.optimize import LinearConstraintProjection, Optimizer


class CallbackState(IOAble, Optimizable):
    """Small savable state for an analytical nonlinear least-squares problem."""

    _io_attrs_ = ["_values"]

    def __init__(self, values):
        self.values = values

    @property
    @optimizable_parameter
    def values(self):
        """ndarray: Optimizable coefficients."""
        return self._values

    @values.setter
    def values(self, value):
        self._values = jnp.asarray(value, dtype=float)


class RosenbrockResidual(_Objective):
    """Residual with independently known positive minimizer (1, 1)."""

    _units = "(~)"
    _print_value_fmt = "Rosenbrock residual: "

    def build(self, use_jit=True, verbose=0):
        """Build the two-component residual."""
        self._dim_f = 2
        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Evaluate the independently defined nonlinear residual."""
        x, y = params["values"][:2]
        return jnp.array([10 * (y - x**2), 1 - x])


def make_problem(constrained=False):
    """Construct the same analytical problem with an optional fixed parameter."""
    state = CallbackState([-1.2, 1, 7] if constrained else [-1.2, 1])
    obj = ObjectiveFunction(RosenbrockResidual(things=state, target=0, normalize=False))
    if constrained:
        con = ObjectiveFunction(FixParameters(state, params={"values": [2]}))
        obj = LinearConstraintProjection(obj, con)
    obj.build(verbose=0)
    return state, obj


def optimize(state, objective, options=None):
    """Solve the small analytical problem with fixed settings."""
    return Optimizer("lsq-exact").optimize(
        state,
        objective,
        x_scale=1,
        options=options,
        verbose=0,
        maxiter=100,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )[1]


@pytest.mark.unit
def test_lsq_callback_observes_accepted_steps_without_changing_trajectory():
    """A false-returning observer preserves the complete no-callback trajectory."""
    state, obj = make_problem()
    baseline = optimize(state, obj)
    state, obj = make_problem()
    explicit_none = optimize(state, obj, {"callback": None})
    state, obj = make_problem()
    snapshots = []

    def observe(x):
        snapshots.append(np.array(x, copy=True))
        return False

    observed = optimize(state, obj, {"callback": observe})
    assert observed.success and baseline.success and explicit_none.success
    np.testing.assert_allclose(observed.x, [1, 1], rtol=0, atol=1e-9)
    assert snapshots
    np.testing.assert_array_equal(observed.allx, baseline.allx)
    np.testing.assert_array_equal(explicit_none.allx, baseline.allx)
    np.testing.assert_array_equal(snapshots, observed.allx[1:])
    residuals = [np.array([10 * (x[1] - x[0] ** 2), 1 - x[0]]) for x in snapshots]
    costs = np.array([np.dot(r, r) / 2 for r in residuals])
    assert np.all(np.diff(costs) <= 1e-14)


@pytest.mark.unit
def test_lsq_callback_stops_at_feasible_roundtrip_snapshot(tmp_path):
    """Reduced callback coordinates recover full fixed-parameter states."""
    state, obj = make_problem(constrained=True)
    original = np.array(state.values, copy=True)
    snapshots = []

    def checkpoint(x):
        assert np.size(x) == 2  # Full state has three parameters.
        full = obj.unpack_state(x, per_objective=False)[0]
        snapshot = state.copy()
        snapshot.params_dict = full
        np.testing.assert_equal(np.array(snapshot.values)[2], 7)
        np.testing.assert_array_equal(state.values, original)
        filename = tmp_path / f"accepted_{len(snapshots)}.h5"
        snapshot.save(filename)
        restored = CallbackState.load(filename)
        np.testing.assert_array_equal(restored.values, snapshot.values)
        snapshots.append(np.array(restored.values, copy=True))
        return len(snapshots) == 2

    result = optimize(state, obj, {"callback": checkpoint})
    assert not result.success
    assert "callback" in result.message.lower()
    assert len(snapshots) == 2
    np.testing.assert_array_equal(snapshots, result.allx[1:])
    np.testing.assert_array_equal(state.values, snapshots[-1])
    assert np.all(np.array(result.allx)[:, 2] == 7)
    assert not np.array_equal(snapshots[-1][:2], [1, 1])


@pytest.mark.unit
def test_lsq_callback_is_not_called_without_an_accepted_step():
    """A solved analytical fixture terminates without a fabricated snapshot."""
    state, obj = make_problem()
    state.values = [1, 1]
    snapshots = []
    result = optimize(state, obj, {"callback": lambda x: snapshots.append(x)})
    assert result.success
    assert snapshots == []
    np.testing.assert_array_equal(result.x, [1, 1])
