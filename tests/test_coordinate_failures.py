"""Failed physical-coordinate inversion must not expose boundary fields."""

import numpy as np
import pytest

from desc.equilibrium import Equilibrium
from desc.grid import Grid


@pytest.mark.unit
@pytest.mark.parametrize("full_output", [False, True])
def test_map_coordinates_rejects_unconverged_exterior(full_output):
    """An exterior query must not silently return fields on the closest LCFS."""
    eq = Equilibrium(L=2, M=2, N=0)
    coords = np.array([[10.3, 0.7, 0.4], [12.0, 0.0, 0.0]])
    mapped = eq.map_coordinates(
        coords,
        inbasis=("R", "phi", "Z"),
        tol=1e-10,
        maxiter=5,
        full_output=full_output,
    )
    if full_output:
        mapped, (residual, _) = mapped
        assert residual[0] <= 1e-10
        assert residual[1] > 1e-10
    assert np.all(np.isnan(mapped[1]))
    data = eq.compute(["R", "phi", "Z"], grid=Grid(mapped[:1], sort=False))
    np.testing.assert_allclose(
        np.array([data[k] for k in ("R", "phi", "Z")]).T, coords[:1], atol=1e-10
    )
