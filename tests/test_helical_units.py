"""Weighted helical units must follow the vector's physical length scaling."""

import re

import numpy as np
import pytest

from desc.compute import data_index
from desc.compute._equil import _e_sup_helical_times_sqrt_g


@pytest.mark.unit
def test_weighted_helical_units_follow_cartesian_length_scaling():
    vectors = []
    for length in (1.0, 7.0):
        basis = length * np.array([[1.0, 0.4, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 0.7]])
        jacobian = np.linalg.det(basis)
        reciprocal = np.linalg.inv(basis)
        btheta, bzeta = 1.3 / length, -0.8 / length
        magnetic = btheta * basis[:, 1] + bzeta * basis[:, 2]
        data = {
            "B^theta": np.array([btheta]),
            "B^zeta": np.array([bzeta]),
            "e^zeta": reciprocal[2][None, :],
            "sqrt(g)": np.array([jacobian]),
            "e^theta*sqrt(g)": (jacobian * reciprocal[1])[None, :],
        }
        _e_sup_helical_times_sqrt_g(None, None, None, data)
        np.testing.assert_allclose(data["e^helical*sqrt(g)"][0], np.cross(magnetic, basis[:, 0]))
        vectors.append(data["e^helical*sqrt(g)"])
    np.testing.assert_allclose(vectors[1], 7 * vectors[0])
    index = data_index["desc.equilibrium.equilibrium.Equilibrium"]
    for name in ("e^helical*sqrt(g)", "|e^helical*sqrt(g)|"):
        power = re.search(r"m(?:\^\{(-?\d+)\})?", index[name]["units"])
        exponent = int(power.group(1) or 1) if power else 0
        np.testing.assert_allclose(7.0**exponent, 7.0)
