"""Signed basis labels and dimensions must describe the computed vectors."""

import re

import numpy as np
import pytest

from desc.compute import data_index
from desc.compute._equil import (
    _e_sup_helical,
    _e_sup_helical_times_sqrt_g,
)

INDEX = data_index["desc.equilibrium.equilibrium.Equilibrium"]


def _fixture(length=1.0):
    # Columns are an independently chosen RH, nonorthogonal coordinate basis.
    basis = length * np.array([[1.0, 0.4, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 0.7]])
    jacobian = np.linalg.det(basis)
    reciprocal = np.linalg.inv(basis)
    btheta, bzeta = 1.3 / length, -0.8 / length
    magnetic = btheta * basis[:, 1] + bzeta * basis[:, 2]
    data = {
        "B^theta": np.array([btheta]),
        "B^zeta": np.array([bzeta]),
        "e^theta": reciprocal[1][None, :],
        "e^zeta": reciprocal[2][None, :],
        "sqrt(g)": np.array([jacobian]),
        "e^theta*sqrt(g)": (jacobian * reciprocal[1])[None, :],
    }
    # Physical B cross covariant e_rho independently fixes sign and SI dimension.
    oracle = np.cross(magnetic, basis[:, 0])
    return data, oracle


def _label_vector(label, data, weighted):
    terms = re.findall(r"B\^\{\\(theta|zeta)\}\s*\\nabla\s*\\(theta|zeta)", label)
    assert len(terms) == 2 and "-" in label
    (b0, e0), (b1, e1) = terms
    vector = data["B^" + b0][:, None] * data["e^" + e0]
    vector -= data["B^" + b1][:, None] * data["e^" + e1]
    return vector * data["sqrt(g)"][:, None] if weighted else vector


@pytest.mark.unit
@pytest.mark.parametrize("length", [0.3, 1.0, 7.0])
def test_helical_labels_describe_signed_cartesian_vector(length):
    """Label and function each match B cross e_rho on a skew RH chart."""
    data, oracle = _fixture(length)
    _e_sup_helical(None, None, None, data)
    _e_sup_helical_times_sqrt_g(None, None, None, data)
    for name, weighted in [("e^helical", False), ("e^helical*sqrt(g)", True)]:
        expected = oracle if weighted else oracle / data["sqrt(g)"][0]
        np.testing.assert_allclose(data[name][0], expected, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(_label_vector(INDEX[name]["label"], data, weighted)[0], expected,
                                   rtol=1e-14, atol=1e-14)

