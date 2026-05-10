"""Truth-table tests for ``_validate_candidate_pool_triple`` (§1.5 C2.1).

The helper is the single source of truth for the candidate-pool invariant; if
its logic is wrong, every PATCH path that uses it is wrong. These tests pin
the invariant table directly so a future refactor can't drift the rules
without breaking obvious cases first.
"""

import pytest

from api.lifecycle.manager import _validate_candidate_pool_triple


@pytest.mark.unit
@pytest.mark.parametrize(
    "s,t,r,p",
    [
        (1, 1, 0, 8),
        (4, 4, 0, 8),
        (4, 0, 4, 8),
        (6, 4, 2, 8),
        (8, 5, 3, 8),  # S == P (saturation upper edge)
        (1, 0, 1, 1),  # P == S == 1, R == S
    ],
)
def test_valid_triples_return_none(s, t, r, p):
    assert _validate_candidate_pool_triple(s, t, r, p) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "s,t,r,p,fragment",
    [
        # S out of range
        (0, 0, 0, 8, "selected_candidates 0 not in"),
        (9, 5, 4, 8, "selected_candidates 9 not in"),
        # Negative components
        (4, -1, 5, 8, "must be >= 0"),
        (4, 5, -1, 8, "must be >= 0"),
        # Component exceeds S
        (4, 5, 0, 8, "each component must be"),
        (4, 0, 5, 8, "each component must be"),
        # Both zero with S > 0
        (4, 0, 0, 8, "cannot both be 0"),
        # Degenerate: T==0 but R != S
        (4, 0, 3, 8, "with top_candidates=0"),
        # Degenerate: R==0 but T != S
        (4, 3, 0, 8, "with random_candidates=0"),
        # Both nonzero but T+R != S
        (4, 1, 1, 8, "must equal S=4"),
        (4, 3, 2, 8, "must equal S=4"),
    ],
)
def test_invalid_triples_return_specific_violation(s, t, r, p, fragment):
    msg = _validate_candidate_pool_triple(s, t, r, p)
    assert msg is not None, f"expected violation for ({s}, {t}, {r}, p={p})"
    assert fragment in msg, f"violation message {msg!r} missing fragment {fragment!r}"
