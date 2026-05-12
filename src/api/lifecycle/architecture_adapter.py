"""Architecture adapter for live dataset swaps (Phase 2 P2-1c).

Implements the additive-only architecture-adaptation logic called for by
``ISSUE_3_PHASE_2_LIVE_DATASET_SWAP_2026-05-09.md`` §3.6: when a live dataset
swap changes the network's input or output dimension, expand the relevant
weight tensors **in place** with zero-initialised new connections.

P2-1c handles equal-dim (no-op) and grow-only (input and/or output)
transitions. Shrink in either dimension raises ``ValueError`` so the route
layer can translate it to HTTP 422 ``shrink_unsupported`` — that surface is
deferred to P2-1d (prepend-adapter-layers, §3.6 "sequential composition rule",
which needs its own design doc per §7).

**Zero-init invariant** (the key property that makes this transition
mathematically safe mid-training):

  After a grow-input swap from ``I_old → I_new``, feeding the post-swap
  network an input of the form ``[x_old, 0, 0, ...]`` (the original I_old
  features padded with zeros for the new slots) produces **exactly** the
  same output as the pre-swap network did on ``x_old`` (across the original
  output dims). This holds because new input-side rows in every weight
  tensor (output layer + every hidden unit) are zero-initialised, so the
  new input slots contribute nothing on day one of the swap.

  After a grow-output swap from ``O_old → O_new``, the first O_old output
  columns of the post-swap forward pass are identical to the pre-swap
  output. The new output columns are pure zero (zero weights + zero bias).

The invariant gives gradient-descent training a clean, low-noise starting
point: the new dimensions look like "dead" inputs/outputs at swap time and
must learn their contribution from scratch, but the network's prior
representation is preserved exactly.

**Tensor inventory** (from the cascade-correlation network):

* ``network.output_weights`` — shape ``[input_size + n_hidden, output_size]``.
  Rows ``0..input_size-1`` are raw-input → output contributions; rows
  ``input_size..input_size+n_hidden-1`` are hidden-unit → output contributions.
* ``network.output_bias`` — shape ``[output_size]``.
* ``network.hidden_units[i]["weights"]`` — 1D, shape ``[input_size + i]``.
  First ``input_size`` entries are raw-input contributions; the next ``i``
  entries are prior-hidden-unit contributions.
* ``network.hidden_units[i]["bias"]`` — scalar (per-unit bias).

Grow-input is therefore a **row insertion** in the middle of ``output_weights``
and each hidden unit's weights, not a simple append: the existing hidden-row
block must shift down to make room for the new input rows. Grow-output is a
plain column-append on ``output_weights`` plus an element-append on
``output_bias`` — hidden units carry no output-dimension state.

See ``_resize_output_layer_for_new_units`` (used by the cascade-add path) for
the precedent this module follows on zero-init + ``requires_grad_(True)``
re-enabling after tensor reassignment.
"""

from dataclasses import dataclass, field
from typing import Any, List, Tuple

import torch


@dataclass(frozen=True)
class ArchChanges:
    """Structured record of an architecture-adapt operation.

    Maps 1:1 onto the ``arch_changes`` block of the §3.3 swap response so
    the lifecycle layer can pass an instance straight through to the route.

    ``appended_nodes`` is the count of new input + output slots added by an
    in-place grow on the outermost adapter layer (in P2-1c this is the
    network itself; in P2-1d it will be the outermost prepended adapter).
    ``prepended_layers`` is reserved for P2-1d's shrink-via-prepend path —
    always empty in P2-1c (any shrink raises).
    """

    input_delta: int
    output_delta: int
    hidden_preserved: int
    appended_nodes: dict = field(default_factory=lambda: {"input": 0, "output": 0})
    prepended_layers: List[Any] = field(default_factory=list)

    def to_response_dict(self) -> dict:
        """Render in the §3.3 response shape. Frozen dataclass + dict copy
        keep callers from mutating the canonical record by accident."""
        return {
            "input_delta": self.input_delta,
            "output_delta": self.output_delta,
            "hidden_preserved": self.hidden_preserved,
            "appended_nodes": dict(self.appended_nodes),
            "prepended_layers": list(self.prepended_layers),
        }


def adapt_for_dataset_swap(
    network: Any,
    before: Tuple[int, int],
    after: Tuple[int, int],
) -> ArchChanges:
    """Adapt ``network`` from ``before=(I_old, O_old)`` to ``after=(I_new, O_new)``.

    Mutates the network in place. Returns an ``ArchChanges`` describing the
    transformation for inclusion in the swap response.

    Contract:
      * Equal-dim → no-op. Returns ``ArchChanges`` with zero deltas.
      * Grow-input — expand ``output_weights`` rows and each hidden unit's
        weight vector via row insertion; new input rows zero-initialised.
      * Grow-output — append columns to ``output_weights`` and elements to
        ``output_bias``; both zero-initialised.
      * Mixed grow (input grows AND output grows) — apply both independently.
      * Any shrink (input or output) — raises ``ValueError`` with a message
        starting ``"shrink_unsupported"``. P2-1d will lift this.

    Raises:
      ``ValueError`` on shrink or non-positive new dims.
    """
    i_old, o_old = before
    i_new, o_new = after

    if i_new <= 0 or o_new <= 0:
        raise ValueError(f"adapter: non-positive new dim (input={i_new}, output={o_new})")

    if i_new < i_old or o_new < o_old:
        raise ValueError(f"shrink_unsupported (P2-1c): input {i_old}→{i_new}, output {o_old}→{o_new}. " "P2-1d will support shrink via prepended adapter layers (§3.6).")

    input_delta = i_new - i_old
    output_delta = o_new - o_old

    if input_delta > 0:
        _grow_input(network, i_old, i_new)
    if output_delta > 0:
        _grow_output(network, o_old, o_new)

    hidden_preserved = len(getattr(network, "hidden_units", []))

    return ArchChanges(
        input_delta=input_delta,
        output_delta=output_delta,
        hidden_preserved=hidden_preserved,
        appended_nodes={"input": input_delta, "output": output_delta},
        prepended_layers=[],
    )


def _grow_input(network: Any, i_old: int, i_new: int) -> None:
    """Expand the input dimension from ``i_old`` to ``i_new`` in place.

    Row-insertion semantics: the existing input-row block stays at rows
    ``0..i_old-1``; the new input rows occupy ``i_old..i_new-1`` zero-init;
    the existing hidden-unit-row block shifts from rows ``i_old..i_old+H-1``
    down to ``i_new..i_new+H-1``. This preserves both the raw-input → output
    contributions and the hidden-unit → output contributions exactly.

    Applied to ``network.output_weights`` and (symmetrically) to every
    ``network.hidden_units[i]["weights"]``. Hidden-unit biases and the
    output bias are unaffected (their shapes are ``[output_size]`` /
    scalar — independent of input width).

    The new ``output_weights`` tensor is allocated with the same dtype +
    device as the original, and ``requires_grad_(True)`` is restored to
    match the convention used by ``_resize_output_layer_for_new_units``.
    Hidden-unit weights are stored detached (cascade-correlation freezes
    them after promotion); ``requires_grad`` is left at ``False`` per
    ``_install_hidden_unit_helper`` (line 3568 of cascade_correlation.py:
    ``weights.clone().detach()``).
    """
    delta = i_new - i_old
    if delta == 0:
        return

    # --- output_weights: shape [i_old + H, O] → [i_new + H, O] ---
    old_w = network.output_weights.detach().clone()
    n_hidden = len(getattr(network, "hidden_units", []))
    o_size = old_w.shape[1]
    new_w = torch.zeros(i_new + n_hidden, o_size, dtype=old_w.dtype, device=old_w.device)
    # Preserve raw-input rows.
    new_w[:i_old, :] = old_w[:i_old, :]
    # Rows i_old..i_new-1 stay zero (new input slots).
    # Shift hidden-unit rows down.
    if n_hidden > 0:
        new_w[i_new : i_new + n_hidden, :] = old_w[i_old : i_old + n_hidden, :]
    new_w.requires_grad_(True)
    network.output_weights = new_w

    # --- hidden_units[i]["weights"]: shape [i_old + i] → [i_new + i] ---
    # Each hidden unit's weight vector mirrors the same row-insertion
    # pattern: preserve raw-input entries, zero-fill the gap, shift the
    # prior-hidden entries down.
    for idx, unit in enumerate(getattr(network, "hidden_units", [])):
        old_uw = unit["weights"].detach().clone()
        new_uw = torch.zeros(i_new + idx, dtype=old_uw.dtype, device=old_uw.device)
        new_uw[:i_old] = old_uw[:i_old]
        if idx > 0:
            new_uw[i_new : i_new + idx] = old_uw[i_old : i_old + idx]
        unit["weights"] = new_uw

    # Update the bookkeeping attribute last so any failure above leaves
    # the network in a self-consistent (pre-swap) state for the caller's
    # rollback path (§3.8). The §3.2 caller wraps adapter calls inside the
    # snapshot/rollback block, so this is belt-and-braces.
    network.input_size = i_new


def _grow_output(network: Any, o_old: int, o_new: int) -> None:
    """Expand the output dimension from ``o_old`` to ``o_new`` in place.

    Column-append on ``output_weights`` and element-append on ``output_bias``.
    Hidden units do not depend on output dimension (their bias is scalar,
    weights are over input + prior-hidden, never over output) — they are
    untouched.

    The new output columns / bias elements are zero-initialised, so the
    first ``o_old`` outputs of the post-swap network are identical to the
    pre-swap outputs; the new ``o_new - o_old`` outputs are pure zero
    until gradient descent gives them a meaningful signal.
    """
    delta = o_new - o_old
    if delta == 0:
        return

    # --- output_weights: shape [I + H, o_old] → [I + H, o_new] ---
    old_w = network.output_weights.detach().clone()
    rows = old_w.shape[0]
    new_w = torch.zeros(rows, o_new, dtype=old_w.dtype, device=old_w.device)
    new_w[:, :o_old] = old_w
    new_w.requires_grad_(True)
    network.output_weights = new_w

    # --- output_bias: shape [o_old] → [o_new] ---
    old_b = network.output_bias.detach().clone()
    new_b = torch.zeros(o_new, dtype=old_b.dtype, device=old_b.device)
    new_b[:o_old] = old_b
    # The bias is read directly in the forward pass (matmul + bias) and
    # is not part of the output-layer optimizer between train calls — the
    # optimizer is rebuilt fresh per ``train_output_layer`` call. So we
    # match the existing convention from ``_resize_output_layer_for_new_units``
    # and leave ``requires_grad`` at the framework default (False on the
    # zero tensor); ``train_output_layer`` re-enables grad on its working
    # copy. See cascade_correlation.py line 1639 for the per-call
    # optimizer rebuild that makes this safe.
    network.output_bias = new_b

    network.output_size = o_new
