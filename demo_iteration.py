#!/usr/bin/env python3
"""Fixed-input demonstration of the adaptive quad-tree 5/3 decomposition.

Run:  python demo_iteration.py

Part 1 builds a deterministic 13x9 integer image (smooth left half, busy
right half), decomposes every subband recursively to depth 3, prunes the
tree with the default explicit integer encoding-length cost, prints the
chosen tree with every leaf's band, shape and cost, and verifies bit-exact
reconstruction.  Part 2 compares against a caller-supplied custom cost.
Part 3 genuinely triggers the rejection of a structurally invalid tree
(missing child band).  Everything is offline, integer-only, computed for
real on each run (no sleep, no pre-recorded output) and finishes in well
under a second -- far below the 90 second budget.
"""
from __future__ import annotations

from dataclasses import replace

from lifting53 import (
    InvalidTreeError,
    adaptive_decompose,
    build_full_tree,
    default_cost,
    inverse_packet,
    leaf_cost_total,
    prune_tree,
)


def make_image(h: int, w: int) -> list[list[int]]:
    """Deterministic signed image: smooth gradient left, busy texture right."""
    img = []
    for i in range(h):
        row = []
        for j in range(w):
            if j < w // 2:
                row.append((i + j) % 5 - 2)            # smooth -> compressible
            else:
                row.append(((i * 7 - j * 13) % 29) - 14)  # busy -> not
        img.append(row)
    return img


def show_tree(node, indent="") -> None:
    mark = "leaf" if node.is_leaf else "node"
    print(f"{indent}{node.band:>4} depth={node.depth} shape={node.shape[0]}x"
          f"{node.shape[1]:<3} origin={node.origin} [{mark}]")
    for child in node.children:
        show_tree(child, indent + "  ")


def main() -> None:
    img = make_image(13, 9)
    print("input image (13x9, signed integers):")
    for row in img:
        print("  " + " ".join(f"{v:4d}" for v in row))
    print()

    print("=== part 1: full tree to depth 3, pruned with the default")
    print("    explicit integer encoding-length cost (an estimate, not a")
    print("    real compressed file size) ===")
    full = build_full_tree(img, 3)
    pruned = prune_tree(full)  # default_cost
    show_tree(pruned)
    print()
    print(f"cost: root-only={default_cost(full.coefficients)}  "
          f"full-depth={leaf_cost_total(full)}  pruned={leaf_cost_total(pruned)}")
    leaves = pruned.leaves()
    print(f"leaves: {len(leaves)}; total leaf sample area = "
          f"{sum(l.shape[0] * l.shape[1] for l in leaves)} (image area 13*9=117)")
    rec = inverse_packet(pruned)
    assert rec == tuple(map(tuple, img))
    print("inverse_packet(pruned) reproduces the input bit-for-bit: OK")
    print()

    print("=== part 2: caller-supplied cost (8 bits per nonzero value plus")
    print("    its magnitude bits; zeros are free) ===")
    def custom_cost(coeffs):
        return sum((8 if v else 0) + abs(v).bit_length()
                   for row in coeffs for v in row)
    pruned2 = prune_tree(full, custom_cost)
    print(f"leaves with custom cost: {len(pruned2.leaves())}, "
          f"total={leaf_cost_total(pruned2, custom_cost)}")
    assert inverse_packet(pruned2) == tuple(map(tuple, img))
    print("inverse_packet(pruned2) reproduces the input bit-for-bit: OK")
    print()

    print("=== part 3: a structurally invalid tree is rejected ===")
    broken = replace(full, children=full.children[:3])  # HH child missing
    try:
        inverse_packet(broken)
    except InvalidTreeError as exc:
        print(f"InvalidTreeError raised as expected: {exc}")
    print()

    print("demo_iteration.py finished (all computations were performed live).")


if __name__ == "__main__":
    main()

