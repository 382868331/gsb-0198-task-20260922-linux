"""Offline demo of the adaptive quadtree 5/3 lifting decomposition.

Real computation on deterministic data only: no network, no accounts, no
keys, no sleep, no prerecorded results.  Runs in well under 90 seconds
(typically a few milliseconds).
"""
from __future__ import annotations

import time

from lifting53 import (
    AdaptiveNode,
    AdaptiveResult,
    InvalidTreeError,
    dwt2d_adaptive,
    idwt2d_adaptive,
)


def make_image(h: int, w: int) -> list[list[int]]:
    """Deterministic integer image: smooth background + textured region."""
    img = []
    for r in range(h):
        row = []
        for c in range(w):
            base = (3 * r + 5 * c) % 23 - 11
            texture = ((r * c + r * r) % 7 - 3) if 8 <= r < 24 and 8 <= c < 24 else 0
            row.append(base + texture)
        img.append(row)
    return img


def render_tree(node: AdaptiveNode, indent: str = "") -> str:
    kept = "KEPT leaf" if node.is_leaf else "expanded"
    n = node.shape[0] * node.shape[1]
    line = f"{indent}{node.band} depth={node.depth} shape={node.shape} [{kept}, {n} coeff(s)]"
    parts = [line]
    for child in node.children or ():
        parts.append(render_tree(child, indent + "  "))
    return "\n".join(parts)


def main() -> None:
    started = time.perf_counter()

    print("=== 1. Adaptive decomposition with the default cost ===")
    image = make_image(32, 32)
    result = dwt2d_adaptive(image, depth=4)
    print(f"image: 32x32, requested depth: {result.depth}")
    print(f"cost model: {result.cost_description}")
    print(f"total cost of pruned representation: {result.total_cost}")
    leaves = result.leaves()
    print(f"kept leaves: {len(leaves)}")
    for leaf in leaves[:8]:
        print(f"  leaf {leaf.band} depth={leaf.depth} shape={leaf.shape}")
    if len(leaves) > 8:
        print(f"  ... and {len(leaves) - 8} more")

    print()
    print("=== 2. Exact inverse from the tree ===")
    reconstructed = idwt2d_adaptive(result)
    ok = [list(r) for r in reconstructed] == image
    print(f"bit-exact reconstruction: {ok}")
    assert ok

    print()
    print("=== 3. Odd shape and a custom cost ===")
    odd = make_image(13, 9)
    per_block = lambda block: 1 + sum(
        abs(v).bit_length() + 1 for row in block for v in row
    )
    odd_result = dwt2d_adaptive(odd, depth=3, cost=per_block)
    odd_back = idwt2d_adaptive(odd_result)
    print(f"13x9 image, depth 3, custom cost: total={odd_result.total_cost}, "
          f"leaves={len(odd_result.leaves())}, "
          f"exact={[list(r) for r in odd_back] == odd}")

    print()
    print("=== 4. Invalid trees are rejected before reconstruction ===")
    root = result.root
    if root.children:
        broken = AdaptiveNode(root.band, root.depth, root.shape,
                              root.coefficients, root.children[:3])
        try:
            idwt2d_adaptive(
                AdaptiveResult(broken, result.shape, result.depth, 0, "demo")
            )
            print("ERROR: broken tree was accepted")
        except InvalidTreeError as exc:
            print(f"missing subblock rejected: {exc}")
    else:
        print("(root was kept as a leaf; building a synthetic broken tree)")
        leaf = AdaptiveNode("LL", 1, (1, 1), ((0,),), None)
        broken = AdaptiveNode("ROOT", 0, (2, 2), ((0, 0), (0, 0)), (leaf,))
        try:
            idwt2d_adaptive(AdaptiveResult(broken, (2, 2), 1, 0, "demo"))
            print("ERROR: broken tree was accepted")
        except InvalidTreeError as exc:
            print(f"missing subblock rejected: {exc}")

    elapsed = time.perf_counter() - started
    print()
    print(f"demo completed in {elapsed:.3f}s (budget: 90s)")


if __name__ == "__main__":
    main()
