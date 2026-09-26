"""Independent tests for the adaptive quad-tree (packet) decomposition.

Expected values come from sources that never call the package under test:

1. Hand-computed constants (default cost, worked pruning examples).
2. A spec-direct reference tree builder written independently in this file
   (index/clamping formulation instead of the implementation's slicing one).
3. Exhaustive enumeration of every pruning of small trees to check that the
   package's pruned tree is cost-optimal and shallowest among optima.

Exact forward-then-inverse round trips are only a supplementary check.
"""
from __future__ import annotations

import math
import unittest
from dataclasses import replace

from lifting53 import (
    CHILD_BANDS,
    IntegerLiftingError,
    InvalidTreeError,
    PacketNode,
    adaptive_decompose,
    build_full_tree,
    default_cost,
    inverse_packet,
    leaf_cost_total,
    prune_tree,
    validate_tree,
)


# ---------------------------------------------------------------------------
# Independent reference (spec-direct; does NOT import lifting53._core)
# ---------------------------------------------------------------------------

def _clamp(i, n):
    return 0 if i < 0 else (n - 1 if i >= n else i)


def ref_fwd1d(x):
    n = len(x)
    if n == 1:
        return [x[0]]
    even = [x[2 * k] for k in range((n + 1) // 2)]
    odd = [x[2 * k + 1] for k in range(n // 2)]
    detail = []
    for j, o in enumerate(odd):
        el = even[_clamp(j, len(even))]
        er = even[_clamp(j + 1, len(even))]
        detail.append(o - math.floor((el + er) / 2))
    approx = []
    for k, e in enumerate(even):
        dl = detail[_clamp(k - 1, len(detail))]
        dr = detail[_clamp(k, len(detail))]
        approx.append(e + math.floor((dl + dr + 2) / 4))
    return approx + detail


def ref_split(block):
    """One spec-direct 2D step: rows then columns; returns {band: block}."""
    h = len(block)
    w = len(block[0]) if h else 0
    a = [list(row) for row in block]
    for r in range(h):
        a[r] = ref_fwd1d(a[r])
    for c in range(w):
        col = ref_fwd1d([a[r][c] for r in range(h)])
        for r in range(h):
            a[r][c] = col[r]
    lr, lc = (h + 1) // 2, (w + 1) // 2
    return {
        "LL": [row[:lc] for row in a[:lr]],
        "LH": [row[lc:] for row in a[:lr]],
        "HL": [row[:lc] for row in a[lr:]],
        "HH": [row[lc:] for row in a[lr:]],
    }


def ref_tree(band, depth, origin, block, max_depth, shape=None):
    """Reference full tree as nested dicts {band, depth, origin, shape, coeffs, kids}."""
    if shape is None:
        shape = (len(block), len(block[0]))
    h, w = shape
    node = {
        "band": band,
        "depth": depth,
        "origin": origin,
        "shape": shape,
        "coeffs": tuple(tuple(row) for row in block) if h and w else (),
        "kids": {},
    }
    if depth < max_depth and h >= 1 and w >= 1 and (h > 1 or w > 1):
        sub = ref_split(block)
        lr, lc = (h + 1) // 2, (w + 1) // 2
        offsets = {"LL": (0, 0), "LH": (0, lc), "HL": (lr, 0), "HH": (lr, lc)}
        shapes = {"LL": (lr, lc), "LH": (lr, w - lc),
                  "HL": (h - lr, lc), "HH": (h - lr, w - lc)}
        for name in ("LL", "LH", "HL", "HH"):
            dr, dc = offsets[name]
            node["kids"][name] = ref_tree(
                name, depth + 1, (origin[0] + dr, origin[1] + dc),
                sub[name], max_depth, shapes[name],
            )
    return node


def make_image(h, w, seed=0):
    return [
        [((i * 7 - j * 5 + seed * 3) % 23) - 11 for j in range(w)]
        for i in range(h)
    ]


def check_against_ref(testcase, pkg_node, ref_node):
    testcase.assertEqual(pkg_node.band, ref_node["band"])
    testcase.assertEqual(pkg_node.depth, ref_node["depth"])
    testcase.assertEqual(pkg_node.origin, ref_node["origin"])
    testcase.assertEqual(pkg_node.shape, ref_node["shape"])
    testcase.assertEqual(pkg_node.coefficients, ref_node["coeffs"])
    testcase.assertEqual(
        tuple(c.band for c in pkg_node.children), tuple(ref_node["kids"])
    )
    for child in pkg_node.children:
        check_against_ref(testcase, child, ref_node["kids"][child.band])


# ---------------------------------------------------------------------------
# Pruning enumeration (independent optimality oracle)
# ---------------------------------------------------------------------------

def enum_best(node, cost):
    """Return (best_cost, frozenset-of-leaf-ids) preferring shallow on ties.

    A node is identified by (depth, origin).  Enumerates keep vs expand at
    every internal node, i.e. every legal pruning of the full tree.
    """
    ident = (node.depth, node.origin)
    keep = cost(node.coefficients)
    if node.is_leaf:
        return keep, frozenset({ident})
    expand = 0
    leaves = frozenset()
    for child in node.children:
        c_cost, c_leaves = enum_best(child, cost)
        expand += c_cost
        leaves |= c_leaves
    if keep <= expand:  # tie -> shallower representation
        return keep, frozenset({ident})
    return expand, leaves


def pseudo_cost(salt):
    """Deterministic content-dependent cost, independent of the package."""
    def cost(coeffs):
        total = 0
        for i, row in enumerate(coeffs):
            for j, v in enumerate(row):
                total += ((v + 9) * (i + 2 * j + 1) + salt) % 17 + 1
        return total
    return cost


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class DefaultCostHandComputed(unittest.TestCase):
    def test_known_values(self):
        # 0 -> 1 bit; +-1 -> 2 bits; 2, -2 -> 3 bits; 7 -> 4 bits
        self.assertEqual(default_cost(((0, 1, -1, 2, -2, 7),)), 1 + 2 + 2 + 3 + 3 + 4)
        self.assertEqual(default_cost(()), 0)
        self.assertEqual(default_cost(((0, 0), (0, 0))), 4)

    def test_sign_symmetry(self):
        self.assertEqual(default_cost(((5, -5),)), 2 * (1 + 3))


class FullTreeVsReference(unittest.TestCase):
    def test_coefficients_match_reference_every_level(self):
        for h, w, depth in [(7, 5, 3), (8, 8, 3), (1, 9, 4), (6, 1, 3),
                            (5, 5, 2), (3, 3, 5)]:
            with self.subTest(shape=(h, w), depth=depth):
                img = make_image(h, w, seed=h + w)
                pkg = build_full_tree(img, depth)
                ref = ref_tree("ROOT", 0, (0, 0), img, depth)
                check_against_ref(self, pkg, ref)

    def test_all_four_subbands_decompose(self):
        # A non-LL subband must be decomposable too: with depth 2 the tree
        # has 1 + 4 + 16 nodes (shape permitting).
        tree = build_full_tree(make_image(8, 8), 2)
        self.assertEqual(len(tree.children), 4)
        for child in tree.children:
            self.assertEqual(len(child.children), 4, child.band)


class PruningOptimality(unittest.TestCase):
    def test_matches_full_enumeration(self):
        for h, w, depth in [(4, 4, 2), (5, 3, 2), (4, 4, 3)]:
            for salt in (0, 5, 11):
                with self.subTest(shape=(h, w), depth=depth, salt=salt):
                    img = make_image(h, w, seed=salt)
                    full = build_full_tree(img, depth)
                    cost = pseudo_cost(salt)
                    pruned = prune_tree(full, cost)
                    best_cost, best_leaves = enum_best(full, cost)
                    self.assertEqual(leaf_cost_total(pruned, cost), best_cost)
                    got = frozenset(
                        (l.depth, l.origin) for l in pruned.leaves()
                    )
                    self.assertEqual(got, best_leaves)

    def test_default_cost_never_worse_than_root_or_full(self):
        img = make_image(9, 7, seed=2)
        full = build_full_tree(img, 3)
        pruned = prune_tree(full)
        self.assertLessEqual(leaf_cost_total(pruned), leaf_cost_total(full))
        self.assertLessEqual(leaf_cost_total(pruned), default_cost(full.coefficients))

    def test_tie_prefers_shallow(self):
        # cost = number of coefficients: keep == expand at every node, so
        # the optimal shallowest tree is the root alone.
        img = make_image(6, 6)
        full = build_full_tree(img, 3)
        count_cost = lambda coeffs: sum(len(row) for row in coeffs)
        pruned = prune_tree(full, count_cost)
        self.assertTrue(pruned.is_leaf)
        self.assertEqual(pruned.band, "ROOT")

    def test_expand_when_children_cheaper(self):
        # cost = 0 for every non-root block, huge for the root: the root
        # must be expanded.
        img = make_image(4, 4)
        full = build_full_tree(img, 2)
        def cost(coeffs):
            n = sum(len(row) for row in coeffs)
            return 10 ** 9 if n == 16 else 0
        pruned = prune_tree(full, cost)
        self.assertFalse(pruned.is_leaf)
        self.assertEqual(leaf_cost_total(pruned, cost), 0)

    def test_bad_cost_values_rejected(self):
        full = build_full_tree(make_image(3, 3), 1)
        for bad in (-1, float("nan"), float("inf"), "x", True):
            with self.subTest(bad=bad):
                with self.assertRaises((TypeError, ValueError)):
                    prune_tree(full, lambda coeffs: bad)

    def test_prune_does_not_mutate_input(self):
        full = build_full_tree(make_image(5, 5), 2)
        before = repr(full)
        prune_tree(full)
        self.assertEqual(repr(full), before)


class InvalidTreeRejection(unittest.TestCase):
    def setUp(self):
        self.img = make_image(6, 6)
        self.full = build_full_tree(self.img, 2)

    def test_missing_child(self):
        bad = replace(self.full, children=self.full.children[:3])
        with self.assertRaises(InvalidTreeError):
            inverse_packet(bad)

    def test_overlapping_leaves_wrong_shape(self):
        ll = self.full.children[0]
        # LL declares the wrong shape: its leaves would overlap the LH block.
        bad_ll = replace(ll, shape=(ll.shape[0] + 1, ll.shape[1]))
        bad = replace(self.full, children=(bad_ll,) + self.full.children[1:])
        with self.assertRaises(InvalidTreeError):
            inverse_packet(bad)

    def test_overlapping_leaves_wrong_origin(self):
        lh = self.full.children[1]
        bad_lh = replace(lh, origin=(0, 0))  # collides with the LL block
        bad = replace(self.full, children=(self.full.children[0], bad_lh)
                      + self.full.children[2:])
        with self.assertRaises(InvalidTreeError):
            inverse_packet(bad)

    def test_swapped_bands(self):
        kids = list(self.full.children)
        kids[0], kids[1] = kids[1], kids[0]
        with self.assertRaises(InvalidTreeError):
            inverse_packet(replace(self.full, children=tuple(kids)))

    def test_coefficient_shape_mismatch(self):
        ll = self.full.children[0]
        bad_ll = replace(ll, coefficients=ll.coefficients[:-1])
        bad = replace(self.full, children=(bad_ll,) + self.full.children[1:])
        with self.assertRaises(InvalidTreeError):
            inverse_packet(bad)

    def test_non_integer_coefficient(self):
        ll = self.full.children[0]
        row0 = (1.5,) + ll.coefficients[0][1:]
        bad_ll = replace(ll, coefficients=(row0,) + ll.coefficients[1:])
        bad = replace(self.full, children=(bad_ll,) + self.full.children[1:])
        with self.assertRaises(TypeError):
            inverse_packet(bad)

    def test_bool_coefficient(self):
        ll = self.full.children[0]
        row0 = (True,) + ll.coefficients[0][1:]
        bad_ll = replace(ll, coefficients=(row0,) + ll.coefficients[1:])
        bad = replace(self.full, children=(bad_ll,) + self.full.children[1:])
        with self.assertRaises(TypeError):
            inverse_packet(bad)

    def test_bad_root(self):
        with self.assertRaises(InvalidTreeError):
            inverse_packet(replace(self.full, band="LL"))
        with self.assertRaises(InvalidTreeError):
            inverse_packet("not a tree")

    def test_children_on_undecomposable_block(self):
        leaf = self.full.children[0].children[0]  # a leaf node
        fake = PacketNode("ROOT", 0, (0, 0), (1, 1), ((5,),),
                          children=(leaf, leaf, leaf, leaf))
        with self.assertRaises(InvalidTreeError):
            validate_tree(fake)

    def test_invalid_tree_error_is_package_error(self):
        self.assertTrue(issubclass(InvalidTreeError, IntegerLiftingError))


class EdgeShapes(unittest.TestCase):
    def test_single_row_column_and_pixel(self):
        cases = [
            ([[3, 1, 4, 1, 5, 9, 2]], 4),
            ([[3], [1], [4], [1], [5]], 3),
            ([[42]], 3),
            ([[-7]], 1),
        ]
        for img, depth in cases:
            with self.subTest(img=img):
                tree = adaptive_decompose(img, depth)
                self.assertEqual(
                    inverse_packet(tree), tuple(map(tuple, img))
                )

    def test_no_padding_artifacts(self):
        # Odd sizes: leaf shapes must tile exactly the image area, nothing
        # padded, nothing cropped.
        img = make_image(7, 5)
        tree = adaptive_decompose(img, 3)
        area = sum(l.shape[0] * l.shape[1] for l in tree.leaves())
        self.assertEqual(area, 35)
        self.assertEqual(tree.shape, (7, 5))

    def test_depth_validation(self):
        with self.assertRaises(ValueError):
            build_full_tree(make_image(3, 3), 0)
        with self.assertRaises(TypeError):
            build_full_tree(make_image(3, 3), True)
        with self.assertRaises(TypeError):
            build_full_tree(make_image(3, 3), 2.0)


class SupplementaryRoundTrips(unittest.TestCase):
    def test_round_trip_many_shapes_and_costs(self):
        costs = [None, pseudo_cost(1), pseudo_cost(7)]
        for h, w in [(1, 1), (1, 8), (8, 1), (3, 3), (7, 5), (8, 8), (13, 11)]:
            img = make_image(h, w, seed=h * w)
            for depth in (1, 2, 4):
                full = build_full_tree(img, depth)
                for cost in costs:
                    with self.subTest(shape=(h, w), depth=depth, cost=cost):
                        pruned = prune_tree(full, cost)
                        self.assertEqual(
                            inverse_packet(pruned), tuple(map(tuple, img))
                        )

    def test_unpruned_full_tree_round_trip(self):
        img = make_image(9, 6, seed=4)
        self.assertEqual(
            inverse_packet(build_full_tree(img, 3)), tuple(map(tuple, img))
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

