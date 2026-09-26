"""Tests for the adaptive quadtree decomposition (lifting53._adaptive).

Three independent verification angles, per the task contract:

* an independent lifting reference (implemented here from the spec, not by
  calling the package under test) checks per-level coefficients;
* small trees are checked against an exhaustive enumeration of all pruning
  choices to prove cost optimality (and the shallow-on-tie rule);
* exact roundtrip is used only as a supplementary acceptance check.
"""
from __future__ import annotations

import unittest
from itertools import product

from lifting53 import (
    INT63_MAX,
    AdaptiveNode,
    AdaptiveResult,
    InvalidTreeError,
    dwt2d_adaptive,
    idwt2d_adaptive,
    integer_coding_length,
)
from lifting53 import max_levels
from lifting53._adaptive import _build, _quadrant_shapes


# ---------------------------------------------------------------------------
# Independent reference implementation (from the written spec only)
# ---------------------------------------------------------------------------

def ref_fwd1d(x):
    n = len(x)
    if n == 1:
        return list(x)
    ev = x[0::2]
    od = x[1::2]
    detail = []
    for j, o in enumerate(od):
        left = ev[j]
        right = ev[j + 1] if j + 1 < len(ev) else ev[j]
        detail.append(o - (left + right) // 2)
    approx = []
    for k, e in enumerate(ev):
        dl = detail[k - 1] if k >= 1 else detail[0]
        dr = detail[k] if k < len(detail) else detail[-1]
        approx.append(e + (dl + dr + 2) // 4)
    return approx + detail


def ref_fwd_block(block):
    h, w = len(block), len(block[0])
    a = [ref_fwd1d(row) for row in block]
    for c in range(w):
        col = ref_fwd1d([a[r][c] for r in range(h)])
        for r in range(h):
            a[r][c] = col[r]
    return a


def ref_tree(block, depth, max_depth):
    """Reference quadtree: {coeffs, children?} built independently."""
    h, w = len(block), len(block[0]) if block else 0
    node = {"coeffs": [list(row) for row in block]}
    if depth < max_depth and h >= 1 and w >= 1 and (h > 1 or w > 1):
        t = ref_fwd_block(block)
        lh, lw = (h + 1) // 2, (w + 1) // 2
        quads = {
            "LL": [row[:lw] for row in t[:lh]],
            "LH": [row[lw:] for row in t[:lh]],
            "HL": [row[:lw] for row in t[lh:]],
            "HH": [row[lw:] for row in t[lh:]],
        }
        node["children"] = {
            name: ref_tree(quads[name], depth + 1, max_depth)
            for name in ("LL", "LH", "HL", "HH")
        }
    return node


def sample_image(h, w, seed=3):
    vals = []
    state = seed
    for _ in range(h * w):
        state = (state * 1103515245 + 12345) % (2**31)
        vals.append(state % 61 - 30)
    return [vals[i * w:(i + 1) * w] for i in range(h)]


def full_tree(image, depth):
    block = tuple(tuple(row) for row in image)
    return _build(block, len(block), len(block[0]), "ROOT", 0, depth)


def node_coeffs(node):
    return [list(row) for row in node.coefficients]


class IndependentReferenceTest(unittest.TestCase):
    """Per-level coefficients must match the independent reference tree."""

    def check_subtree(self, node, ref):
        self.assertEqual(node_coeffs(node), ref["coeffs"])
        if "children" not in ref:
            self.assertIsNone(node.children)
            return
        self.assertIsNotNone(node.children)
        for child, name in zip(node.children, ("LL", "LH", "HL", "HH")):
            self.assertEqual(child.band, name)
            self.check_subtree(child, ref["children"][name])

    def test_depth1_coefficients(self):
        img = sample_image(5, 7)
        tree = full_tree(img, 1)
        self.check_subtree(tree, ref_tree(img, 0, 1))

    def test_depth3_all_subbands_recurse(self):
        img = sample_image(7, 5)
        tree = full_tree(img, 3)
        self.check_subtree(tree, ref_tree(img, 0, 3))
        # The reference really decomposes non-LL subbands: some depth-2 node
        # under an LH/HL/HH parent must itself have children.
        def find_split_high(node):
            for child in node.children or ():
                if child.band != "LL" and child.children:
                    return True
                if find_split_high(child):
                    return True
            return False
        self.assertTrue(find_split_high(tree))

    def test_odd_and_degenerate_shapes(self):
        for h, w in [(1, 7), (9, 1), (3, 3), (1, 1), (2, 5)]:
            img = sample_image(h, w)
            depth = min(3, max(1, max_levels(h, w)))
            tree = full_tree(img, depth)
            self.check_subtree(tree, ref_tree(img, 0, depth))


# ---------------------------------------------------------------------------
# Exhaustive-enumeration optimality
# ---------------------------------------------------------------------------

def internal_nodes(node):
    out = []
    def walk(n):
        if n.children:
            out.append(n)
            for c in n.children:
                walk(c)
    walk(node)
    return out


def enum_best_cost(root, cost):
    """Min total cost over all valid keep/expand assignments (brute force)."""
    internals = internal_nodes(root)
    best = None
    for choices in product([False, True], repeat=len(internals)):
        decision = {id(n): c for n, c in zip(internals, choices)}
        total = 0
        def walk(n, ancestor_expanded):
            nonlocal total
            if not ancestor_expanded:
                return  # pruned away by an ancestor
            if n.children and decision[id(n)]:
                for c in n.children:
                    walk(c, True)
            else:
                total += cost(n.coefficients)
        walk(root, True)
        if best is None or total < best:
            best = total
    return best


def optimal_subtree_costs(node, cost, memo):
    if node.children is None:
        return cost(node.coefficients)
    expanded = sum(optimal_subtree_costs(c, cost, memo) for c in node.children)
    return min(cost(node.coefficients), expanded)


class PruningOptimalityTest(unittest.TestCase):
    def check_optimal(self, img, depth, cost):
        res = dwt2d_adaptive(img, depth, cost=cost)
        root = full_tree(img, depth)
        self.assertEqual(res.total_cost, enum_best_cost(root, cost))
        # Shallow-on-tie: every internal node that was kept expanded must be
        # strictly cheaper expanded; every kept leaf with children in the
        # full tree must not be strictly cheaper expanded.
        def opt(n):
            return optimal_subtree_costs(n, cost, {})
        def walk(pruned, full):
            if pruned.children is None:
                if full.children is not None:
                    keep = cost(full.coefficients)
                    expanded = sum(opt(c) for c in full.children)
                    self.assertLessEqual(keep, expanded)
                return
            keep = cost(full.coefficients)
            expanded = sum(opt(c) for c in full.children)
            self.assertLess(expanded, keep)
            for pc, fc in zip(pruned.children, full.children):
                walk(pc, fc)
        walk(res.root, root)
        # total_cost equals the sum over the reported leaves
        self.assertEqual(
            res.total_cost, sum(cost(l.coefficients) for l in res.leaves())
        )

    def test_default_cost_small_trees(self):
        for h, w, depth in [(4, 4, 2), (3, 4, 2), (5, 3, 2), (4, 3, 2)]:
            img = sample_image(h, w)
            self.check_optimal(img, depth, integer_coding_length)

    def test_custom_cost_small_tree(self):
        img = sample_image(4, 4)
        cost = lambda block: sum(abs(v) for row in block for v in row) + 7
        self.check_optimal(img, 2, cost)

    def test_tie_prefers_shallower(self):
        # Constant cost: keeping any parent ties/beats expanding, so the
        # whole tree must collapse to the root.
        res = dwt2d_adaptive(sample_image(4, 4), 2, cost=lambda block: 5)
        self.assertTrue(res.root.is_leaf)
        self.assertEqual(res.total_cost, 5)

    def test_expansion_when_children_free(self):
        # 1x1 blocks cost nothing, everything else is expensive: a 4x4 image
        # at depth 2 must expand all the way to sixteen 1x1 leaves.
        cost = lambda block: 0 if block == ((0,),) or True else 0
        def real_cost(block):
            if len(block) == 1 and len(block[0]) == 1:
                return 0
            return 1000
        res = dwt2d_adaptive(sample_image(4, 4), 2, cost=real_cost)
        self.assertEqual(res.total_cost, 0)
        self.assertEqual(len(res.leaves()), 16)
        self.assertTrue(all(l.shape == (1, 1) for l in res.leaves()))


# ---------------------------------------------------------------------------
# Roundtrip (supplementary acceptance only)
# ---------------------------------------------------------------------------

class RoundtripTest(unittest.TestCase):
    def check(self, img, depth):
        res = dwt2d_adaptive(img, depth)
        back = idwt2d_adaptive(res)
        self.assertEqual([list(r) for r in back], [list(r) for r in img])

    def test_various_shapes(self):
        for h, w in [(1, 1), (1, 8), (8, 1), (3, 3), (7, 5), (8, 8), (9, 17)]:
            img = sample_image(h, w)
            depth = min(3, max(1, max_levels(h, w)))
            self.check(img, depth)

    def test_negative_and_extreme_values(self):
        big = INT63_MAX // 4
        img = [[-big, big, -1], [0, big, -big]]
        self.check(img, 1)

    def test_overflow_still_rejected(self):
        from lifting53 import IntegerOverflowError
        img = [[-INT63_MAX, -INT63_MAX], [0, 1]]
        with self.assertRaises(IntegerOverflowError):
            dwt2d_adaptive(img, 1)

    def test_input_not_mutated(self):
        img = sample_image(6, 6)
        snapshot = [list(r) for r in img]
        dwt2d_adaptive(img, 2)
        self.assertEqual(img, snapshot)


# ---------------------------------------------------------------------------
# Invalid trees must be rejected before reconstruction
# ---------------------------------------------------------------------------

def valid_result():
    # Cost forces full expansion so the tree has internal nodes to corrupt.
    expand_all = lambda b: 0 if (len(b) == 1 and len(b[0]) == 1) else 10**6
    return dwt2d_adaptive(sample_image(4, 4), 2, cost=expand_all)


class InvalidTreeTest(unittest.TestCase):
    def assert_rejected(self, result):
        with self.assertRaises(InvalidTreeError):
            idwt2d_adaptive(result)

    def replace(self, res, root):
        return AdaptiveResult(root, res.shape, res.depth, 0, "test")

    def test_missing_child(self):
        res = valid_result()
        root = res.root
        self.assertIsNotNone(root.children)
        broken = AdaptiveNode(root.band, root.depth, root.shape,
                              root.coefficients, root.children[:3])
        self.assert_rejected(self.replace(res, broken))

    def test_overlapping_child_shape(self):
        res = valid_result()
        root = res.root
        kids = list(root.children)
        bad = AdaptiveNode(kids[0].band, kids[0].depth,
                           (kids[0].shape[0] + 1, kids[0].shape[1]),
                           kids[0].coefficients, kids[0].children)
        kids[0] = bad
        broken = AdaptiveNode(root.band, root.depth, root.shape,
                              root.coefficients, tuple(kids))
        self.assert_rejected(self.replace(res, broken))

    def test_wrong_band_order(self):
        res = valid_result()
        root = res.root
        kids = list(root.children)
        kids[0], kids[1] = kids[1], kids[0]
        broken = AdaptiveNode(root.band, root.depth, root.shape,
                              root.coefficients, tuple(kids))
        self.assert_rejected(self.replace(res, broken))

    def test_leaf_coefficients_wrong_shape(self):
        res = valid_result()
        root = res.root
        self.assertTrue(root.is_leaf or root.children)
        # Force root into a leaf with a wrong coefficient row count.
        broken = AdaptiveNode("ROOT", 0, res.shape, root.coefficients[:2], None)
        self.assert_rejected(self.replace(res, broken))

    def test_internal_1x1_rejected(self):
        res = valid_result()
        leaf = AdaptiveNode("LL", 1, (1, 1), ((0,),), None)
        inner = AdaptiveNode("ROOT", 0, (1, 1), ((0,),),
                             (leaf, leaf, leaf, leaf))
        self.assert_rejected(AdaptiveResult(inner, (1, 1), 1, 0, "test"))

    def test_depth_mismatch_rejected(self):
        res = valid_result()
        root = res.root
        kid = root.children[0]
        bad = AdaptiveNode(kid.band, kid.depth + 5, kid.shape,
                           kid.coefficients, kid.children)
        kids = (bad,) + tuple(root.children[1:])
        broken = AdaptiveNode(root.band, root.depth, root.shape,
                              root.coefficients, kids)
        self.assert_rejected(self.replace(res, broken))

    def test_non_tree_rejected(self):
        with self.assertRaises(TypeError):
            idwt2d_adaptive(object())
        res = valid_result()
        with self.assertRaises(TypeError):
            idwt2d_adaptive(self.replace(res, root="not a node"))

    def test_valid_tree_still_accepted(self):
        res = valid_result()
        back = idwt2d_adaptive(res)
        self.assertEqual(len(back), 4)


# ---------------------------------------------------------------------------
# Parameter and cost-function validation
# ---------------------------------------------------------------------------

class ParameterValidationTest(unittest.TestCase):
    def test_depth_validation(self):
        img = sample_image(4, 4)
        for bad in [0, -1, True, 2.0, "2"]:
            with self.assertRaises((TypeError, ValueError)):
                dwt2d_adaptive(img, bad)
        with self.assertRaises(ValueError):
            dwt2d_adaptive(img, 100)

    def test_cost_not_callable(self):
        with self.assertRaises(TypeError):
            dwt2d_adaptive(sample_image(4, 4), 1, cost=42)

    def test_cost_return_validation(self):
        img = sample_image(4, 4)
        with self.assertRaises(TypeError):
            dwt2d_adaptive(img, 1, cost=lambda b: True)
        with self.assertRaises(TypeError):
            dwt2d_adaptive(img, 1, cost=lambda b: 1.5)
        with self.assertRaises(ValueError):
            dwt2d_adaptive(img, 1, cost=lambda b: -1)

    def test_default_cost_is_explicit_estimate(self):
        self.assertEqual(integer_coding_length(((0,),)), 1)
        self.assertEqual(integer_coding_length(((1, -2), (7, -8))), 2 + 3 + 4 + 5)
        self.assertEqual(integer_coding_length(()), 0)


if __name__ == "__main__":
    unittest.main()
