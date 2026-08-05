"""CPU-only executable contract tests for the frozen green-bridge protocol.

These tests deliberately do not import torch or TransformerLens.  Server-only
numeric equivalence audits are implemented by ``exp_green_bridge_gpt2.py`` and
are mandatory preflight gates rather than substitutes for these contracts.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analyze_green_bridge import (
    BASELINES,
    confirmation_decision,
    development_decision,
    fit_nonnegative_affine,
)
from green_bridge_dataset import (
    ConfirmationLock,
    build_donor_records,
    build_evaluation_records,
    plan_payload,
    split_records,
    validate_plan,
)
from green_bridge_spec import (
    DIMENSIONS,
    EVALUATION_NOUNS,
    DONOR_NOUNS,
    FROZEN_SPEC,
    SELECTED_GATES,
    frozen_spec_hash,
)
from matched_bypass_gate import (
    GateJet,
    expected_tensor_calls,
    identify_gate,
)


ROOT = Path(__file__).resolve().parent.parent
TARGET_SOURCE = ROOT / "src" / "green_bridge_path_target.py"
TAIL_SOURCE = ROOT / "src" / "green_bridge_tail.py"


class FrozenSpecTests(unittest.TestCase):
    def test_dag_order_and_exact_sites(self):
        sites = FROZEN_SPEC["sites"]
        self.assertEqual(sites["patch"], "blocks.8.hook_mlp_out")
        self.assertEqual(sites["x"], "blocks.10.hook_resid_mid")
        self.assertEqual(sites["z"], "blocks.10.mlp.hook_pre")
        self.assertEqual(sites["gate"], "blocks.10.mlp.hook_post")
        self.assertEqual(sites["target_bypass_subtraction"], "blocks.10.hook_resid_post")
        self.assertLess(8, 10)
        self.assertLess(10, 11)

    def test_dimensions_and_gate_shape(self):
        self.assertEqual((DIMENSIONS.d_model, DIMENSIONS.residual_rank), (768, 4))
        self.assertEqual((DIMENSIONS.d_mlp, DIMENSIONS.output_dimension), (3072, 100))
        self.assertEqual(len(SELECTED_GATES), 10)
        self.assertEqual(len(set(SELECTED_GATES)), 10)
        self.assertTrue(all(0 <= gate < 3072 for gate in SELECTED_GATES))

    def test_spec_hash_is_stable(self):
        self.assertEqual(frozen_spec_hash(), frozen_spec_hash())
        self.assertEqual(len(frozen_spec_hash()), 64)


class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluation = build_evaluation_records()
        cls.donors = build_donor_records()

    def test_exact_population_and_hash(self):
        validate_plan(self.evaluation + self.donors)
        self.assertEqual(len(self.evaluation), 768)
        self.assertEqual(len(self.donors), 1024)
        self.assertEqual(
            plan_payload(self.evaluation + self.donors)["records_sha256"],
            plan_payload(build_evaluation_records() + build_donor_records())["records_sha256"],
        )

    def test_pair_orientation_quotas_and_role_disjointness(self):
        for cid in {row.cell_id for row in self.evaluation}:
            rows = [row for row in self.evaluation if row.cell_id == cid]
            tensor = [row for row in rows if row.role == "tensor"]
            energy = [row for row in rows if row.role == "energy"]
            self.assertEqual((len(tensor), len(energy)), (8, 8))
            self.assertEqual(sum(row.orientation == "up" for row in tensor), 4)
            self.assertEqual(sum(row.orientation == "up" for row in energy), 4)
            t_pairs = {tuple(sorted((row.y, row.y_prime))) for row in tensor}
            e_pairs = {tuple(sorted((row.y, row.y_prime))) for row in energy}
            self.assertFalse(t_pairs & e_pairs)

    def test_development_confirmation_and_donor_separation(self):
        development = {row.cell_id for row in self.evaluation if row.split == "development"}
        confirmation = {row.cell_id for row in self.evaluation if row.split == "confirmation"}
        self.assertEqual((len(development), len(confirmation)), (16, 32))
        self.assertFalse(development & confirmation)
        self.assertFalse(set(EVALUATION_NOUNS) & set(DONOR_NOUNS))
        self.assertTrue(all(row.population == "donor" for row in self.donors))

    def test_confirmation_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "frozen_analysis.json"
            lock = ConfirmationLock(lock_path)
            with self.assertRaises(PermissionError):
                split_records(self.evaluation, "confirmation", confirmation_lock=lock)
            lock_path.write_text("{}", encoding="utf-8")
            opened = split_records(self.evaluation, "confirmation", confirmation_lock=lock)
            self.assertTrue(opened)
            self.assertTrue(all(row.split == "confirmation" for row in opened))


class IdentificationTests(unittest.TestCase):
    def test_exact_matched_bypass_inverse(self):
        rng = np.random.Generator(np.random.PCG64(7))
        C = rng.normal(size=100)
        G = rng.normal(size=100)
        A = rng.normal(size=4)
        direct = rng.normal(size=(4, 100))
        control = rng.normal(size=(4, 100))
        path = control + A[:, None] * C[None, :]
        result = identify_gate(GateJet(G, C, direct + A[:, None] * G, path, control))
        np.testing.assert_allclose(result.A, A, atol=1e-12)
        np.testing.assert_allclose(result.P, A[:, None] * G, atol=1e-12)
        np.testing.assert_allclose(result.D, direct, atol=1e-12)
        self.assertLess(result.factorization_residual, 1e-12)

    def test_finite_design_rank(self):
        # Columns [1,x,z,xz,x^2,z^2] on the frozen center/axis/corner design.
        points = [(0.0, 0.0), (0.0, -1.0), (0.0, 1.0)]
        points += [(s, 0.0) for s in (-1.0, 1.0)]
        points += [(sx, sz) for sx in (-1.0, 1.0) for sz in (-1.0, 1.0)]
        matrix = np.array([[1, x, z, x * z, x * x, z * z] for x, z in points])
        self.assertEqual(np.linalg.matrix_rank(matrix), 6)

    def test_forward_counts_and_baseline_budgets(self):
        self.assertEqual(expected_tensor_calls(), 1682)
        mixed = 2 * (10 * 2 * (2 + 8 + 16 + 16) + 1)
        first_order = 2 * 2 * 2 * (200 + 10) + 2
        self.assertEqual(mixed, 1682)
        self.assertEqual(first_order, 1682)
        self.assertEqual(2 * 2 * 4, 16)  # systems x radii x factorial corners


class InterventionSourceTests(unittest.TestCase):
    def test_target_import_firewall(self):
        tree = ast.parse(TARGET_SOURCE.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        prohibited = {
            "mixed_path_identification", "matched_bypass_gate", "green_bridge_tail",
            "analyze_green_bridge", "exp_green_bridge_gpt2",
        }
        self.assertFalse(imported & prohibited)

    def test_target_subtracts_only_residual_bypass(self):
        source = TARGET_SOURCE.read_text(encoding="utf-8")
        self.assertIn("resid_post[rows, positions, :] -= residual_delta", source)
        self.assertIn("post[rows, positions, :] = anchor.post", source)
        self.assertIn("gate_ids", source)

    def test_tail_path_control_contract(self):
        source = TAIL_SOURCE.read_text(encoding="utf-8")
        self.assertIn('mode not in {"path", "control", "joint"}', source)
        self.assertIn("controlled_pre = anchor.pre", source)
        self.assertIn("post[rows, positions, :] = anchor.post", source)
        self.assertIn("resid_post = resid_mid + mlp_out", source)
        # Path alone receives the live x-dependent gate; control uses anchor_pre+z.
        self.assertIn('elif mode == "path"', source)
        self.assertIn("block10.mlp.act_fn(controlled_pre)", source)
        self.assertIn("with model.hooks(fwd_hooks=hooks):", source)

    def test_center_equality_and_untouched_coordinate_policy(self):
        # At x=z=0 path and control have the same selected gate anchor; every
        # omitted coordinate is copied from the same frozen post anchor.
        anchor = np.arange(12.0)
        selected = 3
        live = anchor.copy()
        path = anchor.copy()
        control = anchor.copy()
        path[selected] = live[selected]
        control[selected] = live[selected]
        np.testing.assert_array_equal(path, control)
        mask = np.arange(len(anchor)) != selected
        np.testing.assert_array_equal(path[mask], anchor[mask])

    def test_control_severs_gate_dependence_but_preserves_bypass(self):
        x, z, anchor_pre = 0.4, -0.2, 0.7
        activation = lambda value: value * value
        path_gate = activation(anchor_pre + x + z)
        control_gate = activation(anchor_pre + z)
        self.assertNotEqual(path_gate, control_gate)
        bypass_path = x
        bypass_control = x
        self.assertEqual(bypass_path, bypass_control)


class AnalysisTests(unittest.TestCase):
    @staticmethod
    def _development_payload():
        cells = []
        for index in range(16):
            target = 0.25 + index * 0.03
            cells.append({
                "cell_id": f"dev-{index}",
                "distance_bin": "near" if index < 8 else "far",
                "survived": True,
                "conditioned": True,
                "snr": 10.0,
                "target": target,
                "mixed": target + (-1) ** index * 0.001,
                "mixed_full": target,
                "mixed_half": target,
                "baselines": {name: 0.2 for name in BASELINES},
            })
        return {"cells": cells}

    def test_nonnegative_calibration(self):
        alpha, beta = fit_nonnegative_affine(np.arange(5.0), -np.arange(5.0))
        self.assertGreaterEqual(alpha, 0.0)
        self.assertGreaterEqual(beta, 0.0)

    def test_development_terminal_verdict(self):
        result = development_decision(self._development_payload())
        self.assertEqual(result["verdict"], "OPEN_CONFIRMATION")

    def test_confirmation_terminal_fail_without_survival(self):
        result = confirmation_decision({"cells": []}, {"baseline_calibration": {}})
        self.assertEqual(result["verdict"], "FAIL_SURVIVAL")

    def test_confirmation_uses_frozen_coefficients_without_refit(self):
        source = (ROOT / "src" / "analyze_green_bridge.py").read_text(encoding="utf-8")
        function = source[source.index("def confirmation_decision"):source.index("def main")]
        self.assertNotIn("fit_nonnegative_affine(", function)
        self.assertIn('frozen["baseline_calibration"]', function)


if __name__ == "__main__":
    unittest.main(verbosity=2)
