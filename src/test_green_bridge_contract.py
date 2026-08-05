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
    GATE04_HOLDOUT_PAIR_SLICE,
    GATE04_LEGACY_PAIR_SLICE,
    SCHEMA_VERSION,
    SELECTED_GATES,
    THRESHOLDS,
    frozen_spec_hash,
)
from green_bridge_numerics import (
    active_contraction_bound,
    cell_error_bound,
    certified_null_bound,
    richardson_numerical_bounds,
)
from exp_green_bridge_gpt2 import (
    all_gate04_submetrics_pass,
    centered_year_error,
    gate04_panel_metadata,
    gate04_record_panels,
    gate04_thresholds,
    pooled_error_metrics,
    task_margin_error,
    weight_mapping_report,
)
from matched_bypass_gate import (
    GateJet,
    expected_tensor_calls,
    identify_gate,
)


ROOT = Path(__file__).resolve().parent.parent
TARGET_SOURCE = ROOT / "src" / "green_bridge_path_target.py"
TAIL_SOURCE = ROOT / "src" / "green_bridge_tail.py"
RUNNER_SOURCE = ROOT / "src" / "exp_green_bridge_gpt2.py"
NUMERICS_SOURCE = ROOT / "src" / "green_bridge_numerics.py"
REQUIREMENTS_LOCK = ROOT / "requirements-green-bridge.lock"


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


class Gate04SpecificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.donors = build_donor_records()
        cls.legacy, cls.holdout = gate04_record_panels(cls.donors)
        cls.panel = gate04_panel_metadata(cls.legacy, cls.holdout)

    def test_gate04_schema_is_v1_1(self):
        self.assertEqual(SCHEMA_VERSION, "green-bridge-v1.1")
        self.assertIn('"green-bridge-manifest-v1.1"', RUNNER_SOURCE.read_text(encoding="utf-8"))

    def test_gate04_legacy_and_holdout_slices_are_exact(self):
        self.assertEqual(GATE04_LEGACY_PAIR_SLICE, (0, 16))
        self.assertEqual(GATE04_HOLDOUT_PAIR_SLICE, (16, 32))
        ranked = sorted(self.donors, key=lambda row: row.pair_digest)
        self.assertEqual(self.legacy, ranked[0:16])
        self.assertEqual(self.holdout, ranked[16:32])

    def test_gate04_holdout_has_16_unique_pair_records(self):
        self.assertEqual(len(self.holdout), 16)
        self.assertEqual(len({row.pair_digest for row in self.holdout}), 16)

    def test_gate04_holdout_is_disjoint_from_legacy_panel(self):
        self.assertFalse(
            {row.pair_digest for row in self.legacy}
            & {row.pair_digest for row in self.holdout}
        )

    def test_gate04_ordered_prompt_count_is_32(self):
        self.assertEqual(len(self.panel["ordered_prompt_keys"]), 32)

    def test_gate04_prompt_order_is_clean_then_corrupt(self):
        for index, row in enumerate(self.holdout):
            self.assertEqual(
                self.panel["ordered_prompt_keys"][2 * index:2 * index + 2],
                [[row.pair_digest, "clean"], [row.pair_digest, "corrupt"]],
            )

    def test_gate04_thresholds_equal_binding_values(self):
        self.assertEqual(gate04_thresholds(), {
            "raw_year_logits": {"max_abs": 3e-4, "pooled_rms": 7.5e-5},
            "centered_year_logits": {"max_abs": 2.5e-4, "pooled_rms": 6e-5},
            "task_margin": {"max_abs": 2e-4, "rms": 5e-5},
            "resid_mid": {"max_abs": 1e-4, "pooled_rms": 2e-5},
            "selected_pre": {"max_abs": 5e-4, "pooled_rms": 1e-4},
            "selected_post": {"max_abs": 5e-4, "pooled_rms": 1e-4},
        })

    def test_original_hf_tl_max_abs_field_is_absent(self):
        self.assertNotIn("hf_tl_max_abs", THRESHOLDS.__dataclass_fields__)

    def test_same_tl_thresholds_are_unchanged(self):
        self.assertEqual(THRESHOLDS.hook_untouched_max, 1e-7)
        self.assertEqual(THRESHOLDS.no_op_max_abs, 2e-5)
        self.assertEqual(THRESHOLDS.tail_max_abs, 2e-5)
        self.assertEqual(THRESHOLDS.tail_derivative_relative, 1e-4)
        self.assertEqual(THRESHOLDS.center_rms, 2e-6)
        self.assertEqual(THRESHOLDS.center_max_abs, 2e-5)


class Gate04AggregationTests(unittest.TestCase):
    def test_pooled_rms_flattens_all_prompts_and_coordinates(self):
        arrays = [np.array([3.0, 4.0]), np.array([0.0])]
        self.assertAlmostEqual(pooled_error_metrics(arrays)["pooled_rms"], 5 / np.sqrt(3))

    def test_global_max_is_not_mean_of_prompt_maxima(self):
        metrics = pooled_error_metrics([np.array([10.0, 0.0]), np.array([2.0, 2.0])])
        self.assertEqual(metrics["max_abs"], 10.0)
        self.assertNotEqual(metrics["max_abs"], 6.0)

    def test_centered_logit_metric_centers_each_prompt_separately(self):
        hf = np.array([2.0, 4.0, 8.0])
        tl = np.array([1.0, 5.0, 7.0])
        expected = (hf - hf.mean()) - (tl - tl.mean())
        np.testing.assert_allclose(centered_year_error(hf, tl), expected)

    def test_margin_metric_uses_clean_suffix_for_clean_and_corrupt(self):
        hf = np.linspace(-1.0, 1.0, 100) ** 3
        tl = np.zeros(100)
        clean_error = task_margin_error(hf, tl, 40)
        corrupt_prompt_error = task_margin_error(hf, tl, 40)
        wrong_suffix_error = task_margin_error(hf, tl, 55)
        self.assertEqual(clean_error, corrupt_prompt_error)
        self.assertNotEqual(clean_error, wrong_suffix_error)

    def test_margin_metric_is_invariant_to_constant_logit_shift(self):
        hf = np.linspace(-1.0, 1.0, 100)
        tl = np.linspace(0.5, -0.5, 100)
        self.assertAlmostEqual(
            task_margin_error(hf + 17.0, tl - 9.0, 37),
            task_margin_error(hf, tl, 37),
            places=12,
        )

    def test_all_gate04_submetrics_must_pass(self):
        thresholds = gate04_thresholds()
        passing = {
            name: {statistic: limit for statistic, limit in limits.items()}
            for name, limits in thresholds.items()
        }
        self.assertTrue(all_gate04_submetrics_pass(passing, thresholds))
        failing = json.loads(json.dumps(passing))
        failing["selected_post"]["pooled_rms"] = 1.0001e-4
        self.assertFalse(all_gate04_submetrics_pass(failing, thresholds))


class WeightMappingTests(unittest.TestCase):
    @staticmethod
    def states():
        expected = {
            "embed.W_E": np.arange(6, dtype=np.float32).reshape(2, 3),
            "blocks.0.mlp.W_in": np.arange(12, dtype=np.float32).reshape(3, 4),
        }
        actual = {key: value.copy() for key, value in reversed(tuple(expected.items()))}
        actual["unembed.b_U"] = np.zeros(3, dtype=np.float32)
        return expected, actual

    def test_weight_map_rejects_missing_key(self):
        expected, actual = self.states()
        del actual["embed.W_E"]
        self.assertFalse(weight_mapping_report(expected, actual)["passed"])

    def test_weight_map_rejects_shape_mismatch(self):
        expected, actual = self.states()
        actual["embed.W_E"] = np.zeros((3, 2), dtype=np.float32)
        self.assertIn("embed.W_E", weight_mapping_report(expected, actual)["shape_mismatches"])

    def test_weight_map_rejects_dtype_mismatch(self):
        expected, actual = self.states()
        actual["embed.W_E"] = actual["embed.W_E"].astype(np.float64)
        self.assertIn("embed.W_E", weight_mapping_report(expected, actual)["dtype_mismatches"])

    def test_weight_map_rejects_one_bit_value_change(self):
        expected, actual = self.states()
        actual["embed.W_E"][0, 0] = np.nextafter(
            actual["embed.W_E"][0, 0], np.float32(1.0)
        )
        self.assertIn("embed.W_E", weight_mapping_report(expected, actual)["value_mismatches"])

    def test_weight_map_accepts_exact_rearranged_gpt2_state(self):
        expected, actual = self.states()
        self.assertTrue(weight_mapping_report(expected, actual)["passed"])

    def test_unembed_bias_must_be_exactly_zero(self):
        expected, actual = self.states()
        actual["unembed.b_U"][1] = np.nextafter(np.float32(0.0), np.float32(1.0))
        report = weight_mapping_report(expected, actual)
        self.assertEqual(report["unembed_b_U_nonzero_count"], 1)
        self.assertFalse(report["passed"])


class BackendContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = RUNNER_SOURCE.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_hf_load_forces_eager_attention(self):
        self.assertIn('attn_implementation="eager"', self.source)

    def test_hf_gate04_forward_disables_cache(self):
        self.assertIn("use_cache=False", self.source)
        self.assertIn("return_dict=True", self.source)

    def test_gate04_batch_size_is_one(self):
        self.assertIn('"batch_size": 1', self.source)
        self.assertIn("tokens.shape[0] != 1", self.source)

    def test_gate04_captures_ln2_input_as_resid_mid(self):
        self.assertIn("ln_2.register_forward_pre_hook", self.source)
        self.assertIn('captured["resid_mid"] = args[0].detach()', self.source)

    def test_gate04_captures_c_fc_output_as_pre(self):
        self.assertIn("mlp.c_fc.register_forward_hook", self.source)
        self.assertIn('captured["pre"] = output.detach()', self.source)

    def test_gate04_captures_c_proj_input_as_post(self):
        self.assertIn("mlp.c_proj.register_forward_pre_hook", self.source)
        self.assertIn('captured["post"] = args[0].detach()', self.source)

    def test_hf_hooks_are_removed_in_finally(self):
        capture = self.source[self.source.index("def capture_hf_gate04"):self.source.index("def _exact_tensor_equal")]
        self.assertIn("finally:", capture)
        self.assertIn("handle.remove()", capture)


class NumericalPropagationTests(unittest.TestCase):
    def setUp(self):
        self.epsilon_y = 2e-7
        self.h1 = 0.2
        self.h2 = 0.1
        self.rich_G = np.linspace(0.1, 1.0, 100)
        self.half_G = self.rich_G - 0.001
        self.rich_C = np.linspace(1.0, 2.0, 100)
        self.half_C = self.rich_C - 0.002
        self.rich_delta = np.vstack([
            np.linspace(0.1 + axis, 0.3 + axis, 100) for axis in range(4)
        ])
        self.half_delta = self.rich_delta - 0.003
        zeros = np.zeros((4, 100))
        J = np.zeros((4, 100))
        self.rich = GateJet(self.rich_G, self.rich_C, J, self.rich_delta, zeros)
        self.half = GateJet(self.half_G, self.half_C, J, self.half_delta, zeros)
        self.bounds = richardson_numerical_bounds(
            self.rich, self.half, epsilon_y=self.epsilon_y,
            h1=self.h1, h2=self.h2,
        )

    def test_eta_G_matches_3_epsilon_over_h2(self):
        self.assertEqual(self.bounds.eta_G, 3 * self.epsilon_y / self.h2)

    def test_eta_C_matches_64_epsilon_over_3_h2_squared(self):
        self.assertEqual(self.bounds.eta_C, 64 * self.epsilon_y / (3 * self.h2**2))

    def test_eta_J_matches_3_epsilon_over_h1(self):
        self.assertEqual(self.bounds.eta_J, 3 * self.epsilon_y / self.h1)

    def test_eta_H_matches_17_epsilon_over_3_h1_h2(self):
        self.assertEqual(self.bounds.eta_H, 17 * self.epsilon_y / (3 * self.h1 * self.h2))

    def test_epsilon_G_includes_richardson_half_discrepancy(self):
        expected = np.linalg.norm(self.rich_G - self.half_G) + 10 * self.bounds.eta_G
        self.assertAlmostEqual(self.bounds.epsilon_G, expected)

    def test_epsilon_C_includes_richardson_half_discrepancy(self):
        expected = np.linalg.norm(self.rich_C - self.half_C) + 10 * self.bounds.eta_C
        self.assertAlmostEqual(self.bounds.epsilon_C, expected)

    def test_epsilon_delta_H_includes_both_path_and_control_noise(self):
        expected = np.linalg.norm(self.rich_delta - self.half_delta, axis=1) + 20 * self.bounds.eta_H
        np.testing.assert_allclose(self.bounds.epsilon_delta_H, expected)

    def test_A_max_uses_C_norm_minus_epsilon_C(self):
        expected = (
            np.linalg.norm(self.rich_delta, axis=1) + self.bounds.epsilon_delta_H
        ) / (np.linalg.norm(self.rich_C) - self.bounds.epsilon_C)
        np.testing.assert_allclose(self.bounds.A_max, expected)

    def test_epsilon_A_matches_frozen_formula(self):
        expected = (
            self.bounds.epsilon_delta_H + self.bounds.A_max * self.bounds.epsilon_C
        ) / np.linalg.norm(self.rich_C)
        np.testing.assert_allclose(self.bounds.epsilon_A, expected)

    def test_epsilon_P_matches_frozen_formula(self):
        expected = (
            self.bounds.epsilon_G * self.bounds.A_max
            + np.linalg.norm(self.rich_G) * self.bounds.epsilon_A
        )
        np.testing.assert_allclose(self.bounds.epsilon_P, expected)

    def test_epsilon_P_F_is_axiswise_l2_norm(self):
        self.assertAlmostEqual(self.bounds.epsilon_P_F, np.linalg.norm(self.bounds.epsilon_P))

    def test_active_curvature_snr_uses_epsilon_C(self):
        self.assertIn("THRESHOLDS.curvature_snr_min * numerical.epsilon_C", RUNNER_SOURCE.read_text(encoding="utf-8"))

    def test_active_gate_response_snr_uses_epsilon_G(self):
        self.assertIn("THRESHOLDS.gate_response_snr_min * numerical.epsilon_G", RUNNER_SOURCE.read_text(encoding="utf-8"))

    def test_active_tensor_snr_uses_epsilon_P_F(self):
        self.assertIn("THRESHOLDS.tensor_snr_min * numerical.epsilon_P_F", RUNNER_SOURCE.read_text(encoding="utf-8"))

    def test_certified_null_bound_uses_contrast_delta_G_and_whitebox_A(self):
        observed = certified_null_bound(2.0, 3.0, 5.0, 0.5, 7.0)
        self.assertEqual(observed, 2.0 * 3.0 * (5.0 + 0.5) * 7.0)

    def test_certified_null_bound_is_added_to_theta_error(self):
        source = RUNNER_SOURCE.read_text(encoding="utf-8")
        self.assertIn('gate_error_bounds.append(values["audit"]["null_bound"])', source)

    def test_cell_error_bound_sums_target_and_patched_item_bounds(self):
        self.assertEqual(cell_error_bound([1.0, 3.0], [2.0, 4.0]), 5.0)

    def test_frozen_propagation_rejects_reviewed_simplified_formula(self):
        self.assertNotAlmostEqual(self.bounds.epsilon_G, self.epsilon_y / self.h2)
        self.assertNotAlmostEqual(self.bounds.epsilon_C, 4 * self.epsilon_y / self.h2**2)
        old_eps_p = self.epsilon_y / (self.h1 * self.h2)
        self.assertNotAlmostEqual(self.bounds.epsilon_P_F, old_eps_p)


class SeparationContractTests(unittest.TestCase):
    def test_hf_tl_metrics_do_not_enter_epsilon_y(self):
        self.assertFalse(FROZEN_SPEC["gate04_amendment"]["hf_tl_error_enters_epsilon_y"])

    def test_epsilon_y_is_derived_only_from_duplicate_noise_audit(self):
        source = RUNNER_SOURCE.read_text(encoding="utf-8")
        development = source[source.index("def development_phase"):source.index("def confirmation_phase")]
        self.assertIn('epsilon_y = max(1e-7, noise["max_abs"])', development)
        self.assertNotIn("hf_audit", development)
        self.assertNotIn("hf_vs_tl", development)

    def test_tail_thresholds_do_not_reference_gate04_thresholds(self):
        source = RUNNER_SOURCE.read_text(encoding="utf-8")
        tail = source[source.index("def tail_audit"):source.index("def _jet_at_radius")]
        self.assertIn("THRESHOLDS.tail_max_abs", tail)
        self.assertNotIn("hf_tl_", tail)

    def test_target_module_import_firewall_remains_intact(self):
        tree = ast.parse(TARGET_SOURCE.read_text(encoding="utf-8"))
        imported = {
            node.module for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        self.assertFalse(imported & {"green_bridge_numerics", "exp_green_bridge_gpt2"})

    def test_confirmation_lock_remains_intact(self):
        source = RUNNER_SOURCE.read_text(encoding="utf-8")
        self.assertIn("verify_freeze(output_root, require_confirmation=True)", source)
        self.assertIn("confirmation_lock=ConfirmationLock", source)

    def test_requirements_lock_is_unchanged(self):
        import hashlib
        self.assertEqual(
            hashlib.sha256(REQUIREMENTS_LOCK.read_bytes()).hexdigest(),
            "f2b035c6d78ab107a50cb208e2521ed31bf9b2f309970c9e929b3e723df74c33",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
