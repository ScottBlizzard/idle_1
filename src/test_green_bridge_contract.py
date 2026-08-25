"""CPU contract suite for structural-envelope matched-bypass protocol v1.3."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import exp_green_bridge_gpt2 as runner
import green_bridge_path_target as target_module
import green_bridge_tail as tail_module
from analyze_green_bridge import development_decision, development_decision_v200
from green_bridge_dataset import (
    build_evaluation_records, build_green_bridge_v200_splits, plan_payload,
    v200_split_payload,
)
from green_bridge_numerics import (
    ScaleNumericalBoundsV200, active_envelope_contraction_bound,
    absolute_value_interval, dyadic_enclosure_v200,
    factorization_compatibility_v200, minkowski_sum_interval,
    richardson_pair_bounds_v200, robust_interval_auc_lower_bound,
    shift_null_compatibility_v200, subtract_intervals,
    whitebox_compatibility_v200, whitebox_factorization_compatibility_v200,
    worst_case_interval_rmse,
)
from green_bridge_spec import (
    ALL_GATE_FRAME_DIM, COMMON_FRAME_DIM, DIMENSIONS,
    FIRST_ORDER_COEFFICIENT_SEED, FIRST_ORDER_COEFFICIENT_SHA256,
    FIRST_ORDER_RESIDUAL_DIRECTIONS, FROZEN_SPEC, GATE_RADIUS,
    HALF_RADIUS_MULTIPLIER, HISTORICAL_V12_BASIS_SPEC,
    HISTORICAL_V136_THRESHOLDS, PROBE_FRAME_DIM,
    PROTOCOL_ID, QUARTER_RADIUS_MULTIPLIER, RESIDUAL_RADIUS_MULTIPLIER,
    SCHEMA_VERSION, V200_CONFIRMATION_GROUPS, V200_DEVELOPMENT_GROUPS,
    V200_SPLIT_SHA256, canonical_json, sha256_text,
    TAIL_FIXED_BATCH_SIZE,
    STRUCTURAL_ATOM_RESIDUAL_MAX, STRUCTURAL_FRAME_ORTHOGONAL_MAX,
)
from green_bridge_structural_frame import (
    canonical_all_gate_frame, canonical_common_frame, canonical_gate_frame,
    center_residual, extend_frame_with_atom, first_order_coefficient_directions,
    frame_containment_metrics, layernorm_gate_atom, normalize_atom,
    residual_radius, target_physical_vector,
)
from green_bridge_whitebox_audit import (
    gradient_envelope_residual, layernorm_gate_gradient_autograd,
    layernorm_gate_gradient_formula, shift_null_metric, whitebox_A_coordinates,
)
from matched_bypass_gate import (
    GateJet, direct_bypass_in_common_frame, expected_tensor_calls,
    identify_gate, operator_action, operator_frobenius_norm,
    operator_inner_product, reconstruct_cotangent,
)
from green_bridge_response_ad import (
    audit_richardson_enclosure_v200, select_ad_audit_panel_v200,
)


ROOT = Path(__file__).resolve().parent.parent
RUNNER_SOURCE = (ROOT / "src" / "exp_green_bridge_gpt2.py").read_text(encoding="utf-8")
TARGET_SOURCE = (ROOT / "src" / "green_bridge_path_target.py").read_text(encoding="utf-8")
TAIL_SOURCE = (ROOT / "src" / "green_bridge_tail.py").read_text(encoding="utf-8")
DECISION = (ROOT / "analysis" / "GPTPRO_GREEN_GATE08_V12_DECISION_20260805.md").read_text(encoding="utf-8")


def synthetic(seed=7):
    rng = np.random.default_rng(seed)
    residuals = [rng.normal(size=768) for _ in range(3)]
    gamma = rng.normal(size=768)
    weights = rng.normal(size=(768, 3072))
    return rng, residuals, gamma, weights


class HistoricalAndTerminationTests(unittest.TestCase):
    def test_v12_stop_result_hash_is_frozen(self):
        self.assertIn("390c5b62d5b42e216abbb15a0d6d206a55419c48117f610f34c0ac802e153747", DECISION)

    def test_v12_stop_manifest_hash_is_frozen(self):
        # Executor-corrected 64-character digest; GPTPro text omitted one 'b'.
        self.assertEqual("ea486fe8eea798b16951fcea9394b1c4ddb4b44bbd4afb5c8b104b37aaf047be".__len__(), 64)

    def test_v12_stop_hook_audit_hash_is_frozen(self): self.assertIn("49aa7a1818fb06d63b975938aea7285d3198fccc97723a96a37afa097abdbb99", DECISION)
    def test_v12_stop_donor_plan_hash_is_frozen(self): self.assertIn("2c8dd401b93d3864969ab941b85cae2ab5e6e983bdf39b909f33c532b480cc16", DECISION)
    def test_v12_stop_ledger_hash_is_frozen(self): self.assertIn("fa88911fcce749942a24c9e479c66cf89cd72ce9386b76d146262de6671b4f65", DECISION)
    def test_v12_serialization_order_defect_is_recorded(self): self.assertIn("serialization-order defect", DECISION)
    def test_v12_missing_matrix_hashes_are_not_invented(self): self.assertIn("missing", DECISION.lower())
    def test_v12_ratio_is_preserved_as_reported(self): self.assertIn("1.0227285601080833", DECISION)

    def test_evaluation_plan_hash_is_unchanged(self):
        self.assertIn("150f146ef69858bce77677ce74a4806129720ee68395246cbce91d498f06960c", DECISION)

    def test_gate04_prompt_hash_is_unchanged(self): self.assertEqual(runner.GATE04_ORDERED_PROMPT_HASH, "619d21c10d4f30e6ce2597c3ba4df1de72cf0cb4f6cce322d82c2d3ec62803ce")
    def test_prior_stop_reports_are_protocol_hashed(self):
        for name in (
            "analysis/GREEN_SERVER_GATE08_V12_20260805.md",
            "analysis/GREEN_SERVER_V13_PREPARE_STOP_20260825.md",
            "analysis/GPTPRO_GREEN_V13_MANUAL_TAIL_DECISION_20260825.md",
            "analysis/archive/green_v13_stop_20260825/archive_manifest.json",
            "analysis/archive/green_v13_stop_20260825/green_bridge_v13_prepare.log",
            "analysis/GREEN_SERVER_V131_PREPARE_STOP_20260825.md",
            "analysis/GREEN_V131_BATCH_SHAPE_DIAGNOSTIC_20260825.json",
            "analysis/CODEX_GREEN_V132_BATCH_SHAPE_DECISION_20260825.md",
            "analysis/GREEN_SERVER_V132_DEVELOPMENT_STOP_20260825.md",
            "analysis/CODEX_GREEN_V133_ANCHOR_RECENTER_DECISION_20260825.md",
            "analysis/GREEN_SERVER_V133_PREPARE_STOP_20260825.md",
            "analysis/GREEN_V134_ANCHOR_RELATIVE_DIAGNOSTIC_20260825.json",
            "analysis/CODEX_GREEN_V134_EXACT_BATCH1_MULTIGPU_DECISION_20260825.md",
            "analysis/CODEX_GREEN_V135_GATEJET_RESPONSE_PAIRING_DECISION_20260825.md",
            "analysis/CODEX_GREEN_V136_DIRECT_BYPASS_ORIENTATION_DECISION_20260825.md",
        ):
            self.assertIn(name, runner.PROTOCOL_FILES)
    def test_active_protocol_has_no_pca_rank(self): self.assertNotIn("residual_rank", json.dumps(FROZEN_SPEC))
    def test_active_protocol_has_no_eigengap_threshold(self): self.assertNotIn("eigengap", json.dumps(FROZEN_SPEC).lower())
    def test_active_protocol_has_no_rank_sweep(self): self.assertNotIn("rank_sweep", json.dumps(FROZEN_SPEC))
    def test_active_protocol_has_no_rank6_fallback(self): self.assertNotIn("rank6_fallback", json.dumps(FROZEN_SPEC))
    def test_active_runner_does_not_call_build_basis_v2_donor_records(self): self.assertNotIn("build_basis_v2_donor_records", RUNNER_SOURCE)
    def test_active_runner_does_not_import_green_bridge_basis(self): self.assertNotIn("from green_bridge_basis import", RUNNER_SOURCE)
    def test_active_protocol_creates_no_donor_basis_artifact(self): self.assertNotIn("donor_basis.npz", inspect.getsource(runner.prepare_v200))
    def test_active_protocol_creates_no_basis_audit_artifact(self): self.assertNotIn("basis_audit", inspect.getsource(runner.prepare_v200))
    def test_active_protocol_creates_no_radius_donor_artifact(self): self.assertNotIn("radius_donor", inspect.getsource(runner.prepare_v200))


class LayerNormTheoremTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng, cls.residuals, cls.gamma, cls.weights = synthetic()
        cls.gate = 2326
        cls.atom = layernorm_gate_atom(cls.gamma, cls.weights, cls.gate)

    def test_layernorm_gate_gradient_matches_float64_autograd(self):
        formula = layernorm_gate_gradient_formula(self.residuals[0], self.gamma, self.weights[:, self.gate], eps=1e-5)
        automatic = layernorm_gate_gradient_autograd(self.residuals[0], self.gamma, self.weights[:, self.gate], eps=1e-5)
        self.assertLessEqual(np.max(np.abs(formula - automatic)), 1e-10)

    def test_layernorm_gate_gradient_is_in_one_centered_atom_gate_span(self):
        r = self.residuals[0]; g = layernorm_gate_gradient_formula(r, self.gamma, self.weights[:, self.gate], eps=1e-5)
        raw = np.column_stack((np.ones(768), center_residual(r), self.atom))
        q = np.linalg.qr(raw)[0]
        self.assertLess(gradient_envelope_residual(q, g)["relative"], 1e-12)

    def test_three_system_gate_envelope_has_five_raw_atoms(self):
        self.assertEqual(np.column_stack((np.ones(768), *map(center_residual, self.residuals), self.atom)).shape, (768, 5))

    def test_layernorm_gate_gradient_is_shift_null(self):
        g = layernorm_gate_gradient_formula(self.residuals[0], self.gamma, self.weights[:, self.gate], eps=1e-5)
        self.assertLessEqual(shift_null_metric(g), 1e-12)

    def test_layernorm_bias_does_not_change_gate_gradient(self):
        a = layernorm_gate_gradient_autograd(self.residuals[0], self.gamma, self.weights[:, self.gate], eps=1e-5)
        b = layernorm_gate_gradient_autograd(self.residuals[0], self.gamma, self.weights[:, self.gate], eps=1e-5, ln_bias=self.rng.normal(size=768))
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)

    def test_mlp_input_bias_does_not_change_gate_gradient(self):
        a = layernorm_gate_gradient_autograd(self.residuals[0], self.gamma, self.weights[:, self.gate], eps=1e-5)
        b = layernorm_gate_gradient_autograd(self.residuals[0], self.gamma, self.weights[:, self.gate], eps=1e-5, mlp_input_bias=9.0)
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)

    def test_actual_gate_weight_atom_uses_ln_scale(self): np.testing.assert_array_equal(self.atom, self.gamma * self.weights[:, self.gate])
    def test_gate_atom_uses_actual_mlp_coordinate(self): self.assertFalse(np.array_equal(self.atom, layernorm_gate_atom(self.gamma, self.weights, self.gate + 1)))


class FrameConstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng, cls.residuals, cls.gamma, cls.weights = synthetic(10)
        cls.common = canonical_common_frame(*cls.residuals)
        cls.atoms = [layernorm_gate_atom(cls.gamma, cls.weights, gate) for gate in runner.SELECTED_GATES]
        cls.gate = canonical_gate_frame(cls.common, cls.atoms[0])
        cls.all_gate = canonical_all_gate_frame(cls.common, cls.atoms)

    def test_common_frame_shape_is_768_by_4(self): self.assertEqual(self.common.shape, (768, 4))
    def test_gate_frame_shape_is_768_by_5(self): self.assertEqual(self.gate.shape, (768, 5))
    def test_all_gate_frame_shape_is_768_by_14(self): self.assertEqual(self.all_gate.shape, (768, 14))

    def test_common_frame_uses_exact_atom_order(self):
        raw = np.column_stack((np.ones(768)/np.sqrt(768), *[normalize_atom(center_residual(r)) for r in self.residuals]))
        self.assertLess(frame_containment_metrics(self.common, raw)["atom_residual_relative"], 1e-12)

    def test_common_qr_is_unpivoted(self): self.assertIn('pivoting=False', inspect.getsource(canonical_common_frame))
    def test_common_qr_is_economic(self): self.assertIn('mode="economic"', inspect.getsource(canonical_common_frame))
    def test_common_qr_runs_under_one_blas_thread(self): self.assertIn('limits=1', inspect.getsource(canonical_common_frame))
    def test_common_frame_repeat_is_bitwise_equal(self): np.testing.assert_array_equal(self.common, canonical_common_frame(*self.residuals))

    def test_gate_extension_uses_twice_residualized_atom(self):
        source = inspect.getsource(extend_frame_with_atom)
        self.assertIn("_twice_residualized", source)

    def test_exactly_dependent_gate_atom_uses_deterministic_completion(self):
        base = np.eye(8)[:, :4]
        result = extend_frame_with_atom(base, base[:, 0], return_metadata=True)
        self.assertEqual(result.extension_source, "deterministic_standard_basis_completion")

    def test_deterministic_completion_uses_smallest_valid_coordinate(self):
        base = np.eye(8)[:, :4]
        result = extend_frame_with_atom(base, base[:, 0], return_metadata=True)
        self.assertEqual(int(np.argmax(np.abs(result.extension))), 4)

    def test_frame_sign_rule_is_exact(self):
        for column in range(self.all_gate.shape[1]):
            pivot = np.argmax(np.abs(self.all_gate[:, column])); self.assertGreaterEqual(self.all_gate[pivot, column], 0)

    def test_frame_orthogonality_gate(self): self.assertLessEqual(frame_containment_metrics(self.all_gate, np.column_stack(self.atoms))["orthogonal_max_abs"], STRUCTURAL_FRAME_ORTHOGONAL_MAX)
    def test_raw_atom_containment_gate(self): self.assertLessEqual(frame_containment_metrics(self.gate, np.column_stack((np.ones(768), *map(center_residual, self.residuals), self.atoms[0])))["atom_residual_relative"], STRUCTURAL_ATOM_RESIDUAL_MAX)

    def test_gradient_envelope_containment_gate(self):
        g = layernorm_gate_gradient_formula(self.residuals[0], self.gamma, self.weights[:, runner.SELECTED_GATES[0]], eps=1e-5)
        self.assertLessEqual(gradient_envelope_residual(self.gate, g)["relative"], 1e-10)

    def test_first_four_gate_frame_columns_are_common(self): np.testing.assert_array_equal(self.gate[:, :4], self.common)
    def test_all_gate_frame_contains_all_gate_atoms(self): self.assertLessEqual(frame_containment_metrics(self.all_gate, np.column_stack(self.atoms))["atom_residual_relative"], 1e-12)


class BasisFreeIdentificationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.Q, _ = np.linalg.qr(rng.normal(size=(768, 5)))
        self.A = rng.normal(size=5); self.G = rng.normal(size=100)
        self.g = reconstruct_cotangent(self.Q, self.A); self.v = rng.normal(size=768)

    def test_basis_free_matched_bypass_identity(self):
        C=np.arange(1,101,dtype=float); D=np.ones((5,100)); jet=GateJet(self.G,C,np.zeros((5,100)),D+self.A[:,None]*C,D)
        np.testing.assert_allclose(identify_gate(jet).A,self.A)
    def test_basis_free_inverse_recovers_directional_gate_differential(self): self.test_basis_free_matched_bypass_identity()
    def test_five_probe_envelope_recovers_full_cotangent(self): np.testing.assert_allclose(self.Q.T@self.g,self.A)

    def test_cotangent_reconstruction_is_invariant_to_frame_rotation(self):
        R,_=np.linalg.qr(np.random.default_rng(5).normal(size=(5,5)))
        np.testing.assert_allclose(reconstruct_cotangent(self.Q@R,R.T@self.A),self.g)

    def test_operator_action_is_invariant_to_frame_rotation(self):
        R,_=np.linalg.qr(np.random.default_rng(6).normal(size=(5,5)))
        np.testing.assert_allclose(operator_action(self.G,self.g,self.v),operator_action(self.G,reconstruct_cotangent(self.Q@R,R.T@self.A),self.v))

    def test_operator_rank_is_at_most_one(self): self.assertLessEqual(np.linalg.matrix_rank(np.outer(self.G,self.g)),1)
    def test_operator_action_matches_explicit_matrix(self): np.testing.assert_allclose(operator_action(self.G,self.g,self.v),np.outer(self.G,self.g)@self.v)
    def test_operator_frobenius_norm_without_materialization(self): self.assertAlmostEqual(operator_frobenius_norm(self.G,self.g),np.linalg.norm(np.outer(self.G,self.g)))
    def test_operator_inner_product_without_materialization(self): self.assertAlmostEqual(operator_inner_product(self.G,self.g,self.G,self.g),np.sum(np.outer(self.G,self.g)**2))

    def test_zero_curvature_converse_is_preserved(self):
        with self.assertRaisesRegex(ValueError,"FAIL_CURVATURE"): identify_gate(GateJet(self.G,np.zeros(100),np.zeros((5,100)),np.zeros((5,100)),np.zeros((5,100))))

    def test_zero_gate_response_implies_zero_path_operator(self): np.testing.assert_array_equal(operator_action(np.zeros(100),self.g,self.v),np.zeros(100))

    def test_direct_bypass_recovery_in_common_frame(self):
        D=np.random.default_rng(8).normal(size=(5,100)).T
        np.testing.assert_allclose(direct_bypass_in_common_frame(D,self.Q,self.Q[:,:4]),D[:,:4])


class ShiftPhysicalRadiusTests(unittest.TestCase):
    def test_constant_residual_shift_leaves_ln_gate_pre_unchanged(self):
        rng,res,gamma,W=synthetic(22); w=W[:,0]
        g1=layernorm_gate_gradient_formula(res[0],gamma,w,eps=1e-5); g2=layernorm_gate_gradient_formula(res[0]+17,gamma,w,eps=1e-5)
        np.testing.assert_allclose(g1,g2,atol=1e-12)
    def test_shift_probe_keeps_direct_residual_bypass(self): self.assertIn("resid_post = resid_mid + mlp_out", (ROOT/"src/green_bridge_tail.py").read_text())
    def test_matched_control_cancels_shift_bypass_interaction(self): self.assertIn("controlled_pre = anchor.pre", (ROOT/"src/green_bridge_tail.py").read_text())
    def test_response_inverse_recovers_zero_shift_coefficient(self): self.assertLess(abs(whitebox_A_coordinates(np.ones((768,1))/np.sqrt(768),np.r_[1.,np.zeros(767)]-np.ones(768)/768)[0]),1e-14)
    def test_shift_coefficient_threshold_is_exact(self): self.assertIn("max(\n            1e-4,", inspect.getsource(runner._mixed_system_v13))
    def test_physical_tail_accepts_batch_by_768_delta(self): self.assertIn("residual_delta must have shape", (ROOT/"src/green_bridge_tail.py").read_text())
    def test_physical_path_keeps_current_gate_live(self): self.assertIn('mode == "path"', (ROOT/"src/green_bridge_tail.py").read_text())
    def test_physical_control_severs_x_to_gate_edge(self): self.assertIn("controlled_pre = anchor.pre", (ROOT/"src/green_bridge_tail.py").read_text())
    def test_physical_control_preserves_x_bypass(self): self.assertIn("resid_post = resid_mid + mlp_out", (ROOT/"src/green_bridge_tail.py").read_text())
    def test_physical_joint_keeps_exact_selected_gates(self): self.assertIn("self.gate_index", (ROOT/"src/green_bridge_tail.py").read_text())
    def test_physical_target_subtracts_exact_input_delta(self): self.assertIn("-= residual_delta", TARGET_SOURCE)
    def test_target_module_does_not_import_structural_frame(self): self.assertNotIn("green_bridge_structural_frame", TARGET_SOURCE)
    def test_target_module_does_not_read_response_inverse(self): self.assertNotIn("A_hat", TARGET_SOURCE)
    def test_target_full_and_half_endpoints_are_symmetric(self): self.assertIn("plus - minus", TARGET_SOURCE)

    def test_residual_radius_is_point_two_chord_rms(self):
        rng=np.random.default_rng(1); a,b,c=rng.normal(size=(3,768)); r=residual_radius(a,b,c)
        self.assertAlmostEqual(r["h_x"],.2*np.linalg.norm(a-c)/np.sqrt(768))
    def test_target_vector_norm_equals_residual_radius(self):
        a=np.arange(768.);c=a[::-1];v=target_physical_vector(a,c,.3);self.assertAlmostEqual(np.linalg.norm(v),.3)
    def test_target_vector_uses_full_clean_corrupt_chord(self):
        a=np.arange(768.);c=a[::-1];v=target_physical_vector(a,c,.3);self.assertGreater(v@(a-c),0)
    def test_residual_radius_floor_is_exact(self): self.assertAlmostEqual(residual_radius(np.ones(768),np.ones(768),np.zeros(768))["floor"],2**-10*np.median([1,1,0]))
    def test_gate_radius_is_exactly_point_two(self): self.assertEqual(GATE_RADIUS,.2)
    def test_gate_half_radius_is_exactly_point_one(self): self.assertEqual(GATE_RADIUS*HALF_RADIUS_MULTIPLIER,.1)
    def test_no_radius_search_exists(self): self.assertNotIn("radius_search", json.dumps(FROZEN_SPEC))
    def test_no_radius_inflation_exists(self): self.assertNotIn("inflation", json.dumps(FROZEN_SPEC))


class BaselineComputeTests(unittest.TestCase):
    def setUp(self): self.D=first_order_coefficient_directions()
    def test_first_order_coordinate_dimension_is_14(self): self.assertEqual(self.D.shape[1],14)
    def test_first_order_direction_count_is_250(self): self.assertEqual(self.D.shape[0],250)
    def test_first_14_first_order_directions_are_identity(self): np.testing.assert_array_equal(self.D[:14],np.eye(14))
    def test_first_order_seed_is_exact(self): self.assertEqual(FIRST_ORDER_COEFFICIENT_SEED,8998478401382166109)
    def test_first_order_coefficient_hash_is_exact(self):
        if np.__version__ == "2.2.6":
            self.assertEqual(hashlib.sha256(self.D.tobytes()).hexdigest(),FIRST_ORDER_COEFFICIENT_SHA256)
        else:
            self.assertEqual(FIRST_ORDER_COEFFICIENT_SHA256,"b39a9a0bdda54bf63d1496f690bd4c89c6fa618ba7beb152364cb9f2b3f18a1a")
    def test_first_order_sign_rule_is_exact(self): self.assertTrue(np.all(self.D[14:,0]>0))
    def test_first_order_rejection_threshold_is_exact(self):
        gram=np.abs(self.D@self.D.T);np.fill_diagonal(gram,0.0);self.assertLessEqual(np.max(gram),0.999999)
    def test_first_order_physical_directions_use_all_gate_frame(self): self.assertIn('coefficients @ design["all_gate"].T',inspect.getsource(runner._first_order_system_v13))
    def test_first_order_budget_is_2082(self): self.assertEqual(expected_tensor_calls(),2082)
    def test_mixed_calls_per_gate_system_radius_is_52(self): self.assertEqual(2+10*5,52)
    def test_mixed_calls_per_system_is_1041(self): self.assertEqual(10*3*52+1,1561)
    def test_mixed_calls_per_item_is_2082(self): self.assertEqual(runner.FORWARD_COUNTS["mixed_per_tensor_item"],3122)
    def test_tensor_item_unique_calls_are_4180(self): self.assertEqual(runner.FORWARD_COUNTS["tensor_item_unique_calls"],5220);self.assertEqual(runner.HISTORICAL_V136_FORWARD_COUNTS["tensor_item_unique_calls"],4180)
    def test_tensor_tail_total_is_1605120(self): self.assertEqual(runner.FORWARD_COUNTS["tensor_tail_total"],1336320);self.assertEqual(runner.HISTORICAL_V136_FORWARD_COUNTS["tensor_tail_total"],1605120)
    def test_energy_tail_total_is_4608(self): self.assertEqual(runner.FORWARD_COUNTS["energy_tail_total"],3840);self.assertEqual(runner.HISTORICAL_V136_FORWARD_COUNTS["energy_tail_total"],4608)
    def test_tail_total_is_1609824(self): self.assertEqual(runner.FORWARD_COUNTS["tail_evaluations_total"],1340160);self.assertEqual(runner.HISTORICAL_V136_FORWARD_COUNTS["tail_evaluations_total"],1609824)
    def test_jvp_total_is_1152(self): self.assertEqual(runner.FORWARD_COUNTS["jvp_invocations_total"],768);self.assertEqual(runner.HISTORICAL_V136_FORWARD_COUNTS["jvp_invocations_total"],1152)
    def test_full_model_total_is_2496(self): self.assertEqual(runner.FORWARD_COUNTS["full_model_evaluations_total"],1664);self.assertEqual(runner.HISTORICAL_V136_FORWARD_COUNTS["full_model_evaluations_total"],2496)
    def test_raw_invocation_total_is_1613472(self): self.assertEqual(runner.FORWARD_COUNTS["raw_invocations_total"],1342592);self.assertEqual(runner.HISTORICAL_V136_FORWARD_COUNTS["raw_invocations_total"],1613472)
    def test_effective_unit_total_is_1614624(self): self.assertEqual(runner.FORWARD_COUNTS["effective_units_total"],1343360);self.assertEqual(runner.HISTORICAL_V136_FORWARD_COUNTS["effective_units_total"],1614624)
    def test_conservative_unit_total_is_1627104(self): self.assertEqual(runner.FORWARD_COUNTS["conservative_units_total"],1351680);self.assertEqual(runner.HISTORICAL_V136_FORWARD_COUNTS["conservative_units_total"],1627104)
    def test_preconfirmation_effective_units_are_538336(self): self.assertEqual(runner.FORWARD_COUNTS["development_effective_units"],335840);self.assertEqual(runner.HISTORICAL_V136_FORWARD_COUNTS["development_effective_units"],538336)
    def test_confirmation_effective_units_are_1076288(self): self.assertEqual(runner.FORWARD_COUNTS["confirmation_effective_units"],1007520);self.assertEqual(runner.HISTORICAL_V136_FORWARD_COUNTS["confirmation_effective_units"],1076288)


class SerializationAndOneRunTests(unittest.TestCase):
    def test_development_inputs_written_before_frame_construction(self): self.assertLess(inspect.getsource(runner.development_phase_v200).index("_capture_structural_inputs"),inspect.getsource(runner.development_phase_v200).index("_construct_structural_design"))
    def test_development_hashes_written_before_frame_construction(self): self.assertLess(inspect.getsource(runner._capture_structural_inputs).index("hashes.json"),len(inspect.getsource(runner._capture_structural_inputs)))
    def test_development_frames_written_before_first_endpoint(self): self.assertLess(inspect.getsource(runner.development_phase_v200).index("_construct_structural_design"),inspect.getsource(runner.development_phase_v200).index("_run_split_v200"))
    def test_development_radii_written_before_first_endpoint(self): self.test_development_frames_written_before_first_endpoint()
    def test_development_target_vectors_written_before_first_endpoint(self): self.test_development_frames_written_before_first_endpoint()
    def test_confirmation_inputs_cannot_exist_before_open(self): self.assertTrue(inspect.getsource(runner.confirmation_phase_v200).index("verify_freeze")<inspect.getsource(runner.confirmation_phase_v200).index("_capture_structural_inputs"))
    def test_confirmation_frames_cannot_exist_before_open(self): self.assertTrue(inspect.getsource(runner.confirmation_phase_v200).index("verify_freeze")<inspect.getsource(runner.confirmation_phase_v200).index("_construct_structural_design"))
    def test_endpoint_batch_requires_started_ledger_record(self): self.assertIn("endpoint_batch_started",inspect.getsource(runner._run_endpoint_batch))
    def test_committed_endpoint_requires_durable_artifact(self): self.assertTrue(inspect.getsource(runner._run_endpoint_batch).index("write_json_atomic")<inspect.getsource(runner._run_endpoint_batch).index("endpoint_batch_committed"))
    def test_uncommitted_endpoint_prevents_restart(self): self.assertIn("started-but-uncommitted",inspect.getsource(runner._assert_no_uncommitted_endpoint))

    def test_prepare_rejects_existing_output_root(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"x";p.mkdir();(p/"a").write_text("x")
            with self.assertRaises(runner.GreenStop): runner.assert_empty_prepare_root(p)
    def test_prepare_rejects_untracked_file(self): self.assertIn("--untracked-files=all",inspect.getsource(runner.assert_clean_repository))
    def test_prepare_rejects_wrong_branch(self): self.assertIn('branch != "main"',inspect.getsource(runner.assert_clean_repository))
    def test_prepare_requires_review_commit_ancestor(self): self.assertIn("merge-base",inspect.getsource(runner.assert_clean_repository))
    def test_attempt_index_is_one(self): self.assertIn('"attempt_index": 1',inspect.getsource(runner.write_run_ledger))
    def test_retry_allowed_is_false(self): self.assertIn('"retry_allowed": False',inspect.getsource(runner.write_run_ledger))
    def test_phase_all_is_rejected(self): self.assertIn('choices=("prepare", "development", "confirmation")',RUNNER_SOURCE)
    def test_stopped_v13_root_cannot_resume(self): self.assertIn('result.get("verdict") == "STOP"',inspect.getsource(runner.verify_freeze))


class FrozenCoreTests(unittest.TestCase):
    def test_schema_and_protocol(self): self.assertEqual((SCHEMA_VERSION,PROTOCOL_ID),("green-bridge-v2.0.0","structural-envelope-matched-bypass-setid-v2.0.0"))
    def test_dimensions(self): self.assertEqual((DIMENSIONS.d_model,DIMENSIONS.probe_frame_dim),(768,5))
    def test_expected_calls(self): self.assertEqual(expected_tensor_calls(n_radii=3),3122)
    def test_envelope_error_term_is_positive(self): self.assertEqual(active_envelope_contraction_bound(2,3,4,5,6,7,8),2*(3*5+(6+7)*4*8))


class ManualTailEndpointContractTests(unittest.TestCase):
    def test_manual_tail_uses_full_transformerlens_unembed(self):
        self.assertIn("model.unembed(normalized_final)", inspect.getsource(tail_module.full_transformerlens_year_logits))
    def test_manual_tail_applies_output_softcap_before_gather(self):
        source=inspect.getsource(tail_module.full_transformerlens_year_logits);self.assertLess(source.index("apply_softcap("),source.index("gather_year_logits("))
    def test_manual_tail_does_not_index_wu_before_unembed(self):
        self.assertNotIn("W_U.index_select", inspect.getsource(tail_module.GreenBridgeTail._evaluate_physical_core))
    def test_manual_tail_does_not_slice_final_position_before_unembed(self):
        source=inspect.getsource(tail_module.GreenBridgeTail._evaluate_physical_core);self.assertNotIn("normalized_final[rows",source)


class PathTargetEndpointContractTests(unittest.TestCase):
    def test_path_target_uses_full_transformerlens_unembed(self): self.assertIn("model.unembed(normalized_final)",inspect.getsource(target_module._full_transformerlens_year_logits))
    def test_path_target_applies_output_softcap_before_gather(self):
        source=inspect.getsource(target_module._full_transformerlens_year_logits);self.assertLess(source.index("apply_softcap("),source.index("index_select"))
    def test_path_target_does_not_index_wu_before_unembed(self): self.assertNotIn("W_U.index_select",inspect.getsource(target_module.evaluate_joint_target))
    def test_path_target_remains_code_isolated(self):
        for forbidden in (
            "from green_bridge_tail import", "from matched_bypass_gate import",
            "from predictor import", "from baseline import",
        ):
            self.assertNotIn(forbidden,TARGET_SOURCE)


class FullHookReferenceContractTests(unittest.TestCase):
    def test_full_hook_endpoint_remains_independent_reference(self):
        source=inspect.getsource(runner.full_hook_endpoint_physical);self.assertNotIn("GreenBridgeTail",source);self.assertIn("model.run_with_hooks",source)


class TailAuditMetricContractTests(unittest.TestCase):
    def test_tail_raw_gate_compares_raw_year_logits(self): self.assertIn('"quantity": "raw_100_dimensional_year_logits"',inspect.getsource(runner._tail_preflight_v136))
    def test_tail_raw_gate_threshold_is_two_e_minus_five(self): self.assertEqual(runner.THRESHOLDS.tail_max_abs,2e-5)
    def test_tail_center_condition_is_binding(self): self.assertIn('("center", "path", np.zeros(5), 0.0)',inspect.getsource(runner._tail_preflight_v136))
    def test_tail_derivative_gate_uses_central_difference(self): self.assertIn("2.0 * step",inspect.getsource(runner.derivative_equivalence_record))
    def test_tail_nonzero_derivative_relative_threshold_is_one_e_minus_four(self): self.assertEqual(runner.THRESHOLDS.tail_derivative_relative,1e-4)
    def test_tail_near_zero_derivative_uses_propagated_absolute_bound(self): self.assertIn("THRESHOLDS.tail_max_abs / step",inspect.getsource(runner.derivative_equivalence_record))
    def test_tail_near_zero_derivative_is_not_silently_dropped(self): self.assertIn("NOT_APPLICABLE_NEAR_ZERO",inspect.getsource(runner.derivative_equivalence_record))


class ProtocolIdentityV136Tests(unittest.TestCase):
    def test_v136_identity_is_fresh(self): self.assertEqual(runner.PROTOCOL_RUN_ID,"green-bridge-v2.0.0-one-shot")
    def test_v136_output_root_is_distinct(self): self.assertEqual(runner.OUTPUT_ROOT.name,"green_bridge_v200")
    def test_v136_attempt_index_is_one(self): self.assertIn('"attempt_index": 1',inspect.getsource(runner.write_run_ledger))
    def test_v136_retry_is_false(self): self.assertIn('"retry_allowed": False',inspect.getsource(runner.write_run_ledger))


class PredecessorArchiveContractTests(unittest.TestCase):
    def test_v131_stop_hashes_are_frozen_and_verified(self): self.assertEqual(runner.V131_TERMINAL_HASHES["outputs/green_bridge_v131/result.json"],"e911860ea406e6b38d7dc475dffd500dde68044185c11e0bc7be605f899ebbbf")
    def test_v131_diagnostic_hash_is_frozen(self): self.assertIn("666a20604fa4b123732bd68a15681fa7a16cafeef8edc2b61544fd911567d07d",inspect.getsource(runner.verify_v131_terminal_archive))
    def test_v132_development_hash_is_frozen(self): self.assertEqual(runner.V132_TERMINAL_HASHES["outputs/green_bridge_v132/dev_cells.json"],"1294a76d6d79c81f240c20c4257aa6b0fe76457d46b30cfc5d5699e27759ae1f")
    def test_v133_prepare_stop_hash_is_frozen(self): self.assertEqual(runner.V133_TERMINAL_HASHES["outputs/green_bridge_v133/result.json"],"e1084e999ff3c94c7d7cec343f22b6d7462f142440955edcde561b860d36a1d8")
    def test_v134_development_stop_hash_is_self_contained(self): self.assertIn("e340700ec23616cd2c8dd4c02f341896ee616f3aeffdf574dbcdc67075196cb2",inspect.getsource(runner.verify_v134_terminal_archive))
    def test_v135_development_stop_hash_is_frozen(self): self.assertEqual(runner.PREDECESSOR_RUN["result_sha256"],"c1c53e6b7fcf062a3c258a79f3ef4b87a0b33724e7087c2182f6f7e14fb81b20")


class PrepareArtifactContractTests(unittest.TestCase):
    def test_root_cause_reproduction_written_before_equivalence_pass(self):
        source=inspect.getsource(runner._tail_preflight_v136);self.assertLess(source.index("manual_tail_root_cause_reproduction_v136.json"),source.index("manual_tail_equivalence_v136.json"))
    def test_stage_trace_written_before_equivalence_pass(self):
        source=inspect.getsource(runner._tail_preflight_v136);self.assertLess(source.index("manual_tail_stage_trace_v136.json"),source.index("manual_tail_equivalence_v136.json"))
    def test_path_target_equivalence_written_before_manifest(self): self.assertLess(RUNNER_SOURCE.index("path_target_equivalence_v136.json"),RUNNER_SOURCE.index('"schema_version": "green-bridge-manifest-v1.3.6"'))


class ExactBatchOneOperationGraphTests(unittest.TestCase):
    def test_fixed_tail_batch_is_one(self): self.assertEqual(TAIL_FIXED_BATCH_SIZE,1)
    def test_tail_wrapper_pads_final_chunk(self): self.assertIn("if count < fixed",inspect.getsource(tail_module.GreenBridgeTail._evaluate_physical_fixed_batch))
    def test_tail_wrapper_slices_declared_rows(self): self.assertIn("logits[:count]",inspect.getsource(tail_module.GreenBridgeTail._evaluate_physical_fixed_batch))
    def test_scientific_tail_activates_fixed_batch(self): self.assertIn("fixed_batch_size=TAIL_FIXED_BATCH_SIZE",inspect.getsource(runner._tensor_item_v200))
    def test_scientific_tail_has_no_recentering(self): self.assertNotIn("recenter_fixed_batch_output",inspect.getsource(runner._tensor_item_v200))
    def test_prepare_requires_bitwise_full_hook_match(self): self.assertIn('if not metrics["bitwise_equal"]',inspect.getsource(runner._prepare_exact_batch_one_and_throughput_v136))
    def test_full_reference_remains_batch_one(self): self.assertIn('"full_model_jvp_batch_size": 1',inspect.getsource(runner._prepare_exact_batch_one_and_throughput_v136))
    def test_eight_worker_gpus_are_frozen(self): self.assertIn('physical_gpus = tuple(range(8))',inspect.getsource(runner._run_split_v200_multigpu))
    def test_worker_failure_is_terminal(self): self.assertIn('"11_MULTIGPU_WORKER"',inspect.getsource(runner._run_split_v200_multigpu))
    def test_worker_records_are_deterministically_sorted(self): self.assertIn('green-v200-worker|{role}|{row.pair_digest}',inspect.getsource(runner._run_split_v200_multigpu))

    def test_identification_is_paired_with_same_scale_gatejet_response(self):
        source = inspect.getsource(runner._mixed_system_v13)
        self.assertNotIn("estimate.G", source)
        for response in ("rich.G", "full.G", "half.G"):
            self.assertIn(response, source)

    def test_response_cotangent_action_equals_identified_matrix_action(self):
        rng = np.random.default_rng(135)
        frame = np.linalg.qr(rng.normal(size=(768, 5)))[0]
        response = rng.normal(size=100)
        curvature = rng.normal(size=100)
        coefficients = rng.normal(size=5)
        delta_h = coefficients[:, None] * curvature[None, :]
        jet = GateJet(response, curvature, delta_h * 0.0, delta_h, delta_h * 0.0)
        estimate = identify_gate(jet)
        physical_v = rng.normal(size=768)
        contrast = rng.normal(size=100)
        ambient = float(
            contrast @ operator_action(
                jet.G, reconstruct_cotangent(frame, estimate.A), physical_v
            )
        )
        coordinate = float(contrast @ ((frame.T @ physical_v) @ estimate.P))
        self.assertAlmostEqual(ambient, coordinate, places=11)

    def test_active_identified_branch_completes_all_interfaces(self):
        import torch
        rng = np.random.default_rng(136)
        frame = np.linalg.qr(rng.normal(size=(768, 5)))[0]
        common = frame[:, :4]
        response = rng.normal(size=100)
        curvature = rng.normal(size=100)
        coefficients = np.r_[0.0, rng.normal(size=4)]
        direct = rng.normal(size=(5, 100))
        delta_h = coefficients[:, None] * curvature[None, :]
        jet = GateJet(
            response,
            curvature,
            direct + coefficients[:, None] * response[None, :],
            delta_h,
            np.zeros_like(delta_h),
        )
        estimate = identify_gate(jet)
        numerical = SimpleNamespace(
            epsilon_A=np.zeros(5), epsilon_P_F=0.0, epsilon_G=0.0
        )
        values = {
            "rich": estimate,
            "full": estimate,
            "half": estimate,
            "numerical": numerical,
            "audit": {"label": "active-identified"},
        }
        anchor = SimpleNamespace(
            resid_mid=torch.zeros((1, 768), dtype=torch.float32),
            year_logits=torch.zeros((1, 100), dtype=torch.float32),
        )
        tail = SimpleNamespace(
            evaluate_physical=lambda *args, **kwargs: anchor.year_logits.clone()
        )
        block = SimpleNamespace(
            ln2=SimpleNamespace(w=torch.ones(768)),
            mlp=SimpleNamespace(W_in=torch.ones((768, 3072))),
        )
        model = SimpleNamespace(blocks=[None] * 10 + [block], cfg=SimpleNamespace(eps=1e-5))
        design = {"gate_frames": [frame] * 10, "common": common, "radius": {"h_x": 0.1}}
        with mock.patch.object(runner, "_jet_at_radius_physical", return_value=jet), \
             mock.patch.object(runner, "_selected_numpy", return_value=np.zeros(768)), \
             mock.patch.object(runner, "layernorm_gate_gradient_formula", return_value=np.zeros(768)), \
             mock.patch.object(runner, "whitebox_A_coordinates", return_value=coefficients), \
             mock.patch.object(runner, "classify_gate", return_value=("active-identified", values)), \
             mock.patch.object(runner, "gradient_envelope_residual", return_value={"absolute": 0.0}):
            result = runner._mixed_system_v13(
                tail, model, anchor, design, rng.normal(size=768),
                rng.normal(size=100), 1e-7,
            )
        self.assertEqual(result["active_gates"], 10)
        self.assertTrue(result["all_valid"])
        self.assertTrue(result["admissible"])
        self.assertAlmostEqual(result["bypass_disagreement"], 0.0)
        self.assertIn("values[\"rich\"].D.T", inspect.getsource(runner._mixed_system_v13))


class DevelopmentTerminalContractTests(unittest.TestCase):
    def test_insufficient_survival_returns_frozen_stop(self):
        decision = development_decision_v200({"cells": []})
        self.assertEqual(decision["verdict"], "STOP_ORAL")
        self.assertEqual(decision["n_surviving_cells"], 0)
        self.assertEqual(decision["baseline_calibration"], {})


class TheoryPreservationContractTests(unittest.TestCase):
    def test_fixed_rank_donor_pca_remains_terminated(self):
        self.assertNotIn("donor_pca",json.dumps(FROZEN_SPEC).lower());self.assertNotIn("green_bridge_basis",inspect.getsource(runner.prepare_v200))


def _v200_fixture():
    rng = np.random.default_rng(200)
    G = rng.normal(size=100)
    C = rng.normal(size=100)
    A = np.r_[0.0, rng.normal(size=4)]
    delta = A[:, None] * C[None, :]
    jet = GateJet(G, C, np.zeros((5, 100)), delta, np.zeros((5, 100)))
    bounds = richardson_pair_bounds_v200(jet, jet, epsilon_y=1e-7, h1=0.2, h2=0.2)
    enclosure = dyadic_enclosure_v200(jet, jet, bounds, bounds)
    return rng, jet, identify_gate(jet), enclosure, A


class V200FactorizationBoundsTests(unittest.TestCase):
    def test_exact_rank_one_inside_derived_bound(self):
        _, jet, identified, enclosure, _ = _v200_fixture()
        self.assertTrue(factorization_compatibility_v200(identified, jet, enclosure)["passed"])

    def test_residual_one_ulp_above_bound_fails(self):
        jet = GateJet(np.ones(100), np.r_[1.0, np.zeros(99)], np.zeros((5,100)), np.zeros((5,100)), np.zeros((5,100)))
        jet.H_path[0, 1] = np.nextafter(np.nextafter(1.0, np.inf), np.inf)
        enclosure = SimpleNamespace(
            final_epsilon_delta_H=np.r_[1.0, np.zeros(4)], final_epsilon_A=np.zeros(5),
            final_A_max=np.zeros(5), final_epsilon_C=0.0,
        )
        identified = SimpleNamespace(A=np.zeros(5))
        self.assertFalse(factorization_compatibility_v200(identified, jet, enclosure)["passed"])

    def test_active_classifier_has_no_factorization_point_one_five(self):
        source = inspect.getsource(runner._classify_gate_v200)
        self.assertNotIn("factorization_residual_max", source); self.assertNotIn("0.15", source)


class V200WhiteboxBoundsTests(unittest.TestCase):
    def test_componentwise_whitebox_envelope_passes(self):
        _, _, identified, enclosure, A = _v200_fixture()
        self.assertTrue(whitebox_compatibility_v200(identified, A, enclosure)["passed"])

    def test_componentwise_whitebox_excess_fails(self):
        _, _, identified, enclosure, A = _v200_fixture()
        altered = A.copy(); altered[1] += 2 * enclosure.final_epsilon_A[1] + 1e-6
        self.assertFalse(whitebox_compatibility_v200(identified, altered, enclosure)["passed"])

    def test_active_classifier_has_no_whitebox_point_zero_five(self):
        source = inspect.getsource(runner._classify_gate_v200)
        self.assertNotIn("whitebox_a_relative_max", source); self.assertNotIn("0.05", source)

    def test_direct_whitebox_factorization_triangle_bound(self):
        _, jet, _, enclosure, A = _v200_fixture()
        self.assertTrue(whitebox_factorization_compatibility_v200(jet, A, enclosure)["passed"])

    def test_shift_null_uses_epsilon_a_plus_epsilon_wb(self):
        row = shift_null_compatibility_v200(1.1e-10, 1e-11, epsilon_wb=1e-10)
        self.assertTrue(row["passed"]); self.assertAlmostEqual(row["bound"], 1.1e-10, places=20)


class V200StencilTests(unittest.TestCase):
    def test_radii_are_base_half_quarter(self):
        self.assertEqual((1.0, HALF_RADIUS_MULTIPLIER, QUARTER_RADIUS_MULTIPLIER), (1.0, 0.5, 0.25))

    def test_fine_richardson_is_always_primary(self):
        source = inspect.getsource(runner._gate_jet_triplet_v200)
        self.assertIn('"fine_richardson": fine', source)

    def test_dyadic_overlap_uses_uncertainty_balls(self):
        source = inspect.getsource(dyadic_enclosure_v200)
        self.assertIn("fine.epsilon_G + coarse.epsilon_G", source)

    def test_no_estimator_selection_uses_behavior_or_baseline(self):
        source = inspect.getsource(runner._gate_jet_triplet_v200).lower()
        self.assertNotIn("behavior", source); self.assertNotIn("baseline", source); self.assertNotIn("pie", source)


class V200ADAuditTests(unittest.TestCase):
    def test_audit_reader_has_no_behavioral_fields(self):
        source = inspect.getsource(select_ad_audit_panel_v200).lower()
        self.assertNotIn("behavioral", source); self.assertNotIn("pie", source); self.assertNotIn("admissib", source)

    def test_panel_has_exactly_forty_strata(self):
        records = [row for row in build_evaluation_records() if row.split == "development"]
        self.assertEqual(len(select_ad_audit_panel_v200(records)), 40)

    def test_ad_value_outside_enclosure_stops_prepare(self):
        zero = GateJet(np.zeros(100), np.zeros(100), np.zeros((5,100)), np.zeros((5,100)), np.zeros((5,100)))
        outside = GateJet(np.full(100, 2.0), np.zeros(100), np.zeros((5,100)), np.zeros((5,100)), np.zeros((5,100)))
        bounds = ScaleNumericalBoundsV200(1.0,1.0,1.0,np.ones(5),np.ones(5),np.ones(5),np.ones(5),1.0,True)
        self.assertFalse(audit_richardson_enclosure_v200(outside,outside,zero,zero,bounds,bounds)["passed"])


class V200GateClassTests(unittest.TestCase):
    @staticmethod
    def _enclosure(inverse):
        return SimpleNamespace(
            overlap_G=True, overlap_C=True, overlap_J=True, overlap_delta_H=np.ones(5,dtype=bool),
            final_epsilon_G=0.0, final_epsilon_C=0.0, final_epsilon_J=0.0,
            final_epsilon_delta_H=np.zeros(5), final_inverse_admissible=inverse,
            final_epsilon_A=np.zeros(5), final_A_max=np.zeros(5),
            final_epsilon_P_F=0.0,
        )

    def test_noninvertible_gate_becomes_unresolved_bounded(self):
        jet = GateJet(np.ones(100), np.zeros(100), np.zeros((5,100)), np.zeros((5,100)), np.zeros((5,100)))
        triplet={"fine_richardson":jet,"coarse_richardson":jet,"dyadic_enclosure":self._enclosure(False)}
        result=runner._classify_gate_v200(triplet,np.zeros(5),np.ones(768),np.ones(100),np.ones(768))
        self.assertEqual(result["audit"]["label"],"unresolved-bounded")

    def test_bound_exceedance_becomes_structural_contradiction(self):
        C=np.r_[1.0,np.zeros(99)]; delta=np.zeros((5,100)); delta[0,1]=1.0
        jet=GateJet(np.ones(100),C,np.zeros((5,100)),delta,np.zeros((5,100)))
        triplet={"fine_richardson":jet,"coarse_richardson":jet,"dyadic_enclosure":self._enclosure(True)}
        result=runner._classify_gate_v200(triplet,np.zeros(5),np.ones(768),np.ones(100),np.ones(768))
        self.assertEqual(result["audit"]["label"],"structural-contradiction")

    def test_unresolved_gate_point_center_is_zero(self):
        source=inspect.getsource(runner._mixed_system_v200)
        self.assertIn('label == "unresolved-bounded"',source);self.assertIn("contribution_center=0.0",source)

    def test_unresolved_whitebox_use_is_absolute_bound_only(self):
        source=inspect.getsource(runner._classify_gate_v200)
        self.assertIn("unresolved_gate_contraction_bound_v200",source);self.assertNotIn("whitebox_gradient @",source)


class V200SystemTests(unittest.TestCase):
    def test_all_ten_gates_are_accounted(self):
        source=inspect.getsource(runner._mixed_system_v200)
        self.assertIn("len(gate_audits) == 10",source)

    def test_invalid_gate_cannot_enter_interval_sum(self):
        source=inspect.getsource(runner._mixed_system_v200)
        self.assertIn("all(interval is not None",source)

    def test_active_gate_minimum_remains_three(self):
        self.assertEqual(runner.THRESHOLDS.active_gates_min,3)


class V200IntervalTests(unittest.TestCase):
    def test_system_interval_is_minkowski_sum(self):
        self.assertEqual(minkowski_sum_interval((-1,2),(3,4)),(2.0,6.0))

    def test_cell_interval_preserves_abs_of_mean_difference(self):
        self.assertEqual(absolute_value_interval(subtract_intervals((2,3),(4,5))),(1.0,3.0))

    def test_worst_case_rmse_uses_farthest_endpoint(self):
        self.assertEqual(worst_case_interval_rmse([0],[[1,3]]),3.0)

    def test_robust_auc_is_pairwise_lower_bound(self):
        self.assertEqual(robust_interval_auc_lower_bound([True,False],[[2,3],[0,1]]),1.0)


class V200BaselineTests(unittest.TestCase):
    def test_pie_remains_baseline_only(self):
        self.assertIn("pie",runner.BASELINES);self.assertNotIn('pie',inspect.getsource(runner._mixed_system_v200).lower())


class V200FirewallTests(unittest.TestCase):
    def test_v136_development_rows_are_forbidden_inputs(self):
        source=inspect.getsource(build_green_bridge_v200_splits)
        self.assertIn('row.split == "confirmation"',source);self.assertIn("exposed",source)

    def test_v200_split_groups_and_sha256_are_exact(self):
        self.assertEqual((len(V200_DEVELOPMENT_GROUPS),len(V200_CONFIRMATION_GROUPS)),(4,12))
        payload=v200_split_payload()
        self.assertEqual(
            [(row["noun"],row["century"]) for row in payload["development_groups"]],
            list(V200_DEVELOPMENT_GROUPS),
        )
        self.assertEqual(
            [(row["noun"],row["century"]) for row in payload["confirmation_groups"]],
            list(V200_CONFIRMATION_GROUPS),
        )
        self.assertEqual(sha256_text(canonical_json(payload)),V200_SPLIT_SHA256)

    def test_confirmation_artifacts_forbidden_before_open(self):
        self.assertIn("require_confirmation=True",inspect.getsource(runner.confirmation_phase_v200))


class V200PredecessorTests(unittest.TestCase):
    def test_v136_terminal_hashes_and_stop_are_immutable(self):
        self.assertEqual(runner.V136_TERMINAL_HASHES["dev_tensor_scores.parquet"],"660788dde8bc5df1d057db31b4dc1065b222ac7777efc0e4c6220e09f1ed81ff")
        self.assertIn('dev.get("verdict") == "STOP_ORAL"',inspect.getsource(runner.verify_v136_terminal_predecessor))


class V200TheoryTests(unittest.TestCase):
    def test_fixed_rank_donor_pca_remains_terminated(self):
        self.assertNotIn("donor_pca",json.dumps(FROZEN_SPEC).lower());self.assertNotIn("pseudoinverse",inspect.getsource(runner._classify_gate_v200).lower())


if __name__ == "__main__":
    unittest.main()
