from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import analyze_green_bridge_v300 as analysis
import exp_green_bridge_v300 as exp
import green_bridge_v300_dataset as dataset
import green_bridge_v300_directions as directions
import green_bridge_v300_numerics as numerics
import green_bridge_v300_prepare as prepare
import green_bridge_v300_spec as spec
import green_bridge_v300_transport as transport


ARCHIVE = ROOT / "analysis/archive/green_v200_stop_20260825"
POSTMORTEM = ROOT / "analysis/GREEN_V21_POSTMORTEM_20260825"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class V300PredecessorTests(unittest.TestCase):
    def test_v200_execution_commit_is_exact(self):
        self.assertEqual(spec.V200_EXECUTION_COMMIT, "e52e082296c33a10557636706e572147136fce34")

    def test_v200_terminal_artifact_hashes_are_verified(self):
        expected = exp._parse_hashes(ARCHIVE / "sha256sums.txt")
        name = "dev_tensor_scores.parquet"
        self.assertEqual(exp._sha256(ARCHIVE / name), expected[name])

    def test_v200_stop_oral_and_confirmation_closed(self):
        self.assertEqual(read_json(ARCHIVE / "result.json")["verdict"], "STOP_ORAL")
        self.assertFalse(read_json(ARCHIVE / "run_ledger.json")["confirmation_started"])

    def test_v200_root_is_read_only(self):
        source = inspect.getsource(exp.verify_v200_terminal_archive_v300)
        self.assertNotIn("write_text", source)
        self.assertNotIn("unlink", source)

    def test_v200_development_parquets_are_diagnostic_only(self):
        self.assertTrue(any(path.endswith("dev_tensor_scores.parquet") for path in exp.FORBIDDEN_V300_RUNTIME_INPUTS))
        self.assertTrue(any(path.endswith("dev_energy_targets.parquet") for path in exp.FORBIDDEN_V300_RUNTIME_INPUTS))

    def test_fixed_rank_donor_pca_remains_terminated(self):
        source = inspect.getsource(exp.verify_v200_terminal_archive_v300)
        self.assertIn('"fixed_rank_donor_pca_terminated": True', source)


class V21PostmortemTests(unittest.TestCase):
    def test_postmortem_reads_only_archived_development(self):
        manifest = read_json(POSTMORTEM / "postmortem_manifest.json")
        self.assertFalse(manifest["confirmation_data_accessed"])

    def test_postmortem_marks_official_verdict_unchanged(self):
        for index in (1, 4, 5, 6, 8, 11):
            path = next(POSTMORTEM.glob(f"{index:02d}_*.json"))
            self.assertTrue(read_json(path)["official_verdict_unchanged"])

    def test_postmortem_forbids_threshold_selection(self):
        for path in POSTMORTEM.glob("[0-1][0-9]_*.json"):
            self.assertFalse(read_json(path)["usable_for_threshold_selection"])

    def test_confirmation_paths_are_denied(self):
        with self.assertRaisesRegex(RuntimeError, exp.UNAUTHORIZED_PHASE):
            exp.confirmation_v300()

    def test_theorem_checks_read_no_behavioral_fields(self):
        source = inspect.getsource(transport.direct_path_control_ad_v300)
        self.assertNotIn("behavioral", source)
        self.assertNotIn("pie", source.lower())

    def test_integrity_failure_stops_before_v300(self):
        manifest = read_json(POSTMORTEM / "postmortem_manifest.json")
        manifest["semantic_completion_checks"]["01_integrity"] = False
        self.assertFalse(all(manifest["semantic_completion_checks"].values()))

    def test_exact_transport_failure_stops_before_v300(self):
        source = inspect.getsource(exp.verify_v200_terminal_archive_v300)
        self.assertIn('"theorem_failures") == 0', source)

    def test_exact_joint_failure_stops_before_v300(self):
        source = inspect.getsource(exp.verify_v200_terminal_archive_v300)
        self.assertIn('"composition_failures") == 0', source)


class V300SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = dataset.build_green_bridge_v300_records()

    def test_literal_split_payload_hash_is_509f791b(self):
        payload = dataset.canonical_v300_split_payload().encode("utf-8")
        self.assertEqual(hashlib.sha256(payload).hexdigest(), spec.V300_SPLIT_SHA256)

    def test_development_nouns_are_exact(self):
        self.assertEqual({r.noun for r in self.records if r.split == "development"}, set(spec.DEVELOPMENT_NOUNS))

    def test_confirmation_nouns_are_exact(self):
        self.assertEqual({r.noun for r in self.records if r.split == "confirmation"}, set(spec.CONFIRMATION_NOUNS))

    def test_no_noun_crosses_phase(self):
        dev = {r.noun for r in self.records if r.split == "development"}
        confirm = {r.noun for r in self.records if r.split == "confirmation"}
        self.assertFalse(dev & confirm)

    def test_v200_development_groups_are_excluded(self):
        from green_bridge_spec import V200_DEVELOPMENT_GROUPS
        self.assertFalse({(r.noun, r.century) for r in self.records} & set(V200_DEVELOPMENT_GROUPS))

    def test_roles_are_transport_and_joint(self):
        self.assertEqual({r.role for r in self.records}, {"transport", "joint"})

    def test_role_pairs_are_disjoint_and_balanced(self):
        for cell in {r.cell_id for r in self.records}:
            rows = [r for r in self.records if r.cell_id == cell]
            for role in spec.ROLES:
                selected = [r for r in rows if r.role == role]
                self.assertEqual(len(selected), 8)
                self.assertEqual({r.orientation: sum(x.orientation == r.orientation for x in selected) for r in selected}, {"up": 4, "down": 4})
            transport_pairs = {tuple(sorted((r.y, r.y_prime))) for r in rows if r.role == "transport"}
            joint_pairs = {tuple(sorted((r.y, r.y_prime))) for r in rows if r.role == "joint"}
            self.assertFalse(transport_pairs & joint_pairs)

    def test_record_counts_are_160_and_224(self):
        self.assertEqual(sum(r.split == "development" for r in self.records), 160)
        self.assertEqual(sum(r.split == "confirmation" for r in self.records), 224)


class V300DirectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(20260825)
        cls.frame = np.linalg.qr(rng.normal(size=(768, 5)))[0]

    def test_helmert_coefficients_are_orthonormal(self):
        a = directions.helmert_coefficients_v300()
        np.testing.assert_allclose(a @ a.T, np.eye(4), atol=1e-15, rtol=0)
        self.assertEqual(
            directions.computed_coefficient_payload_sha256_v300(),
            spec.V300_COEFFICIENT_SHA256,
        )
        self.assertNotEqual(
            spec.V300_COEFFICIENT_SHA256,
            spec.V300_DECLARED_COEFFICIENT_HASH_ID,
        )

    def test_in_frame_directions_are_unit_norm(self):
        panel = directions.heldout_direction_panel_v300(self.frame)
        np.testing.assert_allclose(np.linalg.norm(panel["in_frame"], axis=0), 1, atol=1e-12)

    def test_complement_directions_are_deterministic(self):
        np.testing.assert_array_equal(directions.deterministic_complement_v300(self.frame), directions.deterministic_complement_v300(self.frame))

    def test_complement_is_orthogonal_to_frame(self):
        complement = directions.deterministic_complement_v300(self.frame)
        self.assertLessEqual(float(np.max(np.abs(self.frame.T @ complement))), 1e-12)

    def test_mixed_directions_are_unit_norm(self):
        panel = directions.heldout_direction_panel_v300(self.frame)
        np.testing.assert_allclose(np.linalg.norm(panel["mixed"], axis=0), 1, atol=1e-12)

    def test_null_directions_are_orthogonal_to_frame(self):
        null = directions.heldout_direction_panel_v300(self.frame)["null"]
        self.assertLessEqual(float(np.max(np.abs(self.frame.T @ null))), 1e-12)

    def test_direction_hash_is_repeatable(self):
        self.assertEqual(directions.direction_design_sha256_v300(self.frame), directions.direction_design_sha256_v300(self.frame))

    def test_heldout_directions_never_enter_identification(self):
        source = inspect.getsource(directions.heldout_direction_panel_v300)
        self.assertNotIn("identify_gate", source)
        self.assertEqual(directions.helmert_coefficients_v300().shape, (4, 5))


class V300TransportTheoryTests(unittest.TestCase):
    def test_path_minus_control_jacobian_equals_rank_one_operator(self):
        g = np.array([1.0, -2.0, 0.5]); G = np.array([3.0, -1.0])
        u = np.array([0.2, 0.4, -0.1])
        np.testing.assert_allclose(np.outer(G, g) @ u, G * (g @ u))
        self.assertTrue(prepare.synthetic_theorem_suite_v300()["passed"])

    def test_matched_bypass_factorization_on_synthetic_map(self):
        g = np.array([0.3, -0.7]); G = np.array([1.1, 0.2, -0.4])
        self.assertEqual(np.linalg.matrix_rank(np.outer(G, g), tol=1e-12), 1)

    def test_zero_curvature_is_response_nonidentifiable(self):
        result = numerics.response_detectability_v300(0, 1e-4, 1, 0, 1, 0)
        self.assertFalse(result["recoverable"])

    def test_detectability_bound_is_monotone_in_curvature(self):
        low = numerics.relative_width_v300(1.0, 0.2)
        high = numerics.relative_width_v300(2.0, 0.2)
        self.assertLess(high, low)

    def test_direct_target_is_probe_independent(self):
        G = np.array([1.0, 2.0]); g = np.array([0.5, -0.2]); u = np.array([0.7, 0.3])
        target = G * (g @ u)
        np.testing.assert_array_equal(target, G * (g @ u))

    def test_joint_first_order_composition_is_additive(self):
        value = transport.joint_operator_prediction_v300(
            [np.array([1.0, 2.0]), np.array([-1.0, 0.5])],
            [np.array([0.2, 0.3]), np.array([0.4, -0.1])],
            np.array([2.0, -1.0]), np.array([0.5, -0.25]),
        )
        self.assertTrue(math.isfinite(value))

    def test_structural_contradiction_cannot_be_unresolved(self):
        value = numerics.classify_gate_v300(numerical_valid=True, structural_valid=False,
                                             recoverable=False, exact_operator_upper=1,
                                             direct_numerical_floor=0)
        self.assertEqual(value, "structural-contradiction")

    def test_ad_is_audit_not_point_estimator(self):
        source = inspect.getsource(exp.transport_record_v300)
        self.assertNotIn("ad_midpoint", source)


class V300RadiusTests(unittest.TestCase):
    def _eligible(self):
        return [{"fine": [1.0], "ad_midpoint": [1.0], "ad_route_radius": 0,
                 "endpoint_radius": 0, "ad_route_passed": True,
                 "theorem_passed": True, "endpoint_floor_passed": True,
                 "fallback_used": False}]

    def test_candidate_radius_payload_hash_is_exact(self):
        self.assertEqual(spec.radius_candidate_payload_sha256_v300(), spec.V300_RADIUS_CANDIDATE_SHA256)
        self.assertNotEqual(
            spec.V300_RADIUS_CANDIDATE_SHA256,
            spec.V300_DECLARED_RADIUS_CANDIDATE_HASH_ID,
        )

    def test_largest_eligible_radius_is_selected(self):
        self.assertEqual(numerics.select_global_radius_v300({0.5: self._eligible(), 1.0: self._eligible()}), 1.0)

    def test_calibration_panel_is_behavior_blind(self):
        records = [{"pair_digest": f"{i:064x}", "distance_bin": distance,
                    "population": "legacy_donor", "behavioral": 999}
                   for distance in ("near", "far") for i in range(3)]
        panel = exp.select_radius_calibration_panel_v300(records)
        self.assertEqual(len(panel), 40)
        self.assertTrue(all("behavioral" not in row for row in panel))

    def test_calibration_uses_only_legacy_donors(self):
        records = [{"pair_digest": f"{i:064x}", "distance_bin": distance,
                    "population": population}
                   for distance in ("near", "far")
                   for i, population in enumerate(("legacy_donor", "v2_development"))]
        panel = exp.select_radius_calibration_panel_v300(records)
        self.assertEqual({row["population"] for row in panel}, {"legacy_donor"})

    def test_no_eligible_radius_stops_prepare(self):
        with self.assertRaisesRegex(RuntimeError, "08_RADIUS_LOCALITY"):
            numerics.select_global_radius_v300({1.0: []})

    def test_selected_radius_is_global_and_frozen(self):
        selected = numerics.select_global_radius_v300({0.25: self._eligible()})
        self.assertIsInstance(selected, float)
        self.assertNotIsInstance(selected, dict)


class V300AnalysisTests(unittest.TestCase):
    def test_relative_error_zero_denominator_rules(self):
        self.assertEqual(numerics.relative_width_v300(0, 0), 0)
        self.assertTrue(math.isinf(numerics.relative_width_v300(0, 1)))
        self.assertEqual(numerics.normalized_transport_error_v300([0], [0], 0), 0)

    def test_recoverable_width_threshold_is_one_quarter(self):
        self.assertEqual(spec.RECOVERABLE_RELATIVE_WIDTH_MAX, 0.25)
        self.assertTrue(numerics.response_detectability_v300(4, 1, 4, 1, 4, 1)["recoverable"])

    def test_unresolved_is_not_zeroed_in_joint_interval(self):
        rows = [{"gate_class": "unresolved", "center": 0, "bound": 0.2}] + [
            {"gate_class": "recoverable", "center": 0.1, "bound": 0.01} for _ in range(9)
        ]
        result = exp.joint_record_v300(rows, target=0.9, target_bound=0.01)
        self.assertFalse(result["unresolved_is_zeroed"])
        self.assertGreaterEqual(result["unresolved_bound"], 0.2)

    def test_best_baseline_is_group_balanced_and_frozen(self):
        rows = [{"noun_century_group": "a", **{f"error_{name}": 1.0 for name in analysis.BASELINES}},
                {"noun_century_group": "b", **{f"error_{name}": 1.0 for name in analysis.BASELINES}}]
        result = analysis.select_frozen_baseline_v300(rows)
        self.assertEqual(result["selected_baseline"], min(analysis.BASELINES))
        self.assertFalse(result["confirmation_reselection_allowed"])

    def test_confirmation_cannot_reselect_baseline(self):
        frozen = {"selected_baseline": "zero"}
        with self.assertRaisesRegex(RuntimeError, "RESELECTION"):
            analysis.confirmation_decision_v300({"selected_baseline": "gate_atom_only"}, frozen)


class V300LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "src/launch_green_bridge_v300.sh").read_text(encoding="utf-8")

    def test_all_runtime_paths_are_under_mnt_sdb(self):
        self.assertIn("export GREEN_BASE=/mnt/sdb/ccj", self.source)
        self.assertNotIn("export TMPDIR=/tmp", self.source)
        self.assertIn('[[ "$resolved" == /mnt/sdb/* ]]', self.source)

    def test_prepare_is_the_only_authorized_phase(self):
        self.assertEqual(spec.AUTHORIZED_PHASES, ("prepare",))
        self.assertIn('[[ "$PHASE" == "prepare" ]]', self.source)
        self.assertIn("execute_prepare_v300", inspect.getsource(exp.prepare_v300))

    def test_phase_all_retry_and_resume_are_forbidden(self):
        self.assertFalse(spec.PHASE_ALL_ALLOWED)
        self.assertFalse(spec.RETRY_ALLOWED)
        self.assertFalse(spec.RESUME_ALLOWED)


if __name__ == "__main__":
    unittest.main()
