import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import green_v400_formal_grant_runner as grant_runner
from green_v400_execution_receipts import build_model_session_receipt


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def file_hash(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


class FakeGrantModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def run_with_hooks(self, tokens, fwd_hooks):
        residual = torch.stack(
            [tokens.float(), tokens.float() * 0.5], dim=-1
        ) + 0.0 * self.anchor
        hooks = dict(fwd_hooks)
        candidate = hooks.get("blocks.2.hook_resid_post")
        if candidate is not None:
            residual = candidate(residual, None)
        # The final position already contains a bypass trace of the original
        # corrupt token, so full patching only the candidate position need not
        # restore the downstream target state.
        measurement = residual.clone()
        measurement[:, -1, 0] += tokens[:, 1].float()
        capture = hooks.get("blocks.10.hook_resid_post")
        if capture is not None:
            measurement = capture(measurement, None)
        return torch.zeros(tokens.shape[0], tokens.shape[1], 8)


def fixtures():
    protocol = "GREEN_V400_SILENT_FAILURE_CHALLENGE_PREPARE_V1"
    capture_spec = json.loads(
        (ROOT / "configs/green_v400_grant_capture_spec.json").read_text()
    )
    rows = []
    prediction_jobs = []
    for index in range(4):
        prompt_id = f"{index + 1:064x}"
        site_id = f"{index + 101:064x}"
        rows.append(
            {
                "row_id": prompt_id,
                "clean_token_ids": [3, 5 + index, 7, 9],
                "corrupt_token_ids": [3, 15 + index, 7, 9],
                "signature": [4, 1],
            }
        )
        prediction_jobs.append(
            {
                "job_id": f"{index + 201:064x}",
                "kind": "prediction",
                "role": "development",
                "site_row_id": site_id,
                "prompt_row_id": prompt_id,
                "layer": 2,
                "hook": "resid_post",
            }
        )
    prediction_jobs.sort(key=lambda row: row["site_row_id"])
    cohort_hash = canonical_hash([row["site_row_id"] for row in prediction_jobs])
    grant_job = {
        "job_id": "ab" * 32,
        "kind": "grant_cohort_prediction",
        "role": "development",
        "layer": 2,
        "hook": "resid_post",
        "cohort_site_row_ids_sha256": cohort_hash,
        "cohort_size": 4,
        "grant_capture_spec_sha256": canonical_hash(capture_spec),
        "analysis_seed": 123,
    }
    universe = {"rows": rows}
    sources = {
        "analysis/green_v400_formal_worker.py": "66" * 32,
        grant_runner.RUNNER_SOURCE_PATH: file_hash(grant_runner.RUNNER_SOURCE_PATH),
        grant_runner.GRANT_CORE_SOURCE_PATH: file_hash(
            grant_runner.GRANT_CORE_SOURCE_PATH
        ),
        grant_runner.GRANT_PACKET_SOURCE_PATH: file_hash(
            grant_runner.GRANT_PACKET_SOURCE_PATH
        ),
    }
    plan = {
        "protocol_id": protocol,
        "execution_enabled": True,
        "universe_sha256": canonical_hash(universe),
        "grant_capture_spec_path": grant_runner.CAPTURE_SPEC_PATH,
        "grant_capture_spec_sha256": canonical_hash(capture_spec),
        "model_manifest_sha256": "44" * 32,
        "model_revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "full_model_hash": "55" * 32,
        "source_file_sha256": sources,
        "queues": {
            "development_prediction": prediction_jobs,
            "confirmation_prediction": [],
            "development_grant_cohort_prediction": [grant_job],
            "confirmation_grant_cohort_prediction": [],
        },
    }
    plan["plan_sha256"] = canonical_hash(plan)
    session = build_model_session_receipt(
        plan=plan,
        observed_full_model_hash=plan["full_model_hash"],
        loader_source_sha256="66" * 32,
        process_start_nonce="77" * 32,
        pid=1234,
    )
    return plan, universe, capture_spec, grant_job, session


def test_formal_grant_collects_plan_rows_at_final_position(monkeypatch):
    plan, universe, capture_spec, job, session = fixtures()
    observed = {}

    def fake_packet(**kwargs):
        observed.update(kwargs)
        return {"row_id": kwargs["cohort_id"]}, {"prediction_packet_sha256": "aa" * 32}

    monkeypatch.setattr(
        grant_runner, "compute_grant_divergence_prediction_packet", fake_packet
    )
    packet, commitment = grant_runner.run_formal_grant_prediction(
        plan=plan,
        universe=universe,
        capture_spec=capture_spec,
        grant_job_id=job["job_id"],
        model_session_receipt=session,
        model=FakeGrantModel(),
    )
    assert packet["row_id"] == job["cohort_site_row_ids_sha256"]
    assert commitment["prediction_packet_sha256"]
    assert observed["natural_states"].shape == (4, 2)
    assert observed["intervened_states"].shape == (4, 2)
    assert observed["unpatched_corrupt_states"].shape == (4, 2)
    assert torch.any(observed["natural_states"] != observed["intervened_states"])
    assert observed["formal_execution_binding"]["raw_activation_serialized"] is False
    assert observed["phase"] == "development"


def test_formal_grant_rejects_noncausal_measurement_position(monkeypatch):
    plan, universe, capture_spec, job, session = fixtures()
    for row in universe["rows"]:
        row["signature"][1] = len(row["clean_token_ids"]) - 1
    plan["universe_sha256"] = canonical_hash(universe)
    plan.pop("plan_sha256")
    plan["plan_sha256"] = canonical_hash(plan)
    session = build_model_session_receipt(
        plan=plan,
        observed_full_model_hash=plan["full_model_hash"],
        loader_source_sha256="66" * 32,
        process_start_nonce="88" * 32,
        pid=1235,
    )
    with pytest.raises(ValueError, match="token pair or position"):
        grant_runner.run_formal_grant_prediction(
            plan=plan,
            universe=universe,
            capture_spec=capture_spec,
            grant_job_id=job["job_id"],
            model_session_receipt=session,
            model=FakeGrantModel(),
        )
