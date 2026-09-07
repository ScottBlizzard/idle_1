"""Test the engineering-only CLI without performing real model work."""
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip('fcntl')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]


@pytest.mark.parametrize('derivative', [0, 2])
def test_full_acceptance_checks_ad_and_never_emits_formal_records(tmp_path, monkeypatch, derivative):
    import green_v410_fixed_budget_certificate as fixed
    import green_bridge_v400_tensor_program as program
    import green_bridge_v400_tensor_store as store
    import green_bridge_v400_resources as resources
    graph = tmp_path / 'graphs/job/direction_00'
    graph.mkdir(parents=True)
    (graph / 'graph_manifest.json').write_text(json.dumps({'direction_ordinal': 0}))
    (graph / 'tensor_program.json').write_text('{}')
    capture = tmp_path / 'capture/job/capture_manifest.json'
    capture.parent.mkdir(parents=True)
    names = ('PAT_J', 'PAT_B', 'TAR_J', 'TAR_B')
    capture.write_text(json.dumps({'response_panel': {'ad_derivatives': {
        name: [[derivative, 1]] for name in names}}}))
    output = tmp_path / 'engineering/acceptance.json'
    calls = []
    class Evaluator:
        def __init__(self, *args, **kwargs): pass
        def close(self): pass
    class Recorder:
        def __init__(self, **kwargs):
            self.record = SimpleNamespace(to_dict=lambda: {
                'wall_seconds': 1, 'peak_sampled_tree_rss_kib': 1})
        def __enter__(self): return self
        def __exit__(self, *args): pass
    def certify(evaluator, **kwargs):
        calls.append(kwargs)
        return {'official_intervals': {name: {'lower': [-1, 1], 'upper': [1, 1]}
                                      for name in (*names, 'PSI')}}
    monkeypatch.setattr(fixed, 'FixedBudgetProgramEvaluator', Evaluator)
    monkeypatch.setattr(fixed, 'certify_five_outputs', certify)
    monkeypatch.setattr(program.TensorProgram, 'from_dict', lambda value: value)
    monkeypatch.setattr(store, 'TensorStoreReader', lambda path: None)
    monkeypatch.setattr(resources, 'ProcessTreeResourceRecorder', Recorder)
    monkeypatch.setattr(sys, 'argv', ['probe', '--full', str(graph), 'test-backend',
                                    str(capture), str(output), '0'])
    script = ROOT / 'analysis/green_v410_certificate_repair_probe.py'
    if derivative == 0:
        with pytest.raises(SystemExit) as finished:
            runpy.run_path(str(script), run_name='__main__')
        assert finished.value.code == 0
        record = json.loads(output.read_text())
        assert record['state'] == 'NUMERICAL_ACCEPTANCE_PASS'
        assert record['engineering_only'] is True
        assert record['formal_launch_authorized'] is False
        with pytest.raises(FileExistsError):
            runpy.run_path(str(script), run_name='__main__')
    else:
        with pytest.raises(RuntimeError, match='AD_OVERLAP_FAILED'):
            runpy.run_path(str(script), run_name='__main__')
        assert not output.exists()
        assert output.with_name(output.name + '.failure.json').exists()
        with pytest.raises(RuntimeError, match='prior acceptance launch'):
            runpy.run_path(str(script), run_name='__main__')
    assert calls == [{'leaf_budget': 4}]
    assert not list(tmp_path.rglob('p13_record.json'))
