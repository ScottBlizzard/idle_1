"""Synthetic and real repair acceptance; never writes formal run artifacts."""
from pathlib import Path
import sys
import tempfile
import time
import json
import os
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
from test_green_v410_gpt2_program import _case
from green_v410_fixed_budget_certificate import FixedBudgetProgramEvaluator, certify_five_outputs
from green_bridge_v400_tensor_program import TensorProgram
from green_bridge_v400_tensor_store import TensorStoreReader
from green_bridge_v400_interval import Interval
from green_bridge_v400_mpfr_tensor_executor import jet_exact_payload
from green_v410_artifacts import atomic_no_clobber_json, configure_exact_integer_io


def progress_callback(label):
    last = [0.0]
    def progress(row):
        now = time.monotonic()
        if now - last[0] >= 60:
            print('NODE_PROGRESS', label, os.getpid(), row['ordinal'],
                  row['semantic_id'], row['kernel_id'], flush=True)
            last[0] = now
    return progress


def claim_run(directory):
    import fcntl
    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / '.repair_probe.lock').open('a+')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError('another repair probe owns this output directory') from exc
    return handle


def real_evaluation(arguments):
    configure_exact_integer_io()
    graph, backend, out, lo, hi = arguments
    evaluator = FixedBudgetProgramEvaluator(
        TensorProgram.from_dict(json.loads((graph / 'tensor_program.json').read_text())),
        TensorStoreReader(graph / 'tensor_store.json'), backend,
        successful_node_callback=progress_callback(f'{lo}:{hi}'),
    )
    start = time.monotonic()
    try:
        jets = evaluator.evaluate(Interval.from_bounds(lo, hi, 384))
        payload = {k: jet_exact_payload(j) for k, j in jets.items()}
        atomic_no_clobber_json(out, payload, job_id=out.stem)
        print('DOMAIN_DONE', lo, hi, time.monotonic() - start, flush=True)
    except Exception as exc:
        print('DOMAIN_FAILED', lo, hi, type(exc).__name__, str(exc), flush=True)
        atomic_no_clobber_json(out.with_name(out.name + '.failure.json'), {
            'state': 'DOMAIN_FAILED', 'engineering_only': True,
            'domain': [lo, hi], 'error_type': type(exc).__name__, 'error': str(exc),
        }, job_id=out.stem + '-failure')
        raise
    finally:
        evaluator.close()


def report_real(out):
    from green_bridge_v400_interval_jet import Jet2
    from green_bridge_v400_certificate import DyadicCell
    from green_v410_fixed_budget_certificate import _witnesses, EvaluatedCell
    import gmpy2
    def load(lo, hi):
        raw = json.loads((out / f'jet_{lo}_{hi}.json').read_text())
        return {name: Jet2(*(Interval.from_bounds(gmpy2.mpq(*v[c]['lower']),
                    gmpy2.mpq(*v[c]['upper']), 384)
                    for c in ('value', 'first', 'second'))) for name, v in raw.items()}
    cells = tuple(EvaluatedCell(DyadicCell(Fraction(lo), Fraction(hi)), load(lo, hi))
                  for lo, hi in [(-1, 0), (0, 1)])
    points = {Fraction(t): load(t, t) for t in (-1, 0, 1)}
    try:
        _witnesses(cells, points, Fraction(1))
        print('FIRST_RADIUS_TAYLOR_PASS', flush=True)
    except Exception:
        for name in points[Fraction(0)]:
            m, c, p = [points[Fraction(t)][name] for t in (-1, 0, 1)]
            direct = p.value - c.value - c.first
            curve = cells[1].jets[name].second * Interval.point(gmpy2.mpq(1, 2), 384)
            print(name, 'direct', float(direct.lower), float(direct.upper),
                  'curve', float(curve.lower), float(curve.upper), flush=True)
        raise

if __name__ == '__main__':
    configure_exact_integer_io()
    usage = (
        'usage: probe.py BACKEND TEMP_PARENT\n'
        '       probe.py --real GRAPH BACKEND OUTPUT_DIRECTORY\n'
        '       probe.py --report OUTPUT_DIRECTORY\n'
        '       probe.py --full GRAPH BACKEND CAPTURE OUTPUT_JSON DIRECTION_ORDINAL'
    )
    if len(sys.argv) == 2 and sys.argv[1] in ('-h', '--help'):
        print(usage)
        sys.exit(0)
    expected = {'--real': 5, '--report': 3, '--full': 7}
    if (len(sys.argv) < 2 or
            len(sys.argv) != expected.get(sys.argv[1], 3) or
            (sys.argv[1].startswith('--') and sys.argv[1] not in expected)):
        raise SystemExit(usage)
    if len(sys.argv) == 3 and sys.argv[1] == '--report':
        report_real(Path(sys.argv[2]))
        sys.exit(0)
    if len(sys.argv) == 7 and sys.argv[1] == '--full':
        # This is numerical repair acceptance, NOT a formal P13 worker or a
        # resumption of the failed scientific attempt.  No queue is rewritten.
        graph, backend, capture_path, out = map(Path, sys.argv[2:6])
        ordinal = int(sys.argv[6])
        graph_manifest = json.loads((graph / 'graph_manifest.json').read_text())
        if (not 0 <= ordinal < 8 or graph_manifest['direction_ordinal'] != ordinal
                or graph.parent.name != capture_path.parent.name):
            raise ValueError('repair graph/capture/direction mismatch')
        lock_handle = claim_run(out.parent)
        if out.exists():
            lock_handle.close()
            raise FileExistsError('acceptance output already exists; inspect it, do not rerun')
        start_record = out.with_name(out.name + '.started.json')
        if start_record.exists():
            lock_handle.close()
            raise RuntimeError('prior acceptance launch exists; do not silently restart it')
        atomic_no_clobber_json(start_record, {
            'pid': os.getpid(), 'started_unix_seconds': time.time(),
            'graph': str(graph), 'backend': str(backend), 'engineering_only': True,
        }, job_id='repair-acceptance-start')
        evaluator = FixedBudgetProgramEvaluator(
            TensorProgram.from_dict(json.loads((graph / 'tensor_program.json').read_text())),
            TensorStoreReader(graph / 'tensor_store.json'), backend,
            successful_node_callback=progress_callback('full-L4'),
        )
        started = time.monotonic()
        try:
            from green_bridge_v400_resources import ProcessTreeResourceRecorder
            with ProcessTreeResourceRecorder(sample_interval_seconds=0.01) as recorder:
                certificate = certify_five_outputs(evaluator, leaf_budget=4)
            resource = recorder.record.to_dict()
            if (resource['wall_seconds'] > 85800
                    or resource['peak_sampled_tree_rss_kib'] * 1024 > 68719476736):
                raise RuntimeError('REPAIR_ACCEPTANCE_RESOURCE_CEILING_EXCEEDED')
            capture = json.loads(capture_path.read_text())
            checks = {}
            for name in ('PAT_J', 'PAT_B', 'TAR_J', 'TAR_B'):
                derivative = Fraction(*capture['response_panel']['ad_derivatives'][name][ordinal])
                interval = certificate['official_intervals'][name]
                checks[name] = Fraction(*interval['lower']) <= derivative <= Fraction(*interval['upper'])
            if not all(checks.values()):
                raise RuntimeError('REPAIR_ACCEPTANCE_AD_OVERLAP_FAILED')
            atomic_no_clobber_json(out, {
                'state': 'NUMERICAL_ACCEPTANCE_PASS', 'engineering_only': True,
                'formal_launch_authorized': False, 'graph': str(graph), 'backend': str(backend),
                'capture': str(capture_path), 'direction_ordinal': ordinal,
                'elapsed_seconds': time.monotonic() - started,
                'resource_record': resource,
                'independent_ad_overlaps': checks, 'certificate': certificate,
            }, job_id='repair-full-acceptance')
            print('NUMERICAL_ACCEPTANCE_PASS', out, flush=True)
        except Exception as exc:
            atomic_no_clobber_json(out.with_name(out.name + '.failure.json'), {
                'state': 'NUMERICAL_ACCEPTANCE_FAILED', 'engineering_only': True,
                'error_type': type(exc).__name__, 'error': str(exc),
                'elapsed_seconds': time.monotonic() - started,
                'formal_launch_authorized': False,
            }, job_id='repair-acceptance-failure')
            raise
        finally:
            evaluator.close()
            lock_handle.close()
        sys.exit(0)
    if len(sys.argv) > 3 and sys.argv[1] == '--real':
        graph, backend, out = map(Path, sys.argv[2:5])
        lock_handle = claim_run(out)
        if any(out.glob('*.failure.json')):
            raise RuntimeError('prior diagnostic failure exists; inspect it, do not silently retry')
        domains = [(-1, 0), (0, 1), (-1, -1), (0, 0), (1, 1)]
        args = [(graph, backend, out / f'jet_{lo}_{hi}.json', lo, hi)
                for lo, hi in domains if not (out / f'jet_{lo}_{hi}.json').exists()]
        try:
            with ProcessPoolExecutor(max_workers=5) as pool:
                list(pool.map(real_evaluation, args))
            report_real(out)
        finally:
            lock_handle.close()
        sys.exit(0)
    backend = Path(sys.argv[1])
    with tempfile.TemporaryDirectory(dir=sys.argv[2]) as tmp:
        for layer in (0, 4, 8):
            case = _case(Path(tmp) / str(layer), layer)
            evaluator = FixedBudgetProgramEvaluator(case[6], case[4], backend)
            start = time.monotonic()
            try:
                certify_five_outputs(evaluator, leaf_budget=4)
                print('SYNTHETIC_FULL_CONE_PASS', layer, time.monotonic() - start, flush=True)
            finally:
                evaluator.close()
