"""Small integration fixtures exercise real policy replay without model/network I/O."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eot_stream import cli


@pytest.fixture
def setup_run(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'annotations.jsonl').write_text('source annotations')
    (source / 'manifest.sha256').write_text('source manifest')
    output = tmp_path / 'result'
    cache_root = tmp_path / 'cache'
    events = []
    def turn(identifier, split):
        return dict(id=identifier, split=split, audio=np.ones(16000, dtype=np.float32),
                    eot_s=.5, original_duration_s=1., hold_spans=[],
                    source_audio_sha256=identifier * 8, label_type='pseudo')
    dev = [turn('d1', 'livekit_dev'), turn('d2', 'livekit_dev')]
    pilot = [turn('p1', 'minds_pilot'), turn('p2', 'minds_pilot')]
    monkeypatch.setattr(cli, 'EXPECTED_DEV', 2)
    monkeypatch.setattr(cli, 'EXPECTED_PILOT', 2)
    def build(source_root, output_root):
        path = Path(output_root)
        path.mkdir(parents=True)
        (path / 'manifest.json').write_text('{}')
        return {}
    monkeypatch.setattr(cli.data, 'build_pilot', build)
    monkeypatch.setattr(cli.data, 'load_pilot_turns', lambda path: pilot)
    monkeypatch.setattr(cli.data, 'load_dev_turns', lambda path: dev)
    monkeypatch.setattr(cli.stream, 'SileroFrames', lambda path: SimpleNamespace(metadata={'sha256': 'vad'}))
    class Detector:
        metadata = {'sha256': 'model', 'setting': 'fake'}
        def predict(self, value):
            return .9
        def close(self):
            events.append('closed')
    monkeypatch.setattr(cli.models, 'load_detector', lambda name, **kw: None if name == 'timer' else Detector())
    def trace_path(root, model, split, identifier):
        return Path(root) / model / split / (identifier + '.json')
    def get_trace(turn, model, detector, vad, root, metadata):
        if turn['split'] == 'minds_pilot':
            frozen = json.loads((output / 'frozen_policies.json').read_text())
            assert set(frozen['models']) == {'timer', 'smart-turn', 'livekit-mini'}
        events.append((model, turn['split'], turn['id']))
        if detector:
            detector.predict(turn['audio'])
        frames = [dict(timestamp_s=.72, silence_age_s=.224, region_id=0,
                       p_eot=.9 if detector else None, is_score_event=True, inference_ms=2.)]
        ref = {k: v for k, v in turn.items() if k != 'audio'}
        ref.update(horizon_s=4.5, synthetic_silence_s=3.5)
        path = trace_path(root, model, turn['split'], turn['id'])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'frames': frames, 'ref': ref}))
        return frames, ref
    monkeypatch.setattr(cli.cache, 'get_trace', get_trace)
    monkeypatch.setattr(cli.cache, 'trace_path', trace_path)
    args = ['--source-root', str(source), '--output', str(output), '--cache-dir', str(cache_root),
            '--dev-cache-dir', str(tmp_path / 'dev'), '--model-cache-dir', str(tmp_path / 'models'),
            '--vad-path', str(tmp_path / 'vad.onnx')]
    return SimpleNamespace(args=args, source=source, output=output, dev=dev, pilot=pilot, events=events)


def test_complete_freezes_all_dev_before_pilot_and_reports_common_ids(setup_run):
    s = setup_run
    assert cli.main(s.args) == 0
    manifest = json.loads((s.output / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert manifest['counts']['development']['turns'] == 2
    first_pilot = next(i for i, e in enumerate(s.events) if isinstance(e, tuple) and e[1] == 'minds_pilot')
    assert sum(isinstance(e, tuple) and e[1] == 'livekit_dev' for e in s.events[:first_pilot]) == 6
    summary = json.loads((s.output / 'summary.json').read_text())
    assert len(summary['rows']) == 12
    for model in ['timer', 'smart-turn', 'livekit-mini']:
        index = json.loads((s.output / 'traces' / model / 'index.json').read_text())
        assert [row['id'] for row in index] == ['p1', 'p2']
        assert all(row['sha256'] for row in index)
    assert 'pseudo' in (s.output / 'report.txt').read_text()
    assert 'Silero' in (s.output / 'report.txt').read_text()
    assert (s.output / 'data.json').exists()


def test_nonfinite_score_fails_closed_and_marks_manifest(setup_run, monkeypatch):
    old = cli.cache.get_trace
    def bad(*args):
        frames, ref = old(*args)
        frames[0]['p_eot'] = float('nan')
        return frames, ref
    monkeypatch.setattr(cli.cache, 'get_trace', bad)
    assert cli.main(setup_run.args) == 1
    result = json.loads((setup_run.output / 'manifest.json').read_text())
    assert result['status'] == 'failed'
    assert not (setup_run.output / 'summary.json').exists()
    assert not any(isinstance(e, tuple) and e[1] == 'minds_pilot' for e in setup_run.events)


def test_resume_requires_unchanged_data_and_protocol(setup_run):
    s = setup_run
    assert cli.main(s.args) == 0
    assert cli.main(s.args) == 0
    s.dev[0]['eot_s'] = .6
    assert cli.main(s.args) == 1
    assert json.loads((s.output / 'manifest.json').read_text())['status'] == 'failed'


def test_unclaimed_populated_output_is_rejected(setup_run):
    s = setup_run
    s.output.mkdir()
    (s.output / 'unrelated.txt').write_text('preserve')
    assert cli.main(s.args) == 1
    assert not (s.output / 'manifest.json').exists()
    assert (s.output / 'unrelated.txt').read_text() == 'preserve'


def test_overlap_is_rejected_before_scoring(setup_run):
    s = setup_run
    s.pilot[0]['source_audio_sha256'] = s.dev[0]['source_audio_sha256']
    assert cli.main(s.args) == 1
    assert not s.events
    assert json.loads((s.output / 'manifest.json').read_text())['status'] == 'failed'


def test_resume_rejects_modified_saved_result(setup_run):
    s = setup_run
    assert cli.main(s.args) == 0
    (s.output / 'report.txt').write_text('modified saved result')
    s.events.clear()
    assert cli.main(s.args) == 1
    assert not s.events
    assert json.loads((s.output / 'manifest.json').read_text())['status'] == 'failed'


def test_native_model_metadata_tracks_binary_bytes(tmp_path, monkeypatch):
    native_path = tmp_path / '_native.so'
    native_path.write_bytes(b'first native runtime')
    detector = SimpleNamespace(metadata={'package': 'livekit-local-inference',
                                         'version': '0.2.7', 'context_seconds': 1.2})
    monkeypatch.setattr(cli.importlib, 'import_module',
                        lambda name: SimpleNamespace(__file__=str(native_path)))
    before = cli._runtime_model_metadata(detector)
    native_path.write_bytes(b'replaced native runtime')
    after = cli._runtime_model_metadata(detector)
    assert before['native_binary_file'] == '_native.so'
    assert before['native_binary_sha256'] != after['native_binary_sha256']
    assert detector.metadata == {'package': 'livekit-local-inference',
                                 'version': '0.2.7', 'context_seconds': 1.2}
