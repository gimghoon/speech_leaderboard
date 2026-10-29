"""Development-only policy freezing followed by a common-ID local pilot run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import shlex
import shutil
import sys
import time

import numpy as np
from . import cache, data, models, policy, report, stream

EXPECTED_DEV = 377
EXPECTED_PILOT = 160
BUDGETS = (.05, .10)
MODEL_LABELS = {'timer': 'Silence timer (common causal Silero VAD)',
                'smart-turn': 'Smart Turn v3.2 (CPU INT8)',
                'livekit-mini': 'LiveKit Turn Detector v1-mini (local)'}
RUN_VERSION = 'korean-eot-stream-run-v1'


def _now():
    return datetime.now(timezone.utc).isoformat()


def _turn_metadata(turn):
    return {k: v for k, v in turn.items() if k != 'audio'}


def _runtime_model_metadata(detector):
    if detector is None:
        return dict(model='timer', clock='common_causal_Silero_VAD_audio_timeline', no_model_inference=True)
    metadata = dict(detector.metadata)
    if metadata.get('package') == 'livekit-local-inference':
        native = importlib.import_module('livekit.local_inference._native')
        binary = Path(native.__file__)
        metadata.update(native_binary_file=binary.name, native_binary_sha256=report.sha256(binary))
    return metadata


def _dataset_fingerprint(turns):
    entries = []
    for turn in turns:
        audio = np.asarray(turn['audio'])
        if audio.dtype != np.float32 or audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError('Expected normalized finite mono float32 PCM')
        entries.append(dict(reference=_turn_metadata(turn),
                            normalized_pcm_sha256=hashlib.sha256(audio.tobytes()).hexdigest()))
    return report.canonical_hash(entries)


def _check_turns(dev, pilot):
    for label, turns, expected, split in [('development', dev, EXPECTED_DEV, 'livekit_dev'),
                                          ('pilot', pilot, EXPECTED_PILOT, 'minds_pilot')]:
        ids = [t['id'] for t in turns]
        if len(turns) != expected or len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids):
            raise ValueError(f'{label} requires exactly {expected} distinct IDs')
        if any(t['split'] != split for t in turns):
            raise ValueError(f'Incorrect {label} split')
        if any(not isinstance(t.get('source_audio_sha256'), str) or not t['source_audio_sha256'] for t in turns):
            raise ValueError('Missing source audio digest')
    if {t['id'] for t in dev} & {t['id'] for t in pilot}:
        raise ValueError('Development and pilot IDs overlap')
    if {t['source_audio_sha256'] for t in dev} & {t['source_audio_sha256'] for t in pilot}:
        raise ValueError('Development and pilot source audio hashes overlap')


def _counts(turns):
    return dict(turns=len(turns), eot=len(turns), hold_spans=sum(len(t['hold_spans']) for t in turns),
                hold_turns=sum(bool(t['hold_spans']) for t in turns))


def _trace_set(turns, model_name, detector, vad, cache_root, model_metadata, phase):
    traces, refs, hashes = {}, {}, []
    for number, turn in enumerate(turns, 1):
        frames, ref = cache.get_trace(turn, model_name, detector, vad, cache_root, model_metadata)
        if ref['id'] != turn['id']:
            raise ValueError('Cache returned a different reference ID')
        traces[turn['id']], refs[turn['id']] = frames, ref
        path = cache.trace_path(cache_root, model_name, turn['split'], turn['id'])
        hashes.append(dict(id=turn['id'], path=str(path.resolve()), sha256=report.sha256(path)))
        if number % 25 == 0 or number == len(turns):
            print(f'{model_name}: {phase} {number}/{len(turns)}', flush=True)
    return traces, refs, hashes


def _export_pilot_traces(output, model_name, traces, refs, hashes):
    destination = output / 'traces' / model_name
    destination.mkdir(parents=True, exist_ok=True)
    index = []
    for entry in hashes:
        filename = hashlib.sha256(entry['id'].encode()).hexdigest() + '.json'
        target = destination / filename
        shutil.copyfile(entry['path'], target)
        if report.sha256(target) != entry['sha256']:
            raise ValueError('Exported pilot trace hash mismatch')
        index.append(dict(id=entry['id'], path=str(target.relative_to(output)), sha256=entry['sha256']))
    report.write_json(destination / 'index.json', index)


def _parse(argv):
    parser = argparse.ArgumentParser(description='Local Korean streaming EOT pilot; development-only frozen policies.')
    for name in ('source-root', 'output', 'cache-dir', 'dev-cache-dir', 'model-cache-dir', 'vad-path'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--models', nargs='+', choices=sorted(MODEL_LABELS),
                        default=['timer', 'smart-turn', 'livekit-mini'])
    args = parser.parse_args(argv)
    if len(set(args.models)) != len(args.models):
        parser.error('Duplicate models are not permitted')
    return args


def run(args, command):
    output = args.output.resolve()
    manifest_path = output / 'manifest.json'
    if output.exists() and any(output.iterdir()) and not manifest_path.is_file():
        raise ValueError('Populated output has no matching run manifest; choose a fresh destination')
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    # Refuse protocol/source changes before mutating an existing run directory.
    base = dict(run_version=RUN_VERSION, protocol=stream.PROTOCOL,
                data_version=data.DATA_VERSION, dev_revision=data.DEV_REVISION,
                resampling_method=data.RESAMPLING_METHOD, resampling_delay_s=data.RESAMPLING_DELAY_S,
                source_annotations_sha256=report.sha256(args.source_root / 'annotations.jsonl'),
                source_manifest_sha256=report.sha256(args.source_root / 'manifest.sha256'),
                models=args.models, budgets=list(BUDGETS))
    base_hash = report.canonical_hash(base)
    if previous and previous.get('base_identity_sha256') != base_hash:
        raise ValueError('Resume rejected: source/protocol/model selection differs from manifest')
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(previous or {}, status='running', command=command, updated_at=_now(),
                    started_at=(previous or {}).get('started_at', _now()), base_identity=base,
                    base_identity_sha256=base_hash, protocol=stream.PROTOCOL)
    manifest.pop('error', None)
    report.write_json(manifest_path, manifest)
    loaded = {}
    started = time.perf_counter()
    try:
        for relative, expected in (previous or {}).get('result_files_sha256', {}).items():
            path = output / relative
            if (not path.resolve().is_relative_to(output) or not path.is_file()
                    or path.is_symlink() or report.sha256(path) != expected):
                raise ValueError(f'Resume rejected: saved result hash mismatch: {relative}')
        pilot_root = output / 'data'
        if not (pilot_root / 'manifest.json').exists():
            data.build_pilot(args.source_root, pilot_root)
        dev = data.load_dev_turns(args.dev_cache_dir)
        pilot = data.load_pilot_turns(pilot_root)
        _check_turns(dev, pilot)
        data_hashes = dict(development=_dataset_fingerprint(dev), pilot=_dataset_fingerprint(pilot))
        if previous and previous.get('data_sha256') and previous['data_sha256'] != data_hashes:
            raise ValueError('Resume rejected: normalized data/reference hashes changed')
        manifest.update(data_sha256=data_hashes,
                        ids=dict(development=[t['id'] for t in dev], pilot=[t['id'] for t in pilot]),
                        counts=dict(development=_counts(dev), pilot=_counts(pilot)))
        report.write_json(manifest_path, manifest)
        vad = stream.SileroFrames(args.vad_path)
        if previous and previous.get('vad') and previous['vad'] != vad.metadata:
            raise ValueError('Resume rejected: VAD settings/hash changed')
        manifest['vad'] = vad.metadata
        manifest['models'] = {}
        for name in args.models:
            detector = models.load_detector(name, cache_dir=str(args.model_cache_dir), precision='int8')
            loaded[name] = detector
            metadata = _runtime_model_metadata(detector)
            if previous and previous.get('models', {}).get(name) not in (None, metadata):
                raise ValueError(f'Resume rejected: {name} model settings/hash changed')
            manifest['models'][name] = metadata
            report.write_json(manifest_path, manifest)
        dev_sets, selections = {}, {}
        for name, detector in loaded.items():
            traces, refs, hashes = _trace_set(dev, name, detector, vad, args.cache_dir,
                                             manifest['models'][name], 'development')
            dev_sets[name] = (traces, refs)
            kind = 'timer' if name == 'timer' else 'model'
            selections[name] = dict(model_settings=manifest['models'][name],
                                    dev_trace_hashes=hashes, budgets={})
            for budget in BUDGETS:
                selections[name]['budgets'][f'{budget:.2f}'] = policy.select_policy(traces, refs, kind, budget)
        frozen = dict(run_version=RUN_VERSION, base_identity_sha256=base_hash, data_sha256=data_hashes,
                      vad=vad.metadata, models=selections, selection_split='livekit_dev',
                      selection_rule='premature<=budget; missed<=.05; minimum 4000ms penalized mean delay',
                      dev_ids=[t['id'] for t in dev], pilot_inference_started=False)
        frozen_path = output / 'frozen_policies.json'
        if frozen_path.exists():
            if json.loads(frozen_path.read_text()) != frozen:
                raise ValueError('Resume rejected: frozen development policies/traces changed')
        else:
            report.write_json(frozen_path, frozen)
        frozen_hash = report.sha256(frozen_path)
        manifest.update(frozen_policies_sha256=frozen_hash, phase='pilot')
        report.write_json(manifest_path, manifest)
        rows, pilot_refs = [], None
        for name, detector in loaded.items():
            traces, refs, hashes = _trace_set(pilot, name, detector, vad, args.cache_dir,
                                             manifest['models'][name], 'pilot')
            _export_pilot_traces(output, name, traces, refs, hashes)
            if pilot_refs is not None and refs != pilot_refs:
                raise ValueError('Models received different pilot references')
            pilot_refs = refs
            for split, (split_traces, split_refs) in [('development', dev_sets[name]), ('pilot', (traces, refs))]:
                timing = report.inference_stats(split_traces, name)
                for budget in BUDGETS:
                    selection = selections[name]['budgets'][f'{budget:.2f}']
                    result = policy.evaluate_policy(split_traces, split_refs, selection['policy'])
                    per_turn = [dict(model=name, split=split, budget=budget, dev_feasible=selection['feasible'],
                                     **row, synthetic_silence_s=split_refs[row['id']]['synthetic_silence_s'],
                                     vad_gate_frames=len(split_traces[row['id']]),
                                     score_events=sum(f['is_score_event'] for f in split_traces[row['id']]),
                                     decision_diagnostic=(row['reason'] or ('no_vad_gate_event' if not split_traces[row['id']]
                                                                          else 'no_policy_stop')))
                                for row in result['per_turn']]
                    report.write_csv(output / 'outcomes' / f'{name}-{split}-{int(budget*100):02d}.csv', per_turn)
                    rows.append(dict(model=name, model_label=MODEL_LABELS[name], split=split, budget=budget,
                                     dev_feasible=selection['feasible'], **result['metrics'], **timing,
                                     synthetic_silence_total_s=sum(r['synthetic_silence_s'] for r in split_refs.values())))
        report.write_json(output / 'data.json', dict(data_sha256=data_hashes,
                         development=list(dev_sets[args.models[0]][1].values()), pilot=list(pilot_refs.values())))
        if report.sha256(frozen_path) != frozen_hash:
            raise ValueError('Frozen policies changed during pilot scoring')
        report.write_results(output, rows, manifest, frozen_hash)
        manifest.update(status='complete', phase='complete', updated_at=_now(),
                        current_invocation_elapsed_s=time.perf_counter() - started)
        manifest['result_files_sha256'] = {str(path.relative_to(output)): report.sha256(path)
                                           for path in sorted(output.rglob('*'))
                                           if path.is_file() and path != manifest_path}
        report.write_json(manifest_path, manifest)
        print(f'Complete: {output}', flush=True)
        return manifest
    except Exception as exc:
        manifest.update(status='failed', updated_at=_now(), error=f'{type(exc).__name__}: {exc}',
                        current_invocation_elapsed_s=time.perf_counter() - started)
        report.write_json(manifest_path, manifest)
        raise
    finally:
        for detector in loaded.values():
            close = getattr(detector, 'close', None)
            if callable(close):
                close()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parse(argv)
    try:
        run(args, 'python -m eot_stream.cli ' + shlex.join(argv))
    except Exception as exc:
        print(f'FAILED: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
