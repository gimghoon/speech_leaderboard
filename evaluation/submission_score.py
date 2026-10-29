#!/usr/bin/env python3
"""첫 EOT 예측을 검증하고 평가하는 독립 표준 라이브러리 도구."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

VERSION = 'eot-first-stop-v1'
EPSILON = 1e-9
SPLITS = {'development': 'development', 'pilot': 'pilot', 'livekit_dev': 'development', 'minds_pilot': 'pilot'}


class ValidationError(ValueError):
    def __init__(self, code, path, message):
        super().__init__(f'{path}: {message}')
        self.code = code
        self.path = path


def _fail(code, path, message):
    raise ValidationError(code, path, message)


def _number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _identifier(value):
    return isinstance(value, str) and bool(value.strip())


def _exact_fields(value, fields, path):
    for key in value:
        if key not in fields:
            _fail('UNKNOWN_FIELD', f'{path}.{key}', '지원하지 않는 필드입니다. 제출 형식을 확인하세요.')
    for key in fields:
        if key not in value:
            _fail('MISSING_FIELD', f'{path}.{key}', '필수 필드가 없습니다. 미선언 시간은 null로 표시하세요.')


def _validate_submission(value):
    if not isinstance(value, dict):
        _fail('INVALID_SUBMISSION', 'submission', 'JSON 객체가 필요합니다.')
    _exact_fields(value, ('format_version', 'model_name', 'split', 'budget', 'predictions'), 'submission')
    if value['format_version'] != VERSION:
        _fail('INVALID_SUBMISSION', 'submission.format_version', f'{VERSION} 형식이 필요합니다.')
    if not _identifier(value['model_name']):
        _fail('INVALID_SUBMISSION', 'submission.model_name', '비어 있지 않은 모델 이름이 필요합니다.')
    if value['split'] not in ('development', 'pilot'):
        _fail('INVALID_SUBMISSION', 'submission.split', 'development 또는 pilot이 필요합니다.')
    if not _number(value['budget']) or value['budget'] not in (.05, .1):
        _fail('INVALID_SUBMISSION', 'submission.budget', '숫자 0.05 또는 0.1이 필요합니다.')
    if not isinstance(value['predictions'], list):
        _fail('INVALID_SUBMISSION', 'submission.predictions', '예측 목록이 필요합니다.')


def _validate_references(refs, split):
    if not isinstance(refs, list) or not refs:
        _fail('INVALID_REFERENCES', 'references', '비어 있지 않은 정답 목록이 필요합니다.')
    by_id = {}
    tagged = 0
    for index, ref in enumerate(refs):
        at = f'references[{index}]'
        if not isinstance(ref, dict) or not _identifier(ref.get('id')):
            _fail('INVALID_REFERENCES', at, '문자열 ID가 있는 정답 객체가 필요합니다.')
        if ref['id'] in by_id:
            _fail('DUPLICATE_REFERENCE_ID', f'{at}.id', f"정답 ID가 중복되었습니다: {ref['id']}")
        if (not all(_number(ref.get(key)) for key in ('eot_s', 'original_duration_s', 'horizon_s'))
                or ref['eot_s'] < 0 or ref['original_duration_s'] < ref['eot_s'] - EPSILON
                or abs(ref['horizon_s'] - ref['eot_s'] - 4) > EPSILON):
            _fail('INVALID_REFERENCES', at, '유한한 EOT, 원본 길이, EOT+4초 관찰 끝이 필요합니다.')
        if not isinstance(ref.get('hold_spans'), list):
            _fail('INVALID_REFERENCES', f'{at}.hold_spans', 'HOLD 구간 목록이 필요합니다.')
        for hold_index, span in enumerate(ref['hold_spans']):
            if (not isinstance(span, dict) or not _number(span.get('start')) or not _number(span.get('end'))
                    or not 0 <= span['start'] < span['end'] <= ref['horizon_s']):
                _fail('INVALID_REFERENCES', f'{at}.hold_spans[{hold_index}]', '0 ≤ start < end ≤ horizon_s인 HOLD 구간이 필요합니다.')
        if 'split' in ref:
            if not isinstance(ref['split'], str) or ref['split'] not in SPLITS:
                _fail('INVALID_REFERENCES', f'{at}.split', '알 수 없는 정답 분할입니다.')
            if SPLITS[ref['split']] != split:
                _fail('SPLIT_MISMATCH', f'{at}.split', '제출 분할과 정답 분할이 다릅니다.')
            tagged += 1
        by_id[ref['id']] = ref
    if tagged and tagged != len(refs):
        _fail('INVALID_REFERENCES', 'references', '정답 분할 표시는 모든 행에 일관되게 있어야 합니다.')
    return by_id, tagged == len(refs)


def validate_and_score(submission, refs):
    """Validate full coverage, then return canonical metrics and reference-order rows.

    Prediction-only inputs cannot establish a timeout reason, causal inference,
    model compute, or independent development tuning. Timeout metadata is null.
    """
    _validate_submission(submission)
    references, split_verified = _validate_references(refs, submission['split'])
    predictions = {}
    for index, prediction in enumerate(submission['predictions']):
        at = f'submission.predictions[{index}]'
        if not isinstance(prediction, dict):
            _fail('INVALID_SUBMISSION', at, 'ID와 첫 EOT 시간이 있는 예측 객체가 필요합니다.')
        _exact_fields(prediction, ('id', 'first_eot_s'), at)
        if not _identifier(prediction['id']):
            _fail('INVALID_SUBMISSION', f'{at}.id', '비어 있지 않은 문자열 ID가 필요합니다.')
        identifier = prediction['id']
        if identifier in predictions:
            _fail('DUPLICATE_ID', f'{at}.id', f'예측 ID가 중복되었습니다: {identifier}')
        if identifier not in references:
            _fail('UNKNOWN_ID', f'{at}.id', f'해당 분할의 정답에 없는 ID입니다: {identifier}')
        time = prediction['first_eot_s']
        if time is not None and (not _number(time) or time < 0 or time > references[identifier]['horizon_s']):
            _fail('INVALID_TIME', f'{at}.first_eot_s', f"null 또는 0~{references[identifier]['horizon_s']} 범위의 유한한 초 단위 숫자가 필요합니다.")
        predictions[identifier] = time
    for ref in refs:
        if ref['id'] not in predictions:
            _fail('MISSING_ID', 'submission.predictions', f"예측이 빠진 ID입니다: {ref['id']}. 미선언도 null로 포함하세요.")

    per_turn = []
    premature = missed = correct = eof = hold_turns = hold_premature = hold_spans = 0
    delay_sum = 0.0  # Explicit accumulation gives the same arithmetic order as JS.
    for ref in refs:
        time = predictions[ref['id']]
        has_hold = bool(ref['hold_spans'])
        delay = None
        if time is None:
            outcome = 'missed'
            missed += 1
        elif time < ref['eot_s'] - EPSILON:
            outcome = 'premature'
            premature += 1
            hold_premature += has_hold
        else:
            outcome = 'correct'
            correct += 1
            delay = max(0.0, time - ref['eot_s']) * 1000
            delay_sum += delay
        after_eof = time is not None and time > ref['original_duration_s']
        eof += after_eof
        hold_turns += has_hold
        hold_spans += len(ref['hold_spans'])
        per_turn.append(dict(id=ref['id'], eot_s=ref['eot_s'], decision_s=time, reason=None,
                             outcome=outcome, delay_ms=delay, after_source_eof=after_eof,
                             has_hold=has_hold, first_decision_in_hold=time is not None and any(
                                 span['start'] <= time < span['end'] for span in ref['hold_spans'])))
    count = len(refs)
    return dict(
        metadata=dict(format_version=VERSION, model_name=submission['model_name'], split=submission['split'],
                      budget=submission['budget'], reference_split_verified=split_verified),
        metrics=dict(n_turns=count, n_premature=premature, n_missed=missed, n_correct=correct,
                     premature_rate=premature / count, miss_rate=missed / count,
                     mean_delay_success_ms=delay_sum / correct if correct else None,
                     penalized_mean_delay_ms=(delay_sum + 4000 * (premature + missed)) / count,
                     timeout_rate=None, source_eof_stop_rate=eof / count,
                     n_hold_turns=hold_turns, hold_turn_premature_rate=hold_premature / hold_turns if hold_turns else None,
                     n_hold_spans=hold_spans),
        per_turn=per_turn)


def _strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail('INVALID_JSON', 'json', f'객체 필드가 중복되었습니다: {key}')
        result[key] = value
    return result


def _invalid_constant(value):
    _fail('INVALID_JSON', 'json', f'표준 JSON 숫자가 아닙니다: {value}')


def _read_json(path):
    try:
        content = sys.stdin.read() if path == '-' else Path(path).read_text(encoding='utf-8-sig')
        return json.loads(content, object_pairs_hook=_strict_pairs, parse_constant=_invalid_constant)
    except json.JSONDecodeError as error:
        _fail('INVALID_JSON', path, f'JSON을 읽을 수 없습니다: {error.msg} (행 {error.lineno})')


def _select_references(source, split):
    if isinstance(source, list):
        return source
    if not isinstance(source, dict):
        _fail('INVALID_REFERENCES', 'references', '분할별 정답 객체 또는 정답 목록이 필요합니다.')
    # Both original results/data.json and site-data.json are explicit split maps.
    candidates = source.get('refs', source)
    if not isinstance(candidates, dict) or split not in candidates or not isinstance(candidates[split], list):
        _fail('INVALID_REFERENCES', 'references', f'{split} 정답 목록을 찾을 수 없습니다.')
    if 'split' in source and source['split'] != split:
        _fail('SPLIT_MISMATCH', 'references.split', '정답 파일 메타데이터와 제출 분할이 다릅니다.')
    result = []
    for index, ref in enumerate(candidates[split]):
        if not isinstance(ref, dict):
            _fail('INVALID_REFERENCES', f'references[{index}]', '정답 객체가 필요합니다.')
        # Do not overwrite contradictory row metadata while binding untagged rows.
        if 'split' in ref and (not isinstance(ref['split'], str) or SPLITS.get(ref['split']) != split):
            _fail('SPLIT_MISMATCH', f'references[{index}].split', '정답 파일의 분할과 행의 분할이 다릅니다.')
        result.append(dict(ref, split=split))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='첫 EOT 예측 JSON을 로컬에서 검증하고 점수를 계산합니다.')
    parser.add_argument('--predictions', required=True, help='예측 JSON 파일; - 는 표준 입력')
    parser.add_argument('--references', required=True, help='원본 data.json, site-data.json 또는 정답 목록 JSON')
    parser.add_argument('--output', required=True, help='결과 JSON 경로; - 는 표준 출력 (CSV 생략)')
    args = parser.parse_args(argv)
    try:
        if args.predictions == '-' and args.references == '-':
            _fail('INVALID_ARGUMENT', 'arguments', '예측과 정답을 동시에 표준 입력으로 읽을 수 없습니다.')
        submission = _read_json(args.predictions)
        _validate_submission(submission)
        references = _select_references(_read_json(args.references), submission['split'])
        result = validate_and_score(submission, references)
        content = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + '\n'
        if args.output == '-':
            sys.stdout.write(content)
        else:
            target = Path(args.output)
            csv_target = target.with_name(target.stem + '.per_turn.csv')
            if any(Path(path).resolve() in (target.resolve(), csv_target.resolve())
                   for path in (args.predictions, args.references) if path != '-'):
                _fail('INVALID_ARGUMENT', 'output', '결과 경로가 입력 파일을 덮어쓸 수 없습니다.')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding='utf-8')
            with csv_target.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(result['per_turn'][0]))
                writer.writeheader()
                writer.writerows(result['per_turn'])
        return 0
    except ValidationError as error:
        sys.stderr.write(json.dumps(dict(error=dict(code=error.code, path=error.path, message=str(error))), ensure_ascii=False) + '\n')
        return 2
    except (OSError, UnicodeError) as error:
        sys.stderr.write(json.dumps(dict(error=dict(code='FILE_ERROR', path='file', message=str(error))), ensure_ascii=False) + '\n')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
