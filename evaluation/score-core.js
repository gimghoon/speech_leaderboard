/* First declaration only; no model execution, network request, or registration. */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.EOTScore = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const VERSION = 'eot-first-stop-v1';
  const EPSILON = 1e-9;
  const own = (object, key) => Object.prototype.hasOwnProperty.call(object, key);
  const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
  const number = value => typeof value === 'number' && Number.isFinite(value);
  const identifier = value => typeof value === 'string' && value.trim().length > 0;
  const splitAliases = { development: 'development', pilot: 'pilot', livekit_dev: 'development', minds_pilot: 'pilot' };

  class ValidationError extends Error {
    constructor(code, path, message) {
      super(`${path}: ${message}`);
      this.name = 'ValidationError';
      this.code = code;
      this.path = path;
    }
  }
  function fail(code, path, message) { throw new ValidationError(code, path, message); }
  function exactFields(value, fields, path) {
    for (const key of Object.keys(value)) {
      if (!fields.includes(key)) fail('UNKNOWN_FIELD', `${path}.${key}`, '지원하지 않는 필드입니다. 제출 형식을 확인하세요.');
    }
    for (const key of fields) {
      if (!own(value, key) || value[key] === undefined) fail('MISSING_FIELD', `${path}.${key}`, '필수 필드가 없습니다. 미선언 시간은 null로 표시하세요.');
    }
  }
  function validateSubmission(value) {
    if (!object(value)) fail('INVALID_SUBMISSION', 'submission', 'JSON 객체가 필요합니다.');
    exactFields(value, ['format_version', 'model_name', 'split', 'budget', 'predictions'], 'submission');
    if (value.format_version !== VERSION) fail('INVALID_SUBMISSION', 'submission.format_version', `${VERSION} 형식이 필요합니다.`);
    if (!identifier(value.model_name)) fail('INVALID_SUBMISSION', 'submission.model_name', '비어 있지 않은 모델 이름이 필요합니다.');
    if (!['development', 'pilot'].includes(value.split)) fail('INVALID_SUBMISSION', 'submission.split', 'development 또는 pilot이 필요합니다.');
    if (!number(value.budget) || ![0.05, 0.1].includes(value.budget)) fail('INVALID_SUBMISSION', 'submission.budget', '숫자 0.05 또는 0.1이 필요합니다.');
    if (!Array.isArray(value.predictions)) fail('INVALID_SUBMISSION', 'submission.predictions', '예측 목록이 필요합니다.');
  }
  function validateReferences(refs, split) {
    if (!Array.isArray(refs) || !refs.length) fail('INVALID_REFERENCES', 'references', '비어 있지 않은 정답 목록이 필요합니다.');
    const byId = new Map();
    let tagged = 0;
    refs.forEach((ref, index) => {
      const at = `references[${index}]`;
      if (!object(ref) || !identifier(ref.id)) fail('INVALID_REFERENCES', at, '문자열 ID가 있는 정답 객체가 필요합니다.');
      if (byId.has(ref.id)) fail('DUPLICATE_REFERENCE_ID', `${at}.id`, `정답 ID가 중복되었습니다: ${ref.id}`);
      if (!number(ref.eot_s) || !number(ref.original_duration_s) || !number(ref.horizon_s) ||
          ref.eot_s < 0 || ref.original_duration_s < ref.eot_s - EPSILON ||
          Math.abs(ref.horizon_s - ref.eot_s - 4) > EPSILON) {
        fail('INVALID_REFERENCES', at, '유한한 EOT, 원본 길이, EOT+4초 관찰 끝이 필요합니다.');
      }
      if (!Array.isArray(ref.hold_spans)) fail('INVALID_REFERENCES', `${at}.hold_spans`, 'HOLD 구간 목록이 필요합니다.');
      ref.hold_spans.forEach((span, holdIndex) => {
        if (!object(span) || !number(span.start) || !number(span.end) ||
            !(0 <= span.start && span.start < span.end && span.end <= ref.horizon_s)) {
          fail('INVALID_REFERENCES', `${at}.hold_spans[${holdIndex}]`, '0 ≤ start < end ≤ horizon_s인 HOLD 구간이 필요합니다.');
        }
      });
      if (own(ref, 'split')) {
        if (typeof ref.split !== 'string' || !own(splitAliases, ref.split)) fail('INVALID_REFERENCES', `${at}.split`, '알 수 없는 정답 분할입니다.');
        if (splitAliases[ref.split] !== split) fail('SPLIT_MISMATCH', `${at}.split`, '제출 분할과 정답 분할이 다릅니다.');
        tagged++;
      }
      byId.set(ref.id, ref);
    });
    if (tagged && tagged !== refs.length) fail('INVALID_REFERENCES', 'references', '정답 분할 표시는 모든 행에 일관되게 있어야 합니다.');
    return { byId, splitVerified: tagged === refs.length };
  }
  function validateAndScore(submission, refs) {
    validateSubmission(submission);
    const references = validateReferences(refs, submission.split);
    const predictions = new Map();
    submission.predictions.forEach((prediction, index) => {
      const at = `submission.predictions[${index}]`;
      if (!object(prediction)) fail('INVALID_SUBMISSION', at, 'ID와 첫 EOT 시간이 있는 예측 객체가 필요합니다.');
      exactFields(prediction, ['id', 'first_eot_s'], at);
      if (!identifier(prediction.id)) fail('INVALID_SUBMISSION', `${at}.id`, '비어 있지 않은 문자열 ID가 필요합니다.');
      if (predictions.has(prediction.id)) fail('DUPLICATE_ID', `${at}.id`, `예측 ID가 중복되었습니다: ${prediction.id}`);
      const ref = references.byId.get(prediction.id);
      if (!ref) fail('UNKNOWN_ID', `${at}.id`, `해당 분할의 정답에 없는 ID입니다: ${prediction.id}`);
      const time = prediction.first_eot_s;
      if (time !== null && (!number(time) || time < 0 || time > ref.horizon_s)) {
        fail('INVALID_TIME', `${at}.first_eot_s`, `null 또는 0~${ref.horizon_s} 범위의 유한한 초 단위 숫자가 필요합니다.`);
      }
      predictions.set(prediction.id, time);
    });
    for (const ref of refs) {
      if (!predictions.has(ref.id)) fail('MISSING_ID', 'submission.predictions', `예측이 빠진 ID입니다: ${ref.id}. 미선언도 null로 포함하세요.`);
    }

    const perTurn = [];
    let premature = 0, missed = 0, correct = 0, eof = 0, holdTurns = 0, holdPremature = 0, holdSpans = 0, delaySum = 0;
    for (const ref of refs) {
      const time = predictions.get(ref.id);
      const hasHold = ref.hold_spans.length > 0;
      let outcome, delay = null;
      if (time === null) { outcome = 'missed'; missed++; }
      else if (time < ref.eot_s - EPSILON) { outcome = 'premature'; premature++; if (hasHold) holdPremature++; }
      else { outcome = 'correct'; correct++; delay = Math.max(0, time - ref.eot_s) * 1000; delaySum += delay; }
      const afterEof = time !== null && time > ref.original_duration_s;
      if (afterEof) eof++;
      if (hasHold) holdTurns++;
      holdSpans += ref.hold_spans.length;
      perTurn.push({ id: ref.id, eot_s: ref.eot_s, decision_s: time, reason: null, outcome,
        delay_ms: delay, after_source_eof: afterEof, has_hold: hasHold,
        first_decision_in_hold: time !== null && ref.hold_spans.some(span => span.start <= time && time < span.end) });
    }
    const count = refs.length;
    return {
      metadata: { format_version: VERSION, model_name: submission.model_name, split: submission.split,
        budget: submission.budget, reference_split_verified: references.splitVerified },
      metrics: { n_turns: count, n_premature: premature, n_missed: missed, n_correct: correct,
        premature_rate: premature / count, miss_rate: missed / count,
        mean_delay_success_ms: correct ? delaySum / correct : null,
        penalized_mean_delay_ms: (delaySum + 4000 * (premature + missed)) / count,
        timeout_rate: null, source_eof_stop_rate: eof / count,
        n_hold_turns: holdTurns, hold_turn_premature_rate: holdTurns ? holdPremature / holdTurns : null,
        n_hold_spans: holdSpans },
      per_turn: perTurn,
    };
  }
  return { validateAndScore, ValidationError };
});
