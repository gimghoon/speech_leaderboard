"""Causal common VAD, observed silence clock, and prefix-only endpoint scores."""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import time
import numpy as np

SAMPLE_RATE=16000
FRAME_SAMPLES=512
VAD_SHA256='1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3'
PROTOCOL={'version':'causal-vad-stream-v1','sample_rate':16000,'frame_samples':512,'speech_onset':.5,'speech_offset':.35,'quiet_confirmation_s':.096,'first_score_s':.2,'score_interval_s':.2,'max_model_silence_age_s':3.5,'observation_after_eot_s':4.,'latency_clock':'audio_timeline_excludes_compute','post_source_end':'declared_digital_silence_only','references':'pseudo_Silero_related_not_independent_gold'}


def _audio(value):
    result=np.asarray(value,dtype=np.float32)
    if result.ndim!=1 or not result.size or not np.isfinite(result).all():raise ValueError('Finite nonempty mono audio required')
    return result


class SileroFrames:
    def __init__(self,model_path):
        import onnxruntime as ort
        path=Path(model_path)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=VAD_SHA256:raise ValueError('Silero model hash differs from pinned reference')
        options=ort.SessionOptions();options.inter_op_num_threads=1;options.intra_op_num_threads=1
        self.session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        self.metadata={'model':'snakers4/silero-vad','revision':'066409f3d9db45a30d21dfcbf9147826ffdd7052','sha256':digest,'frame_samples':512,'sample_rate':16000,'providers':self.session.get_providers()}

    def infer(self,audio):
        samples=_audio(audio);state=np.zeros((2,1,128),dtype=np.float32);context=np.zeros((1,64),dtype=np.float32);values=[]
        for start in range(0,len(samples)-FRAME_SAMPLES+1,FRAME_SAMPLES):
            chunk=samples[start:start+FRAME_SAMPLES]
            model_input=np.concatenate((context,chunk.reshape(1,FRAME_SAMPLES)),axis=1)
            output,state=self.session.run(None,{'input':model_input,'state':state,'sr':np.array(SAMPLE_RATE,dtype=np.int64)})
            value=float(np.asarray(output).item())
            if not math.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid VAD probability')
            values.append(value);context=model_input[:,-64:]
        return values


def gate_frames(probabilities):
    probabilities=np.asarray(probabilities,dtype=np.float64)
    if probabilities.ndim!=1 or not np.isfinite(probabilities).all() or ((probabilities<0)|(probabilities>1)).any():raise ValueError('Invalid VAD probabilities')
    ever=False;active=False;quiet_start=None;region=-1;next_score=3200;result=[]
    for index,probability in enumerate(probabilities):
        sample_start=index*FRAME_SAMPLES;sample_end=sample_start+FRAME_SAMPLES
        if probability>=.5:
            ever=True;active=True;quiet_start=None
            continue
        if not ever:continue
        if probability<.35 and quiet_start is None:
            quiet_start=sample_start;region+=1;next_score=3200
        if quiet_start is None:continue
        age_samples=sample_end-quiet_start
        if age_samples>=1536:active=False
        if active or age_samples<3200:continue
        is_score=next_score<=56000 and age_samples>=next_score
        if is_score:next_score+=3200
        result.append({'timestamp_s':sample_end/SAMPLE_RATE,'sample_end':sample_end,'silence_start_s':quiet_start/SAMPLE_RATE,'silence_age_s':age_samples/SAMPLE_RATE,'region_id':region,'speech_probability':float(probability),'is_score_event':bool(is_score)})
    return result


def score_frames(audio,frames,detector):
    samples=_audio(audio);result=[];last_region=None;latest=None;score_timestamp=None;previous_time=-1.
    reset=getattr(detector,'reset',None)
    if callable(reset):reset()
    for frame in frames:
        row=dict(frame);timestamp=float(row['timestamp_s']);end=int(row['sample_end'])
        if not math.isfinite(timestamp) or timestamp<=previous_time or end<=0 or end>len(samples) or abs(end/SAMPLE_RATE-timestamp)>1e-9:raise ValueError('Invalid causal score timestamp')
        previous_time=timestamp
        if row['region_id']!=last_region:
            if not row['is_score_event']:raise ValueError('New silence region requires a fresh score event')
            latest=None;score_timestamp=None;last_region=row['region_id']
        elapsed=0.
        if row['is_score_event'] and detector is not None:
            started=time.perf_counter();value=float(detector.predict(samples[:end].copy()));elapsed=(time.perf_counter()-started)*1000
            if not math.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid model probability')
            latest=value;score_timestamp=timestamp
        row.update(p_eot=latest,score_timestamp_s=score_timestamp,inference_ms=elapsed)
        result.append(row)
    return result


def make_observation(turn):
    samples=_audio(turn['audio']);eot=float(turn['eot_s']);original=float(turn['original_duration_s'])
    if not math.isfinite(eot) or not math.isfinite(original) or eot<0 or eot>original or abs(original-len(samples)/SAMPLE_RATE)>1/SAMPLE_RATE:raise ValueError('Invalid original reference timeline')
    horizon=eot+4.;count=math.floor(horizon*SAMPLE_RATE+1e-7)
    buffer=np.zeros(count,dtype=np.float32);observed=min(len(samples),count);buffer[:observed]=samples[:observed]
    ref={key:value for key,value in turn.items() if key!='audio'}
    ref.update(horizon_s=horizon,synthetic_silence_s=max(0.,(count-len(samples))/SAMPLE_RATE),observed_original_s=observed/SAMPLE_RATE,observed_sample_count=count,reference_label_type=turn.get('label_type','unknown'))
    return buffer,ref
