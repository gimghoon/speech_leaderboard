"""Hash-linked local caches of actual VAD and endpoint inference."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from .stream import PROTOCOL,make_observation,gate_frames,score_frames


def _json(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def _digest(value):return hashlib.sha256(_json(value).encode()).hexdigest()
def _write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(_json(value)+'\n');tmp.replace(path)
def _invalid(value):raise ValueError('Nonfinite cached JSON')
def _read(path):return json.loads(path.read_text(),parse_constant=_invalid)
def _name(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9_.-]+',value):raise ValueError('Unsafe cache namespace')
    return value

def trace_path(cache_root,model_name,split,turn_id):
    token=hashlib.sha256(str(turn_id).encode()).hexdigest()
    return Path(cache_root)/'scores'/_name(model_name)/_name(split)/(token+'.json')


def get_trace(turn,model_name,detector,vad,cache_root,model_metadata):
    audio,ref=make_observation(turn)
    audio_sha=hashlib.sha256(audio.astype('<f4',copy=False).tobytes()).hexdigest()
    directory=Path(__file__).parent
    code_hashes={p:hashlib.sha256((directory/p).read_bytes()).hexdigest() for p in ['stream.py','models.py','data.py'] if (directory/p).exists()}
    common={'protocol':PROTOCOL,'input_pcm_sha256':audio_sha,'reference':ref,'code_hashes':code_hashes,'vad':vad.metadata}
    vad_key=_digest(common);vad_path=trace_path(cache_root,'vad',turn['split'],turn['id'])
    if vad_path.exists() and (item:=_read(vad_path)).get('fingerprint')==vad_key:
        probabilities=item['probabilities']
        if item.get('status')!='complete' or len(probabilities)!=len(audio)//512 or item.get('payload_sha256')!=_digest(probabilities):raise ValueError('Invalid VAD cache')
    else:
        probabilities=vad.infer(audio)
        if len(probabilities)!=len(audio)//512:raise ValueError('Incomplete VAD inference')
        gate_frames(probabilities)
        _write(vad_path,{'id':turn['id'],'status':'complete','fingerprint':vad_key,'metadata':common,'probabilities':probabilities,'payload_sha256':_digest(probabilities)})
    gates=gate_frames(probabilities)
    fingerprint=_digest({'common':common,'model':model_metadata,'model_name':model_name,'vad_payload_sha256':_digest(probabilities)})
    path=trace_path(cache_root,model_name,turn['split'],turn['id'])
    if path.exists() and (item:=_read(path)).get('fingerprint')==fingerprint:
        if item.get('status')!='complete' or item.get('payload_sha256')!=_digest({'frames':item['frames'],'ref':item['ref']}):raise ValueError('Invalid endpoint score cache')
        if item['ref']!=ref or len(item['frames'])!=len(gates):raise ValueError('Cached reference/grid differs')
        return item['frames'],item['ref']
    frames=score_frames(audio,gates,detector)
    _write(path,{'id':turn['id'],'status':'complete','fingerprint':fingerprint,'metadata':{'model':model_metadata,**common},'frames':frames,'ref':ref,'payload_sha256':_digest({'frames':frames,'ref':ref})})
    return frames,ref
