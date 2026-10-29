import json
import numpy as np
import pytest
from eot_stream.cache import get_trace,trace_path

class Vad:
    metadata={'sha256':'test'}
    def __init__(self):self.calls=0
    def infer(self,a):
        self.calls+=1
        return [.9]*8+[.1]*(len(a)//512-8)
class Detector:
    def __init__(self):self.calls=0
    def predict(self,a):self.calls+=1;return .6

def turn():
    return dict(id='x',split='minds_pilot',audio=np.zeros(16000,dtype=np.float32),eot_s=.5,hold_spans=[],original_duration_s=1.,source_audio_sha256='original',label_type='pseudo')

def test_cached_trace_reuses_real_scores_and_vad_without_reinference(tmp_path):
    vad=Vad();d=Detector();a,ref=get_trace(turn(),'mini',d,vad,tmp_path,{'v':1});n=d.calls
    b,ref2=get_trace(turn(),'mini',d,vad,tmp_path,{'v':1})
    assert a==b and ref==ref2 and n>0 and d.calls==n and vad.calls==1

def test_changed_model_metadata_invalidates_score_cache_not_vad(tmp_path):
    vad=Vad();d=Detector();get_trace(turn(),'mini',d,vad,tmp_path,{'v':1});n=d.calls
    get_trace(turn(),'mini',d,vad,tmp_path,{'v':2})
    assert d.calls>n and vad.calls==1

def test_malformed_cached_score_fails_closed(tmp_path):
    vad=Vad();get_trace(turn(),'mini',Detector(),vad,tmp_path,{})
    path=trace_path(tmp_path,'mini','minds_pilot','x');item=json.loads(path.read_text());item['frames'][0]['p_eot']=float('nan');path.write_text(json.dumps(item))
    with pytest.raises(ValueError):get_trace(turn(),'mini',Detector(),vad,tmp_path,{})
