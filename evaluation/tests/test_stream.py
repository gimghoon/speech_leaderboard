import numpy as np
import pytest
from eot_stream.stream import gate_frames, score_frames, make_observation, SileroFrames


def test_leading_quiet_cannot_end_a_turn():
    assert gate_frames([0.0]*50)==[]


def test_first_score_waits_for_200ms_with_anchored_clock():
    frames=gate_frames([.8]*8+[.1]*24)
    assert frames[0]['timestamp_s']==pytest.approx(.480)
    assert frames[0]['silence_age_s']==pytest.approx(.224)
    assert [round(f['timestamp_s'],3) for f in frames if f['is_score_event']]==[.480,.672,.864]


def test_resume_resets_region_and_hysteresis_preserves_pending_quiet():
    frames=gate_frames([.8]*4+[.1]*7+[.4]*7+[.8]*4+[.1]*15)
    regions=sorted({f['region_id'] for f in frames})
    assert len(regions)==2
    second=[f for f in frames if f['region_id']==regions[1]]
    assert second[0]['is_score_event'] is True
    assert second[0]['silence_age_s']==pytest.approx(.224)


@pytest.mark.parametrize('bad',[[np.nan],[1.1],[-.1],[[.2]]])
def test_invalid_vad_probabilities_rejected(bad):
    with pytest.raises(ValueError):gate_frames(bad)


class PrefixDetector:
    def __init__(self):self.prefixes=[]
    def predict(self,audio):
        self.prefixes.append(audio.copy());audio[:]=99
        return .75


def gate():
    return [dict(timestamp_s=.25,sample_end=4000,silence_start_s=0.,silence_age_s=.25,region_id=0,is_score_event=True),dict(timestamp_s=.282,sample_end=4512,silence_start_s=0.,silence_age_s=.282,region_id=0,is_score_event=False),dict(timestamp_s=.45,sample_end=7200,silence_start_s=0.,silence_age_s=.45,region_id=0,is_score_event=True)]


def test_only_prefix_is_sent_and_model_cannot_mutate_source():
    audio=np.arange(8000,dtype=np.float32)/8000;original=audio.copy();detector=PrefixDetector()
    frames=score_frames(audio,gate(),detector)
    assert [len(x) for x in detector.prefixes]==[4000,7200]
    assert np.array_equal(detector.prefixes[0],original[:4000])
    assert np.array_equal(detector.prefixes[1],original[:7200])
    assert np.array_equal(audio,original)
    assert [f['p_eot'] for f in frames]==[.75,.75,.75]
    assert frames[1]['score_timestamp_s']==.25


def test_future_waveform_does_not_change_earlier_prefix():
    audio=np.zeros(8000,dtype=np.float32);changed=audio.copy();changed[5000:]=1
    a=PrefixDetector();b=PrefixDetector();score_frames(audio,gate(),a);score_frames(changed,gate(),b)
    assert np.array_equal(a.prefixes[0],b.prefixes[0])
    assert not np.array_equal(a.prefixes[1],b.prefixes[1])


def test_stale_score_not_carried_to_a_new_region():
    frames=gate();frames[1]['region_id']=1
    with pytest.raises(ValueError):score_frames(np.zeros(8000,dtype=np.float32),frames,PrefixDetector())


@pytest.mark.parametrize('score',[np.nan,1.01,-.01])
def test_invalid_scores_fail_closed(score):
    class Detector:
        def predict(self,a):return score
    with pytest.raises(ValueError):score_frames(np.zeros(8000,dtype=np.float32),gate(),Detector())


def test_future_score_time_rejected():
    with pytest.raises(ValueError):score_frames(np.zeros(3000,dtype=np.float32),gate(),PrefixDetector())


def test_controlled_tail_is_declared_and_original_unchanged():
    audio=np.ones(48000,dtype=np.float32)
    observed,ref=make_observation(dict(id='x',audio=audio,eot_s=2.,hold_spans=[],original_duration_s=3.))
    assert len(observed)==96000
    assert np.array_equal(observed[:48000],audio) and not observed[48000:].any()
    assert ref['synthetic_silence_s']==3. and ref['horizon_s']==6.
    assert audio.all()


def test_recorded_long_tail_is_cropped_without_silence_extension():
    observed,ref=make_observation(dict(id='x',audio=np.ones(128000,dtype=np.float32),eot_s=2.,hold_spans=[],original_duration_s=8.))
    assert len(observed)==96000 and observed.all()
    assert ref['synthetic_silence_s']==0


def test_silero_uses_complete_frames_and_resets_state_per_turn():
    vad=SileroFrames.__new__(SileroFrames)
    class Session:
        def __init__(self):self.seen=[]
        def run(self,output,inputs):
            self.seen.append(inputs)
            return np.array([[.7]],dtype=np.float32),inputs['state']+1
    vad.session=Session();result=vad.infer(np.arange(1025,dtype=np.float32))
    assert len(result)==2 and vad.session.seen[0]['input'].shape==(1,576)
    assert np.array_equal(vad.session.seen[1]['input'][0,:64],np.arange(448,512,dtype=np.float32))
    assert vad.session.seen[1]['state'].sum()==256
    vad.infer(np.ones(512,dtype=np.float32))
    assert vad.session.seen[-1]['state'].sum()==0
