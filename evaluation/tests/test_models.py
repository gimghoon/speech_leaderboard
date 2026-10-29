import numpy as np
import pytest

from eot_stream.models import smart_window, mini_pcm, SmartTurn, LiveKitMini


def test_smart_turn_uses_recent_eight_seconds_with_leading_padding():
    assert np.array_equal(smart_window(np.array([0.2, 0.3], dtype=np.float32))[-2:], [np.float32(0.2), np.float32(0.3)])
    assert not smart_window(np.array([0.2, 0.3], dtype=np.float32))[:-2].any()
    long = np.arange(160000, dtype=np.float32)
    assert np.array_equal(smart_window(long), long[-128000:])


def test_mini_takes_recent_audio_and_clips_pcm_without_wrapping():
    source = np.zeros(32000, dtype=np.float32)
    source[-4:] = [-2.0, -1.0, 1.0, 2.0]
    actual = mini_pcm(source)
    assert len(actual) == 19200
    assert actual.dtype == np.int16
    assert actual.flags.c_contiguous
    assert actual[-4:].tolist() == [-32767, -32767, 32767, 32767]


def test_model_objects_do_not_load_weights_on_import():
    # Import is used by --help and timer-only runs on machines without model extras.
    assert callable(SmartTurn)
    assert callable(LiveKitMini)
