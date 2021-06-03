from ..musicSide.DatasetMusic2emotion.tools.utils import ms2samples, samples2ms

__SAMPLE_RATE = 44100
# dumb test
def test_samples2ms():
    a = samples2ms(samples=22050, sample_rate=__SAMPLE_RATE)
    assert a == 500

def test_ms2samples():
    a = ms2samples(milliseconds=500, sample_rate=__SAMPLE_RATE)
    assert a == 22050
