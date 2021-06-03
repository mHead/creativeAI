from ..musicSide.DatasetMusic2emotion.tools.utils import ms2samples, samples2ms


# dumb test
def test_samples2ms():
    a = samples2ms(samples=22050)
    assert a == 500

def test_ms2samples():
    a = ms2samples(sample_rate=500)
    assert a == 22050
