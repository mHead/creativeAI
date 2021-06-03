from code_root.musicSide.DatasetMusic2emotion.tools.utils import samples2ms, ms2samples


# dumb test
def test_samples2ms():
    a = samples2ms(22050)
    assert a == 500

def test_ms2samples():
    a = ms2samples(500)
    assert a == 22050
