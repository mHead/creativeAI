from ..DatasetMusic2emotion.tools.utils import samples2ms

# dumb test
def test_samples2ms():
    a = samples2ms(22050)
    assert a == 500

