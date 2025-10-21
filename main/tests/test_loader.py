from src.data.loader import loader

#example
def test_loader():
    loader = loader()
    im,l = next(iter(loader))
    assert im[0].shape ==3
