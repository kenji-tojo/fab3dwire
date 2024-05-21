from . import _m


def is_available() -> bool:
    return _m.cuda_is_available()



