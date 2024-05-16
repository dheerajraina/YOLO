class AssertionFailedError(Exception):
    pass


def custom_assert(condition, message="Assertion failed"):
    if not condition:
        raise AssertionFailedError(message)
