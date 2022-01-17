from skimage._shared.testing import setup_test, teardown_test


def pytest_runtest_setup(item):
    setup_test()


def pytest_runtest_teardown(item):
    teardown_test()
