from .version_requirements import is_installed

has_mpl = is_installed("matplotlib", ">=3.0.3")
if has_mpl:
    try:
        # will fail with
        #    ImportError: Failed to import any qt binding
        # if only matplotlib-base is installed
        from matplotlib import pyplot  # noqa
    except ImportError:
        has_mpl = False
