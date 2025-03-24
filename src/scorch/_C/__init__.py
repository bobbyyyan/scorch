# This file imports the compiled C++ extension
try:
    from scorch._C.ops import *
except ImportError:
    # This allows debugging if the extension isn't properly built
    import warnings
    warnings.warn("Could not import C++ extension for scorch. Some functionality will be unavailable.")
