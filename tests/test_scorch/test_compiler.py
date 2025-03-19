"""
Main test file that imports tests from the modular test files.
Tests are organized by tensor dimensionality and operation type.
"""

# Import 1D tensor tests
from tests.test_scorch.test_1d_operations import *

# Import 2D tensor tests
from tests.test_scorch.test_2d_operations import *

# Import matrix multiplication tests
from tests.test_scorch.test_matmul_operations import *

# Import higher-dimensional tensor tests
from tests.test_scorch.test_higher_dim_operations import *
