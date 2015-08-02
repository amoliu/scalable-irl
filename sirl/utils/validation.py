r"""
Validation tools, mainly borrowed from scikit-learn project
"""

from sklearn.utils import check_random_state
from sklearn.utils import assert_all_finite
from sklearn.utils.validation import as_float_array
from sklearn.utils import check_array
from sklearn.utils import column_or_1d

from numpy import asarray


__all__ = [
    'check_random_state',
    'assert_all_finite',
    'as_float_array',
    'check_array',
    'column_or_1d',
    'asarray'
]
