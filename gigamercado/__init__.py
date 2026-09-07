from __future__ import absolute_import

from . import burntiles, uniontiles
from . import edge_finder as edgetiles
from ._accel import HAS_NUMBA
from ._dispatch import HAS_GPU, backend_name
