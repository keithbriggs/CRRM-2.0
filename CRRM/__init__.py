# Keith Briggs 2025-09-25
# This defines the user-level API, while allowing developers to upgrade
# modules in the background.

def get_version():
  'Get the version number of the CRRM module.'
  return '2.0.1'

from .CRRM_core_08 import _Node

from .CRRM_core_08 import CRRM as Simulator
from .CRRM_core_08 import CRRM_parameters as Parameters
from .CRRM_core_08 import Throughput
from .CRRM_core_08 import Antenna_gain
from .CRRM_core_08 import default_UE_locations

from .CRRM_data_02 import CRRM_logger as Logger

from .utilities import *
