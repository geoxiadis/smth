from .selfsimilarity import SelfSimilarityDeficit, SelfSimilarityDeficit2020
from .hybridinduction import HybridInduction
from .rankinehalfbody import RankineHalfBody
from .vortexcylinder import VortexCylinder
from .vortexdipole import VortexDipole
from .rathmann import Rathmann
try:
    from .wakeblaster import WakeBlaster
except ImportError:
    pass
from .noj import NOJDeficit, NOJLocalDeficit, TurboNOJDeficit
from .gaussian import BastankhahGaussianDeficit, IEA37SimpleBastankhahGaussianDeficit, \
    NiayifarGaussianDeficit, ZongGaussianDeficit, CarbajofuertesGaussianDeficit, TurboGaussianDeficit
from .fuga import FugaDeficit, FugaYawDeficit, FugaMultiLUTDeficit
from .gcl import GCLDeficit
from .no_wake import NoWakeDeficit
from .rans_lut import RANSLUTDeficit, RANSLUT
