from .one_vs_one import OVO
from .one_vs_rest import OVR
from .multinomial import Multinomial

KERNELS = {'OVR': OVR(),\
            'OVO': OVO(),\
            'MULTINOMIAL': Multinomial()}