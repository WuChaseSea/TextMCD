from .mmseg_models import MMSegModel
from .ban_models import BANModel
from .changeclip_models import ChangeCLIP


models = {_.__name__: _ for _ in [
    MMSegModel,
    BANModel,
    ChangeCLIP,
]}
