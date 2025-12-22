from .mmseg_models import MMSegModel
from .ban_models import BANModel
from .changeclip_models import ChangeCLIP
from .clipformer_models import CLIPFormer
from .compare_models import CompareModel


models = {_.__name__: _ for _ in [
    MMSegModel,
    BANModel,
    ChangeCLIP,
    CLIPFormer,
    CompareModel
]}
