from .base_dataset import BaseData
from .seg_dataset import SegData
from .change_dataset import ChangeData
from .semantic_change_dataset import SChangeData
from .changeclip_dataset import ChangeClipData
from .pred_dataset import PredictData, PredictIOData, VRTData, PredictListData, SmallPredictData, SmallPredictChangeClipData, SmallPredictTextSCDData, SmallPredictMGCRData, SmallPredictMMChangeData
from .sam_opti_dataset import SamOptiData
from .cpmask_dataset import CPMaskData
from .textmask_dataset import TextMaskData
from .alphaclip_dataset import AlphaClipData, AlphaChangeData, AlphaTextData, RSAlphaClipData, RSAlphaMaskData
from .changealphaclip_dataset import ChangeAlphaClipData
from .textscd_dataset import TextSCDData
from .copy_dataset import CopyData
from .classification_dataset import FolderClassificationData
from .mgcr_dataset import MGCRData
from .mmchange_dataset import MMChangeData

datasets = {_.__name__: _ for _ in [
    BaseData,
    ChangeData,
    SChangeData,
    ChangeClipData,
    TextSCDData,
    SegData,
    PredictData,
    PredictIOData,
    VRTData,
    PredictListData,
    SmallPredictData,
    SmallPredictChangeClipData,
    SmallPredictTextSCDData,
    SmallPredictMGCRData,
    SmallPredictMMChangeData,
    SamOptiData,
    CPMaskData,
    AlphaClipData,
    RSAlphaMaskData,
    AlphaChangeData,
    AlphaTextData,
    ChangeAlphaClipData,
    RSAlphaClipData,
    TextMaskData,
    CopyData,
    FolderClassificationData,
    MGCRData,
    MMChangeData
]}
