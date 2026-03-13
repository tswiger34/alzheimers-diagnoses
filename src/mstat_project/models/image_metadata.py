from typing import Literal

import numpy as np
import numpy.typing as npt
from ltsa.models import SurvivalAnalysisMetadata


class ImageMetadata(SurvivalAnalysisMetadata):
    id: str
    censorship: npt.NDArray[np.float64]
    obs_times: npt.NDArray[np.float64]
    event_time: npt.NDArray[np.float64]
    label: npt.NDArray[np.float64]
    image: npt.NDArray[np.float64]
    sequence_group_broad: Literal["t1_3d_structural", "localizer_or_scout", "other"]
    sequence_group_narrow: Literal["mprage", "ir_spgr", "flair", "localizer", "unknown_other"]
    orientation: Literal["sagittal", "axial", "coronal", "unknown"]
    accelerated: Literal["yes", "no", "unknown"]
    is_repeat: bool
