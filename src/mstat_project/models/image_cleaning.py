from dataclasses import dataclass
from typing import Any

import polars as pl


@dataclass(slots=True)
class ImageCleaningStepOutput:
    df: pl.DataFrame
    metadata: dict[str, Any]
