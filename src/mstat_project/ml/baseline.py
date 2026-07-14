"""Baseline model training and evaluation for the MSTAT project.

Uses the last MRI scan before event/censoring to predict survival.

"""

import os
from pathlib import Path

import polars as pl
import torch
import torch.nn as nn
from torchvision.models import ResNet101_Weights, resnet101

from src.mstat_project.utils import get_db_engine

TENSOR_PATH = Path(os.getenv("IMAGES_PATH", "./images")) / "tensors"


class SingleImageSurvivalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, 1)

    def forward(self, x):
        # x shape: [batch, channels, depth, height, width]
        return self.encoder(x).squeeze(-1)


def cox_ph_loss(risk, time, event):
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order]

    log_cumsum_hazard = torch.logcumsumexp(risk, dim=0)
    loss = -torch.sum((risk - log_cumsum_hazard) * event) / event.sum().clamp_min(1)
    return loss


def get_last_scan() -> pl.DataFrame:
    with get_db_engine().connect() as conn:
        query = """
            SELECT 
                imgs.image_id,
                imgs.ptid,
                imgs.image_date,
                imgs.baseline_diagnosis,
                imgs.final_diagnosis,
                imgs.is_censored,
                imgs.months_to_ad_from_image AS months_to_event,
                tts.split AS train_test_split
            FROM _core.core_image_set  AS imgs
            INNER JOIN _raw.train_test_split AS tts
                ON tts.ptid = imgs.ptid
            WHERE image_date = (
                SELECT MAX(subimg.image_date)
                FROM _core.core_image_set AS subimg
                WHERE subimg.ptid = imgs.ptid
            )
        """
        df = pl.read_database(query, conn)
    return df


def validate_last_scan(df: pl.DataFrame) -> bool:
    n_ptid_imgs = df.group_by("ptid").agg(pl.count("image_id").alias("n_images"))
    img_cnt_max: int = n_ptid_imgs.select(pl.col("n_images")).max().item()
    return img_cnt_max == 1


def df_transform(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("final_diagnosis").eq("AD").alias("is_ad"),
    )


if __name__ == "__main__":
    df = get_last_scan()
    print(df)
    is_valid = validate_last_scan(df)
    print(f"Last scan validation result: {is_valid}")
