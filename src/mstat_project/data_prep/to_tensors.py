import nibabel as nib  # noqa
import os
import json
from .models import NiFTiMetadata  # noqa

NIFTI_IMAGE_PATH = os.getenv("NIFTI_IMAGE_PATH", "data/images/nifti")
TENSOR_OUTPUT_PATH = os.getenv("TENSOR_IMAGE_PATH", "data/images/nifti")


def load_input_image(img_folder: str):

    with open(img_folder, "r", encoding="utf-8") as f:
        metadata = NiFTiMetadata.model_validate_json(json_data=json.load(f))
