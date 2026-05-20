import nibabel as nib  # noqa
import os
import json
from .models import NiFTiMetadata  # noqa
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
IMAGE_PATH = Path(os.getenv("IMAGE_PATH", "data/images"))
NIFTI_IMAGE_PATH = IMAGE_PATH / "nifti"
TENSOR_OUTPUT = IMAGE_PATH / "tensors"


def load_input_image(img_folder: str):

    with open(img_folder, "r", encoding="utf-8") as f:
        metadata = NiFTiMetadata.model_validate_json(json_data=json.load(f))
