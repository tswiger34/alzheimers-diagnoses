"""Image preprocessing logic to get from raw DICOM --> pre-processed image tensors

Steps:
    1. Build `_core.core_image_set` dbt model
    2. Use image IDs from step 1, convert selected images into a workable NIfTI format (`./data/images/NIfTI/`)
    3. Perform image preprocessing (`./data/images/pre_processed/`)
        - GradWarp
        - N4 bias correction
        - Brain Extraction/skull masking
    4. Normalize using within-subject templating (`./data/images/templated/`)
        - Create within-subject longitudinal registration
        - register templates
        - resample/crop
        - Intensity normalization
    5. Split into train/test/val and save model-ready tensors (./data/images/tensors/)
"""
