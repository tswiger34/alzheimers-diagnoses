# Initial Release Roadmap

This section outlines the changes made to make the original  original repo from Holste, Gregory, et al. into a more user friendly and generalizable package. This file is organized by module in the original repo. The primary purpose of this is to help organize the steps in this process

## Models

### Added

- `ResNetEncoder`: Custom ResNet18 image encoder using the new interface
- `SwinEncoder`: Custom SwinTransformer image encoder using the new interface

### Removed

- `LearnedPositionalEncoding`
- `PositionalEncoding` (renamed to `TemporalPositionalEncoding`)
- `ImageSurvivalModel` (renamed to `BaselineImageSurvivalModel`)
- `create_model`

### Changed

- `PositionalEncoding`:
  - Moved to `src/ltsa/tpe.py`
  - Rename: `PositionalEncoding` --> `TemporalPositionalEncoding`
  - Optimizations: Get rid of for loop and support moving loss function to device
- `ImageSurvivalModel`:
  - Rename: `ImageSurvivalModel` --> `BaselineImageSurvivalModel`
- `ImageEncoder`:
  - Moved to `src/ltsa/image_encoder.py`
  - Now the interface for creating custom image encoder blocks
- `LTSA`:
  - Moved to `src/ltsa/ltsa_model.py`
  - Within :method:``__init__``:
    - Removed :param:``args`` and replaced it with explicity key word args
    - Renamed following attributes:
      - encoder --> img_encoder
      - args.max_seq_len --> max_seq_len
    - Added :attr:``device``
    - Only supports legacy :attr:``args.attn_map`` == True
    - Only supports legacy :attr:``args.step_ahead`` == True
    - Only supports legacy :attr:``args.tpe`` == True
    - Removed legacy support for:
      - :attr:``args.learned_pe``
      - :attr:``args.amd_sev_enc``
  - Within :method:``forward``:
    - Output is now a defined dataclass
    - Only supports one output structure
    - Moved src_key_padding_mask compute to it's own method
    - Cleaned up comments

## Losses

### Added

- :method:``cox_surv_loss``
- :param:``**kwargs`` to all loss functions args

### Changed

- Moved :class:``CoxSurvLoss`` computation logic to :method:``cox_surv_loss``
- Removed unused params from :method:``nll_surv_loss``
- Variable naming in functions for clarity
- Used bounded methods for tensor transformations, e.g. -c --> c.neg()

## Train

The train module was very closely tied to the original usecase, so no functionality was used

### Removed

- :module:``train``

## Datasets

The datasets module was very closely tied to the original usecase, so no functionality was used

### Removed

- :module:``datasets``

## Utils

The utils module was very closely tied to the original usecase, so no functionality was used

### Removed

- :module:``utils``

## Transformers

### Added

- :module:``transformer``
- Submodules:
  - :module:``transformer_encoder``
  - :module:``transformer_decoder``
  - :module:``transformer_utils``
  - :module:``transformer``

### Removed

- :module:``transformers``

### Changed

Besides improved documentation and refactoring one large module into several smaller ones, no functional changes were made in order minimize differences in the pytorch patch that could potentially be coming up
